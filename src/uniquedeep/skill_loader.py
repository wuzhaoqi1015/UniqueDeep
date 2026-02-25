#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File: src/uniquedeep/skill_loader.py
@Time: 2026/02/24
@Author: GeorgeWu
@Description: 技能发现和加载机制（Level 1 & 2），管理技能元数据和内容。
'''

"""
Skills 发现和加载器

演示 Skills 三层加载机制的核心实现：
- Level 1: scan_skills() - 扫描并加载所有 Skills 元数据到 system prompt
- Level 2: load_skill(skill_name: str) - 根据skill name加载指定 Skill 的详细指令（只返回 instructions - skill.md文档)）
- Level 3: 由 bash tool 执行脚本（见 tools.py），大模型从指令中自己发现脚本

核心设计理念：
    让大模型成为真正的"智能体"，自己阅读指令、发现脚本、决定执行。
    代码层面不需要特殊处理脚本发现/执行逻辑。

Skills 目录结构：
    my-skill/
    ├── SKILL.md          # 必需：指令和元数据
    ├── scripts/          # 可选：可执行脚本
    ├── references/       # 可选：参考文档
    └── assets/           # 可选：模板和资源

SKILL.md 格式：
    ---
    name: skill-name
    description: 何时使用此 skill 的描述
    ---
    # Skill Title
    详细指令内容...
"""

import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import yaml


# 默认 Skills 搜索路径（项目级优先，用户级兜底）
DEFAULT_SKILL_PATHS = [
    Path.cwd() / ".claude" / "skills",  # 项目级 Skills (.claude/skills/) - 优先
    Path.cwd() / ".agents" / "skills",  # 项目级 Skills (.claude/skills/) - 优先
    Path.home() / ".agents" / "skills",  # 用户级 Skills (~/.claude/skills/) - 兜底
    Path.home() / ".claude" / "skills",  # 用户级 Skills (~/.claude/skills/) - 兜底
]


@dataclass
class SkillMetadata:
    """
    Skill 元数据（Level 1）

    启动时从 YAML frontmatter 解析，用于注入 system prompt。
    每个 skill 约 100 tokens。
    """

    name: str  # skill 唯一名称
    description: str  # 何时使用此 skill 的描述
    skill_path: Path  # skill 目录路径

    def to_prompt_line(self) -> str:
        """生成 system prompt 中的单行描述"""
        return f"- **{self.name}**: {self.description}"


@dataclass
class SkillContent:
    """
    Skill 完整内容（Level 2）

    用户请求匹配时加载，包含 SKILL.md 的完整指令。
    约 5k tokens。

    注意：不收集 scripts 和 additional_docs，让大模型从指令中自己发现。
    这是 Anthropic Skills 的核心设计理念。
    """

    metadata: SkillMetadata
    instructions: str  # SKILL.md body 内容


class SkillLoader:
    """
    Skills 加载器

    核心职责：
    1. scan_skills(): 发现文件系统中的 Skills，解析元数据
    2. load_skill(): 按需加载 Skill 详细内容
    3. build_system_prompt(): 生成包含 Skills 列表的 system prompt

    使用示例：
        loader = SkillLoader()

        # Level 1: 获取 system prompt
        system_prompt = loader.build_system_prompt()

        # Level 2: 加载具体 skill
        skill = loader.load_skill("news-extractor")
        print(skill.instructions)
    """

    def __init__(self, skill_paths: list[Path] | None = None):
        """
        初始化加载器

        Args:
            skill_paths: 自定义 Skills 搜索路径，默认为:
                - .claude/skills/ (项目级，优先)
                - ~/.claude/skills/ (用户级，兜底)
        """
        self.skill_paths = skill_paths or DEFAULT_SKILL_PATHS
        self._metadata_cache: dict[str, SkillMetadata] = {}

    def scan_skills(self) -> list[SkillMetadata]:
        """
        Level 1: 扫描所有 Skills 元数据

        遍历 skill_paths，查找包含 SKILL.md 的目录，
        解析 YAML frontmatter 提取 name 和 description。

        Returns:
            所有发现的 Skills 元数据列表

        示例输出：
            [
                SkillMetadata(name='news-extractor', description='新闻站点内容提取...', ...),
                SkillMetadata(name='slides-generator', description='Generate slides...', ...),
            ]
        """
        skills = []
        seen_names = set()

        for base_path in self.skill_paths:
            if not base_path.exists():
                continue

            # 遍历 skills 目录下的每个子目录
            for skill_dir in base_path.iterdir():
                if not skill_dir.is_dir():
                    continue

                # 检查是否存在 SKILL.md
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue

                # 解析元数据
                metadata = self._parse_skill_metadata(skill_md)
                if metadata and metadata.name not in seen_names:
                    skills.append(metadata)
                    seen_names.add(metadata.name)
                    self._metadata_cache[metadata.name] = metadata

        return skills

    def _parse_skill_metadata(self, skill_md_path: Path) -> Optional[SkillMetadata]:
        """
        解析 SKILL.md 的 YAML frontmatter

        SKILL.md 格式：
            ---
            name: skill-name
            description: Brief description when to use it
            ---
            # Instructions...

        Args:
            skill_md_path: SKILL.md 文件路径

        Returns:
            解析后的元数据，解析失败返回 None
        """
        try:
            content = skill_md_path.read_text(encoding="utf-8")
        except Exception:
            return None

        # 使用正则提取 YAML frontmatter
        # 格式: ---\n...yaml...\n---
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)

        if not frontmatter_match:
            return None

        try:
            # 解析 YAML
            frontmatter = yaml.safe_load(frontmatter_match.group(1))

            name = frontmatter.get("name", "")
            description = frontmatter.get("description", "")

            if not name:
                return None

            return SkillMetadata(
                name=name,
                description=description,
                skill_path=skill_md_path.parent,
            )
        except yaml.YAMLError:
            return None

    def load_skill(self, skill_name: str) -> Optional[SkillContent]:
        """
        Level 2: 加载 Skill 完整内容

        读取 SKILL.md 的完整指令，以及其他 .md 文件和脚本列表。
        这是 load_skill tool 的核心实现。

        Args:
            skill_name: Skill 名称（如 "news-extractor"）

        Returns:
            Skill 完整内容，未找到返回 None
        """
        # 先检查缓存
        metadata = self._metadata_cache.get(skill_name)
        if not metadata:
            # 尝试重新扫描
            self.scan_skills()
            metadata = self._metadata_cache.get(skill_name)

        if not metadata:
            return None

        # 读取 SKILL.md 完整内容
        skill_md = metadata.skill_path / "SKILL.md"
        try:
            content = skill_md.read_text(encoding="utf-8")
        except Exception:
            return None

        # 提取 body（去除 frontmatter）
        body_match = re.match(r'^---\s*\n.*?\n---\s*\n(.*)$', content, re.DOTALL)
        instructions = body_match.group(1).strip() if body_match else content

        # 只返回 instructions，让大模型从指令中自己发现脚本和文档
        return SkillContent(
            metadata=metadata,
            instructions=instructions,
        )

    def build_system_prompt(self, base_prompt: str = "") -> str:
        """
        构建包含 Skills 列表的 system prompt

        这是 Level 1 的核心输出：将所有 Skills 的元数据
        注入到 system prompt 中。

        Args:
            base_prompt: 基础 system prompt（可选）

        Returns:
            完整的 system prompt
        """
        skills = self.scan_skills()

        # 构建 Skills 部分
        if skills:
            skills_section = "## Available Skills\n\n"
            skills_section += "You have access to the following specialized skills:\n\n"
            for skill in skills:
                skills_section += skill.to_prompt_line() + "\n"
            skills_section += "\n"
            skills_section += "### How to Use Skills\n\n"
            skills_section += "1. **Discover**: Review the skills list above\n"
            skills_section += (
                "2. **Load**: When a user request matches a skill's description, "
            )
            skills_section += (
                "use `load_skill(skill_name)` to get detailed instructions\n"
            )
            skills_section += (
                "3. **Execute**: Follow the skill's instructions, which may include "
            )
            skills_section += "running scripts via `bash`\n\n"
            skills_section += "**Important**: Only load a skill when it's relevant to the user's request. "
            skills_section += (
                "Script code never enters the context - only their output does.\n"
            )
        else:
            skills_section = "## Skills\n\nNo skills currently available.\n"

        # 组合完整 prompt
        if base_prompt:
            return f"{base_prompt}\n\n{skills_section}"
        else:
            return f"You are a helpful coding assistant.\n\n{skills_section}"


# 便捷函数
def discover_skills(skill_paths: list[Path] | None = None) -> list[SkillMetadata]:
    """便捷函数：发现所有 Skills"""
    loader = SkillLoader(skill_paths)
    return loader.scan_skills()


def get_skill_content(
    skill_name: str, skill_paths: list[Path] | None = None
) -> Optional[SkillContent]:
    """便捷函数：获取 Skill 内容"""
    loader = SkillLoader(skill_paths)
    return loader.load_skill(skill_name)
