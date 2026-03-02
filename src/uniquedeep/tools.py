#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File: src/uniquedeep/tools.py
@Time: 2026/02/24
@Author: GeorgeWu
@Description: LangChain 工具定义（load_skill, bash 等）和运行时上下文。
'''

"""
LangChain Tools 定义

使用 LangChain 1.0 的 @tool 装饰器和 ToolRuntime 定义工具：
- load_skill: 加载 Skill 详细指令（Level 2）
- bash: 执行命令/脚本（Level 3）
- read_file: 读取文件

ToolRuntime 提供访问运行时信息的统一接口：
- state: 可变的执行状态
- context: 不可变的配置（如 skill_loader）
"""

import subprocess
import sys
import tempfile
import fnmatch
import re
import datetime
import urllib.request
import urllib.parse
from pathlib import Path
from dataclasses import dataclass, field

from langchain.tools import tool, ToolRuntime
from lxml import html

from .skill_loader import SkillLoader
from .stream import resolve_path
from .mcp_client import MCPClient, run_mcp_tool

# Global Bing MCP Client
# Assuming 'npx' is in PATH. The '-y' flag ensures npx doesn't prompt for installation.
bing_mcp_client = MCPClient(command="npx", args=["-y", "bing-cn-mcp"])


@dataclass
class SkillAgentContext:
    """
    Agent 运行时上下文

    通过 ToolRuntime[SkillAgentContext] 在 tool 中访问
    """

    skill_loader: SkillLoader
    working_directory: Path = field(default_factory=Path.cwd)


@tool
def load_skill(skill_name: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Load a skill's detailed instructions.

    This tool reads the SKILL.md file for the specified skill and returns
    its complete instructions. Use this when the user's request matches
    a skill's description from the available skills list.

    The skill's instructions will guide you on how to complete the task,
    which may include running scripts via the bash tool.

    Args:
        skill_name: Name of the skill to load (e.g., 'news-extractor')
    """
    loader = runtime.context.skill_loader

    # 尝试加载 skill
    skill_content = loader.load_skill(skill_name)

    if not skill_content:
        # 列出可用的 skills（从已扫描的元数据中获取）
        skills = loader.scan_skills()
        if skills:
            available = [s.name for s in skills]
            return f"Skill '{skill_name}' not found. Available skills: {', '.join(available)}"
        else:
            return f"Skill '{skill_name}' not found. No skills are currently available."

    # 获取 skill 路径信息
    skill_path = skill_content.metadata.skill_path
    scripts_dir = skill_path / "scripts"

    # 构建路径信息
    path_info = f"""
## Skill Path Info

- **Skill Directory**: `{skill_path}`
- **Scripts Directory**: `{scripts_dir}`

**Important**: When running scripts, use absolute paths like:
```bash
uv run {scripts_dir}/script_name.py [args]
```
"""

    # 返回 instructions 和路径信息
    return f"""# Skill: {skill_name}

## Instructions

{skill_content.instructions}
{path_info}
"""


@tool
def bash(command: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Execute a shell command (bash on Unix/macOS, cmd.exe on Windows).

    Use this for:
    - Running skill scripts (e.g., `uv run path/to/script.py args`)
    - Installing dependencies
    - File operations
    - Any shell command

    Important for Skills:
    - Script code does NOT enter the context, only the output does
    - This is Level 3 of the Skills loading mechanism
    - Follow the skill's instructions for exact command syntax

    Cross-platform Note:
    - On Unix/macOS: Uses /bin/sh (bash-compatible)
    - On Windows: Uses cmd.exe (different syntax, e.g., use 'dir' instead of 'ls')
    - For portable scripts, use Python scripts via `uv run script.py`

    Args:
        command: The shell command to execute
    """
    cwd = str(runtime.context.working_directory)

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 分钟超时
        )

        parts = []

        # 状态标记（与 ToolResultFormatter 配合）
        if result.returncode == 0:
            parts.append("[OK]")
        else:
            parts.append(f"[FAILED] Exit code: {result.returncode}")

        parts.append("")  # 空行分隔

        if result.stdout:
            parts.append(result.stdout.rstrip())

        if result.stderr:
            if result.stdout:
                parts.append("")
            parts.append("--- stderr ---")
            parts.append(result.stderr.rstrip())

        if not result.stdout and not result.stderr:
            parts.append("(no output)")

        return "\n".join(parts)

    except subprocess.TimeoutExpired:
        return "[FAILED] Command timed out after 300 seconds."
    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
def read_file(file_path: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Read the contents of a file.

    Use this to:
    - Read skill documentation files
    - View script output files
    - Inspect any text file

    Args:
        file_path: Path to the file (absolute or relative to working directory)
    """
    path = resolve_path(file_path, runtime.context.working_directory)

    if not path.exists():
        return f"[Error] File not found: {file_path}"

    if not path.is_file():
        return f"[Error] Not a file: {file_path}"

    try:
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # 添加行号
        numbered_lines = []
        for i, line in enumerate(lines[:2000], 1):  # 限制行数
            numbered_lines.append(f"{i:4d}| {line}")

        if len(lines) > 2000:
            numbered_lines.append(f"... ({len(lines) - 2000} more lines)")

        return "\n".join(numbered_lines)

    except UnicodeDecodeError:
        return f"[Error] Cannot read file (binary or unknown encoding): {file_path}"
    except Exception as e:
        return f"[Error] Failed to read file: {str(e)}"


@tool
def write_file(
    file_path: str, content: str, runtime: ToolRuntime[SkillAgentContext]
) -> str:
    """
    Write content to a file.

    Use this to:
    - Save generated content
    - Create new files
    - Modify existing files

    Args:
        file_path: Path to the file (absolute or relative to working directory)
        content: Content to write to the file
    """
    path = resolve_path(file_path, runtime.context.working_directory)

    try:
        # 确保父目录存在
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(content, encoding="utf-8")
        return f"[Success] File written: {path}"

    except Exception as e:
        return f"[Error] Failed to write file: {str(e)}"


@tool
def glob(pattern: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Find files matching a glob pattern.

    Use this to:
    - Find files by name pattern (e.g., "**/*.py" for all Python files)
    - List files in a directory with wildcards
    - Discover project structure

    Args:
        pattern: Glob pattern (e.g., "**/*.py", "src/**/*.ts", "*.md")
    """
    cwd = runtime.context.working_directory

    try:
        # 使用 Path.glob 进行匹配
        matches = sorted(cwd.glob(pattern))

        if not matches:
            return f"No files matching pattern: {pattern}"

        # 限制返回数量
        max_results = 100
        result_lines = []

        for path in matches[:max_results]:
            try:
                rel_path = path.relative_to(cwd)
                result_lines.append(str(rel_path))
            except ValueError:
                result_lines.append(str(path))

        result = "\n".join(result_lines)

        if len(matches) > max_results:
            result += f"\n... and {len(matches) - max_results} more files"

        return f"[OK]\n\n{result}"

    except Exception as e:
        return f"[FAILED] {str(e)}"





@tool
def grep(pattern: str, path: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Search for a pattern in files.

    Use this to:
    - Find code containing specific text or regex
    - Search for function/class definitions
    - Locate usages of variables or imports

    Args:
        pattern: Regular expression pattern to search for
        path: File or directory path to search in (use "." for current directory)
    """
    cwd = runtime.context.working_directory
    search_path = resolve_path(path, cwd)

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"[FAILED] Invalid regex pattern: {e}"

    results = []
    max_results = 50
    files_searched = 0

    try:
        if search_path.is_file():
            files = [search_path]
        else:
            # 搜索所有文本文件，排除常见的二进制/隐藏目录
            files = []
            for p in search_path.rglob("*"):
                if p.is_file():
                    # 排除隐藏文件和常见的非代码目录
                    parts = p.parts
                    if any(
                        part.startswith(".")
                        or part
                        in ("node_modules", "__pycache__", ".git", "venv", ".venv")
                        for part in parts
                    ):
                        continue
                    files.append(p)

        for file_path in files:
            if len(results) >= max_results:
                break

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                lines = content.split("\n")
                files_searched += 1

                for line_num, line in enumerate(lines, 1):
                    if regex.search(line):
                        try:
                            rel_path = file_path.relative_to(cwd)
                        except ValueError:
                            rel_path = file_path
                        results.append(f"{rel_path}:{line_num}: {line.strip()[:100]}")

                        if len(results) >= max_results:
                            break

            except (UnicodeDecodeError, PermissionError, IsADirectoryError):
                continue

        if not results:
            return f"No matches found for pattern: {pattern} (searched {files_searched} files)"

        output = "\n".join(results)
        if len(results) >= max_results:
            output += f"\n... (truncated, showing first {max_results} matches)"

        return f"[OK]\n\n{output}"

    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
def edit(
    file_path: str,
    old_string: str,
    new_string: str,
    runtime: ToolRuntime[SkillAgentContext],
) -> str:
    """
    Edit a file by replacing text.

    Use this to:
    - Modify existing code
    - Fix bugs by replacing incorrect code
    - Update configuration values

    The old_string must match exactly (including whitespace/indentation).
    For safety, the old_string must be unique in the file.

    Args:
        file_path: Path to the file to edit
        old_string: The exact text to find and replace
        new_string: The text to replace it with
    """
    path = resolve_path(file_path, runtime.context.working_directory)

    if not path.exists():
        return f"[FAILED] File not found: {file_path}"

    if not path.is_file():
        return f"[FAILED] Not a file: {file_path}"

    try:
        content = path.read_text(encoding="utf-8")

        # 检查 old_string 是否存在
        count = content.count(old_string)

        if count == 0:
            return f"[FAILED] String not found in file. Make sure the text matches exactly including whitespace."

        if count > 1:
            return f"[FAILED] String appears {count} times in file. Please provide more context to make it unique."

        # 执行替换
        new_content = content.replace(old_string, new_string, 1)
        path.write_text(new_content, encoding="utf-8")

        # 计算变化的行数
        old_lines = len(old_string.split("\n"))
        new_lines = len(new_string.split("\n"))

        return f"[OK]\n\nEdited {path.name}: replaced {old_lines} lines with {new_lines} lines"

    except UnicodeDecodeError:
        return f"[FAILED] Cannot edit file (binary or unknown encoding): {file_path}"
    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
def list_dir(path: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    List contents of a directory.

    Use this to:
    - Explore directory structure
    - See what files exist in a folder
    - Check if files/folders exist

    Args:
        path: Directory path (use "." for current directory)
    """
    dir_path = resolve_path(path, runtime.context.working_directory)

    if not dir_path.exists():
        return f"[FAILED] Directory not found: {path}"

    if not dir_path.is_dir():
        return f"[FAILED] Not a directory: {path}"

    try:
        entries = sorted(
            dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        )

        result_lines = []
        for entry in entries[:100]:  # 限制数量
            if entry.is_dir():
                result_lines.append(f"📁 {entry.name}/")
            else:
                # 显示文件大小
                size = entry.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size // 1024}KB"
                else:
                    size_str = f"{size // (1024 * 1024)}MB"
                result_lines.append(f"   {entry.name} ({size_str})")

        if len(entries) > 100:
            result_lines.append(f"... and {len(entries) - 100} more entries")

        return f"[OK]\n\n{chr(10).join(result_lines)}"

    except PermissionError:
        return f"[FAILED] Permission denied: {path}"
    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
def python(code: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Execute Python code.
    """
    cwd = runtime.context.working_directory
    temp_path = None

    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, dir=cwd, encoding='utf-8'
        ) as f:
            f.write(code)
            temp_path = Path(f.name)

        # Execute the file
        result = subprocess.run(
            [sys.executable, str(temp_path)],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        parts = []
        if result.returncode == 0:
            parts.append("[OK]")
        else:
            parts.append(f"[FAILED] Exit code: {result.returncode}")

        parts.append("")

        if result.stdout:
            parts.append(result.stdout.rstrip())

        if result.stderr:
            if result.stdout:
                parts.append("")
            parts.append("--- stderr ---")
            parts.append(result.stderr.rstrip())

        if not result.stdout and not result.stderr:
            parts.append("(no output)")

        return "\n".join(parts)

    except subprocess.TimeoutExpired:
        return "[FAILED] Execution timed out after 300 seconds."
    except Exception as e:
        return f"[FAILED] {str(e)}"
    finally:
        # Clean up
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@tool
def get_current_time(runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Get the current system time.

    Use this to:
    - Check the current date and time
    - Timestamp events or logs
    """
    now = datetime.datetime.now()
    return f"[OK]\n\nCurrent time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


# @tool
# def web_search(query: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
#     """
#     Search the web for information using Bing.

#     Use this to:
#     - Find current events, news, and real-time information
#     - Look up documentation, libraries, and technical solutions
#     - Verify facts and data

#     Args:
#         query: The search query string
#     """
#     try:
#         # Encode the query
#         encoded_query = urllib.parse.quote(query)
#         url = f"https://www.bing.com/search?q={encoded_query}"
        
#         # Create a request with User-Agent to mimic a browser
#         req = urllib.request.Request(
#             url, 
#             headers={
#                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#             }
#         )
        
#         # Perform the request
#         with urllib.request.urlopen(req, timeout=10) as response:
#             html_content = response.read().decode('utf-8')
            
#         # Parse HTML using lxml
#         tree = html.fromstring(html_content)
        
#         # Bing search results usually have class 'b_algo'
#         results = tree.xpath('//li[@class="b_algo"]')
        
#         if not results:
#             return "No results found on Bing."
            
#         formatted_results = []
#         for i, result in enumerate(results[:5], 1):
#             # Extract title
#             title_node = result.xpath('.//h2/a')
#             if not title_node:
#                 continue
#             title = title_node[0].text_content()
#             link = title_node[0].get('href')
            
#             # Extract snippet (caption)
#             snippet_node = result.xpath('.//div[@class="b_caption"]/p')
#             if not snippet_node:
#                 # Try alternative snippet location
#                 snippet_node = result.xpath('.//p')
                
#             snippet = snippet_node[0].text_content() if snippet_node else "No description available."
            
#             formatted_results.append(f"{i}. {title}\n   Link: {link}\n   Summary: {snippet}\n")
            
#         return f"[OK]\n\n" + "\n".join(formatted_results)
        
#     except Exception as e:
#         return f"[FAILED] Search failed: {str(e)}"


# @tool
# def bing_search(query: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
#     """
#     Search using Bing via MCP (bing-cn-mcp).
    
#     Use this for:
#     - High-quality web search results
#     - Retrieving information via the Bing MCP server
    
#     Args:
#         query: Search query
#     """
#     # The tool name in bing-cn-mcp is 'bing_search'
#     return run_mcp_tool(bing_mcp_client, "bing_search", {"query": query})


ALL_TOOLS = [
    load_skill,
    bash,
    read_file,
    write_file,
    glob,
    grep,
    edit,
    list_dir,
    python,
    get_current_time,
    # web_search,
    # bing_search,
]
