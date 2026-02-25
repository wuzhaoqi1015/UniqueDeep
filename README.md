# UniqueDeep

<div align="center">

![UniqueDeep Logo](docs/images/logo.jpg)

**使用 LangChain 构建的 Skills Agent**  
*实现类似 Anthropic Skills 三层加载机制的底层原理*

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-1.0+-green.svg)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[特性](#-特性) • [快速开始](#-快速开始) • [CLI命令](#-cli-命令)

</div>

---

## ✨ 特性

- 🧠 **Extended Thinking**: 原生支持显示模型的思考过程
- 🌊 **流式输出**: Token 级实时响应，打字机效果。
- 🛠️ **工具调用可视化**: 清晰展示工具名称、参数、执行状态（✅ 成功 / ⚠️ 执行中 / ❌ 失败）。
- 📚 **三层 Skills 加载**: 
  - **Level 1**: 元数据注入（极低 Token 消耗）
  - **Level 2**: 指令按需加载（Lazy Loading）
  - **Level 3**: 脚本沙箱执行（代码不进上下文）

## 🚀 快速开始

### 1. 安装

```bash
# 克隆项目
git clone https://github.com/wuzhaoqi1015/UniqueDeep.git

# 进入项目工作目录
cd UniqueDeep

# 安装依赖环境 (推荐使用 uv)
uv sync

```
### （可选）安装初始skill：find-skills和skill-creator

find-skills：自动下载所需的skill

skill-creator：创建自定义skill

```bash
# 检查npm版本
npm -v

# (若版本低于18, 否则跳过)
npm install -g n
n lts
hash -r
node -v
npm -v

# 安装find-skills
npx skills add https://github.com/vercel-labs/skills --skill find-skills -y

# 安装skill-creator
npx skills add https://github.com/anthropics/skills --skill skill-creator -y


# 若显示超时，则配置ssh令牌，并添加进入
# ssh-add ~/.ssh/id_ed25519
# ssh-add -l
```

### 2. 配置模型

创建 `.env` 文件（参考 `.env.example`）：

```bash
# === 模型选择 ===
LLM_PROVIDER=anthropic  # anthropic, deepseek, openai
LLM_MODEL=claude-opus-4-6

# === 凭证配置 ===
# Anthropic
ANTHROPIC_API_KEY=sk-ant-xxx
ANTHROPIC_BASE_URL=https://api.anthropic.com

# DeepSeek
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# OpenAI
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1

```

### 3. 交互式体验

启动交互式命令行界面：

```bash
uv run uniquedeep --interactive
```


## 🏗️ Skills 三层加载机制

本项目核心在于复刻了高效的 Skills 加载架构：

| 层级 | 时机 | Token 消耗 | 内容 | 作用 |
|------|------|------------|------|------|
| **Level 1** | 启动时 | ~100/Skill | YAML frontmatter (name, description) | 让模型知道有哪些能力可用 |
| **Level 2** | 触发时 | <5000 | SKILL.md 完整指令 | 提供详细的操作 SOP |
| **Level 3** | 执行时 | **0** (仅输出) | 脚本执行结果 | 处理复杂逻辑，结果返回给模型 |

### 演示流程

1. **Level 1**: 启动时扫描 `.agents/skills`，注入元数据。
   ```text
   ✓ Discovered 4 skills
     - gro-seq-pipeline
     - xlsx
   ```

2. **Level 2**: 用户请求 "整理 GRO-Seq 数据"，模型命中 `gro-seq-pipeline` 描述，调用 `load_skill`。
   ```text
   ● Skill(gro-seq-pipeline)
     └ Successfully loaded skill
   ```

3. **Level 3**: 模型根据指令，调用 `bash` 运行脚本。
   ```text
   ● Bash(python create_gro_seq_sop.py)
     └ [OK] GRO-Seq分析流程SOP已保存到: GRO_Seq_Analysis_SOP.xlsx
   ```

## 💻 CLI 命令

| 命令 | 说明 |
|------|------|
| `uv run uniquedeep --interactive` | 启动交互式会话（推荐） |
| `uv run uniquedeep "列出文件"` | 单次执行任务 |
| `uv run uniquedeep --list-skills` | 查看已发现的 Skills |
| `uv run uniquedeep --show-prompt` | 查看注入的 System Prompt |

**交互模式指令**:
- `/skills`: 列出所有技能
- `/prompt`: 显示当前 System Prompt
- `/temp [val]`: 动态调节温度 (0.0-1.0)
- `/exit`: 退出

## 📂 项目结构

```text
UniqueDeep/
├── src/uniquedeep/
│   ├── agent.py          # LangChain Agent (Extended Thinking)
│   ├── cli.py            # CLI 入口 (流式输出)
│   ├── tools.py          # 工具定义 (load_skill, bash, write_file, glob...)
│   ├── skill_loader.py   # Skills 发现和加载
│   └── stream/           # 流式处理模块
│       ├── emitter.py    # 事件发射器
│       ├── tracker.py    # 工具调用追踪（支持增量 JSON）
│       ├── formatter.py  # 结果格式化器
│       └── utils.py      # 常量和工具函数
├── tests/                # 单元测试
│   ├── test_stream.py
│   ├── test_cli.py
│   └── test_tools.py
├── docs/                 # 文档
│   ├── skill_introduce.md
│   └── langchain_agent_skill.md
└── .agent/skills/       # 示例 Skills
    └── gro-seq-pipeline/
        └── SKILL.md
    └── xlsx/
        ├── SKILL.md
        └── scripts/
```

## ⚙️ 环境变量

| 变量 | 说明 | 示例 |
|------|------|--------|
| `LLM_PROVIDER` | 模型厂商 | `anthropic`, `deepseek` |
| `LLM_MODEL` | 模型名称 | `claude-opus-4-6` `deepseek-reasoner` |
| `ENABLE_THINKING` | 启用思考模式 | `true` (仅 Claude 3.7+) |
| `THINKING_BUDGET` | 思考 Token 预算 | `10000` |

## 📚 参考文档

- [Skills 详细介绍](./docs/skill_introduce.md)
- [LangChain 实现原理](./docs/langchain_agent_skill.md)

## 📄 License

MIT © [UniqueDeep](https://github.com/wuzhaoqi1015/UniqueDeep)
