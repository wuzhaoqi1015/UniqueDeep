# 仓库指南

## 项目概述
使用 LangChain 1.0 构建的 Skills Agent，演示 Anthropic Skills 三层加载机制的底层原理。核心设计是让大模型成为真正的"智能体"，自己阅读指令、发现脚本、决定执行。

## 核心架构

### Skills 三层加载机制
| 层级 | 时机 | 实现 |
|------|------|------|
| **Level 1** | 启动时 | `SkillLoader.scan_skills()` 扫描目录，解析 YAML frontmatter，注入 system_prompt |
| **Level 2** | 请求匹配时 | `load_skill` 工具读取 SKILL.md 完整指令 |
| **Level 3** | 执行时 | `bash` 工具执行脚本，脚本代码不进入上下文 |

### 流式处理架构
```
agent.py: stream_events() → 使用 stream_mode="messages" 获取 LangChain 流式输出
    ↓
stream/tracker.py: ToolCallTracker 追踪工具调用，处理增量 JSON (input_json_delta)
    ↓
stream/emitter.py: StreamEventEmitter 生成标准化事件 (thinking/text/tool_call/tool_result/done)
    ↓
stream/formatter.py: ToolResultFormatter 格式化输出，检测 [OK]/[FAILED] 前缀
    ↓
cli.py: Rich Live Display 渲染到终端
```

### 关键流程：工具调用参数处理
LangChain 流式传输中，工具参数可能分批到达：
1. `tool_use` 块先到达（`input` 可能为 `None` 或 `{}`）
2. `input_json_delta` 分批传递参数片段
3. `finalize_all()` 在收到 `tool_result` 前解析完整 JSON

CLI 使用 `tool_id` 去重，允许同一工具调用发送多次（首次显示"执行中"，finalize 后更新完整参数）。

## 项目结构与模块组织
- `src/uniquedeep/` 存放核心包（agent, CLI, tools, skill loader, 以及 `stream/` 格式化/追踪助手）。
- `tests/` 包含基于 pytest 的单元测试，命名为 `test_*.py`。
- `examples/` 提供 CLI 和 agent 用法的可运行演示。
- `docs/` 存放设计笔记和长篇说明。
- `.agent/skills/` 包含示例 Skills，包括 `SKILL.md` 和脚本。
- `pyproject.toml` 定义依赖项和 `uniquedeep` CLI 入口点；`uv.lock` 锁定版本。

### Skills 目录结构
```
.agent/skills/skill-name/
├── SKILL.md          # 必需：YAML frontmatter + 指令
├── scripts/          # 可选：可执行脚本
├── references/       # 可选：参考文档
└── assets/           # 可选：模板和资源
```
Skills 搜索路径（优先级从高到低）：
1. `.agent/skills/` (项目级)
2. `~/.agent/skills/` (用户级)

## 构建、测试与开发命令
- `uv sync`: 安装依赖到本地环境。
- `uv run uniquedeep --interactive`: 运行交互式 CLI 演示。
- `uv run uniquedeep "列出当前目录"`: 运行单个提示词。
- `uv run uniquedeep --list-skills`: 验证 Skills 发现功能。
- `uv run uniquedeep --show-prompt`: 查看 System Prompt。
- `uv run python -m pytest tests/ -v`: 运行测试套件。
- `uv run python -m pytest tests/test_stream.py -v`: 运行单个测试文件。
- `uv run python -m pytest tests/test_stream.py::TestToolCallTracker -v`: 运行特定测试。
- 打包使用 `pyproject.toml` 中的 Hatchling；本地开发不需要单独的构建步骤。

## 代码风格与命名规范
- Python 3.12；使用 4 空格缩进，并尽可能使用类型提示。
- 遵循现有模式：模块/函数使用 `snake_case`，类使用 `CamelCase`。
- 保持 CLI 输出和流式格式化行为与 `src/uniquedeep/stream/` 中的现有工具一致。

### 工具输出格式
bash 工具使用 `[OK]`/`[FAILED]` 前缀标识执行状态：
```
[OK]

output content...

[FAILED] Exit code: 1

--- stderr ---
error message
```

## 测试指南
- 使用 pytest；将新测试放在 `tests/` 下，命名为 `test_*.py`。
- 在更改工具输出格式、流事件解析或 CLI 行为时添加测试。
- 在提交 PR 之前在本地运行相关测试。

## 提交与 Pull Request 指南
- 提交信息遵循历史记录中的 Conventional Commits 风格（例如：`feat: add X`, `refactor: simplify Y`, `docs: update README`）。
- PR 应包含简短摘要、运行的测试命令，以及 CLI/UX 更改的截图或终端输出。
- 有相关 Issue 时请关联。

## 配置与安全
- **主要配置**：项目使用 `models.json` 作为模型配置的唯一真实来源（Single Source of Truth）。
  - 该文件定义了可用的提供商、模型、API 端点（`base_url`）以及模型特定参数，如 `temperature`（温度）、`max_tokens`（最大令牌数）和 `thinking`（思考）模式状态。
  - `models.json` 中的 `active_model` 字段决定了 CLI 使用的默认模型。
- **密钥管理**：
  - 敏感的 API 密钥应存储在 `.env` 中（例如 `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`）。
  - `models.json` 使用 `${VAR_NAME}` 语法引用这些密钥（例如 `"api_key": "${ANTHROPIC_API_KEY}"`）。
  - 不要提交包含硬编码密钥的 `models.json`。使用 `models_template.json` 来分享配置结构。
- **模型切换**：使用 CLI 命令 `/model <name>` 动态切换模型。这将自动更新 `models.json`。
