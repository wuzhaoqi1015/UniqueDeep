# 本地交互式运行
uv run uniquedeep --interactive

# 构建docker并启动（首次运行会自动构建）
docker compose run --rm uniquedeep


# 多智能体
uv run python -m uniquedeep.relay_cli "提供关于新生RNA的解释"  --planner deepseek-reasoner --executor claude-opus-4-6 --executor-provider anthropic