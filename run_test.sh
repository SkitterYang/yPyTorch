#!/bin/bash

# yPyTorch 测试脚本 - 运行所有单元测试

set -e

# 切换到脚本目录
cd "$(dirname "$0")"

echo "=========================================="
echo "yPyTorch 单元测试"
echo "=========================================="
echo ""

# 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo "[ERROR] uv 未安装，请先安装 uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 使用 uv run 运行测试（uv 会自动管理依赖）
uv run pytest tests/ -v --tb=short

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="

