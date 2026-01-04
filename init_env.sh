#!/bin/bash

# yPyTorch 环境初始化脚本
# 使用 uv 创建虚拟环境并安装依赖

set -e

# 切换到脚本目录
cd "$(dirname "$0")"

echo "=========================================="
echo "yPyTorch 环境初始化"
echo "=========================================="
echo ""

# 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo "[ERROR] uv 未安装，请先安装 uv:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  或访问: https://github.com/astral-sh/uv"
    exit 1
fi

echo "[INFO] 检测到 uv 版本: $(uv --version)"
echo ""

# 检查 requirements.txt 是否存在
if [ ! -f "requirements.txt" ]; then
    echo "[ERROR] requirements.txt 文件不存在"
    exit 1
fi

# 使用 uv 创建虚拟环境
echo "[INFO] 创建虚拟环境..."
uv venv

echo ""
echo "[INFO] 安装依赖..."
# 使用 uv 在虚拟环境中安装依赖
uv pip install -r requirements.txt

echo ""
echo "=========================================="
echo "环境初始化完成！"
echo "=========================================="
echo ""
echo "激活虚拟环境:"
echo "  source .venv/bin/activate  # Linux/Mac"
echo "  或"
echo "  .venv\\Scripts\\activate     # Windows"
echo ""

