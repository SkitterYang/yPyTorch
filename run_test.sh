#!/bin/bash

# yPyTorch 测试脚本 - 运行所有单元测试

set -e

# 切换到脚本目录
cd "$(dirname "$0")"

echo "=========================================="
echo "yPyTorch 单元测试"
echo "=========================================="
echo ""

# 检查依赖
python3 -c "import pytest, numpy" 2>/dev/null || {
    echo "[ERROR] 缺少依赖，请安装: pip install -r requirements.txt"
    exit 1
}

# 运行测试
pytest tests/ -v --tb=short

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="

