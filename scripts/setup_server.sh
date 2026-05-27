#!/usr/bin/env bash
# ============================================================================
# 服务器一键环境安装脚本
# 目标环境：Ubuntu 20.04 / Python 3.8 / CUDA 11.8 / PyTorch 2.0.0
# 用法：
#   cd /path/to/final_project
#   bash scripts/setup_server.sh
# ============================================================================

set -euo pipefail

echo "============================================"
echo " Anomaly Detection Benchmark - Server Setup"
echo " Target: Python 3.8 / PyTorch 2.0.0 / CUDA 11.8"
echo "============================================"
echo ""

# ---------- 1. 创建虚拟环境 ----------
if [ ! -d ".venv" ]; then
    echo "[1/6] Creating virtual environment..."
    python3.8 -m venv .venv
else
    echo "[1/6] Virtual environment already exists, skipping."
fi
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# ---------- 2. 安装 PyTorch 2.0.0 + CUDA 11.8 ----------
echo ""
echo "[2/6] Installing PyTorch 2.0.0 + CUDA 11.8..."
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ---------- 3. 安装 PyG (torch-geometric) + 稀疏依赖 ----------
echo ""
echo "[3/6] Installing torch-geometric + pyg-lib + torch-sparse + torch-scatter..."
pip install torch-geometric==2.5.3
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# ---------- 4. 安装 DGL (GADBench 依赖) ----------
echo ""
echo "[4/6] Installing DGL (CUDA 11.8)..."
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html 2>/dev/null || true

# ---------- 5. 安装项目主依赖 ----------
echo ""
echo "[5/6] Installing project dependencies from requirements.txt..."
pip install \
    "numpy>=1.24,<2.0" \
    "scikit-learn>=1.3,<2.0" \
    "pyod>=1.1,<2.0" \
    "xgboost>=2.0,<3.0" \
    "lightgbm>=4.0,<5.0" \
    "stumpy>=1.12,<2.0" \
    "sktime>=0.30,<1.0" \
    "pygod>=1.1,<2.0" \
    "transformers>=4.30,<5.0" \
    "pandas>=2.0"

# TabPFN 单独装（可能需要下载模型权重）
pip install "tabpfn>=2.0,<3.0" || echo "[WARN] tabpfn install failed, will retry later"

# ---------- 6. 安装开发/测试依赖 ----------
echo ""
echo "[6/6] Installing dev dependencies..."
pip install "pytest>=7.0" "hypothesis>=6.0"

# ---------- 验证 ----------
echo ""
echo "============================================"
echo " Verification"
echo "============================================"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

import numpy; print(f'NumPy: {numpy.__version__}')
import sklearn; print(f'scikit-learn: {sklearn.__version__}')
import pyod; print(f'PyOD: {pyod.__version__}')
import xgboost; print(f'XGBoost: {xgboost.__version__}')
import lightgbm; print(f'LightGBM: {lightgbm.__version__}')
import stumpy; print(f'stumpy: {stumpy.__version__}')
import torch_geometric; print(f'PyG: {torch_geometric.__version__}')
import dgl; print(f'DGL: {dgl.__version__}')
import pygod; print(f'PyGOD: {pygod.__version__}')
import transformers; print(f'transformers: {transformers.__version__}')
print()
print('All imports OK!')
"

echo ""
echo "============================================"
echo " Setup complete!"
echo " Activate with: source .venv/bin/activate"
echo " Run smoke test: python -m scripts.smoke_test_tabular"
echo "============================================"
