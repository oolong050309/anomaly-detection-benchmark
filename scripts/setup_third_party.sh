#!/usr/bin/env bash
# 拉取本项目第二阶段需要的 5 个算法所在的官方仓库。
# 默认 clone 到项目同级目录（不放到 final_project 内，避免 git 嵌套）。
#
# 用法：
#   bash scripts/setup_third_party.sh              # 默认到 ../third_party
#   THIRD_PARTY_DIR=/path bash scripts/setup_third_party.sh

set -euo pipefail

THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-../third_party}"
mkdir -p "$THIRD_PARTY_DIR"
cd "$THIRD_PARTY_DIR"

clone_or_update() {
    local url="$1"
    local dir="$2"
    if [[ -d "$dir/.git" ]]; then
        echo "[skip] $dir already cloned; pulling latest"
        git -C "$dir" pull --ff-only || true
    else
        echo "[clone] $url -> $dir"
        git clone "$url" "$dir"
    fi
}

# GADBench：GCN / BWGNN / XGBGraph
clone_or_update "https://github.com/squareRoot3/GADBench.git" "GADBench"

# UNPrompt：图零样本提示
clone_or_update "https://github.com/mala-lab/UNPrompt.git" "UNPrompt"

# DADA：时序基础模型
clone_or_update "https://github.com/iambowen/DADA.git" "DADA"

echo ""
echo "Done. Cloned repos in: $(realpath .)"
echo ""
echo "Next: set THIRD_PARTY_PATH env var or update models/__init__.py:"
echo "  export THIRD_PARTY_PATH=$(realpath .)"
