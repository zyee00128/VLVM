#!/bin/bash

# ==============================================================================
# VLVM 全量评测 · 最终定稿档（数据集两档，默认 hm3d）
# ------------------------------------------------------------------------------
#   hm3d（默认）= HM3D ObjectNav v1 val 全量 = 20 场景 / 2000 集
#   mp3d        = MP3D ObjectNav v1 val 全量 = 11 场景 / 2195 集
# 不传 habitat.dataset.content_scenes=[<scene>]（默认通配全部场景）+ split=val；单档单跑。
# 用法：
#   bash scripts/my_eval.sh              # HM3D val 全量
#   bash scripts/my_eval.sh mp3d         # MP3D val 全量
#   LIMIT=20 bash scripts/my_eval.sh     # 冒烟：前 20 集
#   OUT_DIR=... LABEL=... bash scripts/my_eval.sh
# 跑前先起三个 VLM 服务：bash scripts/launch_vlm_servers.sh
# ==============================================================================

# ----------------- 项目基础路径配置 -----------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlvm}"

cd "$PROJECT_ROOT"

# 加载 Conda 环境（conda 根按 conda info --base → 常见路径顺序探测）
CONDA_PROFILE=""
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
for _c in "$CONDA_BASE" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /root/miniconda3; do
    if [ -n "$_c" ] && [ -f "$_c/etc/profile.d/conda.sh" ]; then
        CONDA_PROFILE="$_c/etc/profile.d/conda.sh"
        break
    fi
done
if [ -n "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"
    conda activate "$CONDA_ENV_NAME" 2>/dev/null || {
        echo "警告: 激活 $CONDA_ENV_NAME 失败，回退环境名 vlfm"
        conda activate vlfm 2>/dev/null || true
    }
else
    echo "警告: 未找到 conda.sh，沿用当前 shell 解释器: $(command -v python)"
fi
echo ">>> 解释器: $(command -v python)"

# ----------------- 声明 GPU 与 EGL 渲染环境变量 -----------------
export CUDA_VISIBLE_DEVICES=0
export EGL_PLATFORM=surfaceless
export FORCE_GLX_USE_EGL=1
export MAGNUM_LOG=quiet
export MAGNUM_GPU_VALIDATION=OFF
export HF_ENDPOINT=https://hf-mirror.com
unset DISPLAY
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
set -o pipefail

# ----------------- 数据集两档 -----------------
DATASET="${1:-hm3d}"
case "$DATASET" in
    hm3d)
        DATA_PATH="data/datasets/objectnav/hm3d/v1/val/val.json.gz"
        N_SCENES=20; N_EPS=2000
        ;;
    mp3d)
        DATA_PATH="data/datasets/objectnav/mp3d/val/val.json.gz"
        N_SCENES=11; N_EPS=2195
        ;;
    *)
        echo "用法: bash scripts/my_eval.sh [hm3d / mp3d]"
        exit 1
        ;;
esac

# ----------------- 输出目录与日志 -----------------
# LOGFILE 在此立即创建（touch），其后脚本的全部终端输出经 tee 实时写入日志：
# 任何阶段（启动失败 / Ctrl+C / python 崩溃）日志文件均已存在，随时可 tail / cp。
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs}"
mkdir -p "$OUT_DIR"
LABEL="${LABEL:-VLVM}"
LOGFILE="$OUT_DIR/eval_${DATASET}_$(date +%Y-%m-%d)_$(date +%H%M%S)_${LABEL}.log"
touch "$LOGFILE"
exec > >(tee -a "$LOGFILE") 2>&1

# ----------------- 服务自检（三服务未起会静默卡住） -----------------
_missing=""
for _p in 12186 12182 12181; do
    timeout 2 bash -c ":</dev/tcp/127.0.0.1/$_p" 2>/dev/null || _missing="$_missing $_p"
done
if [ -n "$_missing" ]; then
    echo "警告: 以下 VLM 服务端口未监听:$_missing"
    echo "      映射: TSP3D=12186 / BLIP2-ITM=12182 / GroundingDINO=12181"
    echo "      请先执行: bash scripts/launch_vlm_servers.sh"
else
    echo ">>> 三服务端口就绪 (12186 / 12182 / 12181)"
fi

# ----------------- 评测 -----------------
echo "================================================="
echo ">>> 数据集: $DATASET（$N_SCENES 场景 / $N_EPS 集，val 全量）"
echo ">>> 档位: 最终定稿档（YAML 基线全开）"
echo ">>> data_path: $DATA_PATH"
echo ">>> 输出目录: $OUT_DIR"
[ -n "${LIMIT:-}" ] && echo ">>> 冒烟模式: 前 $LIMIT 集"
echo "================================================="

args=(
    habitat.dataset.data_path="$DATA_PATH"
    habitat.dataset.split=val
    habitat_baselines.eval.split=val
    habitat_baselines.num_environments=1
    habitat.simulator.habitat_sim_v0.gpu_device_id=0
    habitat.simulator.habitat_sim_v0.gpu_gpu=False
    habitat_baselines.rl.policy.name="HabitatITMPolicyV1"
)
if [ -n "${LIMIT:-}" ]; then
    args+=("habitat_baselines.test_episode_count=$LIMIT")
fi
# 子集调试（可选）：只跑指定场景 = 追加 content_scenes 覆盖（Hydra 列表语法，场景间不加空格）：
#   SCENES="[5cdEh9F2hJL]" bash scripts/my_eval.sh                              # 单场景
#   SCENES="[TEEsavR23oF,mv2HUxq3B53,wcojb4TFT35]" bash scripts/my_eval.sh      # 三场景
# 也可手动在 args 数组内追加：args+=("habitat.dataset.content_scenes=[<scene>]")
SCENES="${SCENES:-}"
if [ -n "$SCENES" ]; then
    args+=("habitat.dataset.content_scenes=$SCENES")
fi

# LOGFILE 已在启动段创建并接管全部输出（exec tee）；`-u` = python 无缓冲逐行落盘
START=$(date +%s)
python -u -m vlfm.run "${args[@]}"
END=$(date +%s)
ELAPSED=$((END - START))

echo ""
echo "================================================="
echo "完成: VLVM @ $DATASET -> $(basename "$LOGFILE")"
echo "用时: $(printf '%d:%02d:%02d' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)))"
echo "结果摘要:"
grep -E "^(Success rate|Oracle Success Rate|Average episode (success|spl|soft_spl|steps_count))" "$LOGFILE" | tail -5 | sed 's/^/    /'
echo "日志: $LOGFILE"
echo "================================================="
# bash ./scripts/launch_vlm_servers.sh
# bash ./scripts/my_eval.sh hm3d
# bash ./scripts/my_eval.sh mp3d
