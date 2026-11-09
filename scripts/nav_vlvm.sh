#!/bin/bash

# ==============================================================================
# VLVM 全量评测 · 最终定稿档（同一导航框架，HM3D / MP3D 两档）
# ------------------------------------------------------------------------------
#   hm3d = HM3D ObjectNav v1 val 全量 = 20 场景 / 2000 集
#   mp3d = MP3D ObjectNav v1 val 全量 = 11 场景 / 2195 集
# 两档只切 benchmark/nav/objectnav 分组与 data_path，其余（YAML 基线、TSP3D 权重、
# 阈值、recheck / explore mode）完全一致；默认 both，缺数据的一档自动跳过。
# 用法：
#   bash scripts/nav_vlvm.sh                 # hm3d + mp3d 依次全量
#   bash scripts/nav_vlvm.sh hm3d            # 只跑 HM3D 全量
#   bash scripts/nav_vlvm.sh mp3d            # 只跑 MP3D 全量
#   LIMIT=20 bash scripts/nav_vlvm.sh hm3d   # 冒烟：前 20 集
#   SCENES="[5cdEh9F2hJL]" bash scripts/nav_vlvm.sh hm3d   # 单场景调试
#   OUT_DIR=... LABEL=... bash scripts/nav_vlvm.sh
# 结果：outputs/<dataset>_full_<时间戳>_<LABEL>/（resolved_config.yaml + eval.log +
#       detections/*.jsonl + tb/），单档单跑。
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

# ----------------- 数据集档位（合法值检查） -----------------
DATASET="${1:-both}"
case "$DATASET" in
    hm3d|mp3d|both) ;;
    *)
        echo "用法: bash scripts/nav_vlvm.sh [hm3d / mp3d / both]"
        exit 1
        ;;
esac

# ----------------- 输出根目录 -----------------
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs}"
LABEL="${LABEL:-VLVM}"
SCENES="${SCENES:-}"
mkdir -p "$OUT_DIR"

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

# ----------------- 单档全量评测 -----------------
# 两档只用 benchmark/nav/objectnav 分组与 data_path 区分，其余 Hydra 参数完全相同。
# 返回码: 0 = run 结束；2 = 数据缺失跳过；其它 = python 失败。
run_one() {
    local dataset="$1"
    local data_path content_dir n_scenes n_eps
    case "$dataset" in
        hm3d)
            data_path="data/datasets/objectnav/hm3d/v1/val/val.json.gz"
            content_dir="data/datasets/objectnav/hm3d/v1/val/content"
            n_scenes=20; n_eps=2000
            ;;
        mp3d)
            data_path="data/datasets/objectnav/mp3d/v1/val/val.json.gz"
            content_dir="data/datasets/objectnav/mp3d/v1/val/content"
            n_scenes=11; n_eps=2195
            ;;
    esac

    # 数据自检：episode 索引（可为空壳）+ per-scene content 必须同时在位
    local missing=""
    [ -f "$data_path" ] || missing="$missing $data_path"
    [ -d "$content_dir" ] || missing="$missing $content_dir"
    if [ -n "$missing" ]; then
        echo ">>> 跳过 $dataset：缺少数据:$missing"
        echo "    期望布局: $data_path + $content_dir/*.json.gz"
        return 2
    fi

    local run_dir="$OUT_DIR/${dataset}_full_$(date +%Y-%m-%d)_$(date +%H%M%S)_${LABEL}"
    mkdir -p "$run_dir/detections"
    local logfile="$run_dir/eval.log"
    touch "$logfile"

    {
        echo "================================================="
        echo ">>> 数据集: $dataset（$n_scenes 场景 / $n_eps 集，val 全量）"
        echo ">>> 档位: 最终定稿档（YAML 基线全开）"
        echo ">>> data_path: $data_path"
        echo ">>> 输出目录: $run_dir"
        [ -n "${LIMIT:-}" ] && echo ">>> 冒烟模式: 前 $LIMIT 集"
        [ -n "$SCENES" ] && echo ">>> 场景子集: $SCENES"
        echo "================================================="

        local args=(
            benchmark/nav/objectnav=objectnav_${dataset}
            habitat.dataset.data_path="$data_path"
            habitat.dataset.split=val
            habitat_baselines.eval.split=val
            habitat_baselines.num_environments=1
            habitat.simulator.habitat_sim_v0.gpu_device_id=0
            habitat.simulator.habitat_sim_v0.gpu_gpu=False
            habitat_baselines.rl.policy.name="HabitatITMPolicyV1"
            habitat_baselines.rl.policy.det_log_dir="$run_dir/detections"
            habitat_baselines.tensorboard_dir="$run_dir/tb"
            habitat_baselines.video_dir="$run_dir/videos"
        )
        [ -n "${LIMIT:-}" ] && args+=("habitat_baselines.test_episode_count=$LIMIT")
        [ -n "$SCENES" ] && args+=("habitat.dataset.content_scenes=$SCENES")

        # 配置快照（Hydra 组合结果）；失败不阻断主流程
        python -m vlfm.run --cfg job "${args[@]}" > "$run_dir/resolved_config.yaml" 2>/dev/null || true

        local start end elapsed
        start=$(date +%s)
        python -u -m vlfm.run "${args[@]}"
        end=$(date +%s)
        elapsed=$((end - start))

        echo ""
        echo "================================================="
        echo "完成: VLVM @ $dataset -> $run_dir"
        echo "用时: $(printf '%d:%02d:%02d' $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)))"
        echo "结果摘要:"
        grep -E "^(Success rate|Oracle Success Rate|Average episode (success|spl|soft_spl|steps_count))" "$logfile" | tail -5 | sed 's/^/    /'
        echo "日志: $logfile"
        echo "================================================="
    } 2>&1 | tee -a "$logfile"
    return "${PIPESTATUS[0]}"
}

# ----------------- 依次跑选中的档位 -----------------
rc=0
case "$DATASET" in
    both)
        run_one hm3d || rc=$?
        run_one mp3d || rc=$?
        ;;
    *)
        run_one "$DATASET" || rc=$?
        ;;
esac
exit "$rc"
