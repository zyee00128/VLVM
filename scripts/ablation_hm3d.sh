#!/bin/bash

# ==============================================================================
# VLVM 消融实验（**全量 HM3D val**）· 基线 = 加入 GD 辅助机制之前的配置
# ------------------------------------------------------------------------------
# 数据集：HM3D ObjectNav v1 val **全量** = 20 场景 / 2000 集
# 基线：occ 门 + density 门 + 回退机制 + S-penalty + world 时序融合；GD 全关
#
# 档位（每档只关掉一个机制；GD 档为最后补齐"完整 VLVM"）：
#   abl_base      基线（GD 关）
#   abl_nofb      基线 − 回退机制              enable_fb=False
#   abl_nosp      基线 − S-penalty            enable_s_penalty=False
#   abl_nofuse    基线 − 时序融合器            fusion_style=none（nofusion）
#   abl_nooccden  基线 − occ 门 + density 门   enable_occ_consistency=False + enable_density_gate=False
#   abl_fullgd    **完整 VLVM**（后续跑）= 基线四机制全开 + GD 辅助机制
#   （备选档）abl_nocam = fusion_style=camera：另一条非时序融合路线，按需追加
#
# 耗时：单场景 99 集 ≈ 2~2.5 h（09-15 实测）⇒ 2000 集 ≈ 40~50 h/档
#       5 档 ≈ 9~11 天；两台机器可拆：
#         机器 1：ARMS="base nofb nosp"    bash scripts/ablation_hm3d.sh
#         机器 2：ARMS="nofuse nooccden"   bash scripts/ablation_hm3d.sh
#       跑前先起三个 VLM 服务：bash scripts/launch_vlm_servers.sh
#
# 判据（对齐 Results_final.md 口径）：
#   SR（Δ ≥ 2 集才算）· Oracle（不得下降）· spl / soft_spl · avg_steps · fps ·
#   逐类 success / fp / fn / nv；GD 档另看 `[GD2]` 计数、条目级 tp 率、`[STOP]` 行
#
# 用法：
#   bash scripts/ablation_hm3d.sh                     # 顺序跑 ARMS 默认档
#   ARMS="base nofb" bash scripts/ablation_hm3d.sh    # 只跑指定档（键名见上）
#   ARMS="fullgd"    bash scripts/ablation_hm3d.sh    # 只跑完整 VLVM（GD 档）
#   LIMIT=3 bash scripts/ablation_hm3d.sh             # 冒烟：每档只跑 3 集
#   SKIP_DONE=1 bash scripts/ablation_hm3d.sh         # 已有日志的档跳过（断点续跑）
#   OUT_DIR=outputs/VLVM-ABL2 bash scripts/ablation_hm3d.sh
# ==============================================================================

# ----------------- 项目基础路径配置 -----------------
PROJECT_ROOT="/root/autodl-tmp/vlvm"
CONDA_ENV_NAME="vlfm"

if [ ! -d "$PROJECT_ROOT" ]; then
    PROJECT_ROOT=$(pwd)
fi

cd "$PROJECT_ROOT"

# 加载 Conda 环境
CONDA_PROFILE="/root/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"
    conda activate "$CONDA_ENV_NAME"
else
    echo "警告: 未找到 $CONDA_PROFILE，尝试使用默认 conda 命令..."
    conda activate "$CONDA_ENV_NAME" 2>/dev/null || true
fi

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

# ----------------- 输出目录 -----------------
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/VLVM-ABL}"
mkdir -p "$OUT_DIR"

# ----------------- 服务自检 -----------------
for port in 12186 12182 12181; do
    if ! timeout 2 bash -c ":</dev/tcp/127.0.0.1/$port" 2>/dev/null; then
        echo "警告: 端口 $port 未监听（TSP3D=12186 / BLIP2-ITM=12182 / Grounding-DINO=12181）"
        echo "      请先执行: bash scripts/launch_vlm_servers.sh"
    fi
done

# ----------------- 档位定义 -----------------
# mk_override <键>：生成该档的全部 hydra 覆盖（自包含，无重复键）
mk_override() {
    local fb=True sp=True occ=True den=True gd=False fuse=world
    case "$1" in
        base)     ;;
        nofb)     fb=False ;;
        nosp)     sp=False ;;
        nofuse)   fuse=none ;;
        nooccden) occ=False; den=False ;;
        # nocam)    fuse=camera ;;
        # fullgd)   gd=True ;;
        *) echo "" ; return 1 ;;
    esac
    local ov="habitat_baselines.rl.policy.fusion_style=$fuse"
    ov="$ov habitat_baselines.rl.policy.det_seed=0"
    ov="$ov habitat_baselines.rl.policy.enable_fb=$fb"
    ov="$ov habitat_baselines.rl.policy.enable_s_penalty=$sp"
    ov="$ov habitat_baselines.rl.policy.enable_occ_consistency=$occ"
    ov="$ov habitat_baselines.rl.policy.enable_density_gate=$den"
    ov="$ov habitat_baselines.rl.policy.gdp_enable=$gd"
    if [ "$gd" = "True" ]; then
        ov="$ov habitat_baselines.rl.policy.gdp_every_n=5"
        ov="$ov habitat_baselines.rl.policy.gdp_max_boxes=1"
        ov="$ov habitat_baselines.rl.policy.gdp_box_thr=0.4"
    fi
    echo "$ov"
}

# 默认队列（GD 档默认**不在**队列里；跑完前面再用 ARMS="fullgd" 补）
# 备选档 abl_nocam 同理，需要时 ARMS="nocam"
ARMS="${ARMS:-base nofb nosp nofuse nooccden}"

echo "================================================="
echo ">>> VLVM 全量消融（HM3D val 全量 = 20 场景 / 2000 集）"
echo ">>> 基线: 加入 GD 辅助机制之前的配置（gdp_enable=False）"
echo ">>> 档位: $ARMS"
echo ">>> 输出: $OUT_DIR"
[ -n "${LIMIT:-}" ] && echo ">>> 冒烟模式: 每档 $LIMIT 集"
[ -n "${SKIP_DONE:-}" ] && echo ">>> 断点续跑: 已有日志的档将跳过"
echo "================================================="

TOTAL_START=$(date +%s)

for KEY in $ARMS; do
    LABEL="abl_$KEY"
    OVERRIDE=$(mk_override "$KEY")
    if [ -z "$OVERRIDE" ]; then
        echo ">>> 跳过未知档位: $KEY（可用: base nofb nosp nofuse nooccden fullgd nocam）"
        continue
    fi

    EXIST=$(ls "$OUT_DIR"/eval_hm3d_*_"$LABEL".log 2>/dev/null | tail -1 || true)
    if [ -n "${SKIP_DONE:-}" ] && [ -n "$EXIST" ]; then
        echo ">>> 跳过 $LABEL（已存在日志: $(basename "$EXIST")）"
        continue
    fi

    echo "================================================="
    echo ">>> 开始实验: $LABEL"
    echo ">>> 覆盖参数: $OVERRIDE"
    echo "================================================="

    args=(
        habitat_baselines.eval.split=val
        habitat_baselines.num_environments=1
        habitat.simulator.habitat_sim_v0.gpu_device_id=0
        habitat.simulator.habitat_sim_v0.gpu_gpu=False
        habitat_baselines.rl.policy.name="HabitatITMPolicyV1"
    )
    for ov in $OVERRIDE; do
        args+=("$ov")
    done
    if [ -n "${LIMIT:-}" ]; then
        args+=("habitat_baselines.test_episode_count=$LIMIT")
    fi

    LOGFILE="$OUT_DIR/eval_hm3d_$(date +%Y-%m-%d)_$(date +%H%M%S)_${LABEL}.log"
    ARM_START=$(date +%s)

    python -m vlfm.run "${args[@]}" 2>&1 | tee "$LOGFILE"

    ARM_END=$(date +%s)
    ELAPSED=$((ARM_END - ARM_START))
    echo ">>> 完成: $LABEL -> $(basename "$LOGFILE")  用时 $(printf '%d:%02d:%02d' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)))"
    echo ">>> 结果摘要（末行）:"
    grep -E "^(Success rate|Oracle Success Rate)" "$LOGFILE" | tail -2 | sed 's/^/    /'
done

TOTAL_END=$(date +%s)
TOTAL=$((TOTAL_END - TOTAL_START))
echo ""
echo "================================================="
echo "队列完成！总用时 $(printf '%d:%02d:%02d' $((TOTAL/3600)) $(((TOTAL%3600)/60)) $((TOTAL%60)))"
echo "日志目录: $OUT_DIR"
echo "--- 各档末行摘要 ---"
for f in "$OUT_DIR"/eval_hm3d_*_abl_*.log; do
    [ -e "$f" ] || continue
    printf "%-52s " "$(basename "$f")"
    grep -E "^(Success rate|Oracle Success Rate)" "$f" | tail -2 | tr '\n' ' '
    echo ""
done
echo "================================================="
# bash ./scripts/launch_vlm_servers.sh
# bash ./scripts/ablation_hm3d.sh
