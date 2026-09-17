#!/bin/bash

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

# ----------------- 输出目录与数据集 -----------------
OUT_DIR="$PROJECT_ROOT/outputs/VLVM-V8_1"
mkdir -p "$OUT_DIR"
# 单场景（5cdE 单场景调参基线）：
SCENES="[5cdEh9F2hJL]"
# 三场景：
# SCENES="[TEEsavR23oF, mv2HUxq3B53, wcojb4TFT35]"

# ----------------- 实验定义（2026-09-16 · 双机两批 · 方向 3）-----------------
#
# 定稿基座（不变）：occ 门 + density 门 + GD 辅助机制（gdp_enable / every_n=5 / max_boxes=1）
#   单场景 5cdE 99 集；对照档 = `d2_anchor`（09-15，51/99，Oracle 56/99）——
#   本批不重跑本地 anchor（用户 09-16 定；d2_anchor 之后的代码改动均为零行为影子/量测）。
#
# 本轮全部实验（共 6 档）分两组、两机并行（每档 ~4 h；依据 §优化方向 3）：
#   用法：服务器 1  `BATCH=A bash scripts/my_eval.sh`
#         服务器 2  `BATCH=B bash scripts/my_eval.sh`
#   09-16 晚修订：d2_scan / d2_lmo3 已跑（均判负收口）；d2_geolock / d2_merg3 / d2_fill
#      撤档（依据 = 影子量测：作用面 / 判别力不足；见 results/VLVM-V8_1.md §四）。
#   09-16 深夜 v2：B 组 = 「参数优化批」（7 档全零代码）。
#   09-16 深夜 v3：A 组 = 影子基线（`d2_shadow`）→ M2 停机帧 GD 复核（`d2_gdv`）；
#      M1/M3 影子随档恒开（§五-5）。
#
# ============================================================================
# 【09-17 09:5x 更新 · 机制批收口 ⇒ A/B 两机 = **参数优化批 3/3 平均分摊**】
#   参数批（单变量、**对拍 V9 新基线 52/99**；每档 ~4 h）：
#     A（服务器 1）= `d2_pick1b`（D2-2′ band=1.0）→ `d2_sig72`（σ_tar 0.70→0.72）→ `d2_merg8`（merge 0.5→0.8）
#     B（服务器 2）= `d2_pick1`（band=0.5）→ `d2_budg3`（susp_hysteresis 2→3）→ `d2_hys3`（hysteresis 5→3）
#   依据：V8_1 §五-6 参数批；`p4` caption 已判负（V73_a3_cap −7.08）不排；`d2_8tv` 保持暂缓。
#   09-17 复核：`d2_conf70` 已撕（空转键——GD 开启时新条目恒标 suspicious，该键不参与决策）；
#     V9 失败结构 = 47 失败中 46 为 false_positive（早停锁假）· 逐类 ora=0：couch 15 / tv 19 / toilet 8 / chair 1 / bed 0
#     ⇒ 瓶颈 = 锁正确率（停止时源分布 tsp3d 50 / both 22 / gd 20）⇒ 录取线档 `d2_sig72`
#       （09-17 修订：原 0.75 与 `fb_suspicious_conf` 重合且历史网格 0.75 起单调降 → 改 0.72）。
#   机制批结果（读参数档时对照）：`lockx` 52/99（+1 · steps −26.5）✓ 采纳为新基线（= 上方 `Stage2_1`）；
#     `lockgeo` ≡ lockx（逐集同、零增量）· `lmd4` 48/99（−3）· `lockfse` ≈ 0（全批仅 7 次事件）⇒ 均判负。
#   多场景批（待开）：SCENES 切三场景后启用阵列末尾的 `ms_v9` / `ms_ref`。
# ============================================================================

# 【A 组 · 服务器 1】（09-16 晚更新：scan 判负 ⇒ 3-2 不排；geolock 撤档）
#   a0) d2_shadow   【09-16 深夜新增，待跑】影子基线：定稿配置 + M1/M3 影子（M2 关）。
#                   给 M1（停机 ITM 裁剪/整帧分可分性）与 M3（关联距离直方图）出干净判据；
#                   兼做 anchor 复现（零行为，期望 ≈51±1）。
#   a1) d2_scan    【已跑 · 判负】43/99（−8）· Oracle 56（±0）—— 预注册门（Oracle ≥ +2、
#                   SR 不降）双败；逐类 couch Ora +3 被 bed −2 / toilet −1 抵消 ⇒ 收口。
#                   日志 outputs/VLVM-V8_1/eval_hm3d_2026-09-16_132627_d2_scan.log。
#   a2) d2_stopg    停止闸 v2（3-9，§3.15.12 重设计形态）：`tsp3d` 源 + 类条件
#                   couch|toilet + n_obs<=1 才拦；二次证据（n_obs 增长 / 票型升级）即放行；
#                   窗口 30 步到期 ⇒ 释放路径 = 丢弃该条目回落 explore；配套
#                   `lock_exempt`（navigate 下 nav goal 条目豁免 near_miss，防"丢条目"）。
#                   判据：`stop_blocked` 类分布命中 couch/toilet、这两类 SR↑、
#                   chair/bed 零回归、`stop_abandoned` 后是否重新找到目标。
#   a3) d2_geolock  【撤档（09-16 晚）】几何独立票锁门（3-5）：锁时刻 `gd` 锁事件仅
#                   9/528（1.7%）带 `geo_ind=1`；末态 tp/fp 均值 1.10/1.07（无判别力）
#                   ⇒ 门近似空操作，不占档。
#   a4) d2_gdv      【09-16 深夜新增，待跑】M2 停机帧 GD 复核（§五-5）：停机瞬间强制
#                   一次 GD（绕过 every_n），目标类框与锁定条目共位（`d2_gdv_dist`=0.5）
#                   ⇒ 放行；未确认 ⇒ 拦停（窗口 `d2_gdv_win`=30 步，到期 `release`=
#                   explore：丢条目回落）；配套影子：M1（`[ITMV]` 停机裁剪/整帧 ITM
#                   打分，每条目一次）与 M3（`[ASSOC]` 写入的最近同类距离直方图）恒开。
#                   判据：`gdv_confirm` 占比、被拦集的 fp→succ 转化、Oracle 不降、SR。
#   a5) d2_itmv    【**判据未过 ⇒ 撤档（09-16 23:1x，d2_shadow 40/99）**】M1 停机裁剪复核闸：
#                   证据 = good 0.2467(n=9) vs bad_fp 0.2342(n=7) ⇒ **gap +0.0125 < 0.05**、
#                   **AUC 0.524 < 0.60**；且 35 个停点里 19 个 `vis=False`（无分可判）⇒ 不排。
#   a6) d2_assoc   【判据倾向过门，但需先实现】M3 对象级关联：`nn0.3-0.5` = **7.40%**（23/311，≥5%）、
#                   外推 99 集 **≈57**（≥50）⇒ 过门；**但当前行是空档（= `$Stage2_1`，与 anchor 重复）**，
#                   须先实现 `box8` + IoU/尺寸判据再用真实开关覆盖后才可启用。
#
# 【B 组 · 服务器 2】（09-16 深夜 v3：**机制优先**，参数优化后置；对拍 d2_anchor / b1）
#   ——— 机制批（复活；均与 `lock_exempt` 组合，相对 b1 取单机制增量）———
#   b1) d2_lockx    锁保持单开（基线）：navigate 下 nav goal 条目免 `near_miss`。
#                   依据：`d2_stopg` 归因（闸 0 救 −1；lock_exempt ≈ +3/−2 + steps −22；
#                   删除 197→158；bed/tv 集内闸不作用 ⇒ 干净归因）。
#                   判据：`hysteresis_exceeded` ↓、steps / soft_spl、逐集。
#   b2) d2_lockgeo  【3-5 复活】b1 + `d2_geo_lock=True`：lock_exempt 语境下 geo-ind
#                   从空操作变有判别力（锁时刻 ind=1 占比 4.2%→11.1%；末态 tp 1.78 vs
#                   fp 1.12，ge2 46% vs 8%）。判据：对拍 b1、逐类 fp、`geo_ind` 分布、Oracle。
#   b3) d2_lmd4     【P2a 复活】b1 + `d2_lock_max_dist=4.0`（远距弱证不锁；本地新实现）：
#                   历史 lmd4 = 46/99 · Oracle 57（+2）从未闭环。判据：逐类 fp、Oracle、
#                   近距瞬锁组（stops_within5）。
#   b4) d2_lockfse  b1 + free_space_erasure（**09-16 夜重写 v2：默认 demote**——盒内
#                   已探索且纯空气 ⇒ 标 `fsv`+suspicious，锁定路径跳过该条目；
#                   `free_erasure_hard=True` 才走 V7 直删）。补 lock_exempt 的风险面（kept-fake）。
#                   判据：`[FSE] episode summary` set/clear/hard/skip_lock、逐类 fp、Oracle。
#   ——— 参数优化批（后置；单变量）———
#   p1) d2_budg3    `fb_suspicious_hysteresis=2→3`（同轴预算）；对拍 b1。
#   p2) d2_pick1    `d2_pick_src_first=True`（D2-2′，band 0.5）：离线 20 改选/5 集。
#   p3) d2_pick1b   【暂缓】band=1.0（35 改选/8 集，含 2 成功集=风险面）⇒ 等 p2 读数。
#   p4) d2_capt     【已判负 09-14 撤档】`V73_a3_cap` = 43/99（−7.08）· Oracle 55.56%：
#                   单类 caption 是"放宽"非"收紧"（target_hit +85%、multi_named→0、gd 单票 422→798、
#                   gd 条目 tp% 44.9→19.8、gd fp 49→146）⇒ 不再排。
#   p5) d2_merg8    【暂缓】`gdp_merge_dist=0.8`（写入侧共位放宽；作用面薄，历史写入侧全负）。
#   备选/暂缓：`fb_suspicious_conf=0.70` / `fb_hysteresis=3` / `d2_8tv`（tv 单类，couch 版曾零效应）。
#
#   历史（已收口）：b0 `d2_lmo3` 判负（47/99·Oracle 55；pick_gd 649→217 但 fp 46→51）⇒
#   锁计数轴关闭；`d2_merg3` / `d2_fill` 撤档（作用面不足）。
#   已关闭参数（用户 09-16 复核）：`use_vlfm_nlp` / `fusion_style` 变体（含 `panoramic`，
#   = `enable_scan` 同族）/ `cap_style` / `lock_min_obs`；`fb_near_radius` = 死键
#   （`_init_fallback` 内被 `max_depth*0.5` 覆盖）。
#
# 两机同代码同配置；各档均对照 `d2_anchor`（跨机噪声带 ±1）。
# 后续批次：B 组备选（fb_suspicious_hysteresis=4 / d2_pick_src_band=1.0 / gdp_merge_dist=0.8 /
# d2_gd_skip_classes=tv）；机制规格（停机裁剪复核 M1 / GD 停机复核 M2 / 对象级关联 M3）=
# results/VLVM-V8_1.md §五-5。3-2（scan 参数）、3-6（锁计数轴）、3-10（cap）= 判负收口。
# 【09-17 定稿基线 V9】= 参照基座 + 锁保持单开（`lockx`：52/99 · Oracle 55 · steps 167.1）。
#   依据机制批终判（V8_1 §五-13）：lockx +1/效率 ✓ 采纳；lockgeo ≡ lockx（逐集同）、lmd4 −3、
#   lockfse ≈0（全批仅 7 次事件）⇒ 全部判负。YAML 默认已同步 `lock_exempt: True`。
# 【09-17 清理】已判负/零效应机制代码全部删除：3-4 帧级补框、3-5 geo_lock、3-9 停止闸 v2、
#   M1 ITM 停机复核、M2 停机帧 GD 复核、M3 关联影子、P2a `lock_max_dist`、free-space erasure v2。
#   下方历史注释档引用的键（d2_gdv_* / d2_itm_* / d2_stop_gate* / free_erasure_* 等）已不存在，
#   仅作留档、不可再启用。当前机制键仅：`d2_pick_src_first` / `d2_pick_src_band` / `d2_gd_skip_classes` / `lock_exempt`。

Stage2_1="habitat_baselines.rl.policy.fusion_style=world habitat_baselines.rl.policy.det_seed=0 habitat_baselines.rl.policy.enable_occ_consistency=True habitat_baselines.rl.policy.enable_density_gate=True habitat_baselines.rl.policy.lock_exempt=True"
BATCH="${BATCH:-A}"
if [ "$BATCH" = "B" ]; then
    # 服务器 2
    EXPERIMENTS=(
        # B-p2 D2-2′ 选点序 band=0.5（修正序 both > gd > tsp3d；离线 20 改选/5 集）
        "d2_pick1|$Stage2_1 habitat_baselines.rl.policy.d2_pick_src_first=True"
        # B-p1 可疑预算：`fb_suspicious_hysteresis` 2→3
        "d2_budg3|$Stage2_1 habitat_baselines.rl.policy.fb_suspicious_hysteresis=3"
        # B-备选 可信目标预算：`fb_hysteresis` 5→3（收紧）
        "d2_hys3|$Stage2_1 habitat_baselines.rl.policy.fb_hysteresis=3"
)
else
    # 服务器 1
    EXPERIMENTS=(
        # A-p3 D2-2′ 选点序 band=1.0（修正序 both > gd > tsp3d；离线 35 改选/8 集，含 2 成功集=风险面）
        "d2_pick1b|$Stage2_1 habitat_baselines.rl.policy.d2_pick_src_first=True habitat_baselines.rl.policy.d2_pick_src_band=1.0"
        # A-p6 【09-17 修订 · σ_tar 0.70→0.72】（原 0.75 弃用：① 与 `fb_suspicious_conf=0.75` 重合
        #   ⇒ 可疑判据作用带 [σ_tar, 0.75) 变空集（回退机制置信分档失效）；② 历史网格搜索：
        #   0.70/0.72 并列最高、**0.75/0.78/0.80 单调降**（results/Analysis_VLVM_vs_VLFM.md L155）。
        #   作用面（V9 日志）：98 个新条目中 23 个 c'<0.72（≈23%）将被拦。
        "d2_sig72|$Stage2_1 habitat_baselines.rl.policy.sigma_tar=0.72"
        # A-p5 写入侧共位放宽：`gdp_merge_dist` 0.5→0.8（`both` 生成率 ↑）
        "d2_merg8|$Stage2_1 habitat_baselines.rl.policy.gdp_merge_dist=0.8"
)
fi

echo ">>> 本机批次: BATCH=$BATCH （共 ${#EXPERIMENTS[@]} 档，顺序执行）"
# ----------------- 运行 -----------------
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r LABEL OVERRIDE <<< "$exp"
    echo "================================================="
    echo ">>> 开始实验: $LABEL"
    echo ">>> 覆盖参数: $OVERRIDE"
    echo "================================================="

    args=(
        habitat.dataset.content_scenes="$SCENES"
        habitat_baselines.eval.split=val
        habitat_baselines.num_environments=1
        habitat.simulator.habitat_sim_v0.gpu_device_id=0
        habitat.simulator.habitat_sim_v0.gpu_gpu=False
        habitat_baselines.rl.policy.name="HabitatITMPolicyV1"
    )
    # 拆分为多个 hydra 覆盖（空格分隔，值无空格）
    for ov in $OVERRIDE; do
        args+=("$ov")
    done

    LOGFILE="$OUT_DIR/eval_hm3d_$(date +%Y-%m-%d)_$(date +%H%M%S)_${LABEL}.log"
    python -m vlfm.run "${args[@]}" 2>&1 | tee "$LOGFILE"
    echo ">>> 完成: $LABEL -> $(basename "$LOGFILE")"
done

echo ""
echo "================================================="
echo "全部实验完成！"
echo "日志目录: $OUT_DIR"
echo "================================================="
# bash ./scripts/launch_vlm_servers.sh
# bash ./scripts/my_eval.sh
