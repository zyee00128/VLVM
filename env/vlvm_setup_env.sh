#!/usr/bin/env bash
# =============================================================================
# VLVM 环境一键脚本：自动配置 / 清理残余包 / 环境校验
# -----------------------------------------------------------------------------
# 仓库布局约定（脚本自动探测，无需手改）：
#   WORKSPACE_ROOT/                 <- 工作区，含下列三部分
#     ├── vlvm/                     <- 仓库根（= VLVM_ROOT，含 pyproject.toml）
#     ├── habitat-lab-<sha>/         <- habitat-lab + habitat-baselines 源码
#     ├── MinkowskiEngine/           <- 稀疏卷积源码（sm_120 已打补丁）
#     └── mmcv/                      <- MMCV 源码（MMCV_WITH_OPS=1 已编译）
#
# 用法：
#   bash vlvm_setup_env.sh                  # = --check，只校验不改动
#   bash vlvm_setup_env.sh --setup           # 自动配置：补 editable / 装缺失直接依赖
#   bash vlvm_setup_env.sh --clean           # 清理残余包（限 conda 环境 vlvm + 仓库 vlvm/）
#   bash vlvm_setup_env.sh --all             # setup → clean → check 全流程
#   bash vlvm_setup_env.sh --fix             # 强制重装 editable 包
#   bash vlvm_setup_env.sh --clean --dry-run # 只列出将删除的项，不实际删除
#
# 清理范围（严格限定，不外溢）：
#   ① conda 环境 `vlvm` 内、且 VLVM 运行链路零引用的 pip 包（系统级安装跳过）
#   ② 仓库 $VLVM_ROOT 内的 __pycache__ / *.egg-info / build 等构建残留
#   工作区其他目录（MinkowskiEngine / mmcv / habitat-lab-*）不在清理范围内。
#
# 覆盖路径（一般无需设置，脚本自动探测）：
#   VLVM_ROOT=<仓库根> WORKSPACE_ROOT=<上层工作区> DEPS_ROOT=<外部源码根>
#   CONDA_ENV=vlvm CONDA_ROOT=<miniconda 根>
# =============================================================================
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLVM_ROOT="${VLVM_ROOT:-$(cd "$SELF_DIR/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$VLVM_ROOT/.." && pwd)}"

# 外部源码根：工作区根，或其中某个含 habitat-lab*/habitat-lab 的子目录
if [[ -z "${DEPS_ROOT:-}" ]]; then
  DEPS_ROOT="$WORKSPACE_ROOT"
  for d in "$WORKSPACE_ROOT"/*/; do
    if compgen -G "${d}habitat-lab*/habitat-lab/habitat" >/dev/null 2>&1; then DEPS_ROOT="${d%/}"; break; fi
  done
fi

# conda 根：环境变量 > conda info --base > 常见路径
if [[ -z "${CONDA_ROOT:-}" ]]; then
  CONDA_ROOT="$(conda info --base 2>/dev/null || true)"
fi
for cand in "$CONDA_ROOT" "$HOME/miniconda3" "$HOME/anaconda3" /root/miniconda3 /opt/conda; do
  [[ -n "$cand" && -d "$cand/envs" ]] && { CONDA_ROOT="$cand"; break; }
done

CONDA_ENV="${CONDA_ENV:-vlvm}"
[[ -d "$CONDA_ROOT/envs/$CONDA_ENV" ]] || CONDA_ENV="${FALLBACK_ENV:-vlfm}"
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-$CONDA_ROOT/envs/$CONDA_ENV}"
PY="${CONDA_PREFIX_DIR}/bin/python"
PIP="${CONDA_PREFIX_DIR}/bin/pip"

DO_SETUP=0; DO_CLEAN=0; DO_CHECK=0; FORCE_EDITABLE=0; DRY_RUN=0; INSTALL_MISSING=0
[[ $# -eq 0 ]] && DO_CHECK=1
for a in "$@"; do
  case "$a" in
    --setup)           DO_SETUP=1 ;;
    --clean)           DO_CLEAN=1 ;;
    --check)           DO_CHECK=1 ;;
    --all)             DO_SETUP=1; DO_CLEAN=1; DO_CHECK=1 ;;
    --fix)             FORCE_EDITABLE=1 ;;
    --install-missing) INSTALL_MISSING=1 ;;
    --dry-run)         DRY_RUN=1 ;;
    -h|--help)         sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "未知参数：$a（-h 查看用法）"; exit 2 ;;
  esac
done

hdr() { printf '\n=== %s ===\n' "$1"; }
ok()  { printf '  [ OK ] %s\n' "$1"; }
bad() { printf '  [FAIL] %s\n' "$1"; FAILED=$((FAILED + 1)); }
warn(){ printf '  [WARN] %s\n' "$1"; }
info(){ printf '  ... %s\n' "$1"; }
FAILED=0

hdr "0. 路径与环境"
echo "  WORKSPACE_ROOT = $WORKSPACE_ROOT"
echo "  VLVM_ROOT      = $VLVM_ROOT"
echo "  DEPS_ROOT      = $DEPS_ROOT"
echo "  CONDA_ROOT     = $CONDA_ROOT"
echo "  CONDA_ENV      = $CONDA_ENV   (prefix: $CONDA_PREFIX_DIR)"
echo "  模式           =$([[ $DO_SETUP -eq 1 ]] && echo ' setup')$([[ $DO_CLEAN -eq 1 ]] && echo ' clean')$([[ $DO_CHECK -eq 1 ]] && echo ' check')$([[ $DRY_RUN -eq 1 ]] && echo ' dry-run')"
if [[ -x "$PY" ]]; then
  ok "python: $("$PY" -V 2>&1)"
else
  if [[ $DO_SETUP -eq 1 && $DRY_RUN -eq 0 && -x "$CONDA_ROOT/bin/conda" ]]; then
    info "conda 环境 $CONDA_ENV 不存在，按指南 §1.1 创建（python=3.9）"
    "$CONDA_ROOT/bin/conda" create -n "$CONDA_ENV" python=3.9 -y \
      && ok "已创建 $CONDA_PREFIX_DIR" || { bad "conda create 失败"; exit 1; }
    "$PY" -V >/dev/null 2>&1 || { bad "创建后仍无 python：$PY"; exit 1; }
    warn "环境为空：请先按 env/vlvm_requirements.txt 安装依赖（指南 §1.2–§2.4 逐步编译），再跑 --setup"
  else
    bad "未找到 python：$PY（先 conda create -n $CONDA_ENV python=3.9，或加 --setup）"; exit 1
  fi
fi
[[ -f "$VLVM_ROOT/pyproject.toml" ]] && ok "vlvm 仓库根（pyproject.toml 存在）" || bad "找不到 $VLVM_ROOT/pyproject.toml"
[[ -d "$VLVM_ROOT/vlfm" ]] && ok "vlfm 包目录存在" || bad "找不到 $VLVM_ROOT/vlfm"
[[ -d "$DEPS_ROOT" ]] && ok "外部依赖根存在（habitat-lab 等）" || bad "找不到 $DEPS_ROOT（见指南 §3.1）"
if command -v nvidia-smi >/dev/null 2>&1; then
  ok "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
else warn "nvidia-smi 不可用（无 GPU 或驱动未装）"; fi
df -h "$WORKSPACE_ROOT" | tail -1 | awk '{printf "  ... 磁盘可用：%s / %s (%s)\n", $4, $2, $5}'

# -----------------------------------------------------------------------------
# 自动配置：editable 包路径对齐（--setup / --fix 生效）
# -----------------------------------------------------------------------------
HAB_SRC=""
for d in "$DEPS_ROOT"/habitat-lab*/; do
  [[ -d "${d}habitat-lab/habitat" ]] && { HAB_SRC="${d%/}"; break; }
done
[[ -z "$HAB_SRC" ]] && warn "在 $DEPS_ROOT 下未找到 habitat-lab-*/habitat-lab（见指南 §3.1）"

# 格式: 显示名|<安装命令的工作目录>|<包源码目录(期望)>|import 名
EDITABLES=(
  "vlfm|$VLVM_ROOT|$VLVM_ROOT/vlfm|vlfm"
  "GroundingDINO|$VLVM_ROOT/GroundingDINO|$VLVM_ROOT/GroundingDINO/groundingdino|groundingdino"
  "habitat-lab|${HAB_SRC}/habitat-lab|${HAB_SRC}/habitat-lab/habitat|habitat"
  "habitat-baselines|${HAB_SRC}/habitat-baselines|${HAB_SRC}/habitat-baselines/habitat_baselines|habitat_baselines"
)
# frontier_exploration / depth_camera_filtering 无公开 pip 包，且源码不随工作区分发：
# 若源码目录存在则恢复 editable，否则退化为「仅校验可导入」（见 §1 与指南 §3.1）。

hdr "1. editable 包重定向"
if [[ $DO_SETUP -eq 0 && $FORCE_EDITABLE -eq 0 ]]; then
  info "未指定 --setup/--fix，仅校验指向"
fi
for row in "${EDITABLES[@]}"; do
  IFS='|' read -r name workdir src mod <<< "$row"
  if [[ -z "$workdir" || ! -d "$workdir" || ! -d "$src" ]]; then bad "$name 源码缺失: $workdir"; continue; fi
  if [[ ! -e "$workdir/setup.py" && ! -e "$workdir/pyproject.toml" ]]; then
    bad "$name 缺打包配置（setup.py / pyproject.toml）: $workdir"; continue
  fi
  cur="$("$PY" -c "import $mod, os; print(os.path.dirname(os.path.abspath($mod.__file__)))" 2>/dev/null)"
  want="$(readlink -f "$src")"
  if [[ "$cur" == "$want" && $FORCE_EDITABLE -eq 0 ]]; then ok "$name 已正确指向 $cur"; continue; fi
  if [[ $DO_SETUP -eq 0 && $FORCE_EDITABLE -eq 0 ]]; then bad "$name 指向错误（$cur ≠ $want），加 --setup 修复"; continue; fi
  echo "  重装 $name  <-  $workdir"
  ( cd "$workdir" && "$PIP" install -e . --no-deps -q ) \
    && ok "$name 已指向 $src" || bad "$name 安装失败（$workdir）"
done
# 源码缺失的两个包：仅提示状态
for pair in "frontier_exploration:$DEPS_ROOT/frontier_exploration" "depth_camera_filtering:$DEPS_ROOT/depth_camera_filtering"; do
  mod="${pair%%:*}"; src="${pair#*:}"
  if "$PY" -c "import $mod" >/dev/null 2>&1; then
    loc="$("$PY" -c "import $mod,os;print(os.path.dirname(os.path.abspath($mod.__file__)))" 2>/dev/null)"
    if [[ -d "$src" ]]; then
      if [[ $DO_SETUP -eq 1 ]]; then
        ( cd "$src" && "$PIP" install -e . --no-deps -q ) && ok "$mod 已 editable 指向 $src" || bad "$mod 重装失败"
      else ok "$mod 可导入（源码在 $src，未重定向）"; fi
    else
      ok "$mod 可导入（site-packages 快照副本：$loc；源码未随工作区迁入）"
    fi
  else
    bad "$mod 不可导入：需从旧环境取回源码后 pip install -e（见指南 §3.1）"
  fi
done

# -----------------------------------------------------------------------------
# 关键包导入 / 版本校验（--check；--install-missing 时对缺失项调用 pip）
# -----------------------------------------------------------------------------
PIP_TARGETS=()   # 缺失且可 pip 直装的条目
check_import() { # 名称|导入语句|可 pip 安装的发行名（可选，,分隔）
  local label="$1" code="$2" pkgspec="${3:-}"
  if "$PY" -c "$code" >/dev/null 2>&1; then ok "$label"; else
    bad "$label 导入失败"
    [[ -n "$pkgspec" ]] && PIP_TARGETS+=("$pkgspec")
  fi
}
check_ver() { # 显示名|发行名|期望前缀
  local label="$1" dist="$2" want="$3"
  local got
  got="$("$PY" -c "import importlib.metadata as m; print(m.version('$dist'))" 2>/dev/null)"
  if [[ -z "$got" ]]; then bad "$label 未安装"; return; fi
  if [[ "$got" == "$want"* ]]; then ok "$label = $got"; else bad "$label = $got（期望 $want*，见指南 §2 版本锁定）"; fi
}

hdr "2. 关键包导入与版本校验"
# 版本期望按 **已装 torch 主版本** 选表：两套 CUDA 栈均已实机验证（见 env/vlvm_requirements.txt）。
TORCH_MAJOR="$("$PY" -c "import torch; print(str(torch.__version__).split('.')[0])" 2>/dev/null || true)"
if [[ "$TORCH_MAJOR" == "2" ]]; then
  STACK="B：RTX 5080 / cu128（torch 2.8）"
  WANT_TORCH="2.8.0"; WANT_TV="0.23.0"; WANT_MMENGINE="0.10.7"
elif [[ "$TORCH_MAJOR" == "1" ]]; then
  STACK="A：RTX 3090 / cu113（torch 1.12）"
  WANT_TORCH="1.12.1"; WANT_TV="0.13.1"; WANT_MMENGINE="0.10.4"
else
  STACK="未知（torch 未安装或无法导入）"
  WANT_TORCH=""; WANT_TV=""; WANT_MMENGINE=""
fi
info "CUDA 栈: $STACK"
check_ver "torch（CUDA 构建）" torch "$WANT_TORCH"
check_ver "torchvision" torchvision "$WANT_TV"
check_ver "numpy（锁 1.x）" numpy "1.26"
check_ver "transformers（锁死）" transformers "4.26.0"
check_ver "tokenizers" tokenizers "0.13.3"
check_ver "timm" timm "0.4.12"
check_ver "MinkowskiEngine" MinkowskiEngine "0.5.4"
check_ver "mmcv" mmcv "2.1.0"
check_ver "mmdet" mmdet "3.2.0"
check_ver "mmdet3d" mmdet3d "1.4.0"
check_ver "mmengine" mmengine "$WANT_MMENGINE"
check_import "torch CUDA 可用" 'import torch; assert torch.cuda.is_available(), "CUDA 不可用"' "torch==${WANT_TORCH:-2.8.0}"
check_import "salesforce-lavis (BLIP-2 ITM)" 'import lavis' "salesforce-lavis==1.0.2"
check_import "habitat / habitat_baselines" 'import habitat, habitat_baselines'
check_import "habitat-sim（EGL 渲染栈）" 'import habitat_sim' "habitat-sim==0.2.4"
check_import "frontier_exploration（配置组注册）" 'import frontier_exploration'
check_import "depth_camera_filtering" 'import depth_camera_filtering'
check_import "groundingdino（GD 辅助源）" 'import groundingdino'
check_import "opencv / open3d / flask" 'import cv2, open3d, flask'
check_import "hydra / omegaconf" 'import hydra, omegaconf'
check_import "gym（habitat 依赖）" 'import gym' "gym==0.23.0"
check_import "项目入口 vlfm.run" "import sys; sys.path.insert(0, '$VLVM_ROOT'); import vlfm.run"
check_import "TSP3D 模型 BeaUTyDETR" "import sys; sys.path.insert(0, '$VLVM_ROOT'); from vlfm.tsp3d_models.bdetr import BeaUTyDETR"
if "$PY" -c 'import psutil' >/dev/null 2>&1; then ok "psutil（可选）"
else warn "psutil 未安装（可选，仅 trainer 内存打印用）"; fi

if [[ ${#PIP_TARGETS[@]} -gt 0 ]]; then
  if [[ $INSTALL_MISSING -eq 1 ]]; then
    warn "按 --install-missing 安装缺失项：${PIP_TARGETS[*]}"
    [[ $DRY_RUN -eq 0 ]] && "$PIP" install "${PIP_TARGETS[@]}"
  else
    warn "缺失项 ${#PIP_TARGETS[@]} 个，可加 --install-missing 由脚本按 env/vlvm_requirements.txt 安装"
  fi
fi

# -----------------------------------------------------------------------------
hdr "3. 权重与数据（仓库内 data/ 为软链或本地产物）"
for f in \
  "data/tsp3d_models/model.safetensors|TSP3D 文本分支 RoBERTa 权重" \
  "data/tsp3d_models/config.json|TSP3D 文本分支配置" \
  "data/groundingdino_swint_ogc.pth|GroundingDINO 权重（GD 辅助源）" \
  "data/pointnav_weights.pth|PointNav 底层策略权重" \
  "data/dummy_policy.pth|habitat-baselines eval 占位权重" ; do
  IFS='|' read -r rel desc <<< "$f"
  if [[ -f "$VLVM_ROOT/$rel" ]]; then ok "$desc  ($(du -h "$VLVM_ROOT/$rel" | cut -f1))"
  else warn "$desc 缺失：$VLVM_ROOT/$rel"; fi
done
# 主权重：按在位文件探测（两机档名不同，见上注释）
_found_ck=""
for _n in tsp3d_scanrefer.pth ckpt_sr3d.pth ckpt_nr3d.pth; do
  [[ -f "$VLVM_ROOT/data/tsp3d_models/$_n" ]] && { _found_ck="$_n"; break; }
done
if [[ -n "$_found_ck" ]]; then
  ok "TSP3D 主权重探测命中: $_found_ck ($(du -h "$VLVM_ROOT/data/tsp3d_models/$_found_ck" | cut -f1))（可用 TSP3D_CHECKPOINT=<path> 显式切换）"
else
  warn "三个候选权重名均未在 $VLVM_ROOT/data/tsp3d_models/ 命中；"
  warn "  启动服务前请：export TSP3D_CHECKPOINT=<实际 .pth 路径>"
fi

# 数据/软链检查：两机布局不同（旧机 hm3d 直链 versioned_data；新机 hm3d_v0.2 名义
# 目录），每个条目给多候选，任一在位即 OK；断链单独报错。返回 0=命中 1=断链 2=未找到。
_check_link() {
  local name="$1"; shift
  local rel tgt
  for rel in "$@"; do
    [[ -e "$VLVM_ROOT/$rel" ]] || continue
    tgt="$(readlink -f "$VLVM_ROOT/$rel" 2>/dev/null || echo "$VLVM_ROOT/$rel")"
    if [[ -L "$VLVM_ROOT/$rel" && ! -e "$tgt" ]]; then bad "$name 断链：$rel → $tgt"; return 1; fi
    ok "$name：$rel → $tgt"; return 0
  done
  return 2
}
_check_link "ObjectNav 数据集" "data/datasets/objectnav/hm3d/v1" "data/datasets"
[[ $? -eq 2 ]] && warn "  ObjectNav 数据集未就位：候选 data/datasets/objectnav/hm3d/v1（见指南 §3.3）"
_check_link "HM3D 场景 train" "data/scene_datasets/hm3d_v0.2/train" "data/scene_datasets/hm3d/train"
[[ $? -eq 2 ]] && info "  HM3D train 未就位（旧机仅需 val 即可跑 eval；要跑 train split 时再补）"
_check_link "HM3D 场景 val" "data/scene_datasets/hm3d_v0.2/val" "data/scene_datasets/hm3d/val"
[[ $? -eq 2 ]] && warn "  HM3D val 未就位：候选 hm3d_v0.2/val 或 hm3d/val（见指南 §3.3）"
_check_link "MP3D 场景" "data/scene_datasets/mp3d"
[[ $? -eq 2 ]] && info "  MP3D 未就位（仅 MP3D 相关实验需要）"
_check_link "LAVIS BERT 目录" "data/bert-base-uncased" "data/tsp3d_models/bert-base-uncased"
if [[ $? -eq 2 ]]; then
  _hf_bert="$(ls -d "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--bert-base-uncased 2>/dev/null | head -1)"
  if [[ -n "$_hf_bert" ]]; then info "  LAVIS BERT 目录未建，将走 HF 缓存：$_hf_bert"
  else warn "  LAVIS BERT 目录未建、HF 缓存亦未命中（BLIP-2 ITM 首次运行需联网下载）"; fi
fi

# -----------------------------------------------------------------------------
# 残余包清理（--clean）
# -----------------------------------------------------------------------------
# 判据：vlfm/ 全量 import 扫描（不产生任何引用的包） + importlib.metadata 反向
#       依赖表。凡被 torch / mmcv / mmdet3d / habitat / lavis 等声明为依赖的包
#       （numba / llvmlite / lmdb / fairscale / webdataset / pycocoevalcap /
#        terminaltables / moviepy / typer ...）一律保留，保证 pip check 干净。
# 保护：GroundingDINO 已编译的 _C.*.so、各包 dist/*.whl（可复用产物）不删。
CLEAN_ROS_RE='^(rclpy|rcutils|rcl-interfaces|rosidl-.*|ros2.*|rosbag2-.*|rqt-.*|ament-.*|sros2|rmw-.*|moveit.*|controller-manager.*|launch.*|tf2-.*|xacro|srdfdom|urdfdom-py|python-qt-binding|qt-gui.*|qt-dotgraph|rpyutils|osrf-pycommon|cv-bridge|image-geometry|laser-geometry|resource-retriever|message-filters|angles|ifcfg|parameterized|generate-parameter-library-py|examples-rclpy.*|demo-nodes-py|action-tutorials.*|quality-of-service-demo-py|logging-demo|topic-monitor|teleop-twist-keyboard|joint-state-publisher.*|turtlesim|interactive-markers|.*-msgs|std-srvs)$'

CLEAN_LIST=(
  # 开发/CI 工具（非运行依赖）
  black flake8 mccabe pycodestyle pyflakes pre_commit cfgv identify nodeenv virtualenv distlib mypy_extensions
  # 数据平台 / 竞赛 SDK（openxlab 依赖链，未参与运行）
  openxlab opendatalab opendatasets kaggle oss2 aliyun-python-sdk-core aliyun-python-sdk-kms crcmod pycryptodome jmespath cloudpathlib smart_open
  # 演示 / 可视化（lavis 与 open3d 的 extras 声明，VLVM 代码未引用）
  streamlit dash plotly altair pydeck narwhals
  # 2D 时代遗留：检测后处理与 spacy 分词链（vlfm/ 零引用）
  supervision objectio decord spacy spacy-legacy spacy-loggers thinc blis cymem preshed murmurhash srsly catalogue confection weasel wasabi langcodes
)
# 明确不清理（运行链路声明依赖，缺失会造成 pip check 缺口）：
#   ifcfg（habitat-baselines）· numba/llvmlite/lmdb（habitat-sim/lab/mmcv）
#   fairscale/webdataset（mmdet/habitat-baselines）· pycocoevalcap/terminaltables（mmdet）
#   moviepy（habitat-baselines）· typer/typer-slim（huggingface_hub）· yapf（groundingdino）

clean_pip_residuals() {
  hdr "4. 清理残余 pip 包（--clean）"
  local row skipped
  # 只处理「实际装在当前 conda 环境内」的包；系统级安装（如 /opt/ros/humble）跳过——
  # 它们由系统包管理器维护且无写权限，删不掉也不影响本环境。
  row="$("$PY" - "$CLEAN_ROS_RE" "$(printf '%s\n' "${CLEAN_LIST[@]}")" "$CONDA_PREFIX_DIR" <<'PYEOF'
import importlib.metadata as md, re, sys
ros_re = re.compile(sys.argv[1])
explicit = [l.strip() for l in sys.argv[2].split("\n") if l.strip()]
prefix = sys.argv[3].rstrip("/") + "/"

def norm(n): return re.sub(r"[-_.]+", "-", n).lower()

installed, where = {}, {}
for d in md.distributions():
    n = d.metadata["Name"]
    if not n: continue
    installed[norm(n)] = n
    try: where[norm(n)] = str(d.locate_file(""))
    except Exception: where[norm(n)] = ""

wanted = {installed[k] for k in installed if ros_re.match(k)}
wanted |= {installed[norm(n)] for n in explicit if norm(n) in installed}
inside  = sorted(n for n in wanted if prefix in where.get(norm(n), ""))
outside = sorted(n for n in wanted if prefix not in where.get(norm(n), ""))
print("TARGETS " + " ".join(inside))
print("OUTSIDE " + " ".join(outside))
PYEOF
)"
  local targets_s outside_s
  targets_s="$(sed -n 's/^TARGETS //p' <<< "$row")"
  outside_s="$(sed -n 's/^OUTSIDE //p' <<< "$row")"
  read -r -a targets <<< "$targets_s"
  if [[ -n "$outside_s" ]]; then
    warn "跳过 $(wc -w <<< "$outside_s") 个环境外包（系统级安装，无写权限，不影响本环境）：$(cut -c1-90 <<< "$outside_s")..."
  fi
  if [[ ${#targets[@]} -eq 0 ]]; then ok "本环境内无残余包"; return; fi
  local before
  before="$(du -sm "$CONDA_PREFIX_DIR" | cut -f1)"
  info "待清理 ${#targets[@]} 个包（开发工具 / 数据平台 / 演示栈 / 2D 遗留）"
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  [dry-run] %s\n' "${targets[@]}"
    info "另将清理仓库构建残留（见 §5）"
    return
  fi
  local n=0
  for p in "${targets[@]}"; do
    if "$PIP" uninstall -y -q "$p" >/dev/null 2>&1; then n=$((n + 1)); else warn "卸载失败：$p"; fi
  done
  local after
  after="$(du -sm "$CONDA_PREFIX_DIR" | cut -f1)"
  ok "已清理 $n/${#targets[@]} 个包，环境体积 ${before} MB → ${after} MB（释放 $((before - after)) MB）"
  if ! "$PIP" check >/dev/null 2>&1; then
    warn "pip check 报告依赖声明缺口（lavis/open3d 的 extras 声明，不影响运行链路）："
    "$PIP" check 2>&1 | sed 's/^/        /' | head -10
  else ok "pip check 通过（无声明缺口）"; fi
}

clean_repo_artifacts() {
  hdr "5. 清理仓库构建残留（--clean，范围限 $VLVM_ROOT）"
  # 清理范围严格限定在 vlvm 仓库内：工作区其他目录（MinkowskiEngine / mmcv /
  # habitat-lab-*）属外部源码，不在本脚本管辖内，保持原样。
  local root="$VLVM_ROOT"
  local total=0 n
  for pat in "__pycache__" "*.egg-info" ".ipynb_checkpoints"; do
    n="$(find "$root" -name "$pat" -not -path "*/.git/*" -not -path "*/dist/*" 2>/dev/null | wc -l)"
    [[ "$n" -eq 0 ]] && continue
    total=$((total + n))
    if [[ $DRY_RUN -eq 1 ]]; then info "[dry-run] $root: $n × $pat"
    else find "$root" -name "$pat" -not -path "*/.git/*" -not -path "*/dist/*" \
           -exec rm -rf {} + 2>/dev/null; ok "删除 $n × $pat"; fi
  done
  # 编译中间产物：build/ 可删；dist/*.whl 为可复用产物，保留
  if [[ -d "$root/GroundingDINO/build" ]]; then
    if [[ $DRY_RUN -eq 1 ]]; then info "[dry-run] 删除 GroundingDINO/build ($(du -sh "$root/GroundingDINO/build" | cut -f1))"
    else rm -rf "$root/GroundingDINO/build"; ok "删除编译中间产物 GroundingDINO/build"; fi
  fi
  [[ -d "$root/GroundingDINO/dist" ]] && info "保留可复用 wheel：$(ls "$root/GroundingDINO/dist" 2>/dev/null | head -1)"
  # 保护性检查：GroundingDINO 的 C 扩展必须仍在
  local so
  so="$(find "$root/GroundingDINO/groundingdino" -maxdepth 1 -name "_C*.so" 2>/dev/null | head -1)"
  [[ -n "$so" ]] && ok "GroundingDINO C 扩展在位：$(basename "$so")" \
                 || bad "GroundingDINO C 扩展 _C*.so 缺失（需重编译：cd GroundingDINO && pip install -e .）"
  # Python 缓存字节码文件
  if [[ $DRY_RUN -eq 0 ]]; then
    find "$root" -name "*.pyc" -not -path "*/.git/*" -delete 2>/dev/null
    ok "已清理散落 *.pyc"
  fi
  info "共处理 $total 个缓存/元数据目录"
}

if [[ $DO_CLEAN -eq 1 ]]; then
  clean_pip_residuals
  clean_repo_artifacts
fi

# -----------------------------------------------------------------------------
hdr "6. 环境变量提示（run 前需设置）"
CKPT_HINT="$VLVM_ROOT/data/tsp3d_models/tsp3d_scanrefer.pth"
[[ -f "$CKPT_HINT" ]] || CKPT_HINT="$VLVM_ROOT/data/tsp3d_models/ckpt_sr3d.pth"
ARCH_HINT="12.0"; CUDA_HINT="/usr/local/cuda-12.8"
[[ "$TORCH_MAJOR" == "2" ]] || { ARCH_HINT="8.6"; CUDA_HINT="/usr/local/cuda-11.3"; }
BERT_HINT=""
for _b in data/bert-base-uncased data/tsp3d_models/bert-base-uncased; do
  [[ -d "$VLVM_ROOT/$_b" ]] && { BERT_HINT="$VLVM_ROOT/$_b"; break; }
done
cat <<EOF
  export PYTHONPATH=$VLVM_ROOT:\$PYTHONPATH
  export HF_ENDPOINT=https://hf-mirror.com
  export TSP3D_DATA_PATH=$VLVM_ROOT/data/tsp3d_models/
  export TSP3D_CHECKPOINT=$CKPT_HINT
  ${BERT_HINT:+export LAVIS_BERT_MODEL_PATH=$BERT_HINT}${BERT_HINT:-# （本机无 LAVIS BERT 目录，可不设；LAVIS 走 HF 缓存）}
  export EGL_PLATFORM=surfaceless
  # 端口：TSP3D_PORT=12186 / BLIP2ITM_PORT=12182 / GROUNDING_DINO_PORT=12181
  # scripts/my_eval.sh 与 scripts/launch_vlm_servers.sh 已内置以上设置
EOF
if [[ $DO_SETUP -eq 1 ]]; then
  if [[ -d "$CUDA_HINT" ]]; then
    if [[ $DRY_RUN -eq 1 ]]; then
      info "[dry-run] conda env config vars set -n $CONDA_ENV CUDA_HOME=$CUDA_HINT TORCH_CUDA_ARCH_LIST=$ARCH_HINT"
    else
      "$CONDA_ROOT/bin/conda" env config vars set -n "$CONDA_ENV" \
        CUDA_HOME="${CUDA_HOME:-$CUDA_HINT}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$ARCH_HINT}" >/dev/null 2>&1 \
        && ok "已写入 conda 环境变量 CUDA_HOME / TORCH_CUDA_ARCH_LIST（重进环境生效）" \
        || warn "写入 conda 环境变量失败（可手动 conda env config vars set）"
    fi
  else
    info "跳过 CUDA 环境变量写入：$CUDA_HINT 不存在（非编译机或路径不同）"
  fi
fi

hdr "7. VLM 服务状态（三个服务：12186 / 12182 / 12181）"
if command -v ss >/dev/null 2>&1; then
  ss -ltn 2>/dev/null | grep -E ":(12186|12182|12181)\b" \
    || echo "  三个端口均未监听（启动：bash scripts/launch_vlm_servers.sh）"
else
  # 容器镜像常缺 iproute2，用 python 兜底探测
  "$PY" - <<'PYEOF'
import socket
for name, port in (("TSP3D", 12186), ("BLIP2ITM", 12182), ("GroundingDINO", 12181)):
    s = socket.socket()
    s.settimeout(0.5)
    alive = s.connect_ex(("127.0.0.1", port)) == 0
    s.close()
    print(f"  port {port} ({name:14s}): {'LISTEN' if alive else 'down'}")
PYEOF
fi

# -----------------------------------------------------------------------------
hdr "结果"
echo "  环境：$CONDA_ENV ($CONDA_PREFIX_DIR)"
echo "  仓库：$VLVM_ROOT"
if [[ "$FAILED" -eq 0 ]]; then
  echo "  全部检查通过。启动：bash $VLVM_ROOT/scripts/launch_vlm_servers.sh"
else
  echo "  有 $FAILED 项失败，处理方式见 vlvm环境配置指南.md"
  exit 1
fi
