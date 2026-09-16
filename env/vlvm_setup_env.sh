#!/usr/bin/env bash
# =============================================================================
# VLVM 环境重定向 / 校验脚本
# -----------------------------------------------------------------------------
# 用途：新机解压环境快照（或按指南 pip 装完依赖）后运行，完成：
#       ① 6 个 editable 包的路径重定向（vlfm / GroundingDINO / habitat-lab /
#          habitat-baselines / frontier_exploration / depth_camera_filtering）
#       ② 关键包导入校验  ③ 权重 / 环境变量 / 服务状态清单
#
# 用法：bash vlvm_setup_env.sh              # 默认本机布局
#       bash vlvm_setup_env.sh --fix        # 强制重装 6 个 editable 包
#       VLVM_ROOT=/path/to/vlvm DEPS_ROOT=/path/to/deps bash vlvm_setup_env.sh
# =============================================================================
set -uo pipefail

VLVM_ROOT="${VLVM_ROOT:-/root/autodl-tmp/vlvm}"
DEPS_ROOT="${DEPS_ROOT:-/root/autodl-tmp/vlfm}"
CONDA_ENV="${CONDA_ENV:-vlfm}"
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-/root/miniconda3/envs/${CONDA_ENV}}"
PY="${CONDA_PREFIX_DIR}/bin/python"
PIP="${CONDA_PREFIX_DIR}/bin/pip"
FIX=0
[[ "${1:-}" == "--fix" ]] && FIX=1

hdr() { printf '\n=== %s ===\n' "$1"; }
ok()  { printf '  [ OK ] %s\n' "$1"; }
bad() { printf '  [FAIL] %s\n' "$1"; FAILED=$((FAILED + 1)); }
warn(){ printf '  [WARN] %s\n' "$1"; }
FAILED=0

hdr "0. 路径与环境"
echo "  VLVM_ROOT   = $VLVM_ROOT"
echo "  DEPS_ROOT   = $DEPS_ROOT"
echo "  CONDA_ENV   = $CONDA_ENV   (prefix: $CONDA_PREFIX_DIR)"
if [[ -x "$PY" ]]; then ok "python: $("$PY" -V 2>&1)"; else bad "未找到 python：$PY"; exit 1; fi
[[ -d "$VLVM_ROOT/vlfm" ]] && ok "vlvm 仓库存在" || bad "找不到 $VLVM_ROOT/vlfm"
[[ -d "$DEPS_ROOT" ]] && ok "外部依赖根存在（habitat-lab 等）" || bad "找不到 $DEPS_ROOT（见指南 §3.1）"

# -----------------------------------------------------------------------------
hdr "1. editable 包重定向"
# 格式: 显示名|<安装命令的工作目录>|<包源码目录(期望)>|import 名
#   vlfm 的打包配置在仓库根（pyproject.toml），包源码在 $VLVM_ROOT/vlfm/
EDITABLES=(
  "vlfm|$VLVM_ROOT|$VLVM_ROOT/vlfm|vlfm"
  "GroundingDINO|$VLVM_ROOT/GroundingDINO|$VLVM_ROOT/GroundingDINO/groundingdino|groundingdino"
  "habitat-lab|$DEPS_ROOT/habitat-lab/habitat-lab|$DEPS_ROOT/habitat-lab/habitat-lab/habitat|habitat"
  "habitat-baselines|$DEPS_ROOT/habitat-lab/habitat-baselines|$DEPS_ROOT/habitat-lab/habitat-baselines/habitat_baselines|habitat_baselines"
  "frontier_exploration|$DEPS_ROOT/frontier_exploration|$DEPS_ROOT/frontier_exploration/frontier_exploration|frontier_exploration"
  "depth_camera_filtering|$DEPS_ROOT/depth_camera_filtering|$DEPS_ROOT/depth_camera_filtering/depth_camera_filtering|depth_camera_filtering"
)
for row in "${EDITABLES[@]}"; do
  IFS='|' read -r name workdir src mod <<< "$row"
  if [[ ! -d "$workdir" || ! -d "$src" ]]; then bad "$name 源码缺失: $workdir"; continue; fi
  if [[ ! -e "$workdir/setup.py" && ! -e "$workdir/pyproject.toml" ]]; then
    bad "$name 缺打包配置（setup.py / pyproject.toml）: $workdir"; continue
  fi
  cur="$("$PY" -c "import $mod, os; print(os.path.dirname(os.path.abspath($mod.__file__)))" 2>/dev/null)"
  if [[ "$FIX" -eq 1 || "$cur" != "$(readlink -f "$src")" ]]; then
    echo "  重装 $name  <-  $workdir"
    ( cd "$workdir" && "$PIP" install -e . --no-deps -q ) \
      && ok "$name 已指向 $src" || bad "$name 安装失败（$workdir）"
  else
    ok "$name 已正确指向 $cur"
  fi
done

# -----------------------------------------------------------------------------
hdr "2. 关键包导入校验"
check_import() { # 名称|导入语句（可多行）
  local label="$1" code="$2"
  if "$PY" -c "$code" >/dev/null 2>&1; then ok "$label"; else bad "$label 导入失败"; fi
}
check_import "torch + CUDA" 'import torch; assert torch.cuda.is_available(); print(torch.__version__)'
check_import "numpy==1.26.4" 'import numpy; assert numpy.__version__.startswith("1.26")'
check_import "transformers==4.26.0" 'import transformers; assert transformers.__version__.startswith("4.26")'
check_import "salesforce-lavis (BLIP-2)" 'import lavis'
check_import "MinkowskiEngine" 'import MinkowskiEngine'
check_import "mmcv / mmdet / mmdet3d / mmengine" 'import mmcv, mmdet, mmdet3d, mmengine'
check_import "habitat / habitat_baselines" 'import habitat, habitat_baselines'
check_import "frontier_exploration（配置组注册）" 'import frontier_exploration'
check_import "depth_camera_filtering" 'import depth_camera_filtering'
check_import "groundingdino" 'import groundingdino'
check_import "opencv / open3d / flask" 'import cv2, open3d, flask'
check_import "hydra / omegaconf" 'import hydra, omegaconf'
check_import "项目入口 vlfm.run" "import sys; sys.path.insert(0, '$VLVM_ROOT'); import vlfm.run"
if "$PY" -c 'import psutil' >/dev/null 2>&1; then ok "psutil（可选）"
else warn "psutil 未安装（可选，仅 trainer 内存打印用）"; fi

# -----------------------------------------------------------------------------
hdr "3. 权重与数据"
for f in \
  "data/tsp3d_models/tsp3d_scanrefer.pth|TSP3D 主权重" \
  "data/groundingdino_swint_ogc.pth|GroundingDINO 权重" \
  "data/pointnav_weights.pth|PointNav 权重" \
  "data/dummy_policy.pth|eval 占位权重" ; do
  IFS='|' read -r rel desc <<< "$f"
  if [[ -f "$VLVM_ROOT/$rel" ]]; then ok "$desc  ($(du -h "$VLVM_ROOT/$rel" | cut -f1))"
  else bad "$desc 缺失：$VLVM_ROOT/$rel"; fi
done
for d in data/datasets/objectnav data/scene_datasets/hm3d data/scene_datasets/mp3d; do
  if [[ -e "$VLVM_ROOT/$d" ]]; then
    tgt="$(readlink -f "$VLVM_ROOT/$d" 2>/dev/null || echo "$VLVM_ROOT/$d")"
    n="$(ls "$VLVM_ROOT/$d" 2>/dev/null | wc -l)"
    ok "$d  ($n 项 → $tgt)"
  else bad "$d 缺失或断链（见指南 §3.4）"; fi
done

# -----------------------------------------------------------------------------
hdr "4. 环境变量提示（run 前需设置）"
cat <<EOF
  export PYTHONPATH=$VLVM_ROOT:\$PYTHONPATH
  export HF_ENDPOINT=https://hf-mirror.com
  export TSP3D_DATA_PATH=$VLVM_ROOT/data/tsp3d_models/
  export TSP3D_CHECKPOINT=$VLVM_ROOT/data/tsp3d_models/tsp3d_scanrefer.pth
  export EGL_PLATFORM=surfaceless
  # 端口：TSP3D_PORT=12186 / BLIP2ITM_PORT=12182 / GROUNDING_DINO_PORT=12181
  # scripts/my_eval.sh 已内置以上设置
EOF

hdr "5. VLM 服务状态（三个服务：12186 / 12182 / 12181）"
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
if [[ "$FAILED" -eq 0 ]]; then
  echo "  全部检查通过。启动：bash $VLVM_ROOT/scripts/launch_vlm_servers.sh"
else
  echo "  有 $FAILED 项失败，处理方式见 vlvm环境配置指南.md §4.2 故障速查表"
  exit 1
fi
