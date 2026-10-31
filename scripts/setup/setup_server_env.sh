#!/usr/bin/env bash
# Ubuntu/Debian x86_64: reproduce this project's environment, excluding ref_thirdparty.
# Run as your normal login user: bash scripts/setup/setup_server_env.sh
set -Eeuo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo '请用 bash 执行安装脚本，不要 source；安装后再 source 生成的激活脚本。' >&2
    return 1
fi

PROJECT_DIR="${SUPERMIMIC_PROJECT_DIR:-/home/jaka/Workspace/SuperMimic-Reproduced-jaka}"
UV_VERSION='0.13.6'
PYTHON_VERSION='3.12.16'
SKIP_SYSTEM=0
SKIP_RENDER=0
CHECK_ONLY=0
TEMP_DIR=''
LOG_FILE='尚未创建日志'

usage() {
    cat <<'HELP'
用法：bash scripts/setup/setup_server_env.sh [选项]

默认项目目录：/home/jaka/Workspace/SuperMimic-Reproduced-jaka
默认安装：系统库、uv 0.13.6、Python 3.12.16、uv.lock 中全部项目/开发依赖。
默认验证：版本、项目导入、PyTorch CUDA/compile、Warp、MuJoCo Warp、EGL 渲染。
需要联网；缺少系统包时会调用 sudo apt-get，可能要求输入 sudo 密码。
不安装 NVIDIA 驱动、系统 CUDA Toolkit、ref_thirdparty 环境，不下载训练数据。
系统包：ca-certificates curl git build-essential pkg-config libegl1 libgl1 libglfw3 ffmpeg

选项：
  --project-dir PATH       指定项目所在目录
  --skip-system-packages  系统依赖已由管理员装好时跳过 apt-get
  --skip-render-check     只验证计算/仿真，不验证 EGL 渲染
  --check-only            仅检查已有 .venv，不安装或同步依赖
  -h, --help              显示帮助

安装成功后：
  cd /home/jaka/Workspace/SuperMimic-Reproduced-jaka
  source .venv/activate_supermimic.sh
  python -c 'import torch; print(torch.__version__, torch.version.cuda)'
HELP
}

die() { echo "[错误] $*" >&2; exit 1; }
cleanup() { if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then rm -rf -- "$TEMP_DIR"; fi; }
on_error() {
    local rc=$?
    echo "[失败] 第 $1 行执行失败，退出码 $rc。完整日志：$LOG_FILE" >&2
    exit "$rc"
}
trap cleanup EXIT
trap 'on_error "$LINENO"' ERR

while (($#)); do
    case "$1" in
        --project-dir)
            (($# >= 2)) || die '--project-dir 缺少路径'
            PROJECT_DIR="$2"; shift 2 ;;
        --skip-system-packages) SKIP_SYSTEM=1; shift ;;
        --skip-render-check) SKIP_RENDER=1; shift ;;
        --check-only) CHECK_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "未知选项：$1（使用 --help 查看帮助）" ;;
    esac
done

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || \
    die '本脚本用于 Linux x86_64；其他平台需要单独适配。'
[[ -d "$PROJECT_DIR" ]] || die "项目目录不存在：$PROJECT_DIR。请先把项目复制到服务器。"
cd -- "$PROJECT_DIR"
PROJECT_DIR="$(pwd -P)"
for required in pyproject.toml uv.lock src/supermimic; do
    [[ -e "$required" ]] || die "缺少 $required，请复制完整项目（包括当前 uv.lock）。"
done

STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="$PROJECT_DIR/logs/setup_server"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$STAMP.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[开始] 项目目录：$PROJECT_DIR"
echo "[日志] $LOG_FILE"

# Avoid importing ROS/Conda/system Python packages or targeting an external venv.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV UV_PROJECT_ENVIRONMENT UV_PYTHON UV_SYSTEM_PYTHON
unset UV_NO_MANAGED_PYTHON
export PYTHONNOUSERSITE=1
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/.venv"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

command -v nvidia-smi >/dev/null || die '找不到 nvidia-smi，请先由管理员配置好 NVIDIA 驱动。'
echo '[检查] NVIDIA GPU 与驱动（CUDA 13.0 至少需要 Linux 驱动 580.65.06）'
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
DRIVERS="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
[[ -n "$DRIVERS" ]] || die 'nvidia-smi 未发现 GPU。'
while IFS= read -r driver; do
    [[ "$driver" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || die "无法解析驱动版本：$driver"
    major=$((10#${BASH_REMATCH[1]}))
    minor=$((10#${BASH_REMATCH[2]}))
    patch=$((10#${BASH_REMATCH[3]}))
    if ((major < 580 || (major == 580 && minor < 65) || (major == 580 && minor == 65 && patch < 6))); then
        die "驱动 $driver 过旧，需要至少 580.65.06；本脚本不会修改驱动。"
    fi
done <<< "$DRIVERS"

if (( ! CHECK_ONLY && ! SKIP_SYSTEM )); then
    command -v apt-get >/dev/null && command -v dpkg-query >/dev/null || \
        die '自动安装系统包需要 Ubuntu/Debian；其他系统请预装帮助中的依赖并用 --skip-system-packages。'
    packages=(ca-certificates curl git build-essential pkg-config libegl1 libgl1 libglfw3 ffmpeg)
    missing=()
    for package in "${packages[@]}"; do
        if [[ "$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true)" != 'install ok installed' ]]; then
            missing+=("$package")
        fi
    done
    if ((${#missing[@]})); then
        echo "[安装] 缺少的系统包：${missing[*]}"
        as_root=()
        if ((EUID != 0)); then
            command -v sudo >/dev/null || die '缺少 sudo。请管理员安装上述系统包后，用 --skip-system-packages 重试。'
            sudo -v
            as_root=(sudo)
        fi
        "${as_root[@]}" apt-get update
        "${as_root[@]}" apt-get install -y --no-install-recommends "${missing[@]}"
    else
        echo '[检查] 所需系统包已安装'
    fi
fi

TEMP_DIR="$(mktemp -d -t supermimic-setup.XXXXXXXX)"
UV_INSTALL_DIR="$HOME/.local/share/supermimic/uv/$UV_VERSION"
UV_BIN=''
uv_version_matches() {
    local version
    version="$("$1" --version)" || return 1
    [[ "$version" == "uv $UV_VERSION" || "$version" == "uv $UV_VERSION "* ]]
}
for candidate in "$UV_INSTALL_DIR/uv" "$(command -v uv || true)"; do
    if [[ -x "$candidate" ]] && uv_version_matches "$candidate"; then
        UV_BIN="$candidate"
        break
    fi
done
if [[ -z "$UV_BIN" ]]; then
    (( ! CHECK_ONLY )) || die "仅检查模式需要 uv $UV_VERSION，请先运行默认安装。"
    command -v curl >/dev/null || die '缺少 curl，请先安装系统依赖。'
    echo "[安装] uv $UV_VERSION（用户独立目录，不修改 shell 配置）"
    curl --fail --location --show-error --retry 5 --connect-timeout 30 \
        "https://astral.sh/uv/$UV_VERSION/install.sh" -o "$TEMP_DIR/install_uv.sh"
    UV_UNMANAGED_INSTALL="$UV_INSTALL_DIR" sh "$TEMP_DIR/install_uv.sh"
    UV_BIN="$UV_INSTALL_DIR/uv"
fi
uv_version_matches "$UV_BIN" || die 'uv 版本不符'
echo "[工具] $("$UV_BIN" --version) ($UV_BIN)"

if (( ! CHECK_ONLY )); then
    for tool in git cc c++; do
        command -v "$tool" >/dev/null || die "缺少 $tool，请安装系统依赖后重试。"
    done
    echo "[安装] Python $PYTHON_VERSION（uv 管理，不替换系统 Python）"
    "$UV_BIN" python install --no-bin "$PYTHON_VERSION"
    PYTHON_BIN="$("$UV_BIN" python find --system --managed-python --no-project "$PYTHON_VERSION")"

    # A copied .venv has absolute paths. Preserve it rather than reuse a broken environment.
    if [[ -e .venv || -L .venv ]]; then
        if [[ ! -x .venv/bin/python || ! -f .venv/bin/activate ]] || \
            ! .venv/bin/python -I -c 'import sys; sys.exit(sys.version_info[:3] != (3, 12, 16))' || \
            ! grep -Fq -- "$PROJECT_DIR/.venv" .venv/bin/activate; then
            BACKUP="$LOG_DIR/venv_backup_$STAMP"
            echo "[备份] 已有 .venv 版本不符、损坏或来自其他目录，移动至 $BACKUP"
            mv -- .venv "$BACKUP"
        fi
    fi

    echo '[安装] 按 uv.lock 安装项目及 dev 依赖；大体积 CUDA wheel 下载可能需要较长时间'
    "$UV_BIN" sync --locked --group dev --python "$PYTHON_BIN" --managed-python \
        --reinstall-package supermimic
fi

VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
[[ -x "$VENV_PYTHON" ]] || die '项目 .venv 不存在，请先运行默认安装。'
echo '[检查] 依赖一致性'
"$UV_BIN" pip check --python "$VENV_PYTHON"

# The verifier is embedded below: only this .sh and the project itself are needed.
# Warp requires its kernels to be in an actual source file, not Python stdin.
cat > "$TEMP_DIR/verify_server_env.py" <<'PY_VERIFY'
"""Verify the locked environment without datasets or ref_thirdparty imports."""

import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import sys
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--skip-render-check", action="store_true")
args = parser.parse_args()
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
if not args.skip_render_check:
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    print("[验证] Python 与锁定依赖版本", flush=True)
    require(sys.version_info[:3] == (3, 12, 16), f"需要 Python 3.12.16，实际 {sys.version}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    expected = {
        "torch": "2.14.1+cu130",
        "mujoco": "3.16.0",
        "mujoco-warp": "3.16.0",
        "warp-lang": "1.18.0",
        "mjlab": "1.6.0",
    }
    for package, version in expected.items():
        actual = metadata.version(package)
        print(f"{package}: {actual}")
        require(actual == version, f"{package} 需要 {version}，实际 {actual}")
    source = metadata.distribution("mjlab").read_text("direct_url.json")
    commit = json.loads(source or "{}").get("vcs_info", {}).get("commit_id")
    require(commit == "8bb2bc9e187797689ec82c78c35019afd2fd0376", f"mjlab git commit不符: {commit}")
    print(f"mjlab commit: {commit}")
    for module in (
        "numpy", "scipy", "tensorboard", "tyro", "huggingface_hub", "pandas",
        "pyarrow", "tqdm", "joblib", "PIL", "pytest", "mujoco_warp", "mjlab",
        "supermimic.training.loop", "supermimic.scripts.train",
    ):
        importlib.import_module(module)
    print("项目训练模块与主要依赖导入通过", flush=True)

    import numpy as np
    import torch
    import warp as wp
    import mujoco
    import mujoco_warp as mw
    from supermimic.data.motion_convert import load_g1_model

    require(torch.version.cuda == "13.0", f"PyTorch CUDA需要13.0，实际 {torch.version.cuda}")
    require(torch.cuda.is_available(), "PyTorch 无可用 CUDA GPU，请检查驱动和 CUDA_VISIBLE_DEVICES")
    ordinal = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(ordinal)
    print(f"GPU: {torch.cuda.get_device_name(ordinal)}, CC={capability}, torch CUDA={torch.version.cuda}")
    require(capability >= (7, 5), f"Warp CUDA13需要 Turing/CC7.5+，实际 {capability}")
    print("[验证] PyTorch CUDA矩阵运算与 torch.compile（首次可能需要几分钟）", flush=True)
    a = torch.arange(256, device="cuda", dtype=torch.float32).reshape(16, 16) / 256
    b = torch.eye(16, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(a @ b, a)
    compiled = torch.compile(lambda x: torch.sin(x) + x * x, fullgraph=True)
    torch.testing.assert_close(compiled(a), torch.sin(a) + a * a)
    torch.cuda.synchronize()
    print("PyTorch CUDA与编译执行通过", flush=True)

    print("[验证] Warp CUDA内核", flush=True)
    wp.init()
    print(f"Warp Toolkit={wp.get_cuda_toolkit_version()}, Driver API={wp.get_cuda_driver_version()}")
    device = wp.get_device(f"cuda:{ordinal}")
    values = wp.zeros(16, dtype=wp.float32, device=device)
    wp.launch(check_warp_kernel, dim=16, inputs=[values], device=device)
    wp.synchronize_device(device)
    np.testing.assert_array_equal(values.numpy(), np.arange(16, dtype=np.float32) + 1)
    print("Warp真实CUDA内核通过", flush=True)

    print("[验证] MuJoCo Warp自由落体10步", flush=True)
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.002" gravity="0 0 -9.81"/>
      <worldbody>
        <light pos="0 -2 4"/>
        <geom type="plane" size="3 3 .1"/>
        <body pos="0 0 2"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
      </worldbody>
    </mujoco>''')
    data = mujoco.MjData(model)
    with wp.ScopedDevice(device):
        gpu_model = mw.put_model(model)
        gpu_data = mw.put_data(model, data, nworld=1, nconmax=8, njmax=16)
        for _ in range(10):
            mw.step(gpu_model, gpu_data)
        wp.synchronize_device(device)
        qpos = gpu_data.qpos.numpy()
        qvel = gpu_data.qvel.numpy()
    require(np.isfinite(qpos).all() and np.isfinite(qvel).all(), "MuJoCo Warp状态出现NaN/Inf")
    require(1.99 < float(qpos[0, 2]) < 2.0, f"自由落体位置异常: {qpos}")
    require(float(qvel[0, 2]) < 0, f"自由落体速度异常: {qvel}")
    print(f"MuJoCo Warp通过，z={qpos[0, 2]:.8f}, vz={qvel[0, 2]:.8f}", flush=True)

    g1 = load_g1_model()
    require((g1.nq, g1.nv) == (36, 35), "G1模型自由度不符合项目要求")
    print(f"G1模型资产加载通过：nq={g1.nq}, nv={g1.nv}", flush=True)
    if not args.skip_render_check:
        print("[验证] EGL无显示器离屏渲染", flush=True)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=64, width=64) as renderer:
            renderer.update_scene(data)
            pixels = renderer.render()
        require(pixels.shape == (64, 64, 3) and pixels.max() > 0, "EGL渲染图像为空或形状错误")
        print("EGL离屏渲染通过", flush=True)
    else:
        print("已按参数跳过EGL渲染检查")
    print("[成功] 锁定版本、项目导入、PyTorch/Warp/MuJoCo Warp GPU检查全部通过", flush=True)


# A real source file is required so Warp can inspect and compile this kernel.
import warp as wp


@wp.kernel
def check_warp_kernel(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = float(i + 1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[失败] 环境验证失败，上一条[验证]信息指出失败阶段：", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
PY_VERIFY
verify_options=()
if ((SKIP_RENDER)); then verify_options+=(--skip-render-check); fi
"$VENV_PYTHON" "$TEMP_DIR/verify_server_env.py" "${verify_options[@]}"

if (( ! CHECK_ONLY )); then
    # Keep machine-specific absolute paths inside the ignored .venv directory.
    {
        echo '#!/usr/bin/env bash'
        echo '# Generated by setup_server_env.sh. Use: source .venv/activate_supermimic.sh'
        echo 'unset PYTHONPATH PYTHONHOME'
        echo 'export PYTHONNOUSERSITE=1'
        printf 'export PATH=%q:"$PATH"\n' "$(dirname "$UV_BIN")"
        printf 'export UV_PROJECT_ENVIRONMENT=%q\n' "$PROJECT_DIR/.venv"
        printf 'source %q\n' "$PROJECT_DIR/.venv/bin/activate"
        echo 'export MUJOCO_GL="${MUJOCO_GL:-egl}"'
        printf 'cd -- %q\n' "$PROJECT_DIR"
    } > "$PROJECT_DIR/.venv/activate_supermimic.sh"
fi

echo '[完成] 环境检查通过：Python 3.12.16 / PyTorch 2.14.1+cu130 / PyTorch CUDA 13.0'
if (( ! CHECK_ONLY )); then
    echo '后续每次登录后执行：'
    printf '  cd -- %q\n' "$PROJECT_DIR"
    echo '  source .venv/activate_supermimic.sh'
    echo '训练数据请另行复制到 data/，机器人资产请保留 assets/。安装日志：'
fi
echo "$LOG_FILE"
