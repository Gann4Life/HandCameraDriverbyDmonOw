# Installs the optional, experimental WiLoR depth on THIS PC (NVIDIA GPU only).
# The app runs it from Add-ons after showing the license notice; it can also be run by hand:
#   powershell -ExecutionPolicy Bypass -File installers\install-depth.ps1 -Env .venv-wilor `
#     -Requirements requirements.txt -Constraints installers\depth-constraints.txt
# Everything is downloaded from the official sources (python.org/winget, PyPI, download.pytorch.org,
# GitHub, Hugging Face) into the -Env folder. Nothing outside it is changed, except Python 3.10 if it
# has to be installed. Re-running it is safe: pip skips what is already there.
param(
    [switch] $Yes,
    [Parameter(Mandatory)] [string] $Env,
    [Parameter(Mandatory)] [string] $Requirements,
    [Parameter(Mandatory)] [string] $Constraints,
    [int] $ComponentVersion = 1
)
$ErrorActionPreference = "Stop"
$py = Join-Path $Env "Scripts\python.exe"
# Pinned to the versions this release was tested with; GitHub archives, so Git is not needed
$chumpy = "https://github.com/mattloper/chumpy/archive/580566eafc9ac68b2614b64d6f7aaa84eebb70da.zip"
$wilor = "https://github.com/warmshao/WiLoR-mini/archive/ebec42f94c389070cdd7dda6fd1bf0b4a659c960.zip"

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Ask($question) {
    if ($Yes) { return $true }
    return (Read-Host "$question (Y/N)") -match '^(y|yes|s|si)$'
}
function Pip {
    & $py -m pip --disable-pip-version-check @args
    if ($LASTEXITCODE -ne 0) { throw "pip failed: $args" }
}
function Find-Python310 {
    try {
        $exe = & py -3.10 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe) { return "$exe".Trim() }
    } catch {}
    foreach ($c in @("$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
                     "$env:ProgramFiles\Python310\python.exe")) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

if (-not $Yes) {
    Write-Host @"
WiLoR depth (EXPERIMENTAL) - steadier distance from the camera, about 200 ms of extra depth lag.

Needs: an NVIDIA GPU with ~1.5 GB of free VRAM, ~6 GB of disk, and an internet connection.

LICENSE - personal, non-commercial use only. This downloads third-party software that is
NOT part of this project and has its own licenses, which you accept by installing it:
  - WiLoR / WiLoR-mini: CC BY-NC-ND 4.0
  - MANO hand model: non-commercial research license (https://mano.is.tue.mpg.de/license.html)
  - Ultralytics YOLO: AGPL-3.0
Do not use it commercially or redistribute it.
"@ -ForegroundColor Yellow
    if (-not (Ask "Do you accept these licenses and want to install it?")) { Write-Host "Cancelled."; exit 1 }
}

Step "GPU"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} elseif (-not (Ask "No NVIDIA GPU driver found (nvidia-smi). WiLoR will not run without one. Continue anyway?")) {
    exit 1
}

if (-not (Test-Path $py)) {
    Step "Python 3.10"
    $base = Find-Python310
    if (-not $base) {
        if ((Get-Command winget -ErrorAction SilentlyContinue) -and (Ask "Python 3.10 not found. Install it now with winget (official python.org build, for this user only)?")) {
            Write-Host "Installing Python 3.10 with winget (python.org build, this user only)..."
            & winget install -e --id Python.Python.3.10 --scope user --silent --accept-package-agreements --accept-source-agreements
            $base = Find-Python310
        }
    }
    if (-not $base) {
        Write-Host "Install Python 3.10 (64-bit) from https://www.python.org/downloads/release/python-31011/ and try again." -ForegroundColor Red
        exit 1
    }
    Write-Host "Using $base"
    & $base -m venv $Env
    if ($LASTEXITCODE -ne 0) { throw "Could not create the environment" }
}

Step "PyTorch with CUDA (about 2.5 GB, takes a while)"
Pip install --upgrade pip
Pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121

Step "WiLoR and the tracker's dependencies"
Pip install "numpy<2" setuptools==65.5.0 wheel -c $Constraints
# chumpy's setup.py imports pip, which fails inside pip's isolated build env
Pip install --no-build-isolation $chumpy -c $Constraints
# wilor-mini declares chumpy as a Git URL; its other dependencies are listed below instead
Pip install --no-deps $wilor
Pip install -r $Requirements -c $Constraints `
    smplx==0.1.28 timm einops ultralytics==8.1.34 huggingface-hub scikit-image roma dill

Step "Download the models and check the GPU"
& $py -c @"
import warnings; warnings.filterwarnings('ignore')
import torch
assert torch.cuda.is_available(), 'CUDA is not available: update the NVIDIA driver'
print('GPU:', torch.cuda.get_device_name(0))
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
WiLorHandPose3dEstimationPipeline(device=torch.device('cuda'), dtype=torch.float16, verbose=False)
print('WiLoR ready')
"@
if ($LASTEXITCODE -ne 0) { throw "WiLoR could not load, see the message above" }

# The app reads this to tell a finished install from a broken or older one
Set-Content -Path (Join-Path $Env "handcam-depth.json") -Value "{`"version`": $ComponentVersion}" -Encoding ascii
Write-Host "`nDone: WiLoR depth is installed." -ForegroundColor Green
