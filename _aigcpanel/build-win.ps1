# Enable echo
$VerbosePreference = "Continue"

# 工具准备
Get-Module -ListAvailable -Name PowerShellGet -ErrorAction SilentlyContinue
Install-Module -Name PowerShellGet -Force -Scope CurrentUser
Install-Module -Name 7Zip4Powershell -Force -Scope CurrentUser
Import-Module 7Zip4Powershell
# 工具准备

# 环境准备
conda 'shell.powershell' 'hook' | Out-String | Invoke-Expression
conda env list
Remove-Item -Recurse -Force ./_aienv -ErrorAction SilentlyContinue -Verbose
conda create --prefix ./_aienv -y python=3.10
conda activate ./_aienv
# 环境准备

# 初始化环境
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install --no-cache-dir -U openmim
mim install "mmengine==0.10.3"
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"
mim install "mmpose==1.3.1"
pip install gradio==3.50.2

#New-Item -ItemType Directory -Force -Path ./models/musetalk/
#New-Item -ItemType Directory -Force -Path ./models/sd-vae-ft-mse/
New-Item -ItemType Directory -Force -Path ./models/whisper/
#New-Item -ItemType Directory -Force -Path ./models/dwpose/
#New-Item -ItemType Directory -Force -Path ./models/face-parse-bisent/
#Invoke-WebRequest -Uri "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin" -OutFile "./models/musetalk/pytorch_model.bin"
#Invoke-WebRequest -Uri "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json" -OutFile "./models/musetalk/musetalk.json"
#Invoke-WebRequest -Uri "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin" -OutFile "./models/sd-vae-ft-mse/diffusion_pytorch_model.bin"
#Invoke-WebRequest -Uri "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json" -OutFile "./models/sd-vae-ft-mse/config.json"
#Invoke-WebRequest -Uri "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt" -OutFile "./models/whisper/tiny.pt"
#Invoke-WebRequest -Uri "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth" -OutFile "./models/dwpose/dw-ll_ucoco_384.pth"
#Invoke-WebRequest -Uri "https://download.pytorch.org/models/resnet18-5c106cde.pth" -OutFile "./models/face-parse-bisent/resnet18-5c106cde.pth"
New-Item -ItemType Directory -Force -Path  ./_cache/torch/hub/checkpoints
Invoke-WebRequest -Uri "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -OutFile "./_cache/torch/hub/checkpoints/s3fd-619a316812.pth"
# 初始化环境

# 构建
#python -m py_compile app.py
#Move-Item -Path "__pycache__\app.cpython-310.pyc" -Destination "app.pyc"
python -m py_compile scripts/inference.py
Move-Item -Path "scripts\__pycache__\inference.cpython-310.pyc" -Destination "scripts\inference.pyc"
# 构建

# 启动服务
#python webui.pyc
# 启动服务

# 清除文件
Remove-Item -Path "app.py" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "scripts/inference.py" -Force -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force asset -ErrorAction SilentlyContinue -Verbose
Remove-Item -Recurse -Force requirements.txt -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\*.md -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\.git -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\.github -ErrorAction SilentlyContinue
Remove-Item -Path "__pycache__" -Recurse -Force -ErrorAction SilentlyContinue
# 清除文件

# 打包服务
$VERSION = python -m _aigcpanel.build
Write-Output "VERSION: $VERSION"
$VERSION_ARCH = ($VERSION -split '-')[0..1] -join '-'
Write-Output "VERSION: $VERSION"
Write-Output "VERSION_ARCH: $VERSION_ARCH"
Invoke-WebRequest -Uri "https://modstart-lib-public.oss-cn-shanghai.aliyuncs.com/aigcpanel-server-launcher/launcher-$VERSION_ARCH" -OutFile "launcher.exe"
Invoke-WebRequest -Uri "https://modstart-lib-public.oss-cn-shanghai.aliyuncs.com/ffmpeg/ffmpeg-$VERSION_ARCH" -OutFile "binary\ffmpeg.exe"
Invoke-WebRequest -Uri "https://modstart-lib-public.oss-cn-shanghai.aliyuncs.com/ffprobe/ffprobe-$VERSION_ARCH" -OutFile "binary\ffprobe.exe"
Remove-Item -Path "_aigcpanel/build*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "_aigcpanel/config.json" -Recurse -Force -ErrorAction SilentlyContinue
Compress-7Zip -Path . -Format Zip -ArchiveFileName "..\aigcpanel-server-musetalk-$VERSION.zip"
Move-Item -Path "..\aigcpanel-server-musetalk-$VERSION.zip" -Destination "aigcpanel-server-musetalk-$VERSION.zip"
# 打包服务
