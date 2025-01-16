# Enable echo
$VerbosePreference = "Continue"

# 环境准备
conda 'shell.powershell' 'hook' | Out-String | Invoke-Expression
conda env list
Remove-Item -Recurse -Force ./_aienv -ErrorAction SilentlyContinue -Verbose
conda create --prefix ./_aienv -y python=3.10
conda activate ./_aienv
# 环境准备

# 初始化环境
pip install -r requirements.txt
pip install --no-cache-dir -U openmim
mim install "mmengine==0.10.3"
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"
mim install "mmpose==1.3.1"
pip install gradio==4.26.0 --no-deps
New-Item -ItemType Directory -Force -Path ./models/musetalk/
New-Item -ItemType Directory -Force -Path ./models/sd-vae-ft-mse/
New-Item -ItemType Directory -Force -Path ./models/whisper/
New-Item -ItemType Directory -Force -Path ./models/dwpose/
New-Item -ItemType Directory -Force -Path ./models/face-parse-bisent/
Invoke-WebRequest -Uri "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin" -OutFile "./models/musetalk/pytorch_model.bin"
Invoke-WebRequest -Uri "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json" -OutFile "./models/musetalk/musetalk.json"
Invoke-WebRequest -Uri "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin" -OutFile "./models/sd-vae-ft-mse/diffusion_pytorch_model.bin"
Invoke-WebRequest -Uri "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json" -OutFile "./models/sd-vae-ft-mse/config.json"
Invoke-WebRequest -Uri "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt" -OutFile "./models/whisper/tiny.pt"
Invoke-WebRequest -Uri "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth" -OutFile "./models/dwpose/dw-ll_ucoco_384.pth"
Invoke-WebRequest -Uri "https://download.pytorch.org/models/resnet18-5c106cde.pth" -OutFile "./models/face-parse-bisent/resnet18-5c106cde.pth"
# 初始化环境

# 启动服务
#python webui.py
# 启动服务

# 清除文件
Remove-Item -Recurse -Force asset -ErrorAction SilentlyContinue -Verbose
Remove-Item -Recurse -Force .\*.md -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force requirements.txt -ErrorAction SilentlyContinue
# 清除文件

# 打包服务
$VERSION = python -m _aigcpanel.build
Write-Output "VERSION: $VERSION"
Get-ChildItem -Path . -Exclude "_aigcpanel" |
    Compress-Archive -DestinationPath "aigcpanel-server-musetalk-$VERSION.zip" -Verbose -Force -ErrorAction Continue
# 打包服务
