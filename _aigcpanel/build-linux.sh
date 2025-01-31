#!/bin/bash

set -x
set -e

# 环境准备
eval "$(conda shell.bash hook)"
conda env list
rm -rfv ./_aienv
conda create --prefix ./_aienv -y python=3.10
conda activate ./_aienv
conda info
# 环境准备

# 初始化环境
pip install -r requirements.txt
pip install --no-cache-dir -U openmim
mim install "mmengine==0.10.3"
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"
mim install "mmpose==1.3.1"
pip install gradio==3.50.2

wget https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin -P ./models/musetalk/
wget https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json -P ./models/musetalk/
wget https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin -P ./models/sd-vae-ft-mse/
wget https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json -P ./models/sd-vae-ft-mse/
wget https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt -P ./models/whisper/
wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth -P ./models/dwpose/
wget https://download.pytorch.org/models/resnet18-5c106cde.pth -P ./models/face-parse-bisent/
# 初始化环境

# 构建
python -m py_compile app.py
mv __pycache__/app.cpython-310.pyc app.pyc
rm -rfv __pycache__ || true
rm -rfv app.py || true
# 构建

# 启动服务
# python app.pyc
# python -m scripts.inference --inference_config configs/inference/tests.yaml
# 启动服务

# 清除文件
rm -rfv asset || true
rm -rfv *.md || true
rm -rfv requirements.txt || true
find . -type d -name "__pycache__" -print -exec rm -r {} +
# 清除文件

# 打包服务
VERSION=$(python -m _aigcpanel.build)
echo "VERSION: ${VERSION}"
zip -rv "./aigcpanel-server-musetalk-${VERSION}.zip" * -x "_aigcpanel/*"
# 打包服务

