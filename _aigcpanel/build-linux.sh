#!/bin/bash

set -x
set -e

# 环境准备
eval "$(conda shell.bash hook)"
conda env list
rm -rfv ./_aienv
conda create --prefix ./_aienv -y python=3.8
conda activate ./_aienv
conda info
# 环境准备

# 初始化环境
git submodule update --init --recursive
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
python download_model.py
# 初始化环境

# 启动服务
# python webui.py
# 启动服务

# 清除文件
rm -rfv build || true
rm -rfv dist || true
rm -rfv asset || true
rm -rfv *.md || true
rm -rfv download_model.py || true
rm -rfv requirements.txt || true
# 清除文件

# 打包服务
VERSION=$(python -m _aigcpanel.build)
echo "VERSION: ${VERSION}"
zip -rv "./aigcpanel-server-cosyvoice-${VERSION}.zip" * -x "_aigcpanel/*"
# 打包服务

