import os, sys
import time
import shutil

import gradio as gr
import re
import subprocess

from gradio.utils import JSON_PATH

if os.path.exists(JSON_PATH):
    os.remove(JSON_PATH)

os.environ["GRADIO_SHARE"] = "false"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
serverPort = 50617
if os.environ.get('AIGCPANEL_SERVER_PORT'):
    serverPort = int(os.environ.get('AIGCPANEL_SERVER_PORT'))


def process_files(image, audio, box):
    print('===== 阶段1/3 准备工作 =====')
    print('image', image)
    print('audio', audio)
    print('box', box)
    print('pwd', os.getcwd())

    root = os.path.dirname(os.path.abspath(__file__))
    ts = str(int(time.time() * 1000))
    resultRoot = root + '/results'
    imagePathRandom = resultRoot + '/video_' + ts + '.mp4'
    audioPathRandom = resultRoot + '/audio_' + ts + '.wav'
    print('imagePathRandom', imagePathRandom)
    print('audioPathRandom', audioPathRandom)
    if not os.path.exists(resultRoot):
        os.makedirs(resultRoot)

    if os.path.exists(imagePathRandom):
        os.remove(imagePathRandom)
    if os.path.exists(audioPathRandom):
        os.remove(audioPathRandom)
    if isinstance(image,str):
        shutil.copyfile(image, imagePathRandom)
        shutil.copyfile(audio, audioPathRandom)
    else:
        os.rename(image.name, imagePathRandom)
        os.rename(audio.name, audioPathRandom)

    # 读取配置文件内容
    with open('./configs/inference/test.yaml', 'r', encoding='utf-8') as file:
        config_content = file.read()

    # 替换路径占位符
    config_content = config_content.replace("@video_path", imagePathRandom)
    config_content = config_content.replace("@audio_path", audioPathRandom)
    config_content = config_content.replace("@bbox_shift", str(box))

    # 保存到新的配置文件
    with open('./configs/inference/test2.yaml', 'w', encoding='utf-8') as file:
        file.write(config_content)

    if sys.platform == 'win32':
        pythonBin = os.path.join(root, '_aienv', 'python.exe')
    else:
        pythonBin = os.path.join(root, '_aienv/bin', 'python')
    # 执行命令生成视频，并指定编码为 UTF-8
    cmd = f"{pythonBin} -m scripts.inference --inference_config ./configs/inference/test2.yaml"
    print('cmd', cmd)
    print('===== 阶段2/3 视频合成 =====')
    # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.test2, universal_newlines=True, encoding='utf-8')
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    # 监听命令执行输出
    output_string = ""
    if process.stdout is not None:
        for line in process.stdout:
            print(line)  # 在控制台输出命令执行过程中的输出
            output_string += line
    if process.stderr is not None:
        for line in process.stderr:
            print(line)

    print('===== 阶段3/3 结果检查 =====')
    print(f'Clean imagePathRandom: {imagePathRandom}')
    os.remove(imagePathRandom)
    print(f'Clean audioPathRandom: {audioPathRandom}')
    os.remove(audioPathRandom)
    # 使用正则表达式匹配路径，确保以特定的文本开头和以.mp4结尾
    match = re.search(r'ResultSaveTo:([.\\/\w_-]+\.mp4)', output_string)
    if match:
        mp4_path = match.group(1)
        print(f'Success {mp4_path}')
        current_dir = os.getcwd()
        full_mp4_path = os.path.join(current_dir, mp4_path)
        print(f"ResultPath {full_mp4_path}")
        return gr.Video(full_mp4_path)
    else:
        print('Failed')
        return 'Failed'


# 创建 Gradio 界面
iface = gr.Interface(
    process_files,
    [
        # gr.File(type="file", label="视频文件"),
        # gr.File(type="file", label="音频文件"),
        gr.Text(label="视频文件"),
        gr.Text(label="音频文件"),
        gr.Slider(-9, 9, value=-7, step=1, label="嘴巴张开度")
    ],
    outputs=gr.Video(label="生成的视频"),
    #    title="MuseTalk：腾讯出品高质量唇形同步数字人",
    description=r"""<h1 style="text-align: center;">MuseTalk：腾讯出品高质量唇形同步数字人</h1><b>请上传视频和音频文件，并设置嘴巴张开度来生成视频。</b>""")

# 启动应用
iface.launch(inbrowser=False, share=False, quiet=True, server_port=serverPort)
