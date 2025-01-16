import os
import gradio as gr
import re
import subprocess

def process_files(image, audio, length):
    # 这里 image 和 audio 为文件的路径
    print('app.process_files', image, audio, length)
    if os.path.exists(image + '.mp4'):
        os.remove(image + '.mp4')
    if os.path.exists(audio + '.wav'):
        os.remove(audio + '.wav')
    os.rename(image, image + '.mp4')
    os.rename(audio, audio + '.wav')
    image = image + '.mp4'
    audio = audio + '.wav'


    image_path = image.replace("\\", "\\\\")
    audio_path = audio.replace("\\", "\\\\")

    # 读取配置文件内容
    with open('./configs/inference/test.yaml', 'r', encoding='utf-8') as file:
        config_content = file.read()
        
    # 替换路径占位符
    config_content = config_content.replace("@video_path", image_path)
    config_content = config_content.replace("@audio_path", audio_path)
    config_content = config_content.replace("@bbox_shift", str(length))
    
    # 保存到新的配置文件
    with open('./configs/inference/test2.yaml', 'w', encoding='utf-8') as file:
        file.write(config_content)

    
    print('process_files.pwd', os.getcwd() )
    # 执行命令生成视频，并指定编码为 UTF-8
    cmd = f".\\.ai\\python.exe -m scripts.inference --inference_config .\\configs\\inference\\test2.yaml"
    print('process_files.cmd', cmd)
    # cmd = f"python -V"
    # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.test2, universal_newlines=True, encoding='utf-8')
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    print('process_files.process.stdout', process.stdout)
    print('process_files.process.stderr', process.stderr)
    
    # 监听命令执行输出
    output_string = ""
    for line in process.stdout:
        print(line)  # 在控制台输出命令执行过程中的输出
        output_string += line
    
    # 使用正则表达式匹配路径，确保以特定的文本开头和以.mp4结尾
    match = re.search(r'result is save to ([.\\/\w_-]+\.mp4)', output_string)

    if match:
        mp4_path = match.group(1)
        mp4_path = mp4_path.replace("/", "\\")
        print(mp4_path)

        current_dir = os.getcwd()
        full_mp4_path = os.path.join(current_dir, mp4_path)
        print("=======视频路径==不正确======")
        print(full_mp4_path)
        return gr.Video(full_mp4_path)
    else:
        return "未找到 .mp4 文件路径"

# 创建 Gradio 界面
iface = gr.Interface(
    process_files,
    [gr.File(type="filepath", label="视频文件"), gr.File(type="filepath", label="音频文件"), gr.Slider(-9, 9, value=-7, step=1, label="嘴巴张开度")],
    outputs=gr.Video(label="生成的视频"),
#    title="MuseTalk：腾讯出品高质量唇形同步数字人",
    description = r"""<h1 style="text-align: center;">MuseTalk：腾讯出品高质量唇形同步数字人</h1>
    <b>请上传视频和音频文件，并设置嘴巴张开度来生成视频。</b>""")

# 启动应用
iface.launch(inbrowser=False, share=False)
