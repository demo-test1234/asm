import os
import sys

import _aigcpanel.base.file
import _aigcpanel.base.result
import _aigcpanel.base.util
import _aigcpanel.base.log

if len(sys.argv) != 2:
    print("Usage: python -u -m aigcpanelrun <config_url>")
    exit(-1)

###### 模型数据开始 ######
import argparse
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import copy
import time
from musetalk.utils.utils import get_file_type, get_video_fps, datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model
import shutil

useCuda = torch.cuda.is_available()
_aigcpanel.base.log.info('开始运行', {'UseCuda': useCuda})
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['XDG_CACHE_HOME'] = os.path.join(ROOT_DIR, '_cache')
os.environ["MODELSCOPE_CACHE"] = os.path.join(ROOT_DIR, '_cache', 'modelscope')
sys.path.append('{}/binary'.format(ROOT_DIR))

audio_processor, vae, unet, pe = load_all_model()
useGpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)

stepCurrent = 1
stepTotal = 7


def stepPrint():
    global stepCurrent
    global stepTotal
    _aigcpanel.base.log.info(f'===== 视频合成MuseTalk {stepCurrent}/{stepTotal} =====')
    stepCurrent += 1


def main():
    outputRoot = "launcher-data/"
    resultDir = "results/"
    resultFps = 25
    use_saved_coord = False
    batch_size = 8

    if not os.path.exists(outputRoot):
        os.makedirs(outputRoot)
        _aigcpanel.base.log.info(f"create dir {outputRoot}")
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)
        _aigcpanel.base.log.info(f"create dir {resultDir}")

    config = _aigcpanel.base.file.contentJson(sys.argv[1])
    _aigcpanel.base.log.info('config', config, sys.argv)
    if not 'id' in config:
        _aigcpanel.base.log.info('config.id not found')
        exit(-1)
    _aigcpanel.base.result.result(config, {'UseCuda': useCuda})
    if not 'mode' in config:
        config['mode'] = 'local'
    modelConfig = config['modelConfig']

    resultMp4File = "result_" + str(int(time.time())) + ".mp4"
    video_path = os.path.join(ROOT_DIR, outputRoot, "video_" + str(int(time.time())) + ".mp4")
    audio_path = os.path.join(ROOT_DIR, outputRoot, "audio_" + str(int(time.time())) + ".wav")
    bbox_shift = modelConfig['box']

    _aigcpanel.base.log.info('video_path', video_path)
    _aigcpanel.base.log.info('audio_path', audio_path)
    _aigcpanel.base.log.info('bbox_shift', bbox_shift)

    modelConfig['_video'] = _aigcpanel.base.file.localCache(modelConfig['video'])
    modelConfig['_audio'] = _aigcpanel.base.file.localCache(modelConfig['audio'])
    shutil.copy(modelConfig["_video"], video_path)
    shutil.copy(modelConfig["_audio"], audio_path)

    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename = os.path.basename(audio_path).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    _aigcpanel.base.log.info('output_basename', output_basename)
    result_img_save_path = os.path.join(resultDir, output_basename)
    crop_coord_save_path = os.path.join(result_img_save_path, input_basename + ".pkl")
    save_dir_full = None

    if os.path.exists(result_img_save_path):
        shutil.rmtree(result_img_save_path)
        _aigcpanel.base.log.info(f"delete dir {result_img_save_path}")

    if os.path.exists(crop_coord_save_path):
        shutil.rmtree(crop_coord_save_path)
        _aigcpanel.base.log.info(f"delete dir {crop_coord_save_path}")

    os.makedirs(result_img_save_path, exist_ok=True)

    _aigcpanel.base.log.info('result_img_save_path', result_img_save_path)
    _aigcpanel.base.log.info('crop_coord_save_path', crop_coord_save_path)

    resultMp4Path = os.path.join(outputRoot, resultMp4File)
    _aigcpanel.base.log.info('resultMp4Path', resultMp4Path)

    if os.path.exists(resultMp4Path):
        _aigcpanel.base.log.info(f'delete file {resultMp4Path}')
        os.remove(resultMp4Path)

    stepPrint()
    _aigcpanel.base.log.info('extract frames from source video')
    ############################################## extract frames from source video ##############################################
    if get_file_type(video_path) == "video":
        save_dir_full = os.path.join(resultDir, input_basename)
        if os.path.exists(save_dir_full):
            shutil.rmtree(save_dir_full)
            _aigcpanel.base.log.info(f"delete dir {save_dir_full}")
        os.makedirs(save_dir_full, exist_ok=True)
        cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
        os.system(cmd)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
        fps = get_video_fps(video_path)
    else:  # input img folder
        input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        fps = resultFps

    # _aigcpanel.base.log.info(input_img_list)
    stepPrint()
    _aigcpanel.base.log.info('extract audio feature')
    ############################################## extract audio feature ##############################################
    whisper_feature = audio_processor.audio2feat(audio_path)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)

    stepPrint()
    _aigcpanel.base.log.info('preprocess input image')
    ############################################## preprocess input image  ##############################################
    if os.path.exists(crop_coord_save_path) and use_saved_coord:
        _aigcpanel.base.log.info("using extracted coordinates")
        with open(crop_coord_save_path, 'rb') as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        _aigcpanel.base.log.info("extracting landmarks... time consuming")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        _aigcpanel.base.log.info("save extracted coordinates")
        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
        _aigcpanel.base.log.info("save extracted coordinates done")

    _aigcpanel.base.log.info('input_latent_list prepare')
    input_latent_list = []
    for bbox, frame in tqdm(zip(coord_list, frame_list), ascii=True):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    # to smooth the first and the last frame
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    ############################################## inference batch by batch ##############################################

    stepPrint()
    _aigcpanel.base.log.info("start inference")
    video_num = len(whisper_chunks)
    gen = datagen(whisper_chunks, input_latent_list_cycle, batch_size)
    res_frame_list = []
    for i, (whisper_batch, latent_batch) in enumerate(
            tqdm(gen, total=int(np.ceil(float(video_num) / batch_size)), ascii=True)):

        tensor_list = [torch.FloatTensor(arr) for arr in whisper_batch]
        audio_feature_batch = torch.stack(tensor_list).to(unet.device)  # torch, B, 5*N,384
        audio_feature_batch = pe(audio_feature_batch)

        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)

    ############################################## pad to full image ##############################################
    stepPrint()
    _aigcpanel.base.log.info("pad talking image to original video")
    for i, res_frame in enumerate(tqdm(res_frame_list, ascii=True)):
        bbox = coord_list_cycle[i % (len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[i % (len(frame_list_cycle))])
        x1, y1, x2, y2 = bbox
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except:
            _aigcpanel.base.log.info(bbox)
            continue

        combine_frame = get_image(ori_frame, res_frame, bbox)
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)

    cmd_img2video = f"ffmpeg -y -v fatal -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
    _aigcpanel.base.log.info('cmd_img2video', cmd_img2video)
    os.system(cmd_img2video)

    stepPrint()
    cmd_combine_audio = f"ffmpeg -y -v fatal -i {audio_path} -i temp.mp4 {resultMp4Path}"
    _aigcpanel.base.log.info('cmd_combine_audio', cmd_combine_audio)
    os.system(cmd_combine_audio)

    os.remove("temp.mp4")
    resultMp4Path = os.path.join(ROOT_DIR, resultMp4Path)
    _aigcpanel.base.log.info(f"ResultSaveTo:{resultMp4Path}")
    _aigcpanel.base.result.result(config, {'url': _aigcpanel.base.file.urlForResult(config, resultMp4Path)})

    _aigcpanel.base.log.info(f'clean {video_path}')
    os.remove(video_path)
    _aigcpanel.base.log.info(f'clean {audio_path}')
    os.remove(audio_path)
    _aigcpanel.base.log.info(f'clean {result_img_save_path}')
    shutil.rmtree(result_img_save_path)
    if save_dir_full is not None:
        _aigcpanel.base.log.info(f'clean {save_dir_full}')
        shutil.rmtree(save_dir_full)


if __name__ == "__main__":
    main()

###### 模型数据结束 ######
