import sys, os
basedirs = [os.getcwd()]

for basedir in basedirs:
    paths_to_ensure = [
        basedir,
        basedir + '/extensions/sd-cn-animation/scripts',
        basedir + '/extensions/SD-CN-Animation/scripts'
        ]

    for scripts_path_fix in paths_to_ensure:
        if not scripts_path_fix in sys.path:
            sys.path.extend([scripts_path_fix])

import torch
import gc
import numpy as np
from PIL import Image

import modules.paths as ph
from modules.shared import devices

from core import utils, flow_utils
from FloweR.model import FloweR

import skimage
import datetime
import cv2
import gradio as gr
import time

FloweR_model = None
DEVICE = 'cpu'
def FloweR_clear_memory():
  global FloweR_model
  del FloweR_model
  gc.collect()
  torch.cuda.empty_cache()
  FloweR_model = None

def FloweR_load_model(w, h):
  global DEVICE, FloweR_model
  DEVICE = devices.get_optimal_device()

  model_path = ph.models_path + '/FloweR/FloweR_0.1.1.pth'
  remote_model_path = 'https://drive.google.com/uc?id=1K7gXUosgxU729_l-osl1HBU5xqyLsALv'

  if not os.path.isfile(model_path):
    from basicsr.utils.download_util import load_file_from_url
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    load_file_from_url(remote_model_path, file_name=model_path)


  FloweR_model = FloweR(input_size = (h, w))
  FloweR_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
  # Move the model to the device
  FloweR_model = FloweR_model.to(DEVICE)



def start_process(*args):
    processing_start_time = time.time()
    args_dict = utils.args_to_dict(*args)
    args_dict = utils.get_mode_args('t2v', args_dict)

    #utils.set_CNs_input_image(args_dict, Image.fromarray(curr_frame))
    processed_frames, _, _, _ = utils.txt2img(args_dict)
    processed_frame = np.array(processed_frames[0])
    processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)
    init_frame = processed_frame.copy()

    # Create an output video file with the same fps, width, and height as the input video
    output_video_name = f'outputs/sd-cn-animation/txt2vid/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4'
    os.makedirs(os.path.dirname(output_video_name), exist_ok=True)
    output_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'mp4v'), args_dict['fps'], (args_dict['width'], args_dict['height']))
    output_video.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))

    stat = f"Frame: 1 / {args_dict['length']}; " + utils.get_time_left(1, args_dict['length'], processing_start_time)
    utils.shared.is_interrupted = False
    yield stat, init_frame, None, None, processed_frame, None, gr.Button.update(interactive=False), gr.Button.update(interactive=True)

    org_size = args_dict['width'], args_dict['height']
    size = args_dict['width'] // 128 * 128, args_dict['height'] // 128 * 128
    FloweR_load_model(size[0], size[1])

    clip_frames = np.zeros((4, size[1], size[0], 3), dtype=np.uint8)

    prev_frame = init_frame

    for ind in range(args_dict['length'] - 1): 
      if utils.shared.is_interrupted: break

      args_dict = utils.args_to_dict(*args)
      args_dict = utils.get_mode_args('t2v', args_dict)

      clip_frames = np.roll(clip_frames, -1, axis=0)
      clip_frames[-1] = cv2.resize(prev_frame[...,:3], size)
      clip_frames_torch = flow_utils.frames_norm(torch.from_numpy(clip_frames).to(DEVICE, dtype=torch.float32))

      with torch.no_grad():
        pred_data = FloweR_model(clip_frames_torch.unsqueeze(0))[0]

      pred_flow = flow_utils.flow_renorm(pred_data[...,:2]).cpu().numpy()
      pred_occl = flow_utils.occl_renorm(pred_data[...,2:3]).cpu().numpy().repeat(3, axis = -1)

      pred_flow = cv2.resize(pred_flow, org_size)
      pred_occl = cv2.resize(pred_occl, org_size)

      pred_flow = pred_flow / (1 + np.linalg.norm(pred_flow, axis=-1, keepdims=True) * 0.05) 
      pred_flow = cv2.GaussianBlur(pred_flow, (31,31), 1, cv2.BORDER_REFLECT_101)
    
      pred_occl = cv2.GaussianBlur(pred_occl, (21,21), 2, cv2.BORDER_REFLECT_101)
      pred_occl = (np.abs(pred_occl / 255) ** 1.5) * 255
      pred_occl = np.clip(pred_occl * 25, 0, 255).astype(np.uint8)

      flow_map = pred_flow.copy()
      flow_map[:,:,0] += np.arange(args_dict['width'])
      flow_map[:,:,1] += np.arange(args_dict['height'])[:,np.newaxis]

      warped_frame = cv2.remap(prev_frame, flow_map, None, cv2.INTER_CUBIC, borderMode = cv2.BORDER_REFLECT_101)

      curr_frame = warped_frame.copy()
      
      args_dict['mode'] = 4
      args_dict['init_img'] = Image.fromarray(curr_frame)
      args_dict['mask_img'] = Image.fromarray(pred_occl)

      #utils.set_CNs_input_image(args_dict, Image.fromarray(curr_frame))
      processed_frames, _, _, _ = utils.img2img(args_dict)
      processed_frame = np.array(processed_frames[0])
      processed_frame = skimage.exposure.match_histograms(processed_frame, init_frame, channel_axis=None)
      processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)

      args_dict['mode'] = 0
      args_dict['init_img'] = Image.fromarray(processed_frame)
      args_dict['mask_img'] = None
      args_dict['denoising_strength'] = args_dict['fix_frame_strength']

      #utils.set_CNs_input_image(args_dict, Image.fromarray(curr_frame))
      processed_frames, _, _, _ = utils.img2img(args_dict)
      processed_frame = np.array(processed_frames[0])
      processed_frame = skimage.exposure.match_histograms(processed_frame, init_frame, channel_axis=None)
      processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)

      output_video.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
      prev_frame = processed_frame.copy()

      
      stat = f"Frame: {ind + 2} / {args_dict['length']}; " + utils.get_time_left(ind+2, args_dict['length'], processing_start_time)
      yield stat, curr_frame, pred_occl, warped_frame, processed_frame, None, gr.Button.update(interactive=False), gr.Button.update(interactive=True)

    output_video.release()
    FloweR_clear_memory()

    curr_frame = gr.Image.update()
    occlusion_mask = gr.Image.update()
    warped_styled_frame_ = gr.Image.update() 
    processed_frame = gr.Image.update()

    yield 'done', curr_frame, occlusion_mask, warped_styled_frame_, processed_frame, output_video_name, gr.Button.update(interactive=True), gr.Button.update(interactive=False)