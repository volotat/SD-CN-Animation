import sys, os

import torch
import gc
import numpy as np
from PIL import Image

import modules.paths as ph
from modules.shared import devices

from scripts.core import utils, flow_utils
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

  model_path = ph.models_path + '/FloweR/FloweR_0.1.2.pth'
  #remote_model_path = 'https://drive.google.com/uc?id=1K7gXUosgxU729_l-osl1HBU5xqyLsALv' #FloweR_0.1.1.pth
  remote_model_path = 'https://drive.google.com/uc?id=1-UYsTXkdUkHLgtPK1Y5_7kKzCgzL_Z6o' #FloweR_0.1.2.pth

  if not os.path.isfile(model_path):
    from basicsr.utils.download_util import load_file_from_url
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    load_file_from_url(remote_model_path, file_name=model_path)


  FloweR_model = FloweR(input_size = (h, w))
  FloweR_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
  # Move the model to the device
  FloweR_model = FloweR_model.to(DEVICE)
  FloweR_model.eval()

def read_frame_from_video(input_video):
  if input_video is None: return None

  # Reading video file
  if input_video.isOpened():
    ret, cur_frame = input_video.read()
    if cur_frame is not None: 
      cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB) 
  else:
    cur_frame = None
    input_video.release()
    input_video = None
  
  return cur_frame

def start_process(*args):
    processing_start_time = time.time()
    args_dict = utils.args_to_dict(*args)
    args_dict = utils.get_mode_args('t2v', args_dict)

    # Open the input video file
    input_video = None
    if args_dict['file'] is not None:
      input_video = cv2.VideoCapture(args_dict['file'].name)

    # Create an output video file with the same fps, width, and height as the input video
    output_video_name = f'outputs/sd-cn-animation/txt2vid/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4'
    output_video_folder = os.path.splitext(output_video_name)[0]
    os.makedirs(os.path.dirname(output_video_name), exist_ok=True)

    #if args_dict['save_frames_check']: 
    os.makedirs(output_video_folder, exist_ok=True)

    # Writing to current params to params.json
    setts_json = utils.export_settings(*args)
    with open(os.path.join(output_video_folder, "params.json"), "w") as outfile:
      outfile.write(setts_json)

    curr_frame = None
    prev_frame = None
    
    def save_result_to_image(image, ind):
      if args_dict['save_frames_check']: 
        cv2.imwrite(os.path.join(output_video_folder, f'{ind:05d}.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def set_cn_frame_input():
      if args_dict['cn_frame_send'] == 0: # Current generated frame"
        pass
      elif args_dict['cn_frame_send'] == 1: # Current generated frame"
        if curr_frame is not None:
          utils.set_CNs_input_image(args_dict, Image.fromarray(curr_frame), set_references=True)
      elif args_dict['cn_frame_send'] == 2: # Previous generated frame
        if prev_frame is not None:
          utils.set_CNs_input_image(args_dict, Image.fromarray(prev_frame), set_references=True)
      elif args_dict['cn_frame_send'] == 3: # Current reference video frame
        if input_video is not None:
          curr_video_frame = read_frame_from_video(input_video)
          curr_video_frame = cv2.resize(curr_video_frame, (args_dict['width'], args_dict['height']))
          utils.set_CNs_input_image(args_dict, Image.fromarray(curr_video_frame), set_references=True)
        else:
          raise Exception('There is no input video! Set it up first.')
      else:
        raise Exception('Incorrect cn_frame_send mode!')

    set_cn_frame_input()

    if args_dict['init_image'] is not None:
      #resize array to args_dict['width'], args_dict['height']
      image_array=args_dict['init_image']#this is a numpy array
      init_frame = np.array(Image.fromarray(image_array).resize((args_dict['width'], args_dict['height'])).convert('RGB'))
      processed_frame = init_frame.copy()
    else:
      processed_frames, _, _, _ = utils.txt2img(args_dict)
      processed_frame = np.array(processed_frames[0])[...,:3]
      #if input_video is not None:
      #  processed_frame = skimage.exposure.match_histograms(processed_frame, curr_video_frame, channel_axis=-1)
      processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)
      init_frame = processed_frame.copy()

    output_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'mp4v'), args_dict['fps'], (args_dict['width'], args_dict['height']))
    output_video.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))

    stat = f"Frame: 1 / {args_dict['length']}; " + utils.get_time_left(1, args_dict['length'], processing_start_time)
    utils.shared.is_interrupted = False

    save_result_to_image(processed_frame, 1)
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
      pred_next = flow_utils.frames_renorm(pred_data[...,3:6]).cpu().numpy()
      
      pred_occl = np.clip(pred_occl * 10, 0, 255).astype(np.uint8)
      pred_next = np.clip(pred_next, 0, 255).astype(np.uint8)
      
      pred_flow = cv2.resize(pred_flow, org_size)
      pred_occl = cv2.resize(pred_occl, org_size)
      pred_next = cv2.resize(pred_next, org_size)

      curr_frame = pred_next.copy()

      '''
      pred_flow = pred_flow / (1 + np.linalg.norm(pred_flow, axis=-1, keepdims=True) * 0.05) 
      pred_flow = cv2.GaussianBlur(pred_flow, (31,31), 1, cv2.BORDER_REFLECT_101)
    
      pred_occl = cv2.GaussianBlur(pred_occl, (21,21), 2, cv2.BORDER_REFLECT_101)
      pred_occl = (np.abs(pred_occl / 255) ** 1.5) * 255
      pred_occl = np.clip(pred_occl * 25, 0, 255).astype(np.uint8)

      flow_map = pred_flow.copy()
      flow_map[:,:,0] += np.arange(args_dict['width'])
      flow_map[:,:,1] += np.arange(args_dict['height'])[:,np.newaxis]

      warped_frame = cv2.remap(prev_frame, flow_map, None, cv2.INTER_NEAREST, borderMode = cv2.BORDER_REFLECT_101)
      alpha_mask = pred_occl / 255.
      #alpha_mask = np.clip(alpha_mask + np.random.normal(0, 0.4, size = alpha_mask.shape), 0, 1)
      curr_frame = pred_next.astype(float) * alpha_mask + warped_frame.astype(float) * (1 - alpha_mask)
      curr_frame = np.clip(curr_frame, 0, 255).astype(np.uint8)
      #curr_frame = warped_frame.copy()
      '''

      set_cn_frame_input()

      args_dict['mode'] = 4
      args_dict['init_img'] = Image.fromarray(pred_next)
      args_dict['mask_img'] = Image.fromarray(pred_occl)
      args_dict['seed'] = -1
      args_dict['denoising_strength'] = args_dict['processing_strength']

      processed_frames, _, _, _ = utils.img2img(args_dict)
      processed_frame = np.array(processed_frames[0])[...,:3]
      #if input_video is not None:
      #  processed_frame = skimage.exposure.match_histograms(processed_frame, curr_video_frame, channel_axis=-1)
      #else: 
      processed_frame = skimage.exposure.match_histograms(processed_frame, init_frame, channel_axis=-1)
      processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)

      args_dict['mode'] = 0
      args_dict['init_img'] = Image.fromarray(processed_frame)
      args_dict['mask_img'] = None
      args_dict['seed'] = -1
      args_dict['denoising_strength'] = args_dict['fix_frame_strength']

      #utils.set_CNs_input_image(args_dict, Image.fromarray(curr_frame))
      processed_frames, _, _, _ = utils.img2img(args_dict)
      processed_frame = np.array(processed_frames[0])[...,:3]
      #if input_video is not None:
      #  processed_frame = skimage.exposure.match_histograms(processed_frame, curr_video_frame, channel_axis=-1)
      #else: 
      processed_frame = skimage.exposure.match_histograms(processed_frame, init_frame, channel_axis=-1)
      processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)

      output_video.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
      prev_frame = processed_frame.copy()

      save_result_to_image(processed_frame, ind + 2)
      stat = f"Frame: {ind + 2} / {args_dict['length']}; " + utils.get_time_left(ind+2, args_dict['length'], processing_start_time)
      yield stat, curr_frame, pred_occl, pred_next, processed_frame, None, gr.Button.update(interactive=False), gr.Button.update(interactive=True)

    if input_video is not None: input_video.release()
    output_video.release()
    FloweR_clear_memory()

    curr_frame = gr.Image.update()
    occlusion_mask = gr.Image.update()
    warped_styled_frame_ = gr.Image.update() 
    processed_frame = gr.Image.update()

    # print('TOTAL TIME:', int(time.time() - processing_start_time))

    yield 'done', curr_frame, occlusion_mask, warped_styled_frame_, processed_frame, output_video_name, gr.Button.update(interactive=True), gr.Button.update(interactive=False)