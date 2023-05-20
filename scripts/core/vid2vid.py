import sys, os

import math
import os
import sys
import traceback

import numpy as np
from PIL import Image

from modules import devices, sd_samplers
from modules import shared, sd_hijack, lowvram

from modules.shared import devices
import modules.shared as shared

import gc
import cv2
import gradio as gr

import time
import skimage
import datetime

from scripts.core.flow_utils import RAFT_estimate_flow, RAFT_clear_memory, compute_diff_map
from scripts.core import utils

class sdcn_anim_tmp:
  prepear_counter = 0
  process_counter = 0
  input_video = None
  output_video = None
  curr_frame = None
  prev_frame = None
  prev_frame_styled = None
  prev_frame_alpha_mask = None
  fps = None
  total_frames = None
  prepared_frames = None
  prepared_next_flows = None
  prepared_prev_flows = None
  frames_prepared = False

def read_frame_from_video():
  # Reading video file
  if sdcn_anim_tmp.input_video.isOpened():
    ret, cur_frame = sdcn_anim_tmp.input_video.read()
    if cur_frame is not None: 
      cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB) 
  else:
    cur_frame = None
    sdcn_anim_tmp.input_video.release()
  
  return cur_frame

def get_cur_stat():
  stat =  f'Frames prepared: {sdcn_anim_tmp.prepear_counter + 1} / {sdcn_anim_tmp.total_frames}; '
  stat += f'Frames processed: {sdcn_anim_tmp.process_counter + 1} / {sdcn_anim_tmp.total_frames}; '
  return stat

def clear_memory_from_sd():
  if shared.sd_model is not None:
    sd_hijack.model_hijack.undo_hijack(shared.sd_model)
    try:
      lowvram.send_everything_to_cpu()
    except Exception as e:
      ...
    del shared.sd_model
    shared.sd_model = None
  gc.collect()
  devices.torch_gc()

def start_process(*args):
  processing_start_time = time.time()
  args_dict = utils.args_to_dict(*args)
  args_dict = utils.get_mode_args('v2v', args_dict)
  
  sdcn_anim_tmp.process_counter = 0
  sdcn_anim_tmp.prepear_counter = 0

  # Open the input video file
  sdcn_anim_tmp.input_video = cv2.VideoCapture(args_dict['file'].name)
  
  # Get useful info from the source video
  sdcn_anim_tmp.fps = int(sdcn_anim_tmp.input_video.get(cv2.CAP_PROP_FPS))
  sdcn_anim_tmp.total_frames = int(sdcn_anim_tmp.input_video.get(cv2.CAP_PROP_FRAME_COUNT))
  loop_iterations = (sdcn_anim_tmp.total_frames-1) * 2

  # Create an output video file with the same fps, width, and height as the input video
  output_video_name = f'outputs/sd-cn-animation/vid2vid/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4'
  output_video_folder = os.path.splitext(output_video_name)[0]
  os.makedirs(os.path.dirname(output_video_name), exist_ok=True)
  
  if args_dict['save_frames_check']:
    os.makedirs(output_video_folder, exist_ok=True)

  def save_result_to_image(image, ind):
    if args_dict['save_frames_check']: 
      cv2.imwrite(os.path.join(output_video_folder, f'{ind:05d}.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

  sdcn_anim_tmp.output_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'mp4v'), sdcn_anim_tmp.fps, (args_dict['width'], args_dict['height']))

  curr_frame = read_frame_from_video()
  curr_frame = cv2.resize(curr_frame, (args_dict['width'], args_dict['height']))
  sdcn_anim_tmp.prepared_frames = np.zeros((11, args_dict['height'], args_dict['width'], 3), dtype=np.uint8)
  sdcn_anim_tmp.prepared_next_flows = np.zeros((10, args_dict['height'], args_dict['width'], 2))
  sdcn_anim_tmp.prepared_prev_flows = np.zeros((10, args_dict['height'], args_dict['width'], 2))
  sdcn_anim_tmp.prepared_frames[0] = curr_frame

  args_dict['init_img'] = Image.fromarray(curr_frame)
  utils.set_CNs_input_image(args_dict, Image.fromarray(curr_frame))
  processed_frames, _, _, _ = utils.img2img(args_dict)
  processed_frame = np.array(processed_frames[0])[...,:3]
  processed_frame = skimage.exposure.match_histograms(processed_frame, curr_frame, channel_axis=None)
  processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)
  #print('Processed frame ', 0)

  sdcn_anim_tmp.curr_frame = curr_frame
  sdcn_anim_tmp.prev_frame = curr_frame.copy()
  sdcn_anim_tmp.prev_frame_styled = processed_frame.copy()
  utils.shared.is_interrupted = False

  save_result_to_image(processed_frame, 1)
  stat = get_cur_stat() + utils.get_time_left(1, loop_iterations, processing_start_time)
  yield stat, sdcn_anim_tmp.curr_frame, None, None, processed_frame, None, gr.Button.update(interactive=False), gr.Button.update(interactive=True)

  for step in range(loop_iterations):
    if utils.shared.is_interrupted: break
    
    args_dict = utils.args_to_dict(*args)
    args_dict = utils.get_mode_args('v2v', args_dict)

    occlusion_mask = None
    prev_frame = None
    curr_frame = sdcn_anim_tmp.curr_frame
    warped_styled_frame_ = gr.Image.update()
    processed_frame = gr.Image.update()

    prepare_steps = 10
    if sdcn_anim_tmp.process_counter % prepare_steps == 0 and not sdcn_anim_tmp.frames_prepared: # prepare next 10 frames for processing
        #clear_memory_from_sd()
        device = devices.get_optimal_device()

        curr_frame = read_frame_from_video()
        if curr_frame is not None: 
            curr_frame = cv2.resize(curr_frame, (args_dict['width'], args_dict['height']))
            prev_frame = sdcn_anim_tmp.prev_frame.copy()

            next_flow, prev_flow, occlusion_mask = RAFT_estimate_flow(prev_frame, curr_frame, device=device)
            occlusion_mask = np.clip(occlusion_mask * 0.1 * 255, 0, 255).astype(np.uint8)

            cn = sdcn_anim_tmp.prepear_counter % 10
            if sdcn_anim_tmp.prepear_counter % 10 == 0:
                sdcn_anim_tmp.prepared_frames[cn] = sdcn_anim_tmp.prev_frame
            sdcn_anim_tmp.prepared_frames[cn + 1] = curr_frame.copy()
            sdcn_anim_tmp.prepared_next_flows[cn] = next_flow.copy()
            sdcn_anim_tmp.prepared_prev_flows[cn] = prev_flow.copy()
            #print('Prepared frame ', cn+1)

            sdcn_anim_tmp.prev_frame = curr_frame.copy()

        sdcn_anim_tmp.prepear_counter += 1
        if sdcn_anim_tmp.prepear_counter % prepare_steps == 0 or \
        sdcn_anim_tmp.prepear_counter >= sdcn_anim_tmp.total_frames - 1 or \
        curr_frame is None:
            # Remove RAFT from memory
            RAFT_clear_memory()
            sdcn_anim_tmp.frames_prepared = True
    else:
        # process frame
        sdcn_anim_tmp.frames_prepared = False

        cn = sdcn_anim_tmp.process_counter % 10 
        curr_frame = sdcn_anim_tmp.prepared_frames[cn+1][...,:3]
        prev_frame = sdcn_anim_tmp.prepared_frames[cn][...,:3]
        next_flow = sdcn_anim_tmp.prepared_next_flows[cn]
        prev_flow = sdcn_anim_tmp.prepared_prev_flows[cn]

        ### STEP 1
        alpha_mask, warped_styled_frame = compute_diff_map(next_flow, prev_flow, prev_frame, curr_frame, sdcn_anim_tmp.prev_frame_styled, args_dict)
        warped_styled_frame_ = warped_styled_frame.copy()

        #fl_w, fl_h = prev_flow.shape[:2]
        #prev_flow_n = prev_flow / np.array([fl_h,fl_w])
        #flow_mask = np.clip(1 - np.linalg.norm(prev_flow_n, axis=-1)[...,None] * 20, 0, 1)
        #alpha_mask = alpha_mask * flow_mask

        if sdcn_anim_tmp.process_counter > 0 and args_dict['occlusion_mask_trailing']:
            alpha_mask = alpha_mask + sdcn_anim_tmp.prev_frame_alpha_mask * 0.5
        sdcn_anim_tmp.prev_frame_alpha_mask = alpha_mask

        alpha_mask = np.clip(alpha_mask, 0, 1)
        occlusion_mask = np.clip(alpha_mask * 255, 0, 255).astype(np.uint8)

        # fix warped styled frame from duplicated that occures on the places where flow is zero, but only because there is no place to get the color from
        warped_styled_frame = curr_frame.astype(float) * alpha_mask + warped_styled_frame.astype(float) * (1 - alpha_mask)

        # process current frame
        # TODO: convert args_dict into separate dict that stores only params necessery for img2img processing
        img2img_args_dict = args_dict #copy.deepcopy(args_dict)
        img2img_args_dict['denoising_strength'] = args_dict['processing_strength']
        if args_dict['step_1_processing_mode'] == 0: # Process full image then blend in occlusions
          img2img_args_dict['mode'] = 0
          img2img_args_dict['mask_img'] = None #Image.fromarray(occlusion_mask)
        elif args_dict['step_1_processing_mode'] == 1: # Inpaint occlusions
          img2img_args_dict['mode'] = 4
          img2img_args_dict['mask_img'] = Image.fromarray(occlusion_mask)
        else:
           raise Exception('Incorrect step 1 processing mode!')
        
        blend_alpha = args_dict['step_1_blend_alpha']
        init_img = warped_styled_frame * (1 - blend_alpha) + curr_frame * blend_alpha
        img2img_args_dict['init_img'] = Image.fromarray(np.clip(init_img, 0, 255).astype(np.uint8))
        img2img_args_dict['seed'] = args_dict['step_1_seed']
        utils.set_CNs_input_image(img2img_args_dict, Image.fromarray(curr_frame))
        processed_frames, _, _, _ = utils.img2img(img2img_args_dict)
        processed_frame = np.array(processed_frames[0])[...,:3]

        # normalizing the colors
        processed_frame = skimage.exposure.match_histograms(processed_frame, curr_frame, channel_axis=None)
        processed_frame = processed_frame.astype(float) * alpha_mask + warped_styled_frame.astype(float) * (1 - alpha_mask)
        
        #processed_frame = processed_frame * 0.94 + curr_frame * 0.06
        processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)
        sdcn_anim_tmp.prev_frame_styled = processed_frame.copy()

        ### STEP 2
        if args_dict['fix_frame_strength'] > 0:
          img2img_args_dict = args_dict #copy.deepcopy(args_dict)
          img2img_args_dict['mode'] = 0
          img2img_args_dict['init_img'] = Image.fromarray(processed_frame)
          img2img_args_dict['mask_img'] = None
          img2img_args_dict['denoising_strength'] = args_dict['fix_frame_strength']
          img2img_args_dict['seed'] = args_dict['step_2_seed']
          utils.set_CNs_input_image(img2img_args_dict, Image.fromarray(curr_frame))
          processed_frames, _, _, _ = utils.img2img(img2img_args_dict)
          processed_frame = np.array(processed_frames[0])
          processed_frame = skimage.exposure.match_histograms(processed_frame, curr_frame, channel_axis=None)

        processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)
        warped_styled_frame_ = np.clip(warped_styled_frame_, 0, 255).astype(np.uint8)
        
        # Write the frame to the output video
        frame_out = np.clip(processed_frame, 0, 255).astype(np.uint8)
        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR) 
        sdcn_anim_tmp.output_video.write(frame_out)

        sdcn_anim_tmp.process_counter += 1
        #if sdcn_anim_tmp.process_counter >= sdcn_anim_tmp.total_frames - 1:
        #    sdcn_anim_tmp.input_video.release()
        #    sdcn_anim_tmp.output_video.release()
        #    sdcn_anim_tmp.prev_frame = None

        save_result_to_image(processed_frame, sdcn_anim_tmp.process_counter + 1)

    stat = get_cur_stat() + utils.get_time_left(step+2, loop_iterations+1, processing_start_time)
    yield stat, curr_frame, occlusion_mask, warped_styled_frame_, processed_frame, None, gr.Button.update(interactive=False), gr.Button.update(interactive=True)

  RAFT_clear_memory()

  sdcn_anim_tmp.input_video.release()
  sdcn_anim_tmp.output_video.release()

  curr_frame = gr.Image.update()
  occlusion_mask = gr.Image.update()
  warped_styled_frame_ = gr.Image.update() 
  processed_frame = gr.Image.update()

  yield get_cur_stat(), curr_frame, occlusion_mask, warped_styled_frame_, processed_frame, output_video_name, gr.Button.update(interactive=True), gr.Button.update(interactive=False)