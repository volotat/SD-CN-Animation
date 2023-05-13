import sys, os

import numpy as np
import cv2

from collections import namedtuple
import torch
import argparse
from RAFT.raft import RAFT
from RAFT.utils.utils import InputPadder

import modules.paths as ph
import gc

RAFT_model = None
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

def background_subtractor(frame, fgbg):
  fgmask = fgbg.apply(frame)
  return cv2.bitwise_and(frame, frame, mask=fgmask)

def RAFT_clear_memory():
  global RAFT_model
  del RAFT_model
  gc.collect()
  torch.cuda.empty_cache()
  RAFT_model = None

def RAFT_estimate_flow(frame1, frame2, device='cuda'):
  global RAFT_model

  org_size = frame1.shape[1], frame1.shape[0]
  size = frame1.shape[1] // 16 * 16, frame1.shape[0] // 16 * 16
  frame1 = cv2.resize(frame1, size)
  frame2 = cv2.resize(frame2, size)

  model_path = ph.models_path + '/RAFT/raft-things.pth'
  remote_model_path = 'https://drive.google.com/uc?id=1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM'

  if not os.path.isfile(model_path):
    from basicsr.utils.download_util import load_file_from_url
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    load_file_from_url(remote_model_path, file_name=model_path)

  if RAFT_model is None:
    args = argparse.Namespace(**{
        'model': ph.models_path + '/RAFT/raft-things.pth',
        'mixed_precision': True,
        'small': False,
        'alternate_corr': False,
        'path': ""
    })

    RAFT_model = torch.nn.DataParallel(RAFT(args))
    RAFT_model.load_state_dict(torch.load(args.model))

    RAFT_model = RAFT_model.module
    RAFT_model.to(device)
    RAFT_model.eval()

  with torch.no_grad():
    frame1_torch = torch.from_numpy(frame1).permute(2, 0, 1).float()[None].to(device)
    frame2_torch = torch.from_numpy(frame2).permute(2, 0, 1).float()[None].to(device)

    padder = InputPadder(frame1_torch.shape)
    image1, image2 = padder.pad(frame1_torch, frame2_torch)

    # estimate optical flow
    _, next_flow = RAFT_model(image1, image2, iters=20, test_mode=True)
    _, prev_flow = RAFT_model(image2, image1, iters=20, test_mode=True)

    next_flow = next_flow[0].permute(1, 2, 0).cpu().numpy()
    prev_flow = prev_flow[0].permute(1, 2, 0).cpu().numpy()

    fb_flow = next_flow + prev_flow
    fb_norm = np.linalg.norm(fb_flow, axis=2)

    occlusion_mask = fb_norm[..., None].repeat(3, axis=-1)

  next_flow = cv2.resize(next_flow, org_size)
  prev_flow = cv2.resize(prev_flow, org_size)

  return next_flow, prev_flow, occlusion_mask

def compute_diff_map(next_flow, prev_flow, prev_frame, cur_frame, prev_frame_styled, args_dict):
  h, w = cur_frame.shape[:2]
  fl_w, fl_h = next_flow.shape[:2]

  # normalize flow
  next_flow = next_flow / np.array([fl_h,fl_w]) 
  prev_flow = prev_flow / np.array([fl_h,fl_w])

  # compute occlusion mask
  fb_flow = next_flow + prev_flow
  fb_norm = np.linalg.norm(fb_flow , axis=2) 

  zero_flow_mask = np.clip(1 - np.linalg.norm(prev_flow, axis=-1)[...,None] * 20, 0, 1)
  diff_mask_flow = fb_norm[..., None] * zero_flow_mask

  # resize flow
  next_flow = cv2.resize(next_flow, (w, h)) 
  next_flow = (next_flow * np.array([h,w])).astype(np.float32)
  prev_flow = cv2.resize(prev_flow, (w, h))
  prev_flow = (prev_flow  * np.array([h,w])).astype(np.float32)

  # Generate sampling grids
  grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
  flow_grid = torch.stack((grid_x, grid_y), dim=0).float()
  flow_grid += torch.from_numpy(prev_flow).permute(2, 0, 1)
  flow_grid = flow_grid.unsqueeze(0)
  flow_grid[:, 0, :, :] = 2 * flow_grid[:, 0, :, :] / (w - 1) - 1
  flow_grid[:, 1, :, :] = 2 * flow_grid[:, 1, :, :] / (h - 1) - 1
  flow_grid = flow_grid.permute(0, 2, 3, 1)

  
  prev_frame_torch = torch.from_numpy(prev_frame).float().unsqueeze(0).permute(0, 3, 1, 2) #N, C, H, W
  prev_frame_styled_torch = torch.from_numpy(prev_frame_styled).float().unsqueeze(0).permute(0, 3, 1, 2) #N, C, H, W

  warped_frame = torch.nn.functional.grid_sample(prev_frame_torch, flow_grid, padding_mode="reflection").permute(0, 2, 3, 1)[0].numpy()
  warped_frame_styled = torch.nn.functional.grid_sample(prev_frame_styled_torch, flow_grid, padding_mode="reflection").permute(0, 2, 3, 1)[0].numpy()

  #warped_frame = cv2.remap(prev_frame, flow_map, None, cv2.INTER_NEAREST, borderMode = cv2.BORDER_REFLECT)
  #warped_frame_styled = cv2.remap(prev_frame_styled, flow_map, None, cv2.INTER_NEAREST, borderMode = cv2.BORDER_REFLECT)

  
  diff_mask_org = np.abs(warped_frame.astype(np.float32) - cur_frame.astype(np.float32)) / 255
  diff_mask_org = diff_mask_org.max(axis = -1, keepdims=True)

  diff_mask_stl = np.abs(warped_frame_styled.astype(np.float32) - cur_frame.astype(np.float32)) / 255
  diff_mask_stl = diff_mask_stl.max(axis = -1, keepdims=True)

  alpha_mask = np.maximum.reduce([diff_mask_flow * args_dict['occlusion_mask_flow_multiplier'] * 10, \
                                  diff_mask_org * args_dict['occlusion_mask_difo_multiplier'], \
                                  diff_mask_stl * args_dict['occlusion_mask_difs_multiplier']]) #
  alpha_mask = alpha_mask.repeat(3, axis = -1)

  #alpha_mask_blured = cv2.dilate(alpha_mask, np.ones((5, 5), np.float32))
  if args_dict['occlusion_mask_blur'] > 0:
    blur_filter_size = min(w,h) // 15 | 1
    alpha_mask = cv2.GaussianBlur(alpha_mask, (blur_filter_size, blur_filter_size) , args_dict['occlusion_mask_blur'], cv2.BORDER_REFLECT)

  alpha_mask = np.clip(alpha_mask, 0, 1)

  return alpha_mask, warped_frame_styled

def frames_norm(occl): return occl / 127.5 - 1

def flow_norm(flow): return flow / 255

def occl_norm(occl): return occl / 127.5 - 1

def flow_renorm(flow): return flow * 255

def occl_renorm(occl): return (occl + 1) * 127.5
