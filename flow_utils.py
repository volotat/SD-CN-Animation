import numpy as np
import cv2

#RAFT dependencies
import sys
sys.path.append('RAFT/core')

from collections import namedtuple
import torch
import argparse
from raft import RAFT
from utils.utils import InputPadder

RAFT_model = None
def RAFT_estimate_flow(frame1, frame2, device = 'cuda'):
  global RAFT_model
  if RAFT_model is None:
    args = argparse.Namespace(**{
      'model': 'RAFT/models/raft-things.pth',
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

    next_flow = next_flow[0].permute(1,2,0).cpu().numpy()
    prev_flow = prev_flow[0].permute(1,2,0).cpu().numpy()

    fb_flow = next_flow + prev_flow
    fb_norm = np.linalg.norm(fb_flow, axis=2)

    occlusion_mask = fb_norm[..., None].repeat(3, axis = -1)

  return next_flow, prev_flow, occlusion_mask

def compute_diff_map(next_flow, prev_flow, prev_frame, cur_frame, prev_frame_styled):
  h, w = cur_frame.shape[:2]

  next_flow = cv2.resize(next_flow, (w, h))
  prev_flow = cv2.resize(prev_flow, (w, h))

  flow_map = -next_flow.copy()
  flow_map[:,:,0] += np.arange(w)
  flow_map[:,:,1] += np.arange(h)[:,np.newaxis]

  warped_frame = cv2.remap(prev_frame, flow_map, None, cv2.INTER_NEAREST)
  warped_frame_styled = cv2.remap(prev_frame_styled, flow_map, None, cv2.INTER_NEAREST)

  # compute occlusion mask
  fb_flow = next_flow + prev_flow
  fb_norm = np.linalg.norm(fb_flow, axis=2)

  occlusion_mask = fb_norm[..., None] 

  diff_mask_org = np.abs(warped_frame.astype(np.float32) - cur_frame.astype(np.float32)) / 255
  diff_mask_org = diff_mask_org.max(axis = -1, keepdims=True)

  diff_mask_stl = np.abs(warped_frame_styled.astype(np.float32) - cur_frame.astype(np.float32)) / 255
  diff_mask_stl = diff_mask_stl.max(axis = -1, keepdims=True)

  alpha_mask = np.maximum(occlusion_mask * 0.3, diff_mask_org * 4, diff_mask_stl * 2)
  alpha_mask = alpha_mask.repeat(3, axis = -1)

  #alpha_mask_blured = cv2.dilate(alpha_mask, np.ones((5, 5), np.float32))
  alpha_mask = cv2.GaussianBlur(alpha_mask, (51,51), 5, cv2.BORDER_DEFAULT)

  alpha_mask = np.clip(alpha_mask, 0, 1)

  return alpha_mask, warped_frame_styled