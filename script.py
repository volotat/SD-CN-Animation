import requests
import cv2
import base64
import numpy as np
from tqdm import tqdm
import os

#RAFT dependencies
import sys
sys.path.append('RAFT/core')

from collections import namedtuple
import torch
import argparse
from raft import RAFT
from utils.utils import InputPadder


INPUT_VIDEO = "/media/alex/ded3efe6-5825-429d-ac89-7ded676a2b6d/media/Fallout_noir_2/benny.mp4"
OUTPUT_VIDEO = "result.mp4"

PROMPT = "RAW photo, Matthew Perry wearing suit and pants in the wasteland, cinematic light, dramatic light, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
N_PROMPT = "blur, blurred, unfocus, obscure, dim, fade, obscure, muddy, black and white image, old, naked, black person, green face, green skin, black and white, slanted eyes, red eyes, blood eyes, deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blurry, ((((mutated hands and fingers)))), watermark, watermarked, oversaturated, censored, distorted hands, amputation, missing hands, obese, doubled face, double hands"
SEED = -1
w,h = 512, 704 # Width and height of the processed image. Note that actual image processed would be a W x 2H resolution. You should have enough VRAM to process it.

START_FROM_IND = 0 # index of a frame to start a processing from. Might be helpful with long animations where you need to restart the script multiple times
SAVE_FRAMES = True # saves individual frames into 'out' folder if set True. Again might be helpful with long animations

BLUR_SIZE = (15, 15)
BLUR_SIGMA = 12

def to_b64(img):
  _, buffer = cv2.imencode('.png', img)
  b64img = base64.b64encode(buffer).decode("utf-8")
  return b64img

class controlnetRequest():
  def __init__(self, b64_cur_img, b64_hed_img, ds = 0.35, w=w, h=h, mask = None):
    self.url = "http://localhost:7860/sdapi/v1/img2img"
    self.body = {
      "init_images": [b64_cur_img],
      "mask": mask,
      "mask_blur": 0,
      "inpainting_fill": 1,
      "inpainting_mask_invert": 0,
      "prompt": PROMPT,
      "negative_prompt": N_PROMPT,
      "seed": SEED,
      "subseed": -1,
      "subseed_strength": 0,
      "batch_size": 1,
      "n_iter": 1,
      "steps": 15,
      "cfg_scale": 7,
      "denoising_strength": ds,
      "width": w,
      "height": h,
      "restore_faces": False,
      "eta": 0,
      "sampler_index": "DPM++ 2S a",
      "control_net_enabled": True,
      "alwayson_scripts": {
        "ControlNet":{
          "args": [
            {
                "input_image": b64_hed_img,
                "module": "hed",
                "model": "control_hed-fp16 [13fee50b]",
                "weight": 1,
                "resize_mode": "Just Resize",
                "lowvram": False,
                "processor_res": 512,
                "guidance": 1,
                "guessmode": False
            }
          ]
        }
      },
    }

  def sendRequest(self):
      r = requests.post(self.url, json=self.body)
      return r.json()
    
DEVICE = 'cuda'
RAFT_model = None
def RAFT_estimate_flow_diff(frame1, frame2, frame1_styled):
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
    RAFT_model.to(DEVICE)
    RAFT_model.eval()

  with torch.no_grad():
    frame1_torch = torch.from_numpy(frame1).permute(2, 0, 1).float()[None].to(DEVICE)
    frame2_torch = torch.from_numpy(frame2).permute(2, 0, 1).float()[None].to(DEVICE)

    padder = InputPadder(frame1_torch.shape)
    image1, image2 = padder.pad(frame1_torch, frame2_torch)

    # estimate and apply optical flow
    _, next_flow = RAFT_model(image1, image2, iters=20, test_mode=True)
    _, prev_flow = RAFT_model(image2, image1, iters=20, test_mode=True)

    next_flow = next_flow[0].permute(1,2,0).cpu().numpy()
    prev_flow = prev_flow[0].permute(1,2,0).cpu().numpy()

    flow_map = prev_flow.copy()
    h, w = flow_map.shape[:2]
    flow_map[:,:,0] += np.arange(w)
    flow_map[:,:,1] += np.arange(h)[:,np.newaxis]

    warped_frame = cv2.remap(frame1, flow_map, None, cv2.INTER_LINEAR)
    warped_frame_styled = cv2.remap(frame1_styled, flow_map, None, cv2.INTER_LINEAR)

    # compute occlusion mask
    fb_flow = next_flow + prev_flow
    fb_norm = np.linalg.norm(fb_flow, axis=2)

    occlusion_mask = fb_norm[..., None] 

    diff_mask = np.abs(warped_frame.astype(np.float32) - frame2.astype(np.float32)) / 255
    diff_mask = diff_mask.max(axis = -1, keepdims=True)

    diff = np.maximum(occlusion_mask * 0.2, diff_mask * 4).repeat(3, axis = -1)
    #diff = diff * 1.5

    diff_blured = cv2.GaussianBlur(diff, BLUR_SIZE, BLUR_SIGMA, cv2.BORDER_DEFAULT)
    diff_frame = np.clip((diff + diff_blured) * 255, 0, 255).astype(np.uint8)

  return warped_frame, diff_frame, warped_frame_styled

''' # old flow estimation algorithm, might be useful later 
def estimate_flow_diff(frame1, frame2, frame1_styled):
  prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
  next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

  # estimate and apply optical flow
  flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
  h, w = flow.shape[:2]
  flow_data = -flow
  flow_data[:,:,0] += np.arange(w)
  flow_data[:,:,1] += np.arange(h)[:,np.newaxis]
  #map_x, map_y = cv2.convertMaps(flow_data, 0, -1, True)
  warped_frame = cv2.remap(frame1, flow_data, None, cv2.INTER_LINEAR)
  warped_frame_styled = cv2.remap(frame1_styled, flow_data, None, cv2.INTER_LINEAR)

  # compute occlusion mask
  flow_back = cv2.calcOpticalFlowFarneback(next, prvs, None, 0.5, 3, 15, 3, 5, 1.2, 0)
  fb_flow = flow + flow_back
  fb_norm = np.linalg.norm(fb_flow, axis=2)
  occlusion_mask = fb_norm[..., None]

  diff_mask = np.abs(warped_frame.astype(np.float32) - frame2.astype(np.float32)) / 255
  diff_mask = diff_mask.max(axis = -1, keepdims=True)

  diff = np.maximum(occlusion_mask, diff_mask).repeat(3, axis = -1)
  #diff = diff * 1.5

  #diff = cv2.GaussianBlur(diff, BLUR_SIZE, BLUR_SIGMA, cv2.BORDER_DEFAULT)
  diff_frame = np.clip(diff * 255, 0, 255).astype(np.uint8)

  return warped_frame, diff_frame, warped_frame_styled
'''

cv2.namedWindow('Out img')

# Open the input video file
input_video = cv2.VideoCapture(INPUT_VIDEO)

# Get useful info from the souce video
fps = int(input_video.get(cv2.CAP_PROP_FPS))
total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output video file with the same fps, width, and height as the input video
output_video = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'MP4V'), fps, (w, h))

prev_frame = None

for ind in tqdm(range(total_frames)):
    # Read the next frame from the input video
    if not input_video.isOpened(): break
    ret, cur_frame = input_video.read()
    if not ret: break

    if ind+1 < START_FROM_IND: continue

    is_keyframe = True
    if prev_frame is not None:
        # Compute absolute difference between current and previous frame
        frames_diff = cv2.absdiff(cur_frame, prev_frame)
        # Compute mean of absolute difference
        mean_diff = cv2.mean(frames_diff)[0]
        # Check if mean difference is above threshold
        is_keyframe = mean_diff > 30

    # Generate course version of a current frame with previous stylized frame as a reference image
    if is_keyframe:
        # Resize the frame to proper resolution 
        frame = cv2.resize(cur_frame, (w, h))

        # Sending request to the web-ui
        data_js = controlnetRequest(to_b64(frame), to_b64(frame), 0.65, w, h, mask = None).sendRequest()
        
        # Convert the byte array to a NumPy array
        image_bytes = base64.b64decode(data_js["images"][0])
        np_array = np.frombuffer(image_bytes, dtype=np.uint8)

        # Convert the NumPy array to a cv2 image
        out_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        diff_mask = out_image.copy()
        diff_mask_blured = out_image.copy()
    else:
        # Resize the frame to proper resolution 
        frame = cv2.resize(cur_frame, (w, h))
        prev_frame = cv2.resize(prev_frame, (w, h))

        # Sending request to the web-ui
        data_js = controlnetRequest(to_b64(frame), to_b64(frame), 0.35, w, h, mask = None).sendRequest()
        
        # Convert the byte array to a NumPy array
        image_bytes = base64.b64decode(data_js["images"][0])
        np_array = np.frombuffer(image_bytes, dtype=np.uint8)

        # Convert the NumPy array to a cv2 image
        out_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)


        _, diff_mask, warped_styled = RAFT_estimate_flow_diff(prev_frame, frame, prev_frame_styled)

        alpha = diff_mask.astype(np.float32) / 255.0
        pr_image = out_image * alpha + warped_styled * (1 - alpha)

        diff_mask_blured = cv2.GaussianBlur(alpha * 255, BLUR_SIZE, BLUR_SIGMA, cv2.BORDER_DEFAULT)
        diff_mask_blured = np.clip(diff_mask_blured + diff_mask, 5, 255).astype(np.uint8)

        # Sending request to the web-ui
        data_js = controlnetRequest(to_b64(pr_image), to_b64(frame), 0.65, w, h, mask = to_b64(diff_mask_blured)).sendRequest()
        
        # Convert the byte array to a NumPy array
        image_bytes = base64.b64decode(data_js["images"][0])
        np_array = np.frombuffer(image_bytes, dtype=np.uint8)

        # Convert the NumPy array to a cv2 image
        out_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Write the frame to the output video
    frame_out = out_image[:h]
    output_video.write(frame_out)

    # show the last written frame - useful to catch any issue with the process
    img_show = cv2.hconcat([out_image, diff_mask, diff_mask_blured])
    cv2.imshow('Out img', img_show)
    if cv2.waitKey(1) & 0xFF == ord('q'): exit() # press Q to close the script while processing


    # Write the frame to the output video
    output_video.write(frame_out)
    prev_frame = cur_frame.copy()
    prev_frame_styled = frame_out.copy()

    
    if SAVE_FRAMES: 
        if not os.path.isdir('out'): os.makedirs('out')
        cv2.imwrite(f'out/{ind+1:05d}.png', frame_out)

# Release the input and output video files
input_video.release()
output_video.release()

# Close all windows
cv2.destroyAllWindows()