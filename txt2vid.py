import requests
import cv2
import base64
import numpy as np
from tqdm import tqdm
import os

import sys
sys.path.append('FloweR/')
sys.path.append('RAFT/core')

import torch
from model import FloweR
from utils import flow_viz

from flow_utils import *
import skimage
import datetime


OUTPUT_VIDEO = f'result_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.mp4'

PROMPT = "RAW photo, bonfire near the camp in the mountains at night, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
N_PROMPT = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, letters, logo, brand, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
w,h = 512, 512 # Width and height of the processed image. Note that actual image processed would be a W x H resolution.

SAVE_FRAMES = True # saves individual frames into 'out' folder if set True. Again might be helpful with long animations

PROCESSING_STRENGTH = 0.85
FIX_STRENGTH = 0.35

CFG_SCALE = 5.5

APPLY_TEMPORALNET = False
APPLY_COLOR = False

VISUALIZE = True
DEVICE = 'cuda'

def to_b64(img):
  img_cliped = np.clip(img, 0, 255).astype(np.uint8)
  _, buffer = cv2.imencode('.png', img_cliped)
  b64img = base64.b64encode(buffer).decode("utf-8")
  return b64img

class controlnetRequest():
  def __init__(self, b64_init_img = None, b64_prev_img = None, b64_color_img = None, ds = 0.35, w=w, h=h, mask = None, seed=-1, mode='img2img'):
    self.url = f"http://localhost:7860/sdapi/v1/{mode}"
    self.body = {
      "init_images": [b64_init_img],
      "mask": mask,
      "mask_blur": 0,
      "inpainting_fill": 1,
      "inpainting_mask_invert": 0,
      "prompt": PROMPT,
      "negative_prompt": N_PROMPT,
      "seed": seed,
      "subseed": -1,
      "subseed_strength": 0,
      "batch_size": 1,
      "n_iter": 1,
      "steps": 15,
      "cfg_scale": CFG_SCALE,
      "denoising_strength": ds,
      "width": w,
      "height": h,
      "restore_faces": False,
      "eta": 0,
      "sampler_index": "DPM++ 2S a",
      "control_net_enabled": True,
      "alwayson_scripts": {
        "ControlNet":{"args": []}
      },
    }

    if APPLY_TEMPORALNET:
      self.body["alwayson_scripts"]["ControlNet"]["args"].append({
        "input_image": b64_prev_img,
        "module": "none",
        "model": "diff_control_sd15_temporalnet_fp16 [adc6bd97]",
        "weight": 0.65,
        "resize_mode": "Just Resize",
        "lowvram": False,
        "processor_res": 512,
        "guidance_start": 0,
        "guidance_end": 0.65,
        "guessmode": False
      })

    if APPLY_COLOR:
      self.body["alwayson_scripts"]["ControlNet"]["args"].append({
        "input_image": b64_prev_img,
        "module": "color",
        "model": "t2iadapter_color_sd14v1 [8522029d]",
        "weight": 0.65,
        "resize_mode": "Just Resize",
        "lowvram": False,
        "processor_res": 512,
        "guidance_start": 0,
        "guidance_end": 0.65,
        "guessmode": False
      })


  def sendRequest(self):
      # Request to web-ui
      data_js = requests.post(self.url, json=self.body).json()

      # Convert the byte array to a NumPy array
      image_bytes = base64.b64decode(data_js["images"][0])
      np_array = np.frombuffer(image_bytes, dtype=np.uint8)

      # Convert the NumPy array to a cv2 image
      out_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
      return out_image
  

  
if VISUALIZE: cv2.namedWindow('Out img')


# Create an output video file with the same fps, width, and height as the input video
output_video = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))

prev_frame = None
prev_frame_styled = None


# Instantiate the model
model = FloweR(input_size = (h, w))
model.load_state_dict(torch.load('FloweR/FloweR_0.1.pth'))
# Move the model to the device
model = model.to(DEVICE)


init_frame = controlnetRequest(mode='txt2img', ds=PROCESSING_STRENGTH, w=w, h=h).sendRequest()

output_video.write(init_frame)
prev_frame = init_frame

clip_frames = np.zeros((4, h, w, 3), dtype=np.uint8)

color_shift = np.zeros((0, 3))
color_scale = np.zeros((0, 3))
for ind in tqdm(range(40)): 
  clip_frames = np.roll(clip_frames, -1, axis=0)
  clip_frames[-1] = prev_frame
  
  clip_frames_torch = frames_norm(torch.from_numpy(clip_frames).to(DEVICE, dtype=torch.float32))

  with torch.no_grad():
    pred_data = model(clip_frames_torch.unsqueeze(0))[0]

  pred_flow = flow_renorm(pred_data[...,:2]).cpu().numpy()
  pred_occl = occl_renorm(pred_data[...,2:3]).cpu().numpy().repeat(3, axis = -1)

  pred_flow = pred_flow / (1 + np.linalg.norm(pred_flow, axis=-1, keepdims=True) * 0.05) 
  pred_flow = cv2.GaussianBlur(pred_flow, (31,31), 1, cv2.BORDER_REFLECT_101)
 

  pred_occl = cv2.GaussianBlur(pred_occl, (21,21), 2, cv2.BORDER_REFLECT_101)
  pred_occl = (np.abs(pred_occl / 255) ** 1.5) * 255
  pred_occl = np.clip(pred_occl * 25, 0, 255).astype(np.uint8)

  flow_map = pred_flow.copy()
  flow_map[:,:,0] += np.arange(w)
  flow_map[:,:,1] += np.arange(h)[:,np.newaxis]

  warped_frame = cv2.remap(prev_frame, flow_map, None, cv2.INTER_CUBIC, borderMode = cv2.BORDER_REFLECT_101)

  out_image = warped_frame.copy()
  
  out_image = controlnetRequest(
      b64_init_img = to_b64(out_image),
      b64_prev_img = to_b64(prev_frame),
      b64_color_img = to_b64(warped_frame),
      mask = to_b64(pred_occl),
      ds=PROCESSING_STRENGTH, w=w, h=h).sendRequest()
 
  out_image = controlnetRequest(
      b64_init_img = to_b64(out_image),
      b64_prev_img = to_b64(prev_frame),
      b64_color_img = to_b64(warped_frame),
      mask = None,
      ds=FIX_STRENGTH, w=w, h=h).sendRequest()
  
  # These step is necessary to reduce color drift of the image that some models may cause
  out_image = skimage.exposure.match_histograms(out_image, init_frame, multichannel=True, channel_axis=-1)
  
  output_video.write(out_image)
  if SAVE_FRAMES: 
      if not os.path.isdir('out'): os.makedirs('out')
      cv2.imwrite(f'out/{ind+1:05d}.png', out_image)

  pred_flow_img = flow_viz.flow_to_image(pred_flow)
  frames_img = cv2.hconcat(list(clip_frames))
  data_img = cv2.hconcat([pred_flow_img, pred_occl, warped_frame, out_image])

  cv2.imshow('Out img', cv2.vconcat([frames_img, data_img]))
  if cv2.waitKey(1) & 0xFF == ord('q'): exit() # press Q to close the script while processing

  prev_frame = out_image.copy()
    
# Release the input and output video files
output_video.release()

# Close all windows
if VISUALIZE: cv2.destroyAllWindows()