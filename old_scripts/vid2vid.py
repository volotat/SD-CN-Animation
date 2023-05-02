import requests
import cv2
import base64
import numpy as np
from tqdm import tqdm
import os

import h5py
from flow_utils import compute_diff_map

import skimage
import datetime

INPUT_VIDEO = "/media/alex/ded3efe6-5825-429d-ac89-7ded676a2b6d/media/Peter_Gabriel/pexels-monstera-5302599-4096x2160-30fps.mp4"
FLOW_MAPS = "/media/alex/ded3efe6-5825-429d-ac89-7ded676a2b6d/media/Peter_Gabriel/pexels-monstera-5302599-4096x2160-30fps.h5"
OUTPUT_VIDEO = f'videos/result_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.mp4'

PROMPT = "Underwater shot Peter Gabriel with closed eyes in  Peter Gabriel's music video. 80's music video. VHS style. Dramatic light, Cinematic light. RAW photo, 8k uhd, dslr, soft lighting, high quality, film grain."
N_PROMPT = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
w,h = 1088, 576 # Width and height of the processed image. Note that actual image processed would be a W x H resolution.

START_FROM_IND = 0 # index of a frame to start a processing from. Might be helpful with long animations where you need to restart the script multiple times
SAVE_FRAMES = True # saves individual frames into 'out' folder if set True. Again might be helpful with long animations

PROCESSING_STRENGTH = 0.95
BLUR_FIX_STRENGTH = 0.15

APPLY_HED = True
APPLY_CANNY = False
APPLY_DEPTH = False
GUESSMODE = False

CFG_SCALE = 5.5

VISUALIZE = True

def to_b64(img):
  img_cliped = np.clip(img, 0, 255).astype(np.uint8)
  _, buffer = cv2.imencode('.png', img_cliped)
  b64img = base64.b64encode(buffer).decode("utf-8")
  return b64img

class controlnetRequest():
  def __init__(self, b64_cur_img, b64_hed_img, ds = 0.35, w=w, h=h, mask = None, seed=-1):
    self.url = "http://localhost:7860/sdapi/v1/img2img"
    self.body = {
      "init_images": [b64_cur_img],
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

    if APPLY_HED:
      self.body["alwayson_scripts"]["ControlNet"]["args"].append({
        "input_image": b64_hed_img,
        "module": "hed",
        "model": "control_hed-fp16 [13fee50b]",
        "weight": 0.65,
        "resize_mode": "Just Resize",
        "lowvram": False,
        "processor_res": 512,
        "guidance_start": 0,
        "guidance_end": 0.65,
        "guessmode": GUESSMODE
      })

    if APPLY_CANNY:
      self.body["alwayson_scripts"]["ControlNet"]["args"].append({
        "input_image": b64_hed_img,
        "module": "canny",
        "model": "control_canny-fp16 [e3fe7712]",
        "weight": 0.85,
        "resize_mode": "Just Resize",
        "lowvram": False,
        "threshold_a": 35,
        "threshold_b": 35,
        "processor_res": 512,
        "guidance_start": 0,
        "guidance_end": 0.85,
        "guessmode": GUESSMODE
      })

    if APPLY_DEPTH:
      self.body["alwayson_scripts"]["ControlNet"]["args"].append({
        "input_image": b64_hed_img,
        "module": "depth",
        "model": "control_depth-fp16 [400750f6]",
        "weight": 0.85,
        "resize_mode": "Just Resize",
        "lowvram": False,
        "processor_res": 512,
        "guidance_start": 0,
        "guidance_end": 0.85,
        "guessmode": GUESSMODE
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

# Open the input video file
input_video = cv2.VideoCapture(INPUT_VIDEO)

# Get useful info from the source video
fps = int(input_video.get(cv2.CAP_PROP_FPS))
total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output video file with the same fps, width, and height as the input video
output_video = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

prev_frame = None
prev_frame_styled = None
#init_image = None

# reading flow maps in a stream manner
with h5py.File(FLOW_MAPS, 'r') as f:
  flow_maps = f['flow_maps']

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

      # Processing current frame with current frame as a mask without any inpainting
      out_image = controlnetRequest(to_b64(frame), to_b64(frame), PROCESSING_STRENGTH, w, h, mask = None).sendRequest()

      alpha_img = out_image.copy()
      out_image_ = out_image.copy()
      warped_styled = out_image.copy()
      #init_image = out_image.copy()
    else:
      # Resize the frame to proper resolution 
      frame = cv2.resize(cur_frame, (w, h))
      prev_frame = cv2.resize(prev_frame, (w, h))

      # Processing current frame with current frame as a mask without any inpainting
      out_image = controlnetRequest(to_b64(frame), to_b64(frame), PROCESSING_STRENGTH, w, h, mask = None).sendRequest()

      next_flow, prev_flow = flow_maps[ind-1].astype(np.float32)
      alpha_mask, warped_styled = compute_diff_map(next_flow, prev_flow, prev_frame, frame, prev_frame_styled)

      # This clipping at lower side required to fix small trailing issues that for some reason left outside of the bright part of the mask, 
      # and at the higher part it making parts changed strongly to do it with less flickering. 
      alpha_mask = np.clip(alpha_mask + 0.05, 0.05, 0.95)
      alpha_img = np.clip(alpha_mask * 255, 0, 255).astype(np.uint8)

      # normalizing the colors
      out_image = skimage.exposure.match_histograms(out_image, frame, multichannel=False, channel_axis=-1)

      out_image = out_image.astype(float) * alpha_mask + warped_styled.astype(float) * (1 - alpha_mask)

      #out_image = skimage.exposure.match_histograms(out_image, prev_frame, multichannel=True, channel_axis=-1)
      #out_image_ = (out_image * 0.65 + warped_styled * 0.35) 
      
      
    # Bluring issue fix via additional processing
    out_image_fixed = controlnetRequest(to_b64(out_image), to_b64(frame), BLUR_FIX_STRENGTH, w, h, mask = None, seed=8888).sendRequest()
    

    # Write the frame to the output video
    frame_out = np.clip(out_image_fixed, 0, 255).astype(np.uint8)
    output_video.write(frame_out)

    if VISUALIZE:
      # show the last written frame - useful to catch any issue with the process
      warped_styled = np.clip(warped_styled, 0, 255).astype(np.uint8)

      img_show_top = cv2.hconcat([frame, warped_styled])
      img_show_bot = cv2.hconcat([frame_out, alpha_img])
      cv2.imshow('Out img', cv2.vconcat([img_show_top, img_show_bot]))
      cv2.setWindowTitle("Out img", str(ind+1))
      if cv2.waitKey(1) & 0xFF == ord('q'): exit() # press Q to close the script while processing

    if SAVE_FRAMES: 
        if not os.path.isdir('out'): os.makedirs('out')
        cv2.imwrite(f'out/{ind+1:05d}.png', frame_out)

    prev_frame = cur_frame.copy()
    prev_frame_styled = out_image.copy()

    
# Release the input and output video files
input_video.release()
output_video.release()

# Close all windows
if VISUALIZE: cv2.destroyAllWindows()
