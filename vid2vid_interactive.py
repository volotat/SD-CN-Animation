import requests
import cv2
import base64
import numpy as np
from tqdm import tqdm
import os
import subprocess 

import h5py
from flow_utils import compute_diff_map

# Ask the user to specify the input file
INPUT_VIDEO = input("Enter the path to the input video file: ")
# Ask the user to specify the flow maps file
flow_maps_input = input("Please enter the path to the flow maps file (flow.h5): ")
# Set the path to the output flow maps file
flow_maps_output = flow_maps_input

input_video = cv2.VideoCapture(INPUT_VIDEO)

# Get the width and height of the input video
w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Set the flow map width and height
flow_width = w
flow_height = h

user_choice = input("Do you want to run compute_flow.py to compute flow maps? (Y/N): ").strip().lower()

if user_choice == 'y':
    import subprocess 
    # Call compute_flow.py with the appropriate arguments
    subprocess.run(
        [
            "python",
            "compute_flow.py",
            "-i", INPUT_VIDEO,
            "-o", flow_maps_output,
            "-v",
            "-W", str(flow_width),
            "-H", str(flow_height),
        ],
        check=True,
    )
else:
    print("Skipping flow map computation...")

# Set the path to the output flow maps file
flow_maps_output = "flow.h5"

# Use the output flow maps file in the main script
FLOW_MAPS = flow_maps_output

# Use the output flow maps file in the main script
FLOW_MAPS = flow_maps_output

OUTPUT_VIDEO = input("Enter the path to the output video file: ")

PROMPT = input("Enter the prompt: ")
default_n_prompt = "(blur, blurred, unfocus, obscure, dim, fade, obscure, muddy, black and white image, old, naked, black person, green face, green skin, black and white, slanted eyes, red eyes, blood eyes, deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blurry, ((((mutated hands and fingers)))), watermark, watermarked, oversaturated, censored, distorted hands, amputation, missing hands, obese, doubled face, double hands, easynegative, bad-hands-5"

user_choice = input("Do you want to use the default negative prompt? (Y/N): ").strip().lower()

if user_choice == 'y':
    N_PROMPT = default_n_prompt
else:
    N_PROMPT = input("Enter your custom negative prompt: ")

# Ask the user to specify the start frame index
# index of a frame to start a processing from. Might be helpful with long animations where you need to restart the script multiple times
start_from_ind = input("Enter the index of the frame to start processing from or press enter for default (default: 0): ").strip()
if not start_from_ind.isdigit():
    START_FROM_IND = 0
else:
    START_FROM_IND = int(start_from_ind)
SAVE_FRAMES = input("Do you want to save individual frames into 'out' folder? might be helpful with long animations (Y/N): ").strip().lower() == 'y'    

PROCESSING_STRENGTH = float(input("Enter the processing strength or press enter for default  (default is 0.85): ") or 0.85)
BLUR_FIX_STRENGTH = float(input("Enter the blur fix strength or press enter for default  (default is 0.15): ") or 0.15)

APPLY_HED = input("Do you want to apply HED? (Y/N): ").strip().lower() == 'y'
APPLY_CANNY = input("Do you want to apply Canny? (Y/N): ").strip().lower() == 'y'
APPLY_DEPTH = input("Do you want to apply Depth? (Y/N): ").strip().lower() == 'y'
GUESSMODE = input("Do you want to use guessmode? (Y/N): ").strip().lower() == 'y'

CFG_SCALE = float(input("Enter the cfg scale or press enter for default (default is 5.5): ") or 5.5)

# Define the list of options
options = {
    "1": "Euler",
    "2": "Euler a",
    "3": "DPM++ 2S a",
    "4": "UniPC",
    "5": "Custom",
}

# Ask the user to select an option
print("Please select an option from the following list:")
for key, value in options.items():
    print(f"{key}. {value}")

user_choice = input("Enter the number corresponding to your choice: ").strip()

# Validate the user input and set the SAMPLER variable
if user_choice in options:
    if user_choice == "5":
        custom_sampler = input("Please enter your custom sampler: ")
        SAMPLER = custom_sampler.strip()
    else:
        SAMPLER = options[user_choice]
else:
    print("Invalid choice. Please enter a number from 1 to 5.")
    
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
      "sampler_index": SAMPLER,
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
        "weight": 0.85,
        "resize_mode": "Just Resize",
        "lowvram": False,
        "processor_res": 512,
        "guidance_start": 0,
        "guidance_end": 0.85,
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

      out_image = out_image.astype(float) * alpha_mask + warped_styled.astype(float) * (1 - alpha_mask)

      out_image_ = (out_image * 0.65 + warped_styled * 0.35) 
      
      
    # Bluring issue fix via additional processing
    out_image_fixed = controlnetRequest(to_b64(out_image_), to_b64(frame), BLUR_FIX_STRENGTH, w, h, mask = None, seed=8888).sendRequest()

    # Write the frame to the output video
    frame_out = np.clip(out_image_fixed, 0, 255).astype(np.uint8)
    output_video.write(frame_out)

    if VISUALIZE:
      # show the last written frame - useful to catch any issue with the process
      img_show = cv2.hconcat([frame_out, alpha_img])
      cv2.imshow('Out img', img_show)
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
