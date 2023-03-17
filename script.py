import requests
import cv2
import base64
import numpy as np
from tqdm import tqdm
import os

INPUT_VIDEO = "video_input.mp4"
OUTPUT_VIDEO = "result.mp4"
REF_IMAGE = "init.png"

PROMPT = "pixarstyle 3D cartoon version of Pulp Fiction apartment Scene, natural skin texture, 4k textures, hdr, intricate, highly detailed, sharp focus, cinematic look, hyperdetailed. White man, black man, strong man, 90's room, gun, burger and cola."
N_PROMPT = "cropped head, black and white, slanted eyes, deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blurry, ((((mutated hands and fingers)))), watermark, watermarked, oversaturated, censored, distorted hands, amputation, missing hands, obese, doubled face, double hands"
SEED = 2901260158
w,h = 1216, 512 # Width and height of the processed image. Note that actual image processed would be a W x 2H resolution. You should have enough VRAM to process it.


START_FROM_IND = 2235 # index of a frame to start a processing from. Might be helpful with long animations where you need to restart the script multiple times
SAVE_FRAMES = True # saves individual frames into 'out' folder if set True. Again might be helpful with long animations

def to_b64(img):
    _, buffer = cv2.imencode('.png', img)
    b64img = base64.b64encode(buffer).decode("utf-8")
    return b64img

mask_img = np.zeros((h * 2,w,3), dtype=np.uint8)
mask_img[:h] = 255
b64_mask_img = to_b64(mask_img)

# load context image to make generations more stable
cont_img = cv2.imread(REF_IMAGE)
cont_img = cv2.resize(cont_img, (w,h))

class controlnetRequest():
    def __init__(self, b64_full_img):
        self.url = "http://localhost:7860/controlnet/img2img"
        self.body = {
            "init_images": [b64_full_img],
            "mask": b64_mask_img,
            "mask_blur": 0,
            "inpainting_fill": 1,
            "prompt": PROMPT,
            "negative_prompt": N_PROMPT,
            "seed": SEED,
            "subseed": -1,
            "subseed_strength": 0,
            "batch_size": 1,
            "n_iter": 1,
            "steps": 20,
            "cfg_scale": 4.5,
            "denoising_strength":0.6,
            "width": w,
            "height": h * 2,
            "restore_faces": False,
            "eta": 0,
            "sampler_index": "DPM++ 2S a",
            "controlnet_units": [
                {
                    "module": "hed",
                    "model": "control_hed-fp16 [13fee50b]",
                    "weight": 0.6,
                    "resize_mode": "Just Resize",
                    "lowvram": False,
                    "processor_res": 64,
                    "threshold_a": 64,
                    "threshold_b": 64,
                    "guidance": 0.6,
                    "guessmode": False
                },
                {
                    "module": "none",
                    "model": "t2iadapter_color_sd14v1 [8522029d]",
                    "weight": 0.6,
                    "resize_mode": "Just Resize",
                    "lowvram": False,
                    "processor_res": 64,
                    "threshold_a": 64,
                    "threshold_b": 64,
                    "guidance": 0.6,
                    "guessmode": False
                }
            ]
        }

    def sendRequest(self):
        r = requests.post(self.url, json=self.body)
        return r.json()

# Open the input video file
input_video = cv2.VideoCapture(INPUT_VIDEO)

# Get useful info from the souce video
fps = int(input_video.get(cv2.CAP_PROP_FPS))
total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output video file with the same fps, width, and height as the input video
output_video = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'MP4V'), fps, (w, h))

for ind in tqdm(range(total_frames)):
    # Read the next frame from the input video
    if not input_video.isOpened(): break
    ret, frame = input_video.read()
    if not ret: break

    if ind+1 < START_FROM_IND: continue

    # Resize the frame to proper resolution 
    frame = cv2.resize(frame, (w,h))

    full_img = cv2.vconcat([frame, cont_img])
    full_img = cv2.resize(full_img, (w,h * 2))
    b64_full_img = to_b64(full_img)

    # Sending request to the web-ui
    data_js = controlnetRequest(b64_full_img).sendRequest()
    
    # Convert the byte array to a NumPy array
    image_bytes = base64.b64decode(data_js["images"][0])
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Convert the NumPy array to a cv2 image
    cv2_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Write the frame to the output video
    cv2_image = cv2_image[:h]
    output_video.write(cv2_image)

    # show the last written frame - useful to catch any issue with the process
    cv2.imshow('Out img', cv2_image)
    
    if SAVE_FRAMES: 
        if not os.path.isdir('out'): os.makedirs('out')
        cv2.imwrite(f'out/{ind+1:05d}.png', cv2_image)
    if cv2.waitKey(1) & 0xFF == ord('q'): break # press Q to close the script while processing

# Release the input and output video files
input_video.release()
output_video.release()

# Close all windows
cv2.destroyAllWindows()