import numpy as np
import cv2
import sys
# Add these imports for U-2-Net
import torch
from torchvision import transforms
sys.path.append('U-2-Net/model')
from u2net import U2NET  # Assuming you have u2net.py in your working directory
import tqdm
# RAFT dependencies
import sys
sys.path.append('RAFT/core')
from collections import namedtuple
import argparse
from raft import RAFT
from utils.utils import InputPadder
from PIL import Image 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RAFT_model = None
u2net_model = None

def load_u2net_model(model_name="u2net"):
    model_dir = "./"
    model_file = model_name + ".pth"
    model_path = model_dir + model_file

    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    return net

u2net_model = load_u2net_model()

def remove_background_u2net(img, u2net_model):
    # convert the image to PIL format and resize it
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_transform = transforms.Compose([
        transforms.Resize((320, 320), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img_tensor = img_transform(img_pil).unsqueeze(0).to(device)

    # run the image through the U2NET model
    with torch.no_grad():
        output = u2net_model(img_tensor)[0] # Updated this line

    # convert the output mask to a numpy array and resize it
    mask = output[0][0].cpu().detach().numpy()
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # apply the mask to the original image
    img_bg_removed = img.copy()
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    img_bg_removed[mask < 0.5] = [0, 0, 0]

    return img_bg_removed



# Replace the 'background_subtractor' function with the 'remove_background_u2net' function
def RAFT_estimate_flow(frame1, frame2, frame1_bg_removed, frame2_bg_removed, device='cuda', subtract_background=True):
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

    if subtract_background:
        frame1 = frame1_bg_removed
        frame2 = frame2_bg_removed

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


    # compute occlusion mask
        fb_flow_diff = next_flow - prev_flow  # Calculate the difference between forward and backward flow
        fb_norm_diff = np.linalg.norm(fb_flow_diff, axis=2)

        occlusion_mask = fb_norm_diff[..., None].repeat(3, axis=-1)
        occlusion_mask = cv2.medianBlur(occlusion_mask, 5)  # Apply a median filter with a kernel size of 5

    return next_flow, prev_flow, occlusion_mask, frame1, frame2


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

def process_video(input_video_path, subtract_background=True):
    video_capture = cv2.VideoCapture(input_video_path)

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for idx in tqdm.tqdm(range(frame_count - 1)):
        ret, frame1 = video_capture.read()
        if not ret: break
        ret, frame2 = video_capture.read()
        if not ret: break

        if subtract_background:
            frame1_bg_removed = remove_background_u2net(frame1, u2net_model)
            frame2_bg_removed = remove_background_u2net(frame2, u2net_model)
        else:
            frame1_bg_removed = frame1.copy()
            frame2_bg_removed = frame2.copy()

        # Get the flow estimation and other results
        next_flow, prev_flow, occlusion_mask, frame1, frame2 = RAFT_estimate_flow(frame1, frame2, frame1_bg_removed, frame2_bg_removed, subtract_background=subtract_background)
        # Remove small values in the optical flow estimation
        next_flow[np.abs(next_flow) < 3] = 0
        prev_flow[np.abs(prev_flow) < 3] = 0

        # Display the original frame, the frame with no background, and the occlusion mask
        occlusion_mask_vis = np.clip(occlusion_mask * 255, 0, 255).astype(np.uint8)
        output_image = cv2.hconcat([frame1, frame1_bg_removed, occlusion_mask_vis])
        cv2.imshow('Connected Windows', output_image)
        key = cv2.waitKey(1) & 0xFF

        # Exit if 'q' is pressed
        if key == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to the input video file", required=True)
    parser.add_argument("-o", "--output", help="Path to the output file", required=True)
    parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("-W", "--width", type=int, help="Output video width", required=True)
    parser.add_argument("-H", "--height", type=int, help="Output video height", required=True)
    parser.add_argument("--subtract_background", action="store_true", help="Remove background using U2-Net")
    args = parser.parse_args()

    process_video(args.input)
