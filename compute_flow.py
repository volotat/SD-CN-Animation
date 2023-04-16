import cv2
import base64
import numpy as np
from tqdm import tqdm
import os

from flow_utils import RAFT_estimate_flow
import h5py

import argparse

def main(args):
    W, H = args.width, args.height
    # Open the input video file
    input_video = cv2.VideoCapture(args.input_video)

    # Get useful info from the source video
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_frame = None

    # create an empty HDF5 file
    with h5py.File(args.output_file, 'w') as f: pass

    # open the file for writing a flow maps into it
    with h5py.File(args.output_file, 'a') as f:
        flow_maps = f.create_dataset('flow_maps', shape=(0, 2, H, W, 2), maxshape=(None, 2, H, W, 2), dtype=np.float16) 

        for ind in tqdm(range(total_frames)):
            # Read the next frame from the input video
            if not input_video.isOpened(): break
            ret, cur_frame = input_video.read()
            if not ret: break

            cur_frame = cv2.resize(cur_frame, (W, H))

            if prev_frame is not None:
                next_flow, prev_flow, occlusion_mask, frame1_bg_removed, frame2_bg_removed = RAFT_estimate_flow(prev_frame, cur_frame)

                # write data into a file
                flow_maps.resize(ind, axis=0)
                flow_maps[ind-1, 0] = next_flow
                flow_maps[ind-1, 1] = prev_flow

                occlusion_mask = np.clip(occlusion_mask * 0.2 * 255, 0, 255).astype(np.uint8)

                if args.visualize:
                    # show the last written frame - useful to catch any issue with the process
                    img_show = cv2.hconcat([cur_frame, frame2_bg_removed, occlusion_mask])
                    cv2.imshow('Out img', img_show)
                    if cv2.waitKey(1) & 0xFF == ord('q'): exit() # press Q to close the script while processing

            prev_frame = cur_frame.copy()

    # Release the input and output video files
    input_video.release()

    # Close all windows
    if args.visualize: cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_video', help="Path to input video file", required=True)
    parser.add_argument('-o', '--output_file', help="Path to output flow file. Stored in *.h5 format", required=True)
    parser.add_argument('-W', '--width', help='Width of the generated flow maps', default=1024, type=int)
    parser.add_argument('-H', '--height', help='Height of the generated flow maps', default=576, type=int)
    parser.add_argument('-v', '--visualize', action='store_true', help='Show proceed images and occlusion maps')
    args = parser.parse_args()

    main(args)
