# SD-CN-Animation
This script allows you to automate video stylization task using StableDiffusion and ControlNet. It uses a simple optical flow estimation algorithm to keep the animation stable and create an inpainting mask that is used to generate the next frame. Here is an example of a video made with this script:

[![IMAGE_ALT](https://img.youtube.com/vi/j-0niEMm6DU/0.jpg)](https://youtu.be/j-0niEMm6DU)

This script can also be using to swap the person in the video like in this example: https://youtube.com/shorts/be93_dIeZWU

## Dependencies
To install all the necessary dependencies, run this command:
```
pip install opencv-python opencv-contrib-python numpy tqdm h5py
```
You have to set up the RAFT repository as it described here: https://github.com/princeton-vl/RAFT . Basically it just comes down to running "./download_models.sh" in RAFT folder to download the models.

## Running the script
This script works on top of [Automatic1111/web-ui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) interface via API. To run this script you have to set it up first. You should also have[sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) extension installed. You need to have control_hed-fp16 model installed. If you have web-ui with ControlNet working correctly, you have to also allow API to work with controlNet. To do so, go to the web-ui settings -> ControlNet tab -> Set "Allow other script to control this extension" checkbox to active and set "Multi ControlNet: Max models amount (requires restart)" to more then 2 -> press "Apply settings".

### Step 1.
To process the video, first of all you would need to precompute optical flow data before running web-ui with this command:
```
python3 compute_flow.py -i {path to your video} -o {path to output *.h5} -v -W {width of the flow map} -H {height of the flow map}
```
The main reason to do this step separately is to save precious GPU memory that will be useful to generate better quality images. Choose W and H parameters as high as your GPU can handle with respect to proportion of original video resolution. Do not worry if it higher or less then the processing resolution, flow maps will be scaled accordingly at the processing stage. This will generate quite a large file that may take up to a several gigabytes on the drive even for minute long video. If you want to process a long video consider splitting it into several parts beforehand. 

### Step 2.
Run web-ui with '--api' flag. It is also better to use '--xformers' flag, as you would need to have the highest resolution possible and using xformers memory optimization will greatly help.   
```
bash webui.sh --xformers --api
```

### Step 3.
Go to the **vid2vid.py** file and change main parameters (INPUT_VIDEO, FLOW_MAPS, OUTPUT_VIDEO, PROMPT, N_PROMPT, W, H) to the ones you need for your project. FLOW_MAPS parameter should contain a path to the flow file that you generated at the first step. The script is pretty simple so you may change other parameters as well, although I would recommend to leave them as is for the first time. Finally run the script with the command:
```
python3 vid2vid.py
```

## Last version changes: v0.4
* Fixed issue with extreme blur accumulating at the static parts of the video.
* The order of processing was changed to achieve the best quality at different domains.
* Optical flow computation isolated into a separate script for better GPU memory management. Check out the instruction for a new processing pipeline.

## Potential improvements
There are several ways overall quality of animation may be improved:
* You may use a separate processing for each camera position to get a more consistent style of the characters and less ghosting.
* Because the quality of the video depends on how good optical flow was estimated it might be beneficial to use high frame rate video as a source, so it would be easier to guess the flow properly.
* The quality of flow estimation might be greatly improved with proper flow estimation model like this one: https://github.com/autonomousvision/unimatch .

## Licence
This repository can only be used for personal/research/non-commercial purposes. However, for commercial requests, please contact me directly at borsky.alexey@gmail.com
