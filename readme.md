# SD+CN Animation Script
This script allows to automate video stylization task using StableDiffusion and ControlNet. It uses simple optical flow estimation algorithm to keep the animation stable and create inpating mask that used to generate the next frame. Here is an example of a video made with this script:

[![IMAGE_ALT](https://img.youtube.com/vi/YW1JBJ57YBQ/0.jpg)](https://youtu.be/YW1JBJ57YBQ)

## Dependencies
To install all necessary dependencies run this command
```
pip install opencv-python opencv-contrib-python numpy tqdm
```

## Running the script
This script works on top of [Automatic1111/web-ui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) interface via API. To run this script you have to set it up first. You also should have [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) extension installed. You need to have t2iadapter_color_sd14v1 and control_hed-fp16 models installed. If you have web-ui with ControlNet working correctly do the following:
1. Go to the web-ui settings -> ControlNet tab -> Set "Allow other script to control this extension" checkbox to active and set "Multi ControlNet: Max models amount (requires restart)" to more then 2 -> press "Apply settings"
2. Run web-ui with '--api' flag. It also better to use '--xformers' flag, as you would need to have the highest resolution possible and using xformers memory optimization will greatly help.   
```bash webui.sh --xformers --api```
3. Go to the script.py file and change main parameters (INPUT_VIDEO, OUTPUT_VIDEO, PROMPT, N_PROMPT, W, H) to the ones you need for your project. The script is pretty simple so you may change other parameters as well, although I would recommend to leave them as is for the first time.
4. Run the script with ```python3 script.py```

## Potential improvements
There are several ways overall quality of animation may be improved:
* You may use a separate reference frame for each camera position to get a more consistent style of the characters and less flickering.
* Because the quality of the video depends on how good optical flow was estimated it might be beneficial to use high frame rate video as a source, so it would be easier to guess the flow properly.
* The quality of flow estimation might be greatly improved with proper flow estimation model like this one: https://github.com/autonomousvision/unimatch
