# Reference based SD+CN Animation Script
This script allows to automate video stylization task using StableDiffusion and ControlNet. There is also reference image trick implemented to make animation more stable. Here is an example of a video made with this script:

[![IMAGE_ALT](https://img.youtube.com/vi/YW1JBJ57YBQ/0.jpg)](https://youtu.be/YW1JBJ57YBQ)

Before running the script you need to create reference image that has the same resolution as the target video. You may generate it using Stable Diffusion by stylizing any frame of the video or via any other means. It is better to process different scenes separately to achieve better quality. For the video above the following reference image was used:
![Reference image!](/init.png "Reference image")

## Dependencies
To install all necessary dependencies run this command
```
pip install opencv-python, numpy, tqdm
```

## Running the script
This script works on top of [Automatic1111/web-ui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) interface via API. To run this script you have to set it up first. You also should have [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) extension installed. You need to have t2iadapter_color_sd14v1 and control_hed-fp16 models installed. If you have web-ui with ControlNet working correctly do the following:
1. Go to the web-ui settings -> ControlNet tab -> Set "Allow other script to control this extension" checkbox to active and set "Multi ControlNet: Max models amount (requires restart)" to more then 2 -> press "Apply settings"
2. Run web-ui with '--api' flag. It also better to use '--xformers' flag, as you would need to have the highest resolution possible and using xformers memory optimization will greatly help.   
```bash webui.sh --xformers --api```
3. Go to the script.py file and change main parameters (INPUT_VIDEO, OUTPUT_VIDEO, REF_IMAGE, PROMPT, N_PROMPT, SEED, W, H) to the ones you need for your project. The script is pretty simple so you may change other parameters as well, although I would recommend to leave them as is for the first time.
4. Run the script with ```python3 script.py```

## Potential improvements
There are several ways overall quality of animation may be improved:
* You may use separate reference frame for each camera position to get more consistent style of the characters and less flickering.
* Face animations may be substantially improved if one would use face-detection to grab them from the frame, align them and then process them separately on a higher resolution. And then you could place processed faces back to its place. It may improve lips movements and face coherency in general.
* You may try to use previous processed frame as a reference frame for next image. It could improve short temporal consistency but may lead to a style drift in a long video.
