import sys, os
basedirs = [os.getcwd()]

for basedir in basedirs:
    paths_to_ensure = [
        basedir,
        basedir + '/extensions/sd-cn-animation/scripts',
        basedir + '/extensions/SD-CN-Animation/scripts'
        ]

    for scripts_path_fix in paths_to_ensure:
        if not scripts_path_fix in sys.path:
            sys.path.extend([scripts_path_fix])

import math
import os
import sys
import traceback

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops

from modules import devices, sd_samplers, img2img
from modules import shared, sd_hijack, lowvram
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, devices
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
import modules.images as images
import modules.scripts

import gc
import cv2
import gradio as gr

import time
import skimage
import datetime

from flow_utils import RAFT_estimate_flow, RAFT_clear_memory, compute_diff_map
from types import SimpleNamespace   

class sdcn_anim_tmp:
    prepear_counter = 0
    process_counter = 0
    input_video = None
    output_video = None
    curr_frame = None
    prev_frame = None
    prev_frame_styled = None
    fps = None
    total_frames = None
    prepared_frames = None
    prepared_next_flows = None
    prepared_prev_flows = None
    frames_prepared = False

def read_frame_from_video():
    # Reading video file
    if sdcn_anim_tmp.input_video.isOpened():
        ret, cur_frame = sdcn_anim_tmp.input_video.read()
        if cur_frame is not None: 
            cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB) 
    else:
        cur_frame = None
        sdcn_anim_tmp.input_video.release()
    
    return cur_frame

def get_cur_stat():
    stat =  f'Frames prepared: {sdcn_anim_tmp.prepear_counter + 1} / {sdcn_anim_tmp.total_frames}; '
    stat += f'Frames processed: {sdcn_anim_tmp.process_counter + 1} / {sdcn_anim_tmp.total_frames}; '
    return stat

def clear_memory_from_sd():
    if shared.sd_model is not None:
        sd_hijack.model_hijack.undo_hijack(shared.sd_model)
        try:
            lowvram.send_everything_to_cpu()
        except Exception as e:
            ...
        del shared.sd_model
        shared.sd_model = None
    gc.collect()
    devices.torch_gc()

def get_device():
    device=devices.get_optimal_device()
    #print('device',device)
    return device

def args_to_dict(*args): # converts list of argumets into dictionary for better handling of it
    args_list = ['id_task', 'mode', 'prompt', 'negative_prompt', 'prompt_styles', 'init_video', 'sketch', 'init_img_with_mask', 'inpaint_color_sketch', 'inpaint_color_sketch_orig', 'init_img_inpaint', 'init_mask_inpaint', 'steps', 'sampler_index', 'mask_blur', 'mask_alpha', 'inpainting_fill', 'restore_faces', 'tiling', 'n_iter', 'batch_size', 'cfg_scale', 'image_cfg_scale', 'denoising_strength', 'seed', 'subseed', 'subseed_strength', 'seed_resize_from_h', 'seed_resize_from_w', 'seed_enable_extras', 'height', 'width', 'resize_mode', 'inpaint_full_res', 'inpaint_full_res_padding', 'inpainting_mask_invert', 'img2img_batch_input_dir', 'img2img_batch_output_dir', 'img2img_batch_inpaint_mask_dir', 'override_settings_texts']

    # set default values for params that were not specified
    args_dict = {
        'mode': 0,
        'prompt': '',
        'negative_prompt': '',
        'prompt_styles': [],
        'init_video': None, # Always required

        'steps': 15,
        'sampler_index': 0, # 'Euler a'    
        'mask_blur': 0,

        'inpainting_fill': 1, # original
        'restore_faces': False,
        'tiling': False,
        'n_iter': 1,
        'batch_size': 1,
        'cfg_scale': 5.5,
        'image_cfg_scale': 1.5,
        'denoising_strength': 0.75,
        'seed': -1,
        'subseed': -1,
        'subseed_strength': 0,
        'seed_resize_from_h': 512,
        'seed_resize_from_w': 512,
        'seed_enable_extras': False,
        'height': 512,
        'width': 512,
        'resize_mode': 1,
        'inpaint_full_res': True,
        'inpaint_full_res_padding': 0,
    }

    args = list(args)

    for i in range(len(args_list)):
        if (args[i] is None) and (args_list[i] in args_dict):
            args[i] = args_dict[args_list[i]] 
        else:
            args_dict[args_list[i]] = args[i]

    args_dict['script_inputs'] = args[len(args_list):]
    return args_dict, args

def start_process(*args):
    args_dict, args_list = args_to_dict(*args) 

    sdcn_anim_tmp.process_counter = 0
    sdcn_anim_tmp.prepear_counter = 0

    # Open the input video file
    sdcn_anim_tmp.input_video = cv2.VideoCapture(args_dict['init_video'].name)
    
    # Get useful info from the source video
    sdcn_anim_tmp.fps = int(sdcn_anim_tmp.input_video.get(cv2.CAP_PROP_FPS))
    sdcn_anim_tmp.total_frames = int(sdcn_anim_tmp.input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create an output video file with the same fps, width, and height as the input video
    output_video_name = f'outputs/sd-cn-animation/vid2vid/{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.mp4'
    os.makedirs(os.path.dirname(output_video_name), exist_ok=True)
    sdcn_anim_tmp.output_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'mp4v'), sdcn_anim_tmp.fps, (args_dict['width'], args_dict['height']))

    curr_frame = read_frame_from_video()
    curr_frame = cv2.resize(curr_frame, (args_dict['width'], args_dict['height']))
    sdcn_anim_tmp.prepared_frames = np.zeros((11, args_dict['height'], args_dict['width'], 3), dtype=np.uint8)
    sdcn_anim_tmp.prepared_next_flows = np.zeros((10, args_dict['height'], args_dict['width'], 2))
    sdcn_anim_tmp.prepared_prev_flows = np.zeros((10, args_dict['height'], args_dict['width'], 2))
    sdcn_anim_tmp.prepared_frames[0] = curr_frame

    #args_dict['init_img'] = cur_frame
    args_list[5] = Image.fromarray(curr_frame)
    processed_frames, _, _, _ = modules.img2img.img2img(*args_list) #img2img(args_dict)
    processed_frame = np.array(processed_frames[0])
    processed_frame = skimage.exposure.match_histograms(processed_frame, curr_frame, multichannel=False, channel_axis=-1)
    processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)
    #print('Processed frame ', 0)
    

    sdcn_anim_tmp.curr_frame = curr_frame
    sdcn_anim_tmp.prev_frame = curr_frame.copy()
    sdcn_anim_tmp.prev_frame_styled = processed_frame.copy()
    yield get_cur_stat(), sdcn_anim_tmp.curr_frame, None, None, processed_frame, ''

    # TODO: SOLVE PROBLEM with wrong prev frame on the start on new processing iterations

    for step in range((sdcn_anim_tmp.total_frames-1) * 2):
        args_dict, args_list = args_to_dict(*args) 

        occlusion_mask = None
        prev_frame = None
        curr_frame = sdcn_anim_tmp.curr_frame
        warped_styled_frame = gr.Image.update()
        processed_frame = gr.Image.update()

        prepare_steps = 10
        if sdcn_anim_tmp.process_counter % prepare_steps == 0 and not sdcn_anim_tmp.frames_prepared: # prepare next 10 frames for processing
            #clear_memory_from_sd()
            device = get_device()

            curr_frame = read_frame_from_video()
            if curr_frame is not None: 
                curr_frame = cv2.resize(curr_frame, (args_dict['width'], args_dict['height']))
                prev_frame = sdcn_anim_tmp.prev_frame.copy()

                next_flow, prev_flow, occlusion_mask, frame1_bg_removed, frame2_bg_removed = RAFT_estimate_flow(prev_frame, curr_frame, subtract_background=False, device=device)
                occlusion_mask = np.clip(occlusion_mask * 0.1 * 255, 0, 255).astype(np.uint8)

                cn = sdcn_anim_tmp.prepear_counter % 10
                if sdcn_anim_tmp.prepear_counter % 10 == 0:
                    sdcn_anim_tmp.prepared_frames[cn] = sdcn_anim_tmp.prev_frame
                sdcn_anim_tmp.prepared_frames[cn + 1] = curr_frame.copy()
                sdcn_anim_tmp.prepared_next_flows[cn] = next_flow.copy()
                sdcn_anim_tmp.prepared_prev_flows[cn] = prev_flow.copy()
                #print('Prepared frame ', cn+1)

                sdcn_anim_tmp.prev_frame = curr_frame.copy()

            sdcn_anim_tmp.prepear_counter += 1
            if sdcn_anim_tmp.prepear_counter % prepare_steps == 0 or \
               sdcn_anim_tmp.prepear_counter >= sdcn_anim_tmp.total_frames - 1 or \
               curr_frame is None:
                # Remove RAFT from memory
                RAFT_clear_memory()
                sdcn_anim_tmp.frames_prepared = True
        else:
            # process frame
            sdcn_anim_tmp.frames_prepared = False

            cn = sdcn_anim_tmp.process_counter % 10 
            curr_frame = sdcn_anim_tmp.prepared_frames[cn+1]
            prev_frame = sdcn_anim_tmp.prepared_frames[cn]
            next_flow = sdcn_anim_tmp.prepared_next_flows[cn]
            prev_flow = sdcn_anim_tmp.prepared_prev_flows[cn]

            # process current frame
            args_list[5] = Image.fromarray(curr_frame)
            args_list[24] = -1
            processed_frames, _, _, _ = modules.img2img.img2img(*args_list)
            processed_frame = np.array(processed_frames[0])


            alpha_mask, warped_styled_frame = compute_diff_map(next_flow, prev_flow, prev_frame, curr_frame, sdcn_anim_tmp.prev_frame_styled)
            alpha_mask = np.clip(alpha_mask + 0.05, 0.05, 0.95)

            fl_w, fl_h = prev_flow.shape[:2]
            prev_flow_n = prev_flow / np.array([fl_h,fl_w])
            flow_mask = np.clip(1 - np.linalg.norm(prev_flow_n, axis=-1)[...,None], 0, 1)

            # fix warped styled frame from duplicated that occures on the places where flow is zero, but only because there is no place to get the color from
            warped_styled_frame = curr_frame.astype(float) * alpha_mask * flow_mask + warped_styled_frame.astype(float) * (1 - alpha_mask * flow_mask)
            
            # This clipping at lower side required to fix small trailing issues that for some reason left outside of the bright part of the mask, 
            # and at the higher part it making parts changed strongly to do it with less flickering. 
            
            occlusion_mask = np.clip(alpha_mask * 255, 0, 255).astype(np.uint8)

            # normalizing the colors
            processed_frame = skimage.exposure.match_histograms(processed_frame, curr_frame, multichannel=False, channel_axis=-1)
            processed_frame = processed_frame.astype(float) * alpha_mask + warped_styled_frame.astype(float) * (1 - alpha_mask)
            
            processed_frame = processed_frame * 0.9 + curr_frame * 0.1
            processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)
            sdcn_anim_tmp.prev_frame_styled = processed_frame.copy()

            args_list[5] = Image.fromarray(processed_frame)
            args_list[23] = 0.15
            args_list[24] = 8888
            processed_frames, _, _, _ = modules.img2img.img2img(*args_list)
            processed_frame = np.array(processed_frames[0])

            processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)
            warped_styled_frame = np.clip(warped_styled_frame, 0, 255).astype(np.uint8)
            

            # Write the frame to the output video
            frame_out = np.clip(processed_frame, 0, 255).astype(np.uint8)
            frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR) 
            sdcn_anim_tmp.output_video.write(frame_out)

            sdcn_anim_tmp.process_counter += 1
            if sdcn_anim_tmp.process_counter >= sdcn_anim_tmp.total_frames - 1:
                sdcn_anim_tmp.input_video.release()
                sdcn_anim_tmp.output_video.release()
                sdcn_anim_tmp.prev_frame = None

        #print(f'\nEND OF STEP {step}, {sdcn_anim_tmp.prepear_counter}, {sdcn_anim_tmp.process_counter}')
        yield get_cur_stat(), curr_frame, occlusion_mask, warped_styled_frame, processed_frame, ''

    #sdcn_anim_tmp.input_video.release()
    #sdcn_anim_tmp.output_video.release()

    return get_cur_stat(), curr_frame, occlusion_mask, warped_styled_frame, processed_frame, ''

'''
# id_task: str, mode: int, prompt: str, negative_prompt: str, prompt_styles: list, init_img, sketch, init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, init_img_inpaint, init_mask_inpaint, steps: int, sampler_index: int, mask_blur: int, mask_alpha: float, inpainting_fill: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, image_cfg_scale: float, denoising_strength: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, img2img_batch_output_dir: str, img2img_batch_inpaint_mask_dir: str, override_settings_texts, *args
def img2img(args_dict):  
    args = SimpleNamespace(**args_dict)
    print('override_settings:', args.override_settings_texts)
    override_settings = create_override_settings_dict(args.override_settings_texts)

    is_batch = args.mode == 5

    if args.mode == 0:  # img2img
        image = args.init_img.convert("RGB")
        mask = None
    elif args.mode == 1:  # img2img sketch
        image = args.sketch.convert("RGB")
        mask = None
    elif args.mode == 2:  # inpaint
        image, mask = args.init_img_with_mask["image"], args.init_img_with_mask["mask"]
        alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
        mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
        image = image.convert("RGB")
    elif args.mode == 3:  # inpaint sketch
        image = args.inpaint_color_sketch
        orig = args.inpaint_color_sketch_orig or args.inpaint_color_sketch
        pred = np.any(np.array(image) != np.array(orig), axis=-1)
        mask = Image.fromarray(pred.astype(np.uint8) * 255, "L")
        mask = ImageEnhance.Brightness(mask).enhance(1 - args.mask_alpha / 100)
        blur = ImageFilter.GaussianBlur(args.mask_blur)
        image = Image.composite(image.filter(blur), orig, mask.filter(blur))
        image = image.convert("RGB")
    elif args.mode == 4:  # inpaint upload mask
        image = args.init_img_inpaint
        mask = args.init_mask_inpaint
    else:
        image = None
        mask = None

    # Use the EXIF orientation of photos taken by smartphones.
    if image is not None:
        image = ImageOps.exif_transpose(image)

    assert 0. <= args.denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        styles=args.prompt_styles,
        seed=args.seed,
        subseed=args.subseed,
        subseed_strength=args.subseed_strength,
        seed_resize_from_h=args.seed_resize_from_h,
        seed_resize_from_w=args.seed_resize_from_w,
        seed_enable_extras=args.seed_enable_extras,
        sampler_name=sd_samplers.samplers_for_img2img[args.sampler_index].name,
        batch_size=args.batch_size,
        n_iter=args.n_iter,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        width=args.width,
        height=args.height,
        restore_faces=args.restore_faces,
        tiling=args.tiling,
        init_images=[image],
        mask=mask,
        mask_blur=args.mask_blur,
        inpainting_fill=args.inpainting_fill,
        resize_mode=args.resize_mode,
        denoising_strength=args.denoising_strength,
        image_cfg_scale=args.image_cfg_scale,
        inpaint_full_res=args.inpaint_full_res,
        inpaint_full_res_padding=args.inpaint_full_res_padding,
        inpainting_mask_invert=args.inpainting_mask_invert,
        override_settings=override_settings,
    )

    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args.script_inputs

    print('script_inputs 1:', args.script_inputs[1].__dict__)
    #print('script_inputs 2:', args.script_inputs[1])

    if shared.cmd_opts.enable_console_prompts:
        print(f"\nimg2img: {args.prompt}", file=shared.progress_print_out)

    if mask:
        p.extra_generation_params["Mask blur"] = args.mask_blur

    if is_batch:
        ...
    #    assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"
    #    process_batch(p, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir, args.script_inputs)
    #    processed = Processed(p, [], p.seed, "")
    else:
        processed = modules.scripts.scripts_img2img.run(p, *args.script_inputs)
        if processed is None:
            processed = process_images(p)
    
    p.close()

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images[0] #, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments)'''