class shared:
  is_interrupted = False
  v2v_custom_inputs_size = 0
  t2v_custom_inputs_size = 0

def get_component_names():
  components_list = [
    'glo_sdcn_process_mode',
    'v2v_file', 'v2v_width', 'v2v_height', 'v2v_prompt', 'v2v_n_prompt', 'v2v_cfg_scale', 'v2v_seed', 'v2v_processing_strength', 'v2v_fix_frame_strength', 
    'v2v_sampler_index', 'v2v_steps', 'v2v_override_settings',
    'v2v_occlusion_mask_blur', 'v2v_occlusion_mask_trailing', 'v2v_occlusion_mask_flow_multiplier', 'v2v_occlusion_mask_difo_multiplier', 'v2v_occlusion_mask_difs_multiplier',
    'v2v_step_1_processing_mode', 'v2v_step_1_blend_alpha', 'v2v_step_1_seed', 'v2v_step_2_seed',
    't2v_width', 't2v_height', 't2v_prompt', 't2v_n_prompt', 't2v_cfg_scale', 't2v_seed', 't2v_processing_strength', 't2v_fix_frame_strength',
    't2v_sampler_index', 't2v_steps', 't2v_length', 't2v_fps',
    'glo_save_frames_check'
  ]

  return components_list

def args_to_dict(*args): # converts list of argumets into dictionary for better handling of it
  args_list = get_component_names()

  # set default values for params that were not specified
  args_dict = {
    # video to video params
    'v2v_mode': 0,
    'v2v_prompt': '',
    'v2v_n_prompt': '',
    'v2v_prompt_styles': [],
    'v2v_init_video': None, # Always required

    'v2v_steps': 15,
    'v2v_sampler_index': 0, # 'Euler a'    
    'v2v_mask_blur': 0,

    'v2v_inpainting_fill': 1, # original
    'v2v_restore_faces': False,
    'v2v_tiling': False,
    'v2v_n_iter': 1,
    'v2v_batch_size': 1,
    'v2v_cfg_scale': 5.5,
    'v2v_image_cfg_scale': 1.5,
    'v2v_denoising_strength': 0.75,
    'v2v_fix_frame_strength': 0.15,
    'v2v_seed': -1,
    'v2v_subseed': -1,
    'v2v_subseed_strength': 0,
    'v2v_seed_resize_from_h': 512,
    'v2v_seed_resize_from_w': 512,
    'v2v_seed_enable_extras': False,
    'v2v_height': 512,
    'v2v_width': 512,
    'v2v_resize_mode': 1,
    'v2v_inpaint_full_res': True,
    'v2v_inpaint_full_res_padding': 0,
    'v2v_inpainting_mask_invert': False,

    # text to video params
    't2v_mode': 4,
    't2v_prompt': '',
    't2v_n_prompt': '',
    't2v_prompt_styles': [],
    't2v_init_img': None,
    't2v_mask_img': None,

    't2v_steps': 15,
    't2v_sampler_index': 0, # 'Euler a'    
    't2v_mask_blur': 0,

    't2v_inpainting_fill': 1, # original
    't2v_restore_faces': False,
    't2v_tiling': False,
    't2v_n_iter': 1,
    't2v_batch_size': 1,
    't2v_cfg_scale': 5.5,
    't2v_image_cfg_scale': 1.5,
    't2v_denoising_strength': 0.75,
    't2v_fix_frame_strength': 0.15,
    't2v_seed': -1,
    't2v_subseed': -1,
    't2v_subseed_strength': 0,
    't2v_seed_resize_from_h': 512,
    't2v_seed_resize_from_w': 512,
    't2v_seed_enable_extras': False,
    't2v_height': 512,
    't2v_width': 512,
    't2v_resize_mode': 1,
    't2v_inpaint_full_res': True,
    't2v_inpaint_full_res_padding': 0,
    't2v_inpainting_mask_invert': False,

    't2v_override_settings': [],
    #'t2v_script_inputs': [0],

    't2v_fps': 12,
  }

  args = list(args)

  for i in range(len(args_list)):
    if (args[i] is None) and (args_list[i] in args_dict):
      #args[i] = args_dict[args_list[i]] 
      pass
    else:
      args_dict[args_list[i]] = args[i]

  args_dict['v2v_script_inputs'] = args[len(args_list):len(args_list)+shared.v2v_custom_inputs_size]
  #print('v2v_script_inputs', args_dict['v2v_script_inputs'])
  args_dict['t2v_script_inputs'] = args[len(args_list)+shared.v2v_custom_inputs_size:]
  #print('t2v_script_inputs', args_dict['t2v_script_inputs'])
  return args_dict

def get_mode_args(mode, args_dict):
  mode_args_dict = {}
  for key, value in args_dict.items():
    if key[:3] in [mode, 'glo'] :
      mode_args_dict[key[4:]] = value

  return mode_args_dict

def set_CNs_input_image(args_dict, image):
  for script_input in args_dict['script_inputs']:
    if type(script_input).__name__ == 'UiControlNetUnit':
      script_input.batch_images = [image]

import time
import datetime

def get_time_left(ind, length, processing_start_time):
  s_passed = int(time.time() - processing_start_time)
  time_passed = datetime.timedelta(seconds=s_passed)
  s_left = int(s_passed / ind * (length - ind))
  time_left = datetime.timedelta(seconds=s_left)
  return f"Time elapsed: {time_passed}; Time left: {time_left};"

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops
from types import SimpleNamespace  

from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.processing import Processed, StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img, process_images
import modules.processing as processing
from modules.ui import plaintext_to_html
import modules.images as images
import modules.scripts
from modules.shared import opts, devices, state
from modules import devices, sd_samplers, img2img
from modules import shared, sd_hijack, lowvram

# TODO: Refactor all the code below

def process_img(p, input_img, output_dir, inpaint_mask_dir, args):
    processing.fix_seed(p)

    #images = shared.listfiles(input_dir)
    images = [input_img]

    is_inpaint_batch = False
    #if inpaint_mask_dir:
    #    inpaint_masks = shared.listfiles(inpaint_mask_dir)
    #    is_inpaint_batch = len(inpaint_masks) > 0
    #if is_inpaint_batch:
    #    print(f"\nInpaint batch is enabled. {len(inpaint_masks)} masks found.")

    #print(f"Will process {len(images)} images, creating {p.n_iter * p.batch_size} new images for each.")

    save_normally = output_dir == ''

    p.do_not_save_grid = True
    p.do_not_save_samples = not save_normally

    state.job_count = len(images) * p.n_iter

    generated_images = []
    for i, image in enumerate(images):
        state.job = f"{i+1} out of {len(images)}"
        if state.skipped:
            state.skipped = False

        if state.interrupted:
            break

        img = image #Image.open(image)
        # Use the EXIF orientation of photos taken by smartphones.
        img = ImageOps.exif_transpose(img)
        p.init_images = [img] * p.batch_size

        #if is_inpaint_batch:
        #    # try to find corresponding mask for an image using simple filename matching
        #    mask_image_path = os.path.join(inpaint_mask_dir, os.path.basename(image))
        #    # if not found use first one ("same mask for all images" use-case)
        #    if not mask_image_path in inpaint_masks:
        #        mask_image_path = inpaint_masks[0]
        #    mask_image = Image.open(mask_image_path)
        #    p.image_mask = mask_image

        proc = modules.scripts.scripts_img2img.run(p, *args)
        if proc is None:
            proc = process_images(p)
            generated_images.append(proc.images[0])

        #for n, processed_image in enumerate(proc.images):
        #    filename = os.path.basename(image)

        #    if n > 0:
        #        left, right = os.path.splitext(filename)
        #        filename = f"{left}-{n}{right}"

        #    if not save_normally:
        #        os.makedirs(output_dir, exist_ok=True)
        #        if processed_image.mode == 'RGBA':
        #            processed_image = processed_image.convert("RGB")
        #        processed_image.save(os.path.join(output_dir, filename))

    return generated_images

def img2img(args_dict):  
    args = SimpleNamespace(**args_dict)
    override_settings = create_override_settings_dict(args.override_settings)

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
        #image = args.init_img_inpaint
        #mask = args.init_mask_inpaint

        image = args.init_img.convert("RGB")
        mask = args.mask_img.convert("L")
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
        negative_prompt=args.n_prompt,
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

    p.scripts = modules.scripts.scripts_img2img
    p.script_args = args.script_inputs

    #if shared.cmd_opts.enable_console_prompts:
    #    print(f"\nimg2img: {args.prompt}", file=shared.progress_print_out)

    if mask:
        p.extra_generation_params["Mask blur"] = args.mask_blur
    
    '''
    if is_batch:
        ...
    #    assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"
    #    process_batch(p, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir, args.script_inputs)
    #    processed = Processed(p, [], p.seed, "")
    else:
        processed = modules.scripts.scripts_img2img.run(p, *args.script_inputs)
        if processed is None:
            processed = process_images(p)
    '''

    generated_images = process_img(p, image, None, '', args.script_inputs)
    processed = Processed(p, [], p.seed, "")
    p.close()

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    #if opts.samples_log_stdout:
    #    print(generation_info_js)

    #if opts.do_not_show_images:
    #    processed.images = []

    return generated_images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments)

def txt2img(args_dict):  
    args = SimpleNamespace(**args_dict)
    override_settings = create_override_settings_dict(args.override_settings)

    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=args.prompt,
        styles=args.prompt_styles,
        negative_prompt=args.n_prompt,
        seed=args.seed,
        subseed=args.subseed,
        subseed_strength=args.subseed_strength,
        seed_resize_from_h=args.seed_resize_from_h,
        seed_resize_from_w=args.seed_resize_from_w,
        seed_enable_extras=args.seed_enable_extras,
        sampler_name=sd_samplers.samplers[args.sampler_index].name,
        batch_size=args.batch_size,
        n_iter=args.n_iter,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        width=args.width,
        height=args.height,
        restore_faces=args.restore_faces,
        tiling=args.tiling,
        #enable_hr=args.enable_hr,
        #denoising_strength=args.denoising_strength if enable_hr else None,
        #hr_scale=hr_scale,
        #hr_upscaler=hr_upscaler,
        #hr_second_pass_steps=hr_second_pass_steps,
        #hr_resize_x=hr_resize_x,
        #hr_resize_y=hr_resize_y,
        override_settings=override_settings,
    )

    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args.script_inputs

    #if cmd_opts.enable_console_prompts:
    #    print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    processed = modules.scripts.scripts_txt2img.run(p, *args.script_inputs)

    if processed is None:
        processed = process_images(p)

    p.close()

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    #if opts.samples_log_stdout:
    #    print(generation_info_js)

    #if opts.do_not_show_images:
    #    processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments)
