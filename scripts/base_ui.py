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

import gradio as gr
from types import SimpleNamespace

from modules import script_callbacks, shared
from modules.shared import cmd_opts, opts
from webui import wrap_gradio_gpu_call

from modules.ui_components import ToolButton, FormRow, FormGroup
from modules.ui import create_override_settings_dropdown
import modules.scripts as scripts

from modules.sd_samplers import samplers_for_img2img
from modules.ui import setup_progressbar, create_sampler_and_steps_selection, ordered_ui_categories, create_output_panel


from vid2vid import *

def V2VArgs():
    seed = -1
    width = 1024
    height = 576
    cfg_scale = 5.5
    steps = 15
    prompt = ""
    n_prompt = "text, letters, logo, brand, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    processing_strength = 0.85
    fix_frame_strength = 0.15
    return locals()

def T2VArgs():
    seed = -1
    width = 768
    height = 512
    cfg_scale = 5.5
    steps = 15
    prompt = ""
    n_prompt = "text, letters, logo, brand, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    processing_strength = 0.85
    fix_frame_strength = 0.35
    return locals()

def setup_common_values(mode, d):
    with gr.Row():
        width = gr.Slider(label='Width', minimum=64, maximum=2048, step=64, value=d.width, interactive=True)
        height = gr.Slider(label='Height', minimum=64, maximum=2048, step=64, value=d.height, interactive=True)
    with gr.Row(elem_id=f'{mode}_prompt_toprow'):
        prompt = gr.Textbox(label='Prompt', lines=3, interactive=True, elem_id=f"{mode}_prompt", placeholder="Enter your prompt here...")
    with gr.Row(elem_id=f'{mode}_n_prompt_toprow'):
        n_prompt = gr.Textbox(label='Negative prompt', lines=3, interactive=True, elem_id=f"{mode}_n_prompt", value=d.n_prompt)
    with gr.Row():
        #steps = gr.Slider(label='Steps', minimum=1, maximum=100, step=1, value=d.steps, interactive=True)
        cfg_scale = gr.Slider(label='CFG scale', minimum=1, maximum=100, step=1, value=d.cfg_scale, interactive=True)
    with gr.Row():
        seed = gr.Number(label='Seed (this parameter controls how the first frame looks like and the color distribution of the consecutive frames as they are dependent on the first one)', value = d.seed, Interactive = True, precision=0)
    with gr.Row():
        processing_strength = gr.Slider(label="Processing strength", value=d.processing_strength, minimum=0, maximum=1, step=0.05, interactive=True)
        fix_frame_strength = gr.Slider(label="Fix frame strength", value=d.fix_frame_strength, minimum=0, maximum=1, step=0.05, interactive=True)

    return width, height, prompt, n_prompt, cfg_scale, seed, processing_strength, fix_frame_strength

def inputs_ui():
    v2v_args = SimpleNamespace(**V2VArgs())
    t2v_args = SimpleNamespace(**T2VArgs())
    with gr.Tab('vid2vid') as tab_vid2vid:
        with gr.Row():
            gr.HTML('Put your video here')
        with gr.Row():
            vid2vid_file = gr.File(label="Input video", interactive=True, file_count="single", file_types=["video"], elem_id="vid_to_vid_chosen_file")
            #init_img = gr.Image(label="Image for img2img", elem_id="img2img_image", show_label=False, source="upload", interactive=True, type="pil", image_mode="RGBA")
        #with gr.Row():
        #    gr.HTML('Alternative: enter the relative (to the webui) path to the file')
        #with gr.Row():
        #    vid2vid_frames_path = gr.Textbox(label="Input video path", interactive=True, elem_id="vid_to_vid_chosen_path", placeholder='Enter your video path here, or upload in the box above ^')

        width, height, prompt, n_prompt, cfg_scale, seed, processing_strength, fix_frame_strength = setup_common_values('vid2vid', v2v_args)
        #with gr.Row():
        #    strength = gr.Slider(label="denoising strength", value=d.strength, minimum=0, maximum=1, step=0.05, interactive=True)
        #    vid2vid_startFrame=gr.Number(label='vid2vid start frame',value=d.vid2vid_startFrame)
        
    with gr.Tab('txt2vid') as tab_txt2vid:
        gr.Markdown('Work in progress...') 
    #    width, height, prompt, n_prompt, steps, cfg_scale, seed, processing_strength, fix_frame_strength = setup_common_values('txt2vid', t2v_args)

    #with gr.Tab('settings') as tab_setts:        
    #    gr.Markdown('Work in progress...')         
         
    return locals()

def on_ui_tabs():
    modules.scripts.scripts_current = modules.scripts.scripts_img2img
    modules.scripts.scripts_img2img.initialize_scripts(is_img2img=True)

    with gr.Blocks(analytics_enabled=False) as sdcnanim_interface:
        components = {}
        
        #dv = SimpleNamespace(**T2VOutputArgs())
        with gr.Row(elem_id='v2v-core').style(equal_height=False, variant='compact'):
            with gr.Column(scale=1, variant='panel'):
                with gr.Row(variant='compact'):
                    run_button = gr.Button('Generate', elem_id=f"sdcn_anim_generate", variant='primary')

                with gr.Tabs():
                    components = inputs_ui()
                    print('components', components['processing_strength'])
                    #do_vid2vid = gr.State(value=0)

                    for category in ordered_ui_categories():
                        if category == "sampler":
                            steps, sampler_index = create_sampler_and_steps_selection(samplers_for_img2img, "vid2vid")

                        elif category == "override_settings":
                            with FormRow(elem_id="vid2vid_override_settings_row") as row:
                                override_settings = create_override_settings_dropdown("vid2vid", row)

                        elif category == "scripts":
                            with FormGroup(elem_id=f"script_container"):
                                custom_inputs = scripts.scripts_img2img.setup_ui()
    
            with gr.Column(scale=1, variant='compact'):
                with gr.Column(variant="panel"):
                    sp_progress = gr.HTML(elem_id="sp_progress", value="")
                    sp_progress.update()
                    #sp_outcome = gr.HTML(elem_id="sp_error", value="")
                    #sp_progressbar = gr.HTML(elem_id="sp_progressbar")
                    #setup_progressbar(sp_progressbar, sp_preview, 'sp', textinfo=sp_progress)
                    
                    with gr.Row(variant='compact'):
                        img_preview_curr_frame = gr.Image(label='Current frame', elem_id=f"img_preview_curr_frame", type='pil').style(height=240)
                        img_preview_curr_occl = gr.Image(label='Current occlusion', elem_id=f"img_preview_curr_occl", type='pil').style(height=240)
                    with gr.Row(variant='compact'):
                        img_preview_prev_warp = gr.Image(label='Previous frame warped', elem_id=f"img_preview_curr_frame", type='pil').style(height=240)
                        img_preview_processed = gr.Image(label='Processed', elem_id=f"img_preview_processed", type='pil').style(height=240)
                    #with gr.Row(variant='compact'):
                    
                    html_log = gr.HTML(elem_id=f'html_log_vid2vid')
                
                with gr.Row(variant='compact'):
                    dummy_component = gr.Label(visible=False)
            

            # Define parameters for the action methods. Not all of them are included yet
            method_inputs = [
                dummy_component,                    # send None for task_id
                dummy_component,                    # mode
                components['prompt'],               # prompt
                components['n_prompt'],             # negative_prompt
                dummy_component,                    # prompt_styles
                components['vid2vid_file'],         # input_video
                dummy_component,                    # sketch
                dummy_component,                    # init_img_with_mask
                dummy_component,                    # inpaint_color_sketch
                dummy_component,                    # inpaint_color_sketch_orig
                dummy_component,                    # init_img_inpaint
                dummy_component,                    # init_mask_inpaint
                steps,                              # steps
                sampler_index,                      # sampler_index
                dummy_component,                    # mask_blur
                dummy_component,                    # mask_alpha
                dummy_component,                    # inpainting_fill
                dummy_component,                    # restore_faces
                dummy_component,                    # tiling
                dummy_component,                    # n_iter
                dummy_component,                    # batch_size
                components['cfg_scale'],            # cfg_scale
                dummy_component,                    # image_cfg_scale
                components['processing_strength'],  # denoising_strength
                components['seed'],                 # seed
                dummy_component,                    # subseed
                dummy_component,                    # subseed_strength
                dummy_component,                    # seed_resize_from_h
                dummy_component,                    # seed_resize_from_w
                dummy_component,                    # seed_enable_extras
                components['height'],               # height
                components['width'],                # width
                dummy_component,                    # resize_mode
                dummy_component,                    # inpaint_full_res
                dummy_component,                    # inpaint_full_res_padding
                dummy_component,                    # inpainting_mask_invert
                dummy_component,                    # img2img_batch_input_dir
                dummy_component,                    # img2img_batch_output_dir
                dummy_component,                    # img2img_batch_inpaint_mask_dir
                override_settings,                  # override_settings_texts
            ] + custom_inputs

            method_outputs = [
                sp_progress,
                img_preview_curr_frame,
                img_preview_curr_occl,
                img_preview_prev_warp,
                img_preview_processed,
                html_log,
            ]

            run_button.click(
                fn=start_process, #wrap_gradio_gpu_call(start_process, extra_outputs=[None, '', '']), 
                inputs=method_inputs,
                outputs=method_outputs,
                show_progress=True,
            )

        modules.scripts.scripts_current = None

        # define queue - required for generators
        sdcnanim_interface.queue(concurrency_count=1)
    return [(sdcnanim_interface, "SD-CN-Animation", "sd_cn_animation_interface")]


script_callbacks.on_ui_tabs(on_ui_tabs)
