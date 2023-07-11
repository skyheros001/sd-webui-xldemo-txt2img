import gradio as gr

from modules.shared import opts
from modules.ui_components import ToolButton

from xldemo_txt2img import XLDEMO_HUGGINGFACE_ACCESS_TOKEN, XLDEMO_LOAD_REFINER_ON_STARTUP, xldemo_txt2img_1model, xldemo_txt2img_2model, xldemo_txt2img_zdy
from xldemo_txt2img import do_xldemo_txt2img_infer, do_xldemo_txt2img_refine
from xldemo_txt2img_ui_common import create_seed_inputs, create_output_panel, connect_reuse_seed, gr_show


import modules.shared as shared
import modules.styles
from modules.ui_common import create_refresh_button

switch_values_symbol = '\U000021C5'  # ⇅

apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001f5d1\ufe0f'  # 🗑️

def apply_styles(prompt, prompt_neg, styles):
    prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, styles)
    prompt_neg = shared.prompt_styles.apply_negative_styles_to_prompt(prompt_neg, styles)

    return [gr.Textbox.update(value=prompt), gr.Textbox.update(value=prompt_neg), gr.Dropdown.update(value=[])]
    
    
def make_ui():
    id_part = 'xldemo_txt2img'

    if ((XLDEMO_HUGGINGFACE_ACCESS_TOKEN is None or XLDEMO_HUGGINGFACE_ACCESS_TOKEN.strip() == '') and xldemo_txt2img_zdy == False) or (xldemo_txt2img_zdy == True and (xldemo_txt2img_1model == '' or xldemo_txt2img_2model =='')):
        with gr.Blocks(analytics_enabled=False) as ui_component:
            gr.HTML(value="""<div style='font-size: 1.4em; margin-bottom: 0.7em'><ul>
            <li>*** 强制GPU运行 ***</li>
            <li>1) 如果你下载好了本地模型，请在设置中为XL设置好本地路径即可。设置好后，点击扩展中的“应用更改并重载前端”即可，然后以下的操作你都可以省略。</li>
            <li>2) 如果你还没有下载模型，可以向QQ群713069905索要模型下载的种子文件。</li>
            <li>***</li>
            <li>***以下的第二种方法（不推荐）是让你获取XL模型的授权并远程下载***</li>
            <li>1)登陆你的抱脸网址</li>
            <li>2) 获取模型授权 <b><a href='https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9/tree/main'>点击这里</a></b></li>
            <li>3) 创建访问TOKEN <b><a href='https://huggingface.co/settings/tokens'>点击这里</a></b></li>
            <li>4) 在设置里面设置抱脸网的TOKEN.</li>
            <li>5) 确认重启</li>
            <li>6) 或者在设置里面指定你的模型路径</li>
            </ul></div>""")
            return ui_component

    else:
        with gr.Blocks(analytics_enabled=False) as ui_component:

            with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
                with gr.Column(elem_id=f"{id_part}_prompt_container", scale=6):
                    with gr.Row():
                        with gr.Column(scale=80):
                            with gr.Row():
                                xldemo_txt2img_prompt = gr.Textbox(label="词汇", elem_id=f"{id_part}_prompt", show_label=False, lines=3,
                                                                   placeholder="词汇 (按 Ctrl+Enter 或 Alt+Enter 生成图像)", elem_classes=["prompt"])
                                xldemo_txt2img_dummy_component = gr.Label(
                                    visible=False)

                    with gr.Row():
                        with gr.Column(scale=80):
                            with gr.Row():
                                xldemo_txt2img_negative_prompt = gr.Textbox(label="负面词汇", elem_id=f"{id_part}_neg_prompt", show_label=False,
                                                                            lines=3, placeholder="负面词汇 (按 Ctrl+Enter 或 Alt+Enter 生成图像)", elem_classes=["prompt"])

                with gr.Column(scale=1, elem_id=f"{id_part}_actions_column"):
                    with gr.Row(elem_id=f"{id_part}_generate_box", elem_classes="generate-box"):
                        xldemo_txt2img_submit = gr.Button(
                            '生成图像', elem_id=f"{id_part}_generate", variant='primary')
                        
                    with gr.Row(elem_id=f"{id_part}_refine_box", elem_classes="refine-box"):
                        xldemo_txt2img_refine = gr.Button(                            
                            '精炼图像', interactive=False, elem_id=f"{id_part}_refine", variant='primary')
                    with gr.Row(elem_id=f"{id_part}_tools"):
                        clear_prompt_button = ToolButton(value=clear_prompt_symbol, elem_id=f"{id_part}_clear_prompt")
                        prompt_style_apply = ToolButton(value=apply_style_symbol, elem_id=f"{id_part}_style_apply")
                        
                        clear_prompt_button.click(
                            fn=lambda *x: x,
                            _js="confirm_clear_prompt",
                            inputs=[xldemo_txt2img_prompt, xldemo_txt2img_negative_prompt],
                            outputs=[xldemo_txt2img_prompt, xldemo_txt2img_negative_prompt],
                        )
                    with gr.Row(elem_id=f"{id_part}_styles_row"):
                        prompt_styles = gr.Dropdown(label="Styles", elem_id=f"{id_part}_styles", choices=[k for k, v in shared.prompt_styles.styles.items()], value=[], multiselect=True)
                        create_refresh_button(prompt_styles, shared.prompt_styles.reload, lambda: {"choices": [k for k, v in shared.prompt_styles.styles.items()]}, f"refresh_{id_part}_styles")
                
            with gr.Row():
                with gr.Column():
                    xldemo_txt2img_steps = gr.Slider(minimum=1, maximum=150, step=1,
                                                     elem_id=f"{id_part}_steps", label="采样步数", value=20)

                    with gr.Row():
                        with gr.Column(elem_id="xldemo_txt2img_column_size", scale=4):
                            xldemo_txt2img_width = gr.Slider(
                                minimum=64, maximum=2048, step=8, label="宽", value=1024, elem_id="xldemo_txt2img_width")
                            xldemo_txt2img_height = gr.Slider(
                                minimum=64, maximum=2048, step=8, label="高", value=1024, elem_id="xldemo_txt2img_height")

                        with gr.Column(elem_id="xldemo_txt2img_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                            xldemo_txt2img_res_switch_btn = ToolButton(
                                value=switch_values_symbol, elem_id="xldemo_txt2img_res_switch_btn", label="切换维度")

                        with gr.Column(elem_id="txt2img_column_batch"):
                            xldemo_txt2img_batch_size = gr.Slider(
                                minimum=1, maximum=8, step=1, label='样本数量', value=1, elem_id=f"{id_part}_batch_size")

                    xldemo_txt2img_cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5,
                                                         label='CFG Scale', value=7.0, elem_id=f"{id_part}_cfg_scale")

                    xldemo_txt2img_seed, xldemo_txt2img_reuse_seed = create_seed_inputs(id_part)

                    xldemo_txt2img_enable_Autorefiner = gr.Checkbox(
                            visible=True,
                            label='文生图后自动精炼', value=True, elem_id=f"{id_part}_enable_Autorefiner")
                    xldemo_txt2img_enable_refiner = gr.Checkbox(
                            visible=XLDEMO_LOAD_REFINER_ON_STARTUP,
                            label='精炼', value=False, elem_id=f"{id_part}_enable_refiner")
                            
                    with gr.Row(visible=False, elem_id="xldemo_txt2img_refiner_group", variant="compact") as xldemo_txt2img_refiner_group:
                        with gr.Column():
                            xldemo_txt2img_image_to_refine = gr.Image(label="图片", type='pil')

                        with gr.Column():
                            xldemo_txt2img_refiner_steps = gr.Slider(minimum=1, maximum=150, step=1,
                                                            elem_id=f"{id_part}_refiner_steps", label="精炼步数", value=20)

                            xldemo_txt2img_refiner_strength = gr.Slider(
                                interactive=XLDEMO_LOAD_REFINER_ON_STARTUP,
                                label="精炼强度", minimum=0, maximum=1.0, value=0.3, step=0.1, elem_id=f"{id_part}_refiner_strength")

                with gr.Column():
                    xldemo_txt2img_gallery, xldemo_txt2img_generation_info, xldemo_txt2img_html_info, xldemo_txt2img_html_log = create_output_panel(
                        id_part, opts.outdir_txt2img_samples)

                connect_reuse_seed(xldemo_txt2img_seed, xldemo_txt2img_reuse_seed,
                                   xldemo_txt2img_generation_info, xldemo_txt2img_dummy_component, is_subseed=False)

                xldemo_txt2img_res_switch_btn.click(fn=None, _js="function(){switchWidthHeight('xldemo_txt2img')}", inputs=None, outputs=None, show_progress=False)

                xldemo_txt2img_enable_refiner.change(
                    fn=lambda x: gr_show(x),
                    inputs=[xldemo_txt2img_enable_refiner],
                    outputs=[xldemo_txt2img_refiner_group],
                    show_progress = False,
                )

                xldemo_txt2img_enable_refiner.change(
                    fn=lambda x: gr.update(interactive=x),
                    inputs=[xldemo_txt2img_enable_refiner],
                    outputs=[xldemo_txt2img_refine],
                    show_progress = False,
                )
            prompt_style_apply.click(
                    fn=apply_styles,
                    _js="",
                    inputs=[xldemo_txt2img_prompt, xldemo_txt2img_negative_prompt, prompt_styles],
                    outputs=[xldemo_txt2img_prompt, xldemo_txt2img_negative_prompt, prompt_styles],
            )
            
            xldemo_txt2img_submit.click(fn=do_xldemo_txt2img_infer, inputs=[
                xldemo_txt2img_prompt,
                xldemo_txt2img_negative_prompt,
                xldemo_txt2img_width,
                xldemo_txt2img_height,
                xldemo_txt2img_cfg_scale,
                xldemo_txt2img_seed,
                xldemo_txt2img_batch_size,
                xldemo_txt2img_steps,
                xldemo_txt2img_enable_Autorefiner,
                xldemo_txt2img_refiner_steps,
                xldemo_txt2img_refiner_strength
            ], outputs=[
                xldemo_txt2img_gallery,
                xldemo_txt2img_generation_info,
                xldemo_txt2img_html_info,
                xldemo_txt2img_html_log,
            ], api_name="do_xldemo_txt2img_infer")

            
            xldemo_txt2img_refine.click(fn=do_xldemo_txt2img_refine, inputs=[
                xldemo_txt2img_prompt,
                xldemo_txt2img_negative_prompt,
                xldemo_txt2img_seed,
                xldemo_txt2img_refiner_steps,
                xldemo_txt2img_enable_refiner,
                xldemo_txt2img_image_to_refine,
                xldemo_txt2img_refiner_strength
            ], outputs=[
                xldemo_txt2img_gallery,
                xldemo_txt2img_generation_info,
                xldemo_txt2img_html_info,
                xldemo_txt2img_html_log,
            ], api_name="do_xldemo_txt2img_refine")


            return ui_component
