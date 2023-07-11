from modules.shared import opts, OptionInfo
from modules import script_callbacks
from xldemo_txt2img_ui import make_ui


def on_ui_tabs():
    return [(make_ui(), "SDXL 0.9 Demo", "xldemo_txt2img")]


def on_ui_settings():
    section = ("xldemo_txt2img", "SDXL 0.9 Demo")

    opts.add_option(
        "xldemo_txt2img_huggingface_access_token", OptionInfo(
            "", "抱脸网访问Token", section=section)
    )
    opts.add_option(
        "xldemo_txt2img_1model", OptionInfo(
            "", "XL基础模型路径", section=section)
    )
    opts.add_option(
        "xldemo_txt2img_2model", OptionInfo(
            "", "XL精炼模型路径", section=section)
    )
    opts.add_option(
        "xldemo_txt2img_zdy", OptionInfo(
            True, "使用自定义路径", section=section)
    )
    opts.add_option(
        "xldemo_txt2img_load_refiner_on_startup", OptionInfo(
            True, "启动时加载精炼模型", section=section)
    )

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)

