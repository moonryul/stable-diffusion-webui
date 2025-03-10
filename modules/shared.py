import sys

import gradio as gr

from modules import shared_cmd_options, shared_gradio_themes, options, shared_items, sd_models_types

#MJ: note that the last line of shared_items.py is sys.modules['modules.shared'].__class__ = Shared
# So, yes, the last line of the module is executed during the import process, 
# and it modifies the behavior of the modules.shared module as described in your comments. 
# This change in behavior affects how you can interact with the modules.shared module when it's imported in other Python files.

from modules.paths_internal import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir  # noqa: F401
from modules import util

cmd_opts = shared_cmd_options.cmd_opts
parser = shared_cmd_options.parser

batch_cond_uncond = True  # old field, unused now in favor of shared.opts.batch_cond_uncond
parallel_processing_allowed = True
styles_filename = cmd_opts.styles_file
config_filename = cmd_opts.ui_settings_file
hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}

demo = None

device = None

weight_load_location = None

xformers_available = False

hypernetworks = {}

loaded_hypernetworks = []

state = None

prompt_styles = None

interrogator = None

face_restorers = []

options_templates = None
opts = None
restricted_opts = None

sd_model: sd_models_types.WebuiSdModel = None

#In sd_models_types.py, class WebuiSdModel(LatentDiffusion):
#    """This class is not actually instantinated, but its fields are created and fieeld by webui"""

# This property is set to None by default. However, the last line in the shared_items.py module
# modifies the behavior of the modules.shared module:  this property can be dynamically changed or set
# elsewhere.
# Given this information, it is indeed possible that mentioning shared in your code, 
# as you do in the CheckpointInfo class when accessing shared.cmd_opts, can indirectly 
# involve mentioning shared.sd_model 
# or trigger changes related to it, depending on how it's used or modified elsewhere in your codebase.

#

settings_components = None
"""assinged from ui.py, a mapping on setting names to gradio components repsponsible for those settings"""

tab_names = []

latent_upscale_default_mode = "Latent"
latent_upscale_modes = {
    "Latent": {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
    "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
    "Latent (nearest)": {"mode": "nearest", "antialias": False},
    "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
}

sd_upscalers = []

clip_model = None

progress_print_out = sys.stdout

gradio_theme = gr.themes.Base()

total_tqdm = None

mem_mon = None

options_section = options.options_section
OptionInfo = options.OptionInfo
OptionHTML = options.OptionHTML

natural_sort_key = util.natural_sort_key
listfiles = util.listfiles
html_path = util.html_path
html = util.html
walk_files = util.walk_files
ldm_print = util.ldm_print

reload_gradio_theme = shared_gradio_themes.reload_gradio_theme

list_checkpoint_tiles = shared_items.list_checkpoint_tiles
refresh_checkpoints = shared_items.refresh_checkpoints
list_samplers = shared_items.list_samplers
reload_hypernetworks = shared_items.reload_hypernetworks
