import os
from abc import abstractmethod

import PIL
from PIL import Image

import modules.shared
from modules import modelloader, shared

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
NEAREST = (Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST)


class Upscaler:
    name = None
    model_path = None
    model_name = None
    model_url = None
    enable = True
    filter = None
    model = None
    user_path = None
    scalers: []
    tile = True

    def __init__(self, create_dirs=False):
        self.mod_pad_h = None
        self.tile_size = modules.shared.opts.ESRGAN_tile #MJ = 192
        self.tile_pad = modules.shared.opts.ESRGAN_tile_overlap #MJ: = 8
        self.device = modules.shared.device
        self.img = None
        self.output = None
        self.scale = 1
        self.half = not modules.shared.cmd_opts.no_half
        self.pre_pad = 0
        self.mod_scale = None
        self.model_download_path = None

        if self.model_path is None and self.name:
            self.model_path = os.path.join(shared.models_path, self.name)
        if self.model_path and create_dirs:
            os.makedirs(self.model_path, exist_ok=True)

        try:
            import cv2  # noqa: F401
            self.can_tile = True
        except Exception:
            pass

    @abstractmethod
    def do_upscale(self, img: PIL.Image, selected_model: str):
        return img

    def upscale(self, img: PIL.Image, scale, selected_model: str = None):
        self.scale = scale
        
        #MJ: The width and height are floor-divided by 8 and then multiplied by 8, possibly to maintain a certain alignment
        
        dest_w = int((img.width * scale) // 8 * 8)
        dest_h = int((img.height * scale) // 8 * 8)

        for _ in range(3): #
            shape = (img.width, img.height) #MJ: shape = (1024, 1024)

            img = self.do_upscale(img, selected_model) #MJ:img: shape=(4096, 4096;) 

            if shape == (img.width, img.height): #MJ: if the upscaler = None, the input img and the output img will be the same; break
                break

            if img.width >= dest_w and img.height >= dest_h: #MJ: dest_w = 2048; dest_h = 2048; img: shape=(4096, 4096;)=>  break
                break
        #for _ in range(3)
        
        #MJ: If, after exiting the loop, the image dimensions are not equal to the destination dimensions, it resizes the image
        # to exactly match the destination dimensions using a Lanczos resampling filter.
        
# Intermediate Upscaling:

# First, upscale the image by 2x using ESRGAN_4x (or any other method).
# Then, downscale the result back to 2x of the original using a good quality downscaling method
# like bicubic interpolation or Lanczos filtering.
# While this approach might sound counterintuitive, the ESRGAN model can still produce sharper
# and more detailed images that can be fine-tuned using the downscaling process.

        if img.width != dest_w or img.height != dest_h: #MJ: dest_w = 2048; dest_h = 2048; img: shape=(4096, 4096;)
            img = img.resize((int(dest_w), int(dest_h)), resample=LANCZOS) #MJ: downscale by Lancsos filtering
        
        return img

    @abstractmethod
    def load_model(self, path: str):
        pass

    def find_models(self, ext_filter=None) -> list:
        return modelloader.load_models(model_path=self.model_path, model_url=self.model_url, command_path=self.user_path, ext_filter=ext_filter)

    def update_status(self, prompt):
        print(f"\nextras: {prompt}", file=shared.progress_print_out)


class UpscalerData:
    name = None
    data_path = None
    scale: int = 4
    scaler: Upscaler = None
    model: None

    def __init__(self, name: str, path: str, upscaler: Upscaler = None, scale: int = 4, model=None):
        self.name = name
        self.data_path = path
        self.local_data_path = path
        self.scaler = upscaler
        self.scale = scale
        self.model = model


class UpscalerNone(Upscaler):
    name = "None"
    scalers = []

    def load_model(self, path):
        pass

    def do_upscale(self, img, selected_model=None):
        return img

    def __init__(self, dirname=None):
        super().__init__(False)
        self.scalers = [UpscalerData("None", None, self)]


class UpscalerLanczos(Upscaler):
    scalers = []

    def do_upscale(self, img, selected_model=None):
        return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=LANCZOS)

    def load_model(self, _):
        pass

    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "Lanczos"
        self.scalers = [UpscalerData("Lanczos", None, self)]


class UpscalerNearest(Upscaler):
    scalers = []

    def do_upscale(self, img, selected_model=None):
        return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=NEAREST)

    def load_model(self, _):
        pass

    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "Nearest"
        self.scalers = [UpscalerData("Nearest", None, self)]
