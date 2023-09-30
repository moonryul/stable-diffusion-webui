import math

import modules.scripts as scripts
import gradio as gr
from PIL import Image

from modules import processing, shared, images, devices
from modules.processing import Processed
from modules.shared import opts, state


class Script(scripts.Script): #Batch size and batch count: https://techtactician.com/batch-size-vs-batch-count-stable-diffusion/
    def title(self):
        return "SD upscale"

    def show(self, is_img2img):
        return is_img2img  #MJ: The script "sd_upscale.py" is visible when img2img is selected

    def ui(self, is_img2img):
        info = gr.HTML("<p style=\"margin-bottom:0.75em\">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>")
        overlap = gr.Slider(minimum=0, maximum=256, step=16, label='Tile overlap', value=64, elem_id=self.elem_id("overlap"))
        scale_factor = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label='Scale Factor', value=2.0, elem_id=self.elem_id("scale_factor"))
        upscaler_index = gr.Radio(label='Upscaler', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index", elem_id=self.elem_id("upscaler_index"))

        return [info, overlap, upscaler_index, scale_factor]

    def run(self, p, _, overlap, upscaler_index, scale_factor):
        if isinstance(upscaler_index, str):
            upscaler_index = [x.name.lower() for x in shared.sd_upscalers].index(upscaler_index.lower())
        processing.fix_seed(p)
        upscaler = shared.sd_upscalers[upscaler_index]  #MJ: defined in  modules.shared; upscaler_index=3= ESRGAN_4x
        #MJ: upscaler = LSDR object
        p.extra_generation_params["SD upscale overlap"] = overlap
        p.extra_generation_params["SD upscale upscaler"] = upscaler.name

        initial_info = None
        seed = p.seed

        init_img = p.init_images[0]  #MJ: get the image to upscale: (1024,1024)
        init_img = images.flatten(init_img, opts.img2img_background_color)

        if upscaler.name != "None": #MJ: "upscaler" is other than None, use it to generate the initial upscaled image
            img = upscaler.scaler.upscale(init_img, scale_factor, upscaler.data_path) #MJ: scale_factor is used only here,  not when upscaler=None
        else: #MJ: If "upscaler" = None, set img to init_img; upscaler.scaler= <ldsr_model.py.UpscalerLDSR object at 0x7f2033c628f0>
            img = init_img
        #MJ: img.size = (2048,2048) <= 2 * (1024,1024)
        devices.torch_gc()
        #MJ: img is the image that is upscaled two or four times: def split_grid(image, tile_w=512, tile_h=512, overlap=64):
        grid = images.split_grid(img, tile_w=p.width, tile_h=p.height, overlap=overlap) #MJ: modules.images; p.width = 1024
        #MJ: =>  grid.tiles.append([y, tile_h, row_images]) #MJ: tile_h = 512 by default, 1024 in our 1K => 2K experiment; grid = 3 x 3
        
        batch_size = p.batch_size  #MJ:Batch size is essentially a value that defines the amount of images to be generated in one batch = n_samples
        #Batch count: Number of times you run the image generation pipeline. (the process to generate starts again for every picture)
        
        upscale_count = p.n_iter  #MJ: = 1: batch_count = n_iter:  images in one batch are processed in parallel, 
        #  the individual batches are processed one after another p.n_iter times
         #MJ: increasing n_iter on the other hand just increases number of batches (each of size --n_samples) 
        # being processed sequentially.  #MJ: in general you should set your batch size setting first,
        # and use the batch count setting only in case you run out of VRAM during generation 
        p.n_iter = 1 
        #  
        
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        work = []

        #MJ:  Add each tile (from the upscaled img) to queue/list work:
        for _y, _h, row in grid.tiles: #MJ: for each row tile in row_images: [y, tile_h, row_images]
            for tiledata in row: #MJ: [x, tile_w, tile]: tiledata =  row_images.append([x, tile_w, tile]) #MJ: tile_w = 512
                work.append(tiledata[2])  #MJ: tiledata[2] refers to the tile image: 1024 x 1024 => 2048 x 2048: len(work)= 9 = 3x3 tiles

        batch_count = math.ceil(len(work) / batch_size)
        #MJ: In this script, batch_count is created from the number of tiles, not given by the user.
        # It is because the len(work)  tiles are considered samples for image generation. If these are more than the batch_size,
        # we need to create multiple batch-count.
        #batch_count refers to the number of batches in one iteration of image generation.
        state.job_count = batch_count * upscale_count

        print(f"SD upscaling will process a total of {len(work)} images tiled as {len(grid.tiles[0][2])}x{len(grid.tiles)} per upscale in a total of {state.job_count} batches.")

        result_images = []
        #MJ: process each tile from the upscaled img:
        for n in range(upscale_count): #upscale_count = original batch_count = original n_iter
            
            #MJ: The variable upscale_count = n_iter therefore dictates how many times this series of operations will be performed,
            # with each iteration potentially producing slightly different results
            # due to the stochastic elements of the process (such as random seeds).
            
            start_seed = seed + n
            
            p.seed = start_seed

            work_results = []
            
            for i in range(batch_count): #MJ: i refers to each batch: batch_count = 9
                p.batch_size = batch_size
                
                #MJ: Get the init_images of tiles from the upscaled image: p.init_images[0].size=(1024,1024)
                p.init_images = work[i * batch_size: (i + 1) * batch_size] #MJ:len(work) = 9; the 0th batch= 0: batch_size; the 1th batch =batch_size: 2*batch_size'...

                state.job = f"Batch {i + 1 + n * batch_count} out of {state.job_count}"
                
                processed = processing.process_images(p) #MJ: p will generate  p.init_images images (= batch size/num-samples) for each i in batch-count

                if initial_info is None:
                    initial_info = processed.info

                p.seed = processed.seed + 1
                work_results += processed.images
            #for i in range(batch_count)
            
            image_index = 0
            for _y, _h, row in grid.tiles:
                for tiledata in row: #MJ: setting tiledata[2] is a part of row, which has its location within the image
                    tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (p.width, p.height))
                    image_index += 1
            #for _y, _h, row in grid.tiles
            
            #MJ: now grid.tiles contain the results of img2img processing and combine them 
            combined_image = images.combine_grid(grid)
            
            result_images.append(combined_image)

            if opts.samples_save:
                images.save_image(combined_image, p.outpath_samples, "", start_seed, p.prompt, opts.samples_format, info=initial_info, p=p)
        #for n in range(upscale_count)
        
        processed = Processed(p, result_images, seed, initial_info)

        return processed
