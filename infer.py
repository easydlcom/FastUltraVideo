import os
import random

import pandas as pd
import subprocess
from multiprocessing import Pool
import multiprocessing as mp
from filelock import FileLock
import time
import json
from datetime import datetime
import argparse
import glob
from more_itertools import divide
import yaml
import numpy as np
import cv2
import os
import cv2
import json
import time
import subprocess

import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download
import torch.distributed as dist
from xfuser.core.distributed import (initialize_model_parallel,
                                     init_distributed_environment)


class VideoGenerator:

    def __init__(self, cfg_terminal):
        model_dir = cfg_terminal.model_dir

        if cfg_terminal.mode == 'full' and cfg_terminal.model_path:
            model_path = cfg_terminal.model_path
        else:
            model_path = f"{model_dir}/diffusion_pytorch_model.safetensors"
        # Load models
        model_manager = ModelManager(device="cpu")
        model_manager.load_models(
            [
                model_path,
                f"{model_dir}/models_t5_umt5-xxl-enc-bf16.pth",
                f"{model_dir}/Wan2.1_VAE.pth",
            ],
            torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
        )
        print(model_path)
        if cfg_terminal.mode == 'lora':
            model_manager.load_lora(cfg_terminal.model_path, lora_alpha=cfg_terminal.lora_alpha)
            print('lora: ', cfg_terminal.model_path)

        if cfg_terminal.usp == 1:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
            )

            init_distributed_environment(
                rank=dist.get_rank(), world_size=dist.get_world_size())

            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=1,
                ulysses_degree=dist.get_world_size(),
            )
            torch.cuda.set_device(dist.get_rank())

            self.pipe = WanVideoPipeline.from_model_manager(
                model_manager,
                torch_dtype=torch.bfloat16,
                device=f"cuda:{dist.get_rank()}",
                use_usp=True if dist.get_world_size() > 1 else False)
            self.pipe.enable_vram_management(num_persistent_param_in_dit=None)
        else:
            self.pipe = WanVideoPipeline.from_model_manager(
                model_manager,
                torch_dtype=torch.bfloat16,
                device=f"cuda")
            self.pipe.enable_vram_management(num_persistent_param_in_dit=None)

    def process_prompt(self, prompt, size):
        video = self.pipe(
            prompt=prompt,
            negative_prompt="Oversaturated, overexposed, static, details are blurry, subtitles, style, works, paintings, pictures, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, grotesque limbs, finger fusion, motionless images, messy backgrounds, three legs, many people in the background, walking backwards",
            num_inference_steps=50,
            seed=0, tiled=True,
            height=size[0],
            width=size[1],
            num_frames=size[2],
        )
        return video

    def process_image(self, image_path, size, prompt=None):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        video = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt="Oversaturated, overexposed, static, details are blurry, subtitles, style, works, paintings, pictures, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, grotesque limbs, finger fusion, motionless images, messy backgrounds, three legs, many people in the background, walking backwards",
            num_inference_steps=50,
            seed=0, tiled=True,
            height=size[0],
            width=size[1],
            num_frames=size[2],
        )
        return video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='ultrawan_weights/Wan2.1-T2V-1.3B')
    parser.add_argument('--model_path', default='')
    parser.add_argument('--mode', default='full')
    parser.add_argument('--lora_alpha', type=float, default=1.0)
    parser.add_argument('--usp', type=int, default=0)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=832)
    parser.add_argument('--num_frames', type=int, default=81)
    parser.add_argument('--out_dir', default='output/ori')
    parser.add_argument('--image_path', default='', type=str, help='Path to input image for image2video')
    parser.add_argument('--prompt', default='', type=str, help='Prompt for image2video or text2video')
    cfg_terminal = parser.parse_args()

    # dir
    out_dir = cfg_terminal.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # init model
    generator = VideoGenerator(cfg_terminal)

    sizes = [[cfg_terminal.height, cfg_terminal.width, cfg_terminal.num_frames]]

    # Image to video (supports image+prompt)
    if cfg_terminal.image_path:
        for size in sizes:
            video = generator.process_image(cfg_terminal.image_path, size=size, prompt=cfg_terminal.prompt if cfg_terminal.prompt else None)
            save_video(video, f"{out_dir}/image2video_{size[0]}_{size[1]}_{size[2]}.mp4", fps=30, quality=8)
    else:
        # Text to video (supports single prompt or default prompt list)
        if cfg_terminal.prompt:
            prompts = [cfg_terminal.prompt]
        else:
            prompts = [
                "An astronaut wearing a spacesuit rides a mechanical horse on the surface of Mars, facing the camera. The red desolate terrain stretches into the distance, dotted with huge craters and strange rock formations. The mechanical horse strides steadily, kicking up faint dust, showcasing the perfect blend of future technology and primal exploration. The astronaut holds a control device, looking determined, as if pioneering new territory for humanity. The background is the deep universe and the blue Earth, the scene is both sci-fi and full of hope, inspiring visions of future interstellar life.",
                "Documentary photography style: a lively puppy runs quickly on a lush green lawn. The puppy has brown-yellow fur, two upright ears, and a focused, joyful expression. Sunlight shines on its body, making its fur look especially soft and shiny. The background is a wide expanse of grass, occasionally dotted with wildflowers, with blue sky and a few white clouds faintly visible in the distance. The perspective is clear, capturing the dynamic movement of the puppy and the vitality of the surrounding grass. Medium shot, side-moving view."
            ]
        for idx, prompt in enumerate(prompts):
            for size in sizes:
                video = generator.process_prompt(prompt, size=size)
                save_video(video, f"{out_dir}/{idx}_{size[0]}_{size[1]}_{size[2]}.mp4", fps=30, quality=8)

if __name__ == '__main__':
    main()