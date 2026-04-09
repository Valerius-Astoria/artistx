import torch
import os
import uuid
import argparse
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from controlnet_aux import LineartDetector

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

input_image_dir = "data/input_images"
sketches_dir = "data/sketches"
generated_images_dir = "data/generated_images"
lora_dir = "loras"

_MODEL_CACHE = {}


def load_model(lora_name: str = None):
    """
    加载并缓存 Lineart 处理器与 SD+ControlNet pipeline，避免重复加载模型。
    """
    # 使用 Controlnet Lineart 模型，具体请参考“https://github.com/lllyasviel/ControlNet-v1-1-nightly”
    checkpoint = "ControlNet-1-1-preview/control_v11p_sd15_lineart"

    # 优先使用GPU运行，如GPU不可用，则自动切换至CPU运行
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    cache_key = (device, str(dtype), lora_name or "")

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # 加载线稿检测模型
    processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

    # 加载ControlNet模型
    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=dtype)

    # 加载Stable Diffusion ControlNet模型，如果模型不存在，则从Hugging Face下载(仅首次运行时下载)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=dtype
    )

    # 仅当 lora_dir 和 lora_name 都提供时才加载 LoRA
    if lora_dir and lora_name:
        pipe.load_lora_weights(lora_dir, weight_name=lora_name)

    # 设置调度器
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # 如果GPU可用，则将模型固定在GPU上，确保推理走CUDA
    if device == "cuda":
        pipe = pipe.to("cuda")
    # 如果GPU不可用，则将模型保持在CPU上
    else:
        pipe = pipe.to("cpu")

    _MODEL_CACHE[cache_key] = (processor, pipe, device)
    return _MODEL_CACHE[cache_key]

def generate_image(
    name: str,
    lora_name: str,
    prompt: str,
    seed: int = None,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    negative_prompt: str = None,
    num_images_per_prompt: int = 1,
    controlnet_conditioning_scale: float = 1.0,
    guess_mode: bool = False,
    lora_scale: float = 0.8,
):
    """
    name: 输入图像名称(含后缀)
    lora_name: LoRA 权重文件名称
    prompt: 生成图像提示词
    seed: 随机种子
    num_inference_steps: 生成图像步数
    guidance_scale: 文字引导强度
    negative_prompt: 负向提示词
    num_images_per_prompt: 生成图像数量
    controlnet_conditioning_scale: 线稿引导强度
    guess_mode: 是否使用猜测模式
    lora_scale: LoRA 引导强度
    """
    # 复用已加载模型，避免重复初始化
    processor, pipe, device = load_model(lora_name=lora_name)

    # 加载输入图像
    image = load_image(
        f"{input_image_dir}/{name}"
    )
    # image = image.resize((width, height)) # 将输入图像缩放至指定尺寸

    # 生成线稿
    control_image = processor(image)
    control_image.save(f"{sketches_dir}/control_{name}") # 保存线稿至sketches文件夹
    sketch_width, sketch_height = control_image.size

    # 设置随机种子
    seed = seed if seed is not None else np.random.randint(0, 1000000)
    generator = torch.manual_seed(seed) 

    # 生成图像
    images = pipe(prompt, 
    image=control_image, 
    height=sketch_height,
    width=sketch_width,
    num_inference_steps=num_inference_steps, 
    guidance_scale=guidance_scale, 
    negative_prompt=negative_prompt, 
    num_images_per_prompt=num_images_per_prompt, 
    generator=generator, 
    controlnet_conditioning_scale=controlnet_conditioning_scale, 
    guess_mode=guess_mode,
    cross_attention_kwargs={"scale": lora_scale} if lora_name else None,).images

    # 将生成图像随机命名后保存至generated_images文件夹 
    for image in images:
        random_name = f"{uuid.uuid4().hex}.png"
        image.save(f"{generated_images_dir}/{random_name}")

    # 打印每张图像对应的随机种子
    print(f"{seed}: {random_name}")


def main():
    generate_image(
       name="megan fox.png",
       lora_name="loish_style.safetensors",
       prompt="""
       a beautiful woman
       digital art, concept art
       loishstyle
       high detail
       """,
       num_inference_steps=30,
       guidance_scale=2,
       negative_prompt="""
       owres, bad anatomy, bad hands, extra fingers, deformed face, asymmetrical eyes, distorted, oversaturated, overexposed
       """,
       num_images_per_prompt=1,
       controlnet_conditioning_scale=0.8,
       guess_mode=False,
       lora_scale=0.8
    )


if __name__ == "__main__":
    main()

