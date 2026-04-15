import uuid
from pathlib import Path

import numpy as np
import torch
from controlnet_aux import LineartDetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image

BASE_DIR = Path(__file__).resolve().parent
input_image_dir = BASE_DIR / "data" / "input_images"
sketches_dir = BASE_DIR / "data" / "sketches"
generated_images_dir = BASE_DIR / "data" / "generated_images"
lora_dir = BASE_DIR / "loras"

_MODEL_CACHE: dict[tuple[str, str, str], tuple[LineartDetector, StableDiffusionControlNetPipeline, str]] = {}


def _resolve_lora_path(lora_name: str) -> Path:
    normalized = lora_name.strip()
    if not normalized:
        raise ValueError(
            "LoRA file name is required. Put your .safetensors file in loras/ and set lora_name in main()."
        )

    lora_path = lora_dir / normalized
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    if lora_path.suffix.lower() != ".safetensors":
        raise ValueError(f"LoRA file must be a .safetensors file: {lora_path.name}")

    return lora_path


def load_model(lora_name: str):
    """
    加载并缓存 Lineart 处理器与 SD+ControlNet pipeline，避免重复加载模型。
    """
    # 使用 Controlnet Lineart 模型，具体请参考“https://github.com/lllyasviel/ControlNet-v1-1-nightly”
    checkpoint = "ControlNet-1-1-preview/control_v11p_sd15_lineart"

    # 优先使用GPU运行，如GPU不可用，则自动切换至CPU运行
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    lora_path = _resolve_lora_path(lora_name)
    cache_key = (device, str(dtype), lora_path.name)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # 加载线稿检测模型
    processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

    # 加载ControlNet模型。
    # 显式关闭 low_cpu_mem_usage，避免某些环境下出现 meta tensor -> device 迁移报错。
    controlnet = ControlNetModel.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )

    # 加载Stable Diffusion ControlNet模型，如果模型不存在，则从Hugging Face下载(仅首次运行时下载)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )

    # 加载 LoRA
    pipe.load_lora_weights(str(lora_dir), weight_name=lora_path.name)

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

def _generate_images_once(
    name: str,
    lora_name: str,
    prompt: str,
    seed: int = None,
    num_inference_steps: int = 30,
    guidance_scale: float = 2,
    negative_prompt: str = None,
    num_images_per_prompt: int = 1,
    controlnet_conditioning_scale: float = 0.8,
    guess_mode: bool = False,
    lora_scale: float = 0.8,
)-> tuple[list[str], int]:
    """
    name: 输入图像名称(含后缀)
    lora_name: LoRA 权重文件名称
    prompt: 生成图像提示词
    seed: 随机种子（None 时随机）
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
    image = load_image(str(input_image_dir / name))
    # image = image.resize((width, height)) # 将输入图像缩放至指定尺寸

    # 生成线稿
    control_image = processor(image)
    control_image.save(str(sketches_dir / f"control_{name}")) # 保存线稿至sketches文件夹
    sketch_width, sketch_height = control_image.size

    # 设置随机种子
    current_seed = seed if seed is not None else np.random.randint(0, 1000000)
    generator = torch.manual_seed(current_seed)

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
    cross_attention_kwargs={"scale": lora_scale},).images

    # 将生成图像随机命名后保存至generated_images文件夹 
    saved_files: list[str] = []
    for image in images:
        random_name = f"{uuid.uuid4().hex}.png"
        image.save(str(generated_images_dir / random_name))
        saved_files.append(random_name)

    # 打印每张图像对应的随机种子
    print(f"{current_seed}: {', '.join(saved_files)}")
    return saved_files, int(current_seed)

def generate_images(
    num: int,
    name: str,
    lora_name: str,
    prompt: str,
    seed: int = None,
    num_inference_steps: int = 30,
    guidance_scale: float = 2,
    negative_prompt: str = None,
    num_images_per_prompt: int = 1,
    controlnet_conditioning_scale: float = 0.8,
    guess_mode: bool = False,
    lora_scale: float = 0.8,
):
    """
    num: 生成图像数量
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
    _resolve_lora_path(lora_name)
    sketches_dir.mkdir(parents=True, exist_ok=True)
    generated_images_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(num):
        # seed 为空时，每轮随机一个新的 seed；seed 有值时固定使用该 seed（可复现）
        current_seed = None if seed is None else seed
        _generate_images_once(
            name=name,
            lora_name=lora_name,
            prompt=prompt,
            seed=current_seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guess_mode=guess_mode,
            lora_scale=lora_scale,
        )
        
def main():
    # 必须先将你的 LoRA 权重文件放到 loras/ 目录，再填写下面的参数。
    generate_images(
        num=5,
        name="megan fox.png",
        # TODO: Fill your LoRA file name in loras/, e.g. "your_style.safetensors"
        lora_name="",
        # TODO: Fill your prompt
        prompt="",
        num_inference_steps=30,
        guidance_scale=2,
        # TODO: Fill your negative prompt
        negative_prompt="",
        num_images_per_prompt=1,
        controlnet_conditioning_scale=0.8,
        guess_mode=False,
        lora_scale=0.8,
    )


if __name__ == "__main__":
    main()

