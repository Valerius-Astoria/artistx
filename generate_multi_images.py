from generate_image import load_model, generate_image


def main():
    image_name = "scene.png"
    lora_name = "neil_ross_style.safetensors"
    prompt = """
    A painting of a discarded robot in the wild
    digital art, concept art
    neil_ross_style
    high detail
    """
    negative_prompt = """
    blurry, lowres, noisy, jpeg artifacts, watermark, text, logo, cropped, out of frame, duplicate objects, bad anatomy, bad hands, extra fingers, deformed face, asymmetrical eyes, distorted, oversaturated, overexposed
    """

    # 先加载模型，后续生成会复用缓存
    load_model(lora_name=lora_name)

    for i in range(5):
        generate_image(
            name=image_name,
            lora_name=lora_name,
            prompt=prompt,
            seed=None,
            num_inference_steps=30,
            guidance_scale=2,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            controlnet_conditioning_scale=0.8,
            guess_mode=False,
            lora_scale=0.8,
        )


if __name__ == "__main__":
    main()
