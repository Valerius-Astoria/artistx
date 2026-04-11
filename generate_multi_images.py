from generate_image import load_model, generate_image


def main():
    image_name = "michal sawtyruk car.jpeg"
    lora_name = "michal_sawtyruk_style.safetensors"
    prompt = """
    michal_sawtyruk_style a retro blue sedan parked on a wet street, colorful modern buildings, evening city scene, cinematic lighting, digital painting
    """
    negative_prompt = """
    lowres, blurry, bad anatomy, distorted car, extra wheels, text, watermark, logo
    """
    # 先加载模型，后续生成会复用缓存
    load_model(lora_name=lora_name)

    for i in range(5):
        generate_image(
            name=image_name,
            lora_name=lora_name,
            prompt=prompt,
            seed=None,
            num_inference_steps=45,
            guidance_scale=2,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            controlnet_conditioning_scale=0.8,
            guess_mode=False,
            lora_scale=0.8,
        )


if __name__ == "__main__":
    main()
