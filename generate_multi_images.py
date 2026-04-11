from generate_image import load_model, generate_image


def main():
    image_name = "006_Via_Mazanti._Verona._by_Neil_Ross_elEm9Z_1.jpg"
    lora_name = "neil_ross_style.safetensors"
    prompt = """
    neil_ross_style sunlit narrow Verona street, old Italian facades, cafe umbrellas, painterly urban scene, warm shadows
    """
    negative_prompt = """
    lowres, blurry, worst quality, normal quality, jpeg artifacts, text, watermark, logo, oversharpen, overexposed, underexposed, distorted buildings, warped perspective, deformed people, noisy
    """
    # 先加载模型，后续生成会复用缓存
    load_model(lora_name=lora_name)

    for i in range(5):
        generate_image(
            name=image_name,
            lora_name=lora_name,
            prompt=prompt,
            seed=None,
            num_inference_steps=60,
            guidance_scale=2,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            controlnet_conditioning_scale=0.8,
            guess_mode=False,
            lora_scale=0.8,
        )


if __name__ == "__main__":
    main()
