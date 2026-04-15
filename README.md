# ArtistX

基于 Stable Diffusion v1.5 + ControlNet Lineart + LoRA 的风格化图像生成项目。  
你可以上传原图，选择 LoRA 风格，并快速生成具有特定艺术风格的结果。

> 开始前必做：先将你自己的 LoRA 文件（`.safetensors`）放入 `loras/` 目录，再到 `generate_images.py` 的 `main()` 中修改 `lora_name` 与提示词参数。
>
> 重要：本项目基座模型为 `Stable Diffusion v1.5`，你使用的 LoRA 必须与 `SD v1.5` 适配，否则可能出现风格失效、画面异常或生成失败。

项目提供：
- 脚本方式生成（`generate_images.py`）
- 本地 Web UI 交互生成（`run_web.py`）
- LoRA 条件生成示例流程

## 效果展示（3组：1张原图 + 3张生成图）

### 组 1

| 原图 | 生成图 1 | 生成图 2 | 生成图 3 |
| --- | --- | --- | --- |
| ![](assets/showcase/megan_original_resized.png) | ![](assets/showcase/megan_generated_1.png) | ![](assets/showcase/megan_generated_2.png) | ![](assets/showcase/megan_generated_3.png) |

### 组 2

| 原图 | 生成图 1 | 生成图 2 | 生成图 3 |
| --- | --- | --- | --- |
| ![](assets/showcase/car_original_resized.png) | ![](assets/showcase/car_generated_1.png) | ![](assets/showcase/car_generated_2.png) | ![](assets/showcase/car_generated_3.png) |

### 组 3

| 原图 | 生成图 1 | 生成图 2 | 生成图 3 |
| --- | --- | --- | --- |
| ![](assets/showcase/city_original_resized.png) | ![](assets/showcase/city_generated_1.png) | ![](assets/showcase/city_generated_2.png) | ![](assets/showcase/city_generated_3.png) |

---

## 1. 环境准备

推荐使用 Miniconda（参考官方文档安装）。

创建环境：

```bash
conda env create -f environment.yaml
```

激活环境：

```bash
conda activate artistx
```

---

## 2. 目录说明

- `data/input_images`：用户上传或输入的原图
- `data/sketches`：ControlNet 线稿中间结果
- `data/generated_images`：最终生成图像
- `loras`：LoRA 权重文件（`.safetensors`）
- `webapp`：Web UI 前端与后端页面逻辑

---

## 3. 快速开始（推荐：Web UI）

启动 Web UI：

```bash
conda run -n artistx python run_web.py
```

浏览器打开：

```text
http://127.0.0.1:7860
```

在页面中可完成以下流程：
1. 上传原图（会保存到 `data/input_images`）
2. 选择 LoRA 风格（读取 `loras` 中权重）
3. 设置生成参数（`Num` 为循环生成次数）
4. 浏览生成结果并下载到本地（图片保存到 `data/generated_images`）

> Seed 逻辑：  
> - 留空：每轮随机新 seed（结果更有变化）  
> - 填写具体值：每轮固定同一 seed（结果可复现）

---

## 4. LoRA 权重说明（版权）

出于艺术家版权与模型分发合规考虑，仓库**不提供**任何 `loras/*.safetensors` 文件，也不提供下载链接。  
你可以自行训练 LoRA 权重，并将文件放置到 `loras/` 目录后再运行本项目。
请确保 LoRA 是基于 `Stable Diffusion v1.5` 训练或明确兼容 `SD v1.5`。

将 LoRA 文件保存到 `loras/` 目录下即可。

训练与参数参考仍以本文档当前内容为准，重点可参考：

- `7. LoRA 使用建议`
- `8. LoRA 训练参考`

---

## 5. 脚本方式生成

你也可以直接修改 `generate_images.py` 中 `main()` 的参数后运行：

```bash
python generate_images.py
```

核心函数：`generate_images(...)`

---

## 6. 运行与性能说明

- 优先使用 GPU；若无 GPU 会自动回退到 CPU
- 速度参考：
  - GPU：约 4s/张（视显卡与参数而定）
  - CPU：约 7min/张
- 当前环境中 PyTorch 使用 CUDA 12.8；如与你机器驱动不匹配，可在 `environment.yaml` 中调整

---

## 7. LoRA 使用建议

以下为通用参数建议，可作为起点后按场景微调。

- 推荐参数：
  - `num_inference_steps=30`
  - `guidance_scale=3`
  - `negative_prompt="owres, bad anatomy, bad hands, extra fingers, deformed face, asymmetrical eyes, distorted, oversaturated, overexposed"`
  - `num_images_per_prompt=1`
  - `controlnet_conditioning_scale=0.8`
  - `guess_mode=False`
  - `lora_scale=0.8`
- 参数说明：
  - 上述组合可让 ControlNet 更稳定地遵循原始草稿结构，同时叠加你指定的 LoRA 风格进行生成。
  - `guess_mode=False` + 合理的 `controlnet_conditioning_scale` 有助于减少结构漂移与细节幻觉。
  - 对复杂度较高的场景，可将 `num_inference_steps` 提升至 `45` 或 `60` 以增强稳定性与细节还原。

---

## 8. LoRA 训练参考

- [kohya_ss](https://github.com/bmaltais/kohya_ss)
- [How to train Stable Diffusion LoRA models](https://techtactician.com/how-to-train-stable-diffusion-lora-models/)