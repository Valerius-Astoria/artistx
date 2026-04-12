# ArtistX

基于 Stable Diffusion v1.5 + ControlNet Lineart + LoRA 的风格化图像生成项目。  
你可以上传原图，选择 LoRA 风格，并快速生成具有特定艺术风格的结果。

项目提供：
- 脚本方式生成（`generate_images.py`）
- 本地 Web UI 交互生成（`run_web.py`）
- 3 种 LoRA 风格示例（Loish / Michal Sawtyruk / Neil Ross）

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

## 4. 脚本方式生成

你也可以直接修改 `generate_images.py` 中 `main()` 的参数后运行：

```bash
python generate_images.py
```

核心函数：`generate_images(...)`

---

## 5. 运行与性能说明

- 优先使用 GPU；若无 GPU 会自动回退到 CPU
- 速度参考：
  - GPU：约 4s/张（视显卡与参数而定）
  - CPU：约 7min/张
- 当前环境中 PyTorch 使用 CUDA 12.8；如与你机器驱动不匹配，可在 `environment.yaml` 中调整

---

## 6. LoRA 使用建议

以下为当前项目中验证过的推荐参数起点，可按效果微调。

### 6.1 Loish Style

- Prompt 倾向：`loish style, woman, digital art, concept art, high detail`
- 可增加风格词：`painting`（偏厚涂、偏奇幻）
- 推荐 Seed：`51717`、`411936`
- 推荐参数：
  - `num_inference_steps=30`
  - `guidance_scale=3`
  - `negative_prompt="owres, bad anatomy, bad hands, extra fingers, deformed face, asymmetrical eyes, distorted, oversaturated, overexposed"`
  - `num_images_per_prompt=1`
  - `controlnet_conditioning_scale=0.8`
  - `guess_mode=False`
  - `lora_scale=0.8`

### 6.2 Neil Ross Style

- Prompt 倾向：`neil_ross_style, [scene description], digital art, concept art, neil_ross_style, high detail`
- 可增加风格词：`painting`
- 推荐参数：
  - `num_inference_steps=45`
  - `guidance_scale=3`
  - `num_images_per_prompt=1`
  - `controlnet_conditioning_scale=0.8`
  - `guess_mode=False`
  - `lora_scale=0.8`

### 6.3 Michal Sawtyruk Style

- Prompt 倾向：`michal_sawtyruk_style, [scene description]`
- 推荐参数：
  - `num_inference_steps=45`
  - `guidance_scale=3`
  - `num_images_per_prompt=1`
  - `controlnet_conditioning_scale=0.8`
  - `guess_mode=False`
  - `lora_scale=0.8`

---

## 7. LoRA 训练参考

- [kohya_ss](https://github.com/bmaltais/kohya_ss)
- [How to train Stable Diffusion LoRA models](https://techtactician.com/how-to-train-stable-diffusion-lora-models/)