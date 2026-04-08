本机所需环境安装命令：
<!-- sudo apt install python3.11 python3.11-venv git -->
miniconda（参考官方文档）

依赖安装命令：
conda env create -f environment.yaml

文件结构说明：
generated_images：存放模型生成图像
input_images：存放输入图像
sketches：存放模型根据输入图像生成的线稿

LoRA训练：
参考：https://techtactician.com/how-to-train-stable-diffusion-lora-models/

CPU & GPU配置说明：
- 优先使用GPU，在GPU不可用时使用CPU
- GPU图像生成时间约4s，CPU图像生成时间约7min
- 为避免 PyTorch 的 CUDA 运行版本高于当前驱动可支持的版本，将 PyTorch 的 CUDA 版本调整为12.8。你可以根据你的机器实际支持的 CUDA 版本在 environment.yaml 中进行调整