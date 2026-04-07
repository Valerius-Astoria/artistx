本机所需环境安装命令：
sudo apt install python3.11 python3.11-venv git
miniconda（参考官方文档）

依赖安装命令：
conda env create -f environment.yaml

文件结构说明：
generated_images：存放模型生成图像
input_images：存放输入图像
sketches：存放模型根据输入图像生成的线稿

