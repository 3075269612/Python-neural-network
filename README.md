# Python 神经网络课程项目

本仓库用于《人工智能》课程课外自学项目，核心目标是基于教材逐章完成实验，并沉淀为可复现、可追踪、可提交的作业仓库。

## 项目目标
- 按章节完成实验代码实现。
- 规范保存模型、日志、图像和报告素材。
- 通过 GitHub 持续记录实验过程与结论。

## 仓库结构

```text
.
|- deep-research-report.md
|- README.md
|- requirements.txt
|- environment.yml
|- experiments/
|  |- README.md
|  |- ch1/
|  |- ch2/
|  |  `- book_notebooks/
|  |- ch3/
|  |  `- book_notebooks/
|- data/
|  |- raw/
|  |  |- book_mnist_csv/
|  |  `- book_own_images/
|  |- processed/
|- models/
|- outputs/
|  |- figures/
|  |- logs/
|- reports/
|  |- lab-report-template.md
|- scripts/
|  |- run_quick_start.ps1
|  |- run_chapter3.ps1
|- third_party/
|  `- makeyourownneuralnetwork/
`- .vscode/
```

## Python 环境说明
已按你的要求固定为以下解释器：
- D:/code/Python/ai_learn/python.exe
- Python 3.10.x

可使用以下命令验证环境：

```powershell
python -c "import sys, numpy, matplotlib, torch, torchvision; print('python:', sys.version.split()[0]); print('numpy:', numpy.__version__); print('matplotlib:', matplotlib.__version__); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__)"
```

## 教材源码整合（已完成）
- 教材原始仓库已按章节分流到：
	- experiments/ch2/book_notebooks
	- experiments/ch3/book_notebooks
- 教材 CSV 和手写图片样例已迁移到：
	- data/raw/book_mnist_csv
	- data/raw/book_own_images
- 教材版权与说明文档保留在：
	- third_party/makeyourownneuralnetwork
- Notebook 中原有相对路径已改成项目统一数据目录路径，可直接运行。

## 教材 Notebook 入口

### Chapter 2（教材版）
```text
experiments/ch2/book_notebooks/part2_neural_network_mnist_data.ipynb
```

### Chapter 3（教材版）
```text
experiments/ch3/book_notebooks/part3_neural_network_mnist_data_with_rotations.ipynb
experiments/ch3/book_notebooks/part3_neural_network_mnist_and_own_single_image.ipynb
experiments/ch3/book_notebooks/part3_neural_network_mnist_backquery.ipynb
```

## 依赖安装

```powershell
conda activate D:\code\Python\ai_learn
pip install -r requirements.txt
```

如需从环境文件重建：

```powershell
conda env create -f environment.yml
conda activate ai_learn
```

## 一键快速开始

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_quick_start.ps1
```

该脚本会依次执行：
- 环境检查。
- 第1章感知机实验。
- 第1章 XOR 三层网络实验。
- 第2章 MNIST 小样本训练。

## 分章运行命令

### 第1章
```powershell
python .\experiments\ch1\perceptron_numpy.py
python .\experiments\ch1\mlp_xor_numpy.py
```

### 第2章
```powershell
python .\experiments\ch2\mnist_mlp_pytorch.py --epochs 5
```

### 第3章
```powershell
python .\experiments\ch3\rotate_augmentation.py
python .\experiments\ch3\rotate_augmentation.py --use-rotation --rotation-degree 20
python .\experiments\ch3\visualize_weights.py
python .\experiments\ch3\custom_digit_test.py --image <你的手写数字图片路径>
```

## 产物位置
- 训练日志：outputs/logs/*.csv
- 图像结果：outputs/figures/*.png
- 模型文件：models/*.pt

## 报告编写建议
1. 先运行实验得到日志、图和模型。
2. 复制 reports/lab-report-template.md 作为每次实验报告模板。
3. 将 deep-research-report.md 作为任务规划参考。
4. 每完成一个实验小节就提交一次 Git 记录。

## GitHub 提交建议

```powershell
git checkout -b feat/ch1-baseline
git add .
git commit -m "init: chapter scaffold and baseline scripts"
git push -u origin feat/ch1-baseline
```
