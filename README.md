# Python 神经网络课程项目

本仓库面向《人工智能》课程中的神经网络实验学习，内容围绕教材章节组织，提供可直接运行的脚本、实验报告与图表产物。

## 项目内容
- Chapter 1：感知机线性分类与三层网络反向传播基础。
- Chapter 2：基于 MNIST 的全连接神经网络训练与评估。
- Chapter 3：Backquery 可视化与旋转数据增强效果对比。

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
|  |  |- MNIST/raw/
|  |  |- book_mnist_csv/
|  |  `- book_own_images/
|  `- processed/
|- outputs/
|  |- figures/
|  `- logs/
|- reports/
|- chapter_packages/
|- models/
`- third_party/
```

## 环境与依赖
- Python 3.10.x
- 推荐环境：Conda

安装依赖：

```powershell
conda activate D:\code\Python\ai_learn
pip install -r requirements.txt
```

通过环境文件重建：

```powershell
conda env create -f environment.yml
conda activate ai_learn
```

## 快速运行

### Chapter 1
```powershell
python .\experiments\ch1\1.1_perceptron_linear_classifier.py
python .\experiments\ch1\1.2_three_layer_neural_network_backprop.py
```

### Chapter 2
```powershell
python .\experiments\ch2\2.1_neural_network_mnist_data.py --epochs 5
```

### Chapter 3
```powershell
python .\experiments\ch3\3.1_neural_network_mnist_backquery.py
python .\experiments\ch3\3.2_neural_network_mnist_rotation_augmentation.py
```

## 教材 Notebook 入口

Chapter 2：

```text
experiments/ch2/book_notebooks/part2_neural_network_mnist_data.ipynb
```

Chapter 3：

```text
experiments/ch3/book_notebooks/part3_neural_network_mnist_data_with_rotations.ipynb
experiments/ch3/book_notebooks/part3_neural_network_mnist_and_own_single_image.ipynb
experiments/ch3/book_notebooks/part3_neural_network_mnist_backquery.ipynb
```

## 数据与产物说明
- 训练数据：`data/raw/MNIST/raw`（IDX 全量）
- 教材示例数据：`data/raw/book_mnist_csv`、`data/raw/book_own_images`
- 实验图表：`outputs/figures`
- 实验日志：`outputs/logs`
- 实验报告：`reports`
- 章节打包：`chapter_packages`

## 参考资料
- 教材配套源码及说明保存在 `third_party/makeyourownneuralnetwork`。
- 课程实验规划与方法总结见 `deep-research-report.md`。
