$ErrorActionPreference = "Stop"

Write-Host "[1/4] Checking core packages"
python -c "import sys, numpy, matplotlib, torch, torchvision; print('python:', sys.version.split()[0]); print('numpy:', numpy.__version__); print('matplotlib:', matplotlib.__version__); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__)"

Write-Host "[2/4] Running Chapter 1 perceptron"
python .\experiments\ch1\perceptron_numpy.py --epochs 50 --lr 0.1

Write-Host "[3/4] Running Chapter 1 XOR MLP"
python .\experiments\ch1\mlp_xor_numpy.py --epochs 5000 --lr 0.8

Write-Host "[4/4] Running Chapter 2 quick MNIST training"
python .\experiments\ch2\mnist_mlp_pytorch.py --epochs 1 --train-subset 5000 --test-subset 1000

Write-Host "Quick start complete."
