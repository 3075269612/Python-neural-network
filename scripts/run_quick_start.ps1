$ErrorActionPreference = "Stop"

Write-Host "[1/4] Checking environment"
python .\1.py

Write-Host "[2/4] Running Chapter 1 perceptron"
python .\experiments\ch1\perceptron_numpy.py --epochs 50 --lr 0.1

Write-Host "[3/4] Running Chapter 1 XOR MLP"
python .\experiments\ch1\mlp_xor_numpy.py --epochs 5000 --lr 0.8

Write-Host "[4/4] Running Chapter 2 quick MNIST training"
python .\experiments\ch2\mnist_mlp_pytorch.py --epochs 1 --train-subset 5000 --test-subset 1000

Write-Host "Quick start complete."
