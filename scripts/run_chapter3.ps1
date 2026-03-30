$ErrorActionPreference = "Stop"

Write-Host "Training baseline model (no augmentation)"
python .\experiments\ch3\rotate_augmentation.py --epochs 2 --train-subset 10000 --test-subset 2000

Write-Host "Training rotation-augmented model"
python .\experiments\ch3\rotate_augmentation.py --use-rotation --rotation-degree 20 --epochs 2 --train-subset 10000 --test-subset 2000

Write-Host "Visualizing first-layer weights"
python .\experiments\ch3\visualize_weights.py

Write-Host "Chapter 3 script finished."
