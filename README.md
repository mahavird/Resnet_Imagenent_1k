# ResNet-50 ImageNet-1K Training Project

A modular PyTorch training setup for ResNet-50 on ImageNet-1K dataset.

## Layout
```
Resnet_Imagenet_1k/
├── configs/
│   └── experiment.toml
├── src/
│   ├── data/
│   │   └── datasets.py
│   ├── engine/
│   │   └── engine.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── bottleneck.py  # Contains Bottleneck, ResNet, resnet50/101/152
│   ├── utils/
│   │   └── utils.py
│   ├── legacy/
│   │   └── original_script.py
│   └── train.py
├── checkpoints/     # Model checkpoints
├── logs/           # Training logs
├── scripts/
│   └── train.sh
├── test_setup.py   # Setup verification script
└── README.md
```

## Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Verify Setup
```bash
python test_setup.py
```

### 3. Prepare Data
Organize your data in ImageFolder format:
```
data/
├── train/
│   ├── class_0/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class_1/
│       ├── image1.jpg
│       └── image2.jpg
└── val/
    ├── class_0/
    │   └── image1.jpg
    └── class_1/
        └── image1.jpg
```

### 4. Configure Training
Edit `configs/experiment.toml`:
```toml
[data]
train_dir = "/content/data/imagenet_subtrain"
val_dir   = "/content/data/imagenet_validation"
batch_size = 32  # Adjust based on GPU memory
num_workers = 4

[train]
epochs = 100
amp = true  # Mixed precision training

[optim]
lr = 0.1  # Learning rate
```

### 5. Start Training
```bash
python src/train.py
```

### 6. Monitor Training
- Checkpoints saved when validation accuracy improves
- Use `Ctrl+C` to stop training gracefully

## Model

This project uses **ResNet-50** with 25.6M parameters for ImageNet-1K classification (1000 classes).

Other ResNet variants are available but not configured by default:
- `resnet101(num_classes=1000)` - ResNet-101 (44.5M parameters)  
- `resnet152(num_classes=1000)` - ResNet-152 (60.2M parameters)

To switch models, edit `src/train.py`:
```python
from src.models import resnet101  # or resnet152
model = resnet101(num_classes=num_classes, **cfg.get("model", {})).to(device)
```

## Training Options

### Configuration Parameters
Edit `configs/experiment.toml` to customize training:

```toml
# Data settings
[data]
train_dir = "/content/data/imagenet_subtrain"
val_dir = "/content/data/imagenet_validation"
img_size = 224        # Input image size
batch_size = 64       # Batch size (adjust for GPU memory)
num_workers = 4       # Data loading workers

# Model settings  
[model]
width_per_group = 64  # ResNet width multiplier

# Optimizer settings
[optim]
lr = 0.1             # Learning rate
momentum = 0.9       # SGD momentum
weight_decay = 5e-4  # Weight decay

# Training settings
[train]
epochs = 100         # Number of epochs
amp = true          # Mixed precision (requires CUDA)

# Logging settings
[logging]
save_freq = 5       # Save checkpoint every N epochs
log_freq = 100      # Log metrics every N batches
```

### Hardware Recommendations
- **GPU Memory**: 8GB+ VRAM recommended for ResNet-50
- **Batch Size**: Start with 32-64, reduce for OOM errors
- **Mixed Precision**: 1.5-2x speedup on modern GPUs

### Training Tips
1. **Learning Rate**: Start with 0.1, reduce by 10x every 30 epochs
2. **Data Augmentation**: RandomResizedCrop and RandomHorizontalFlip included
3. **Monitoring**: Watch both training and validation metrics
4. **Early Stopping**: Stop manually if validation accuracy plateaus
