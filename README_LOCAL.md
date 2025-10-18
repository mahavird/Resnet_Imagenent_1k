# ImageNet POC - Local Version

This document describes the local version of the ImageNet training notebook (`imagenet_poc_local.ipynb`).

## What's New

A local-execution-ready copy of `imagenet_poc.ipynb` has been created as `imagenet_poc_local.ipynb` with the following changes:

### Changes from Colab Version:

✅ **Updated Data Paths**:
- Uses ILSVRC2012 folder structure by default
- Training data: `./ILSVRC2012/train/`
- Validation data: `./ILSVRC2012/val_sorted/`
- Path validation included to check if data exists
- Easy to configure ROOT path for different locations

✅ **Enhanced for Local Use**:
- Pure Python environment (no Colab dependencies)
- Device-agnostic checkpoint loading (CPU/CUDA/MPS)
- Automatic class detection from dataset
- Automatic validation data filtering (prevents class mismatch errors)
- Model name changed to `resnet_152_sgd1_local`
- Better logging and progress indicators
- Kernel environment detection and debugging tools


## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements_imagenet_poc.txt
```

The requirements file includes:
- PyTorch & TorchVision
- TensorBoard
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn
- tqdm

### 2. Prepare Your Data

#### Understanding ILSVRC2012 Dataset Structure

**Dataset Overview:**
- **1000 object categories** (low-level synsets from WordNet)
- **Training images**: 1,281,167 total (732-1300 images per class)
- **Validation images**: 50,000 total (50 images per class)
- **File naming**: 
  - Training: `<WNID>_<number>.JPEG` (e.g., `n01440764_18.JPEG`)
  - Validation: `ILSVRC2012_val_00000001.JPEG` to `ILSVRC2012_val_00050000.JPEG`

**Important**: Validation images are provided as a flat directory and must be sorted into class folders using the ground truth file (`data/ILSVRC2012_validation_ground_truth.txt`). The original Colab notebook (Cell 6) includes a script to perform this sorting.

#### Expected Folder Structure

The notebook expects ILSVRC2012 dataset organized as follows:

```
ILSVRC2012/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_*.JPEG
│   │   └── ...
│   ├── n01443537/
│   └── ... (1000 classes)
│
└── val_sorted/
    ├── n01440764/
    │   ├── ILSVRC2012_val_*.JPEG
    │   └── ...
    ├── n01443537/
    └── ... (1000 classes)
```

**Training Data Structure:**
- Already organized in class folders (one folder per WNID)
- Each folder named by WordNet ID (e.g., `n01440764` for tench fish)
- Images named as `<WNID>_<number>.JPEG`

**Validation Data Requirements:**
- ⚠️ **Raw validation images are flat** (not in class folders)
- Must be sorted into class folders using ground truth labels
- Use the sorting script from original notebook Cell 6, or
- Obtain pre-sorted `val_sorted/` directory
- Each class should have ~50 validation images

**Configuration:**
- Update the `ROOT` path in Cell 12 to point to your ILSVRC2012 folder location
- Notebook supports both full dataset (1000 classes) and subsets (automatically detected)

**Alternative paths**:
If your data is in a different location, simply update this line in Cell 12:
```python
ROOT = Path('./ILSVRC2012')  # Change to your actual path
# Examples:
# ROOT = Path('/data/imagenet/ILSVRC2012')     # Linux/Mac absolute path
# ROOT = Path('D:/ImageNet/ILSVRC2012')        # Windows external drive
# ROOT = Path('E:/datasets/ImageNet/ILSVRC2012') # USB drive
# ROOT = Path('Z:/shared/ILSVRC2012')          # Network drive

# For testing with sample data:
# training_folder_name = './sample-data-train'  # 5-11 class subset
# val_folder_name = './sample-data-val'
```


### 3. Run the Notebook

Open `imagenet_poc_local.ipynb` in Jupyter:

```bash
jupyter notebook imagenet_poc_local.ipynb
```

Or in JupyterLab:

```bash
jupyter lab imagenet_poc_local.ipynb
```

### 4. Configure Parameters

Adjust these parameters in Cell 2 based on your hardware:

```python
class Params:
    def __init__(self):
        self.batch_size = 16       # Reduce if GPU memory is limited
        self.workers = 4           # Set to number of CPU cores
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr_step_size = 30
        self.lr_gamma = 0.1
```

### 5. Validate Setup (Optional but Recommended)

Before training, the notebook includes a validation step using a pretrained ResNet18 to verify your data pipeline is correct.

**Expected output:**
```
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to C:\Users\...\checkpoints\resnet18-f37072fd.pth
100%|██████████| 44.7M/44.7M [00:01<00:00, 40.0MB/s]

Test Error: 
 Accuracy: 69.5%, Avg loss: 1.254693 

Test Error: 
 Accuracy-5: 88.9%, Avg loss: 1.254693 

Elapsed: 137.06 seconds
```

**✅ Key Metrics:**
- **Top-1 Accuracy**: ~70% (pretrained ResNet18 on ImageNet validation)
- **Top-5 Accuracy**: ~89% (correct class in top 5 predictions)
- **Processing Time**: ~2-3 minutes on GPU (10+ minutes on CPU)

If you see similar results, your setup is working correctly! 🎉

### 6. Monitor Training

View training progress with TensorBoard:

```bash
tensorboard --logdir=runs
```

Then open http://localhost:6006 in your browser.

## Output Locations

- **Checkpoints**: `./checkpoints/resnet_152_sgd1_local/`
  - `checkpoint.pth` - Latest checkpoint (for resuming)
  - `model_{epoch}.pth` - Per-epoch checkpoints

- **TensorBoard Logs**: `./runs/resnet_152_sgd1_local/`
  - Training loss
  - Test accuracy (top-1 and top-5)
  - Test loss

## Model Architecture

- **Model**: ResNet152 (customizable number of classes)
- **Architecture**: `[3, 8, 36, 3]` bottleneck blocks
- **Parameters**: ~60M (depends on number of classes)

## Training Configuration

- **Optimizer**: SGD with momentum (0.9)
- **Initial LR**: 0.1
- **LR Schedule**: Step decay by 0.1 every 30 epochs
- **Weight Decay**: 1e-4
- **Epochs**: 100 (configurable)

## Expected Results

### Validation with Pretrained ResNet18

Before training your custom ResNet152, the notebook validates the setup with a pretrained ResNet18:

| Metric | Value | Description |
|--------|-------|-------------|
| **Top-1 Accuracy** | **69.5%** | Single best prediction is correct |
| **Top-5 Accuracy** | **88.9%** | Correct class in top 5 predictions |
| Test Loss | 1.254693 | Cross-entropy loss |
| Time (GPU) | ~137 seconds | On CUDA-enabled GPU |

These baseline results confirm:
- ✅ Data loading and transforms are correct
- ✅ GPU acceleration is working
- ✅ Model inference pipeline is functional
- ✅ Ready to train custom ResNet152

### ResNet152 Training (Expected)

Training ResNet152 from scratch on ImageNet typically achieves:
- **Top-1 Accuracy**: 76-78% (after 90 epochs)
- **Top-5 Accuracy**: 93-95% (after 90 epochs)
- **Training Time**: 3-7 days on single GPU (depends on hardware)

**Note**: Results vary based on dataset size, hyperparameters, and hardware.

## Resume Training

The notebook automatically resumes from the last checkpoint if:
1. `resume_training = True` (set in Cell 2)
2. A checkpoint exists at `./checkpoints/resnet_152_sgd1_local/checkpoint.pth`

To start fresh training:
- Set `resume_training = False`, or
- Delete/rename the checkpoint file

## Hardware Requirements

**Minimum**:
- GPU: 8GB VRAM (for batch_size=16)
- RAM: 16GB
- Storage: 50GB+ for data + checkpoints

**Recommended**:
- GPU: 16GB+ VRAM (for larger batches)
- RAM: 32GB+
- Storage: 100GB+

For CPU-only training:
- Reduce `batch_size` to 4-8
- Set `workers = 0` or `workers = 1`
- Training will be significantly slower

## Performance Benchmarks

### System Performance Reference

Based on the validation run with pretrained ResNet18:

**With GPU (CUDA 12.x + PyTorch 2.7+):**
- Validation time: ~137 seconds for 50,000 images
- Throughput: ~365 images/second
- Batch size: 64
- Workers: 4

**Expected Training Speed (ResNet152):**
- ~10-15 minutes per epoch (depends on dataset size)
- ~17-25 hours for 100 epochs (single GPU)

**CPU-only (slower):**
- Validation time: ~15-20 minutes for 50,000 images  
- Training: 10-20x slower than GPU
- Not recommended for full training

### GPU Memory Usage

| Batch Size | ResNet152 Memory | Recommended VRAM |
|------------|------------------|------------------|
| 8          | ~6 GB            | 8 GB             |
| 16         | ~10 GB           | 12 GB            |
| 32         | ~18 GB           | 24 GB            |
| 64         | ~32 GB           | 40 GB            |

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size` in Params class
- Reduce `workers` (number of data loader processes)
- Close other applications

### Data Not Found
- Check and update the `ROOT` path in Cell 12 to point to your ILSVRC2012 folder
- Ensure data is extracted and organized correctly
- Verify folder structure matches expected format:
  - `ILSVRC2012/train/` with 1000 class subfolders (WNIDs like n01440764)
  - `ILSVRC2012/val_sorted/` with 1000 class subfolders (50 images per class)
- If validation data is flat (not in class folders), you need to sort it:
  - Use the sorting script from the original notebook Cell 6
  - Requires `data/meta.mat` and `data/ILSVRC2012_validation_ground_truth.txt`
  - Or download pre-sorted validation data

### Validation Data Not Sorted
If you have flat validation images (`ILSVRC2012_val_*.JPEG` in single folder):
1. Ensure you have `devkit/data/meta.mat` and `devkit/data/ILSVRC2012_validation_ground_truth.txt`
2. Run the sorting script from original notebook Cell 6
3. This creates `val_sorted/` with 1000 class folders
4. Each folder will contain ~50 validation images for that class

### Checkpoint Load Error
- Ensure checkpoint was saved on compatible PyTorch version
- Check that model architecture matches
- Try deleting checkpoint and starting fresh

### CUDA Error: device-side assert triggered
This typically means class mismatch between model and data:
- **Cause**: Model outputs N classes, but data has labels 0 to M (where M > N-1)
- **Solution**: The notebook automatically filters validation data (Cell 18)
- **Check**: Ensure training and validation data have matching class counts
- Example: If training on 11 classes, validation should also use only those 11 classes

### Low Accuracy (<50%)
If validation accuracy is significantly lower than expected:
- Check data paths are correct
- Verify images are loading properly (run EDA cell)
- Ensure transforms match ImageNet preprocessing
- Confirm model architecture is correct

## File Comparison

| File | Description |
|------|-------------|
| `imagenet_poc.ipynb` | Original Colab version |
| `imagenet_poc_local.ipynb` | ✅ Local execution version |
| `requirements_imagenet_poc.txt` | Dependencies for both notebooks |

## Support

For issues specific to:
- **Data preparation**: Check ImageNet dataset documentation
- **PyTorch setup**: Visit https://pytorch.org/get-started/locally/
- **CUDA issues**: Verify CUDA installation and PyTorch CUDA version match

---

## Data Path Configuration

The notebook is configured to use ILSVRC2012 folder structure by default:

```python
ROOT = Path('./ILSVRC2012')
training_folder_name = str(ROOT / 'train')
val_folder_name = str(ROOT / 'val_sorted')
```

**Common configurations**:

1. **ILSVRC2012 in current directory**:
   ```python
   ROOT = Path('./ILSVRC2012')
   ```

2. **ILSVRC2012 on external drive**:
   ```python
   ROOT = Path('D:/ImageNet/ILSVRC2012')         # Windows

   ```

3. **Using sample data for testing** (5-11 class subset):
   ```python
   # Override ROOT-based paths with sample data folders
   training_folder_name = './ILSVRC2012/train'  # 5-11 classes for quick testing
   val_folder_name = './sample-data-val'         # Matching validation subset
   ```

4. **Network or external drive**:
   ```python
   ROOT = Path('Z:/shared/datasets/ILSVRC2012')  # Network drive

   ```

---

**Note**: The notebook automatically detects the number of classes from your dataset, so it works with both:
- **Full ILSVRC2012**: 1000 classes, 1,281,167 training images, 50,000 validation images
- **Subsets**: Any number of classes (e.g., 5-11 classes in `/ILSVRC2012/train/`)

---

## ILSVRC2012 Dataset Details

### Official Dataset Statistics

| Component | Count | Details |
|-----------|-------|---------|
| **Classes** | 1000 | Low-level WordNet synsets (leaf nodes) |
| **Training Images** | 1,281,167 | 732-1300 images per class |
| **Validation Images** | 50,000 | 50 images per class |
| **Test Images** | 100,000 | Not publicly available |

### File Naming Conventions

**Training Images:**
```
n01440764/
├── n01440764_18.JPEG
├── n01440764_36.JPEG
└── ... (732-1300 images)
```

**Validation Images (before sorting):**
```
ILSVRC2012_val_00000001.JPEG
ILSVRC2012_val_00000002.JPEG
...
ILSVRC2012_val_00050000.JPEG
```

**After Sorting:**
```
val_sorted/
├── n01440764/
│   ├── ILSVRC2012_val_00000293.JPEG
│   ├── ILSVRC2012_val_00002138.JPEG
│   └── ... (50 images)
├── n01443537/
└── ...
```

### Ground Truth Files

- **Training**: Implicit in folder structure (folder name = class WNID)
- **Validation**: `data/ILSVRC2012_validation_ground_truth.txt`
  - One ILSVRC2012_ID per line (1-1000)
  - Order matches alphabetical order of validation filenames
- **Meta Data**: `data/meta.mat`
  - Contains synset information, WNIDs, and class hierarchy
  - Used for validation sorting script

### Additional Resources

**Bounding Boxes** (optional):
- Available for all validation/test images and many training images
- In PASCAL VOC XML format
- Can be used for localization tasks
- Not required for classification training

**References:**
- J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database." IEEE CVPR, 2009.

