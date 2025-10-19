# TensorBoard Graphs

This folder contains screenshots from TensorBoard training visualizations.

## How to Generate Graphs

### Step 1: Start TensorBoard
```bash
cd z:\era-v4\era4-assign9
tensorboard --logdir=runs/resnet_152_sgd1_local
```

### Step 2: Open Browser
Navigate to: http://localhost:6006

### Step 3: Take Screenshots

Capture the following graphs and save them here:

1. **training_loss.png**
   - Graph: Training Loss (over steps/epochs)
   - Shows: Loss decreasing over time

2. **test_accuracy.png**
   - Graph: Test Accuracy (SCALARS tab)
   - Shows: Top-1 accuracy improving from 7.6% to 76.9%

3. **test_accuracy5.png**
   - Graph: Test Accuracy5 (SCALARS tab)
   - Shows: Top-5 accuracy reaching 99.1%

4. **test_loss.png**
   - Graph: Test Loss (SCALARS tab)
   - Shows: Validation loss converging to 0.574

### Step 4: Uncomment Image Links

After saving screenshots here, edit `../../logs.md` and uncomment the image sections (lines ~161-177):

Change from:
```markdown
<!--
### Training Loss
![Training Loss](docs/images/training_loss.png)
-->
```

To:
```markdown
### Training Loss
![Training Loss](docs/images/training_loss.png)
*Training loss decreased from 2.4 to ~0.5 over 100 epochs*
```

## Tips for Good Screenshots

- Use TensorBoard's zoom feature to focus on interesting regions
- Enable "smoothing" slider (0.6-0.8) for cleaner curves
- Capture at full resolution
- Include axis labels in the screenshot
- Use light theme for better visibility in documents

## Current Training Session

**Model**: ResNet152 ([3, 8, 36, 3])  
**Dataset**: ILSVRC2012 Subset (11 classes)  
**Training Completed**: October 18, 2025  
**Final Results**: Top-1: 76.9%, Top-5: 99.1%

---

**Generated**: October 19, 2025

