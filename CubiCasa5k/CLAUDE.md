# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CubiCasa5K is a floorplan image analysis pipeline. It trains a multi-task neural network to predict three outputs simultaneously from a floorplan image: **room segmentation** (12 classes), **icon segmentation** (11 classes), and **heatmaps** (21 channels for wall junctions, corners, and door/window endpoints).

The model is based on the Hourglass (Furukawa) architecture and uses a multi-task uncertainty loss (`UncertaintyLoss`) that learns per-task loss weighting automatically.

## Commands

### Data Preparation
```bash
# Download dataset from https://zenodo.org/record/2613548 and extract to data/cubicasa5k/
# Then build the LMDB cache (~105GB):
python create_lmdb.py --txt val.txt
python create_lmdb.py --txt test.txt
python create_lmdb.py --txt train.txt
```

### Training
```bash
python train.py
# Key options:
#   --arch hg_furukawa_original   (default)
#   --n-classes 44                (21 heatmap + 12 room + 11 icon)
#   --optimizer adam-patience-previous-best  (default)
#   --l-rate 1e-3
#   --batch-size 26
#   --image-size 256
#   --weights path/to/checkpoint.pkl   # resume from checkpoint
#   --furukawa-weights path/to/furukawa.pkl  # init from Furukawa pretrained
#   --scale      # enable resize+crop augmentation
#   --debug      # single worker, useful for debugging
#   --plot-samples  # write prediction images to TensorBoard

tensorboard --logdir runs_cubi/
```

### Evaluation
```bash
# Download weights from Google Drive (see README) or use a checkpoint
python eval.py --weights model_best_val_loss_var.pkl
# Results written to runs_cubi/<timestamp>/eval.log
```

### Docker
```bash
docker build -t cubi -f Dockerfile .
docker run --rm -it --init --runtime=nvidia --ipc=host \
  --publish 1111:1111 \
  --user="$(id -u):$(id -g)" \
  --volume=$PWD:/app \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  cubi jupyter-lab --port 1111 --ip 0.0.0.0 --no-browser
```

## Architecture

### Data Pipeline

**`floortrans/loaders/svg_loader.py` — `FloorplanSVG`**  
The main PyTorch `Dataset`. Supports two formats:
- `format='txt'`: Parses SVG files on-the-fly via `House` (slow, for debugging)
- `format='lmdb'`: Reads pre-serialized samples from the LMDB cache (fast, used for training)

Each sample is a dict: `{'image': (3,H,W), 'label': (23,H,W), 'heatmaps': dict, 'folder': str, 'scale': float}`.  
The label tensor has 23 channels: 21 heatmap channels + 1 room class index + 1 icon class index.

**`floortrans/loaders/house.py` — `House`**  
Parses `model.svg` annotation files. Maps the 80+ raw SVG room types down to the 12 training classes and 11 icon classes. Generates both segmentation tensors and 2D Gaussian heatmaps for structural points (junctions, corners, door/window endpoints).

**`floortrans/loaders/augmentations.py`**  
Augmentation pipeline operating on the sample dict. Key transforms: `RandomCropToSizeTorch`, `ResizePaddedTorch`, `RandomRotations`, `ColorJitterTorch`, `DictToTensor`.

### Model

**`floortrans/models/hg_furukawa_original.py` — `hg_furukawa_original`**  
Stacked Hourglass network. Takes a `(B,3,H,W)` image, outputs `(B,44,H/4,W/4)`.

Output channel layout: `[0:21]` heatmaps · `[21:33]` room logits · `[33:44]` icon logits.

When loading from Furukawa pretrained weights (51-class), the final `conv4_` and `upsample` layers are replaced to produce 44 outputs.

### Loss

**`floortrans/losses/uncertainty_loss.py` — `UncertaintyLoss`**  
Combines three task losses with learned log-variance weighting (Kendall et al. 2018):
- Heatmap: per-channel MSE with 21 learned `log_vars_mse` parameters
- Room segmentation: cross-entropy with 1 learned `log_vars[0]`
- Icon segmentation: cross-entropy with 1 learned `log_vars[1]`

The `criterion` parameters (log-vars) are optimized alongside model parameters and saved/loaded with the checkpoint.

### Post-Processing

**`floortrans/post_prosessing.py`**  
Converts raw model outputs into vector representation: detects wall lines from heatmaps, extracts polygon shapes for rooms/icons, and applies geometry cleanup (remove overlapping walls, fix corners). Used during evaluation to compute polygon-segmentation metrics.

### Metrics

**`floortrans/metrics.py` — `runningScore`**  
Confusion-matrix-based segmentation metrics: Overall Acc, Mean Acc, Mean IoU, FreqW Acc. The evaluation script reports both direct segmentation and polygon-segmentation variants for rooms and icons.

## Data Layout

```
data/cubicasa5k/
├── train.txt / val.txt / test.txt   # lists of sample folder paths
├── cubi_lmdb/                        # LMDB cache (created by create_lmdb.py)
└── <sample_folder>/
    ├── F1_scaled.png                 # input image (scaled to fit model)
    ├── F1_original.png              # original resolution image
    └── model.svg                    # polygon annotations
```

Checkpoints are saved to `runs_cubi/<timestamp>/` as:
- `model_best_val_loss_var.pkl` — best validation loss (with variance weighting, primary)
- `model_best_val_loss.pkl` — best validation loss (unweighted)
- `model_best_val_acc.pkl` — best mean pixel accuracy
- `model_last_epoch.pkl` — final epoch

## Key Implementation Notes

- The model outputs at **1/4 resolution** (`H/4 × W/4`). Labels are interpolated to match during both training (`F.interpolate` in the loss) and validation.
- `avg_loss` in the training loop is overwritten with `np.inf` after computation — this means `model_best_train_loss_var.pkl` is never actually updated (existing bug).
- The `format='lmdb'` loader sets `is_transform=False`, so image normalization to `[-1, 1]` is **not applied** when using LMDB. Normalization must be baked in during `create_lmdb.py` or handled externally.
- `samples.ipynb` demonstrates inference on individual floorplan images without needing the full dataset.
