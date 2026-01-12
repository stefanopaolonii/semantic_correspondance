# Semantic Correspondence with Foundation Models

A comprehensive comparative study of vision foundation models (DINOv2, DINOv3, SAM, CLIP) for semantic correspondence, evaluating their effectiveness in matching keypoints across images with different object instances from the same category.

## Overview

This project implements and evaluates **four state-of-the-art vision foundation models** on the semantic correspondence task:

- **DINOv2** (Self-supervised Vision Transformer)
- **DINOv3** (Enhanced DINO with improved pretraining)
- **SAM** (Segment Anything Model)
- **CLIP** (Vision-Language Model)

### Key Features

- **Baseline Evaluation**: Training-free correspondence using direct feature matching
- **Fine-tuning**: Selective layer fine-tuning with keypoint supervision using Gaussian Cross-Entropy Loss
- **Multiple Matching Strategies**:
  - **Argmax**: Direct peak similarity matching
  - **Windowed Soft-argmax**: Refined prediction with local smoothing
- **Comprehensive Metrics**: PCK (Percentage of Correct Keypoints) evaluation by image and by point
- **Three Benchmark Datasets**: SPair-71k, PF-PASCAL, PF-WILLOW

### Evaluation Protocol

The project follows a standard evaluation protocol with four phases:

1. **Phase 1**: Baseline evaluation with pretrained models (argmax matching)
2. **Phase 2**: Fine-tuning of models with keypoint supervision
3. **Phase 3**: Evaluation of baseline and fine-tuned models with windowed soft-argmax matching
4. **Phase 4**: Cross-dataset generalization evaluation

---

## Environment Setup

```bash
pip install -r requirements.txt
```

---

## Data Preparation

Download the datasets you need under the `asset/dataset/` folder.

### Automated Download

Use the automated script to download all datasets:
```bash
python tools/download_datasets.py
```

Or download specific datasets:
```bash
python tools/download_datasets.py --dataset spair
python tools/download_datasets.py --dataset pfpascal
python tools/download_datasets.py --dataset pfwillow
```

### Manual Download

#### SPair-71k

1. Download SPair-71k dataset from the [official link](https://cvlab.postech.ac.kr/research/SPair-71k/).
2. Extract to `asset/dataset/SPair-71k/`

### PF-PASCAL

1. Download PF-PASCAL dataset from [link](https://www.di.ens.fr/willow/research/proposalflow/).
2. Rename the outermost directory from `PF-dataset-PASCAL` to `pf-pascal`.
3. Download lists for image pairs from [link](https://www.robots.ox.ac.uk/~xinghui/sd4match/pf-pascal_image_pairs.zip).
4. Place the lists for image pairs under `pf-pascal` directory.

### PF-WILLOW

1. Download PF-Willow dataset from the [link](https://www.di.ens.fr/willow/research/proposalflow/).
2. Rename the outermost directory from `PF-dataset` to `pf-willow`.
3. Download lists for image pairs from [link](https://www.robots.ox.ac.uk/~xinghui/sd4match/test_pairs.csv).
4. Place the lists for image pairs under `pf-willow` directory.

### Expected Directory Structure

```
asset/
├── dataset/
│   ├── SPair-71k/
│   │   ├── ImageAnnotation/
│   │   ├── JPEGImages/
│   │   ├── Layout/
│   │   ├── PairAnnotation/
│   │   └── ...
│   ├── pf-pascal/
│   │   ├── PF-dataset-PASCAL/
│   │   │   ├── Annotations/
│   │   │   └── JPEGImages/
│   │   ├── test_pairs.csv
│   │   ├── trn_pairs.csv
│   │   └── val_pairs.csv
│   └── pf-willow/
│       ├── PF-dataset/
│       └── test_pairs.csv
└── weights/
    ├── dinov2/
    ├── dinov3/
    ├── sam/
    └── clip/
```

### Download Model Weights

#### Automated Download

Use the automated script to download all weights:
```bash
python tools/download_weights.py
```

Or download specific models:
```bash
python tools/download_weights.py --model dinov2
python tools/download_weights.py --model sam
python tools/download_weights.py --model clip
```

**Note**: DINOv3 weights must be downloaded manually from the official repository as they require authentication or are not available through direct download links.

#### Manual Download

Weights can be manually downloaded from the official repositories:

- **DINOv2**: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) → `asset/weights/dinov2/`
- **DINOv3**: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) → `asset/weights/dinov3/`
- **SAM**: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) → `asset/weights/sam/`
- **CLIP**: [openai/CLIP](https://github.com/openai/CLIP) → `asset/weights/clip/`

#### Fine-tuned Weights

Pre-trained fine-tuned weights (trained on SPair-71k) are available for download:

- **Fine-tuned models** (DINOv2, SAM, CLIP): [Download Link](https://drive.google.com/drive/folders/1BhwwRF6wbmyAWvyPnBgbVmXxdRge2mWy?usp=sharing) → `asset/weights/{model}/trained/`

**Note**: Fine-tuned DINOv3 weights cannot be publicly distributed due to Meta AI's licensing restrictions on the pretrained model.

---

## Model Architectures

All supported models and their variants (including fine-tuned versions with 2, 4, or 6 layers) are defined in their respective adapter classes:

- **DINOv2**: `src/dinov2_adapter.py` - Vision Transformer variants (vits14, vitb14, vitl14)
- **DINOv3**: `src/dinov3_adapter.py` - Vision Transformer variants (vits16, vitb16, vitl16)
- **SAM**: `src/sam_adapter.py` - Segment Anything Model variants (vitb16, vitl16, vith16)
- **CLIP**: `src/clip_adapter.py` - Vision-Language Model variants (vitb32, vitb16, vitl14)

Fine-tuned model architectures are specified using the suffix `_ft2`, `_ft4`, or `_ft6` (e.g., `vits14_ft2`) to indicate the number of last layers fine-tuned.

---

## Experimental Protocol

### Phase 1: Baseline Evaluation

Evaluate pretrained models with argmax matching:

```bash
python evaluate.py \
    --phase 1 \
    --model dinov2 \
    --model_arch vits14 \
    --dataset spair \
    --split test \
    --match_method argmax
```

For all available evaluation parameters, run:
```bash
python evaluate.py --help
```

All evaluation configurations are available in `config/base.py`.

### Phase 2: Fine-tuning

Train models with keypoint supervision using Gaussian Cross-Entropy Loss:

#### Training

```bash
python train.py \
    --model dinov2 \
    --model_arch vits14 \
    --fine_tune_layers 2 \
    --dataset spair \
    --epochs 3
```

For all available training parameters, run:
```bash
python train.py --help
```

All training configurations are available in `config/base.py`.

**Fine-tuned weights are saved to:** `asset/weights/{model}/trained/`

#### Evaluation

After training, evaluate fine-tuned models with argmax matching:

```bash
python evaluate.py \
    --phase 2 \
    --model dinov2 \
    --model_arch vits14_ft2 \
    --dataset spair \
    --split test \
    --match_method argmax
```

### Phase 3: Windowed Soft-argmax Evaluation

Evaluate baseline and fine-tuned models with windowed soft-argmax matching:

```bash
# Baseline
python evaluate.py \
    --phase 3 \
    --model dinov2 \
    --model_arch vits14 \
    --dataset spair \
    --split test \
    --match_method windowed_softargmax

# Fine-tuned
python evaluate.py \
    --phase 3 \
    --model dinov2 \
    --model_arch vits14_ft2 \
    --dataset spair \
    --split test \
    --match_method windowed_softargmax
```

### Phase 4: Cross-Dataset Generalization

Evaluate baseline and fine-tuned models on PF-Pascal and PF-Willow datasets:

```bash
# Baseline
python evaluate.py \
    --phase 4 \
    --model dinov2 \
    --model_arch vits14 \
    --dataset pfpascal \
    --split test \
    --match_method windowed_softargmax

# Fine-tuned
python evaluate.py \
    --phase 4 \
    --model dinov2 \
    --model_arch vits14_ft2 \
    --dataset pfwillow \
    --split test \
    --match_method windowed_softargmax
```

---

## Results Output

**Results Location:** `asset/results/{model}/phase{phase}/{arch}/{dataset}/`

**Generated Reports:**
- `phase{X}_{method}_image_report.txt`: PCK by image
- `phase{X}_{method}_kpt_report.txt`: PCK by keypoint

---

## Visualization

Generate visualizations of semantic correspondence predictions:

```bash
python visualization_pairs.py \
    --model dinov2 \
    --model_arch vitb14 \
    --model_arch_tuned vitb14_ft2 \
    --dataset spair \
    --split test \
    --sample_idx 1644 \
    --match_method argmax \
    --match_method_tuned windowed_softargmax
```

This will generate visualizations showing:
- Source and target images
- Ground truth correspondences
- Predicted correspondences (baseline and/or tuned)

---

## Acknowledgements

Parts of this codebase (dataset loaders, loss function, and evaluation utilities) are adapted from the [SD4Match repository](https://github.com/ActiveVisionLab/SD4Match).

**Paper**: "SD4Match: Learning to Prompt Stable Diffusion Model for Semantic Matching" (CVPR 2024)

This project uses the following foundation models:
- **DINOv2**: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- **DINOv3**: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- **SAM**: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
- **CLIP**: [openai/CLIP](https://github.com/openai/CLIP)

---

## License

This project is for **academic and educational purposes only**.

---

**Last Updated**: January 2026
