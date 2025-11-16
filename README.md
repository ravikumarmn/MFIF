# Multi-Focus Image Fusion (MFIF)

A PyTorch implementation of Multi-Focus Image Fusion using Siamese Networks with attention mechanisms.

## Overview

This project implements a simplified version of the Multi-Focus Image Fusion system described in the project requirements. The Phase 1 implementation focuses on the core Siamese architecture with basic fusion capabilities.

### Key Features

- **Siamese Encoder Architecture**: Shared encoder for symmetric feature extraction
- **GenClean Block (GCB)**: Denoising and preprocessing module
- **Attention-based Fusion**: Multi-scale attention fusion at different feature levels
- **Comprehensive Training Pipeline**: L1 + SSIM loss with tensorboard logging
- **Evaluation Metrics**: SSIM, PSNR, and L1 loss calculations
- **Inference Tools**: Single image and batch processing capabilities

## Project Structure

```
MFIF/
├── src/
│   ├── models.py              # Model architectures
│   ├── data_loader.py         # Dataset handling
│   ├── train.py               # Training script (full dataset)
│   ├── inference.py           # Inference script
│   └── metrics.py             # Evaluation metrics
├── test/
│   ├── test_setup.py          # Architecture testing
│   ├── test_single_image.py   # Model inference testing
│   └── README.md              # Test documentation
├── dataset/
│   ├── sourceA/               # Near-focus images
│   ├── sourceB/               # Far-focus images
│   └── groundtruth/           # All-focus ground truth
├── checkpoints/               # Saved model checkpoints
├── logs/                      # Training logs
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Installation

1. **Clone the repository** (if applicable) or ensure you're in the project directory
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Verify setup**:

   ```bash
   python test_setup.py
   ```

## Dataset Structure

The dataset should follow this structure:

- `dataset/sourceA/`: Images with near focus
- `dataset/sourceB/`: Images with far focus
- `dataset/groundtruth/`: All-focus ground truth images

Each image triplet should have the same filename (e.g., `2007_000032_1.jpg`).

## Usage

### Training

Basic training with default parameters:

```bash
python src/train.py
```

Training with custom parameters:

```bash
python src/train.py --batch_size 8 --num_epochs 200 --learning_rate 0.0005
```

### Inference

Single image fusion:

```bash
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --source_a dataset/sourceA/2007_000032_1.jpg \
    --source_b dataset/sourceB/2007_000032_1.jpg \
    --output results/fused_result.jpg
```

Batch processing:

```bash
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --source_a_dir dataset/sourceA \
    --source_b_dir dataset/sourceB \
    --output_dir results/
```

## Model Architecture

### Siamese Encoder

- Shared encoder for both input images
- GenClean Block for initial denoising
- Multi-scale feature extraction (64, 128, 256, 512 channels)
- Skip connections for better gradient flow

### Attention Fusion

- Channel-wise attention at each feature level
- Learnable fusion weights between source images
- Multi-scale fusion from coarse to fine features

### Decoder

- Progressive upsampling with skip connections
- Feature fusion at multiple scales
- Final output with Tanh activation

## Training Configuration

Default training parameters:

- **Batch Size**: 8
- **Image Size**: 256×256
- **Learning Rate**: 0.0005
- **Epochs**: 100
- **Loss Weights**: L1 (1.0) + SSIM (0.5)
- **Optimizer**: Adam with StepLR scheduling

## Evaluation Metrics

- **SSIM**: Structural Similarity Index
- **PSNR**: Peak Signal-to-Noise Ratio
- **L1 Loss**: Mean Absolute Error

## Future Enhancements

1. **Advanced Attention Mechanisms**

   - Implement DFEB (Deep Feature Extraction Block)
   - Add DCA (Dynamic Channel Adjustment)
   - Multi-scale dilated attention
2. **GAN Integration**

   - PatchGAN discriminator
   - Adversarial loss for perceptual realism
   - Enhanced texture preservation
3. **Advanced Loss Functions**

   - VGG perceptual loss
   - Feature matching loss
   - Gradient-based losses
4. **Performance Optimizations**

   - Model pruning and quantization
   - TensorRT optimization
   - Multi-GPU training support

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size or image size
2. **Dataset not found**: Ensure dataset structure matches requirements
3. **Poor fusion quality**: Try adjusting loss weights or learning rate
4. **Training instability**: Reduce learning rate or add gradient clipping

## Contributing

This is an implementation. Future contributions should focus on:

- Adding advanced attention mechanisms
- Implementing GAN-based training
- Improving evaluation metrics
- Adding more robust data augmentation

## License

This project is part of the MFIF research implementation.
