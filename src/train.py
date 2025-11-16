#!/usr/bin/env python3
"""
Exact Paper Implementation Training Script
Multi-Focus Image Fusion using GAN with Multi-Scale Attention and Siamese Architecture

This script implements the EXACT training procedure as described in the paper with:
- Exact hyperparameters from paper
- Complete GAN training with all loss components
- Paper-specific evaluation metrics
- Exact learning rate scheduling
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import time

from models import SiameseMFIFGAN, VGGPerceptualLoss
from data_loader import create_data_loaders
from metrics import (
    SSIMLoss,
    calculate_l1_loss,
    evaluate_model_comprehensive,
    calculate_ssim,
    calculate_psnr,
)


def get_optimal_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class PaperExactTrainer:
    """Exact paper implementation trainer"""

    def __init__(self, config):
        self.config = config
        self.device = get_optimal_device()

        # Initialize model
        self.model = SiameseMFIFGAN(in_channels=3, out_channels=3).to(self.device)

        # Loss functions (exact paper weights)
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = VGGPerceptualLoss().to(self.device)
        self.adversarial_loss = nn.BCEWithLogitsLoss()

        # Loss weights from paper
        self.lambda_l1 = 1.0  # λ₁ (pixel)
        self.lambda_ssim = 0.5  # λ₂ (SSIM)
        self.lambda_perceptual = 0.1  # λ₃ (perceptual)
        self.lambda_adversarial = 0.01  # λ₄ (adversarial)

        # Optimizers (exact paper settings)
        self.optimizer_G = optim.Adam(
            self.model.generator.parameters(),
            lr=config["learning_rate"],
            betas=(0.5, 0.999),  # Paper uses β₁=0.5, β₂=0.999
        )

        self.optimizer_D = optim.Adam(
            self.model.discriminator.parameters(),
            lr=config["learning_rate"],
            betas=(0.5, 0.999),
        )

        # Learning rate schedulers (exact paper: decay=0.88 per 2 epochs)
        self.scheduler_G = optim.lr_scheduler.StepLR(
            self.optimizer_G, step_size=2, gamma=0.88
        )
        self.scheduler_D = optim.lr_scheduler.StepLR(
            self.optimizer_D, step_size=2, gamma=0.88
        )

        # Training state
        self.epoch = 0
        self.best_metrics = {"qnmi": 0, "qg": 0, "qcb": 0, "qpiella": 0, "ssim": 0}

        print(f"✅ Trainer initialized on device: {self.device}")
        print(
            f"📊 Generator parameters: {sum(p.numel() for p in self.model.generator.parameters()):,}"
        )
        print(
            f"📊 Discriminator parameters: {sum(p.numel() for p in self.model.discriminator.parameters()):,}"
        )

    def train_epoch(self, train_loader, epoch):
        """Train one epoch with exact paper procedure"""
        self.model.train()

        epoch_metrics = {
            "g_loss": 0,
            "d_loss": 0,
            "l1_loss": 0,
            "ssim_loss": 0,
            "perceptual_loss": 0,
            "adversarial_loss": 0,
        }

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')

        for batch_idx, batch in enumerate(pbar):
            source_a = batch["source_a"].to(self.device)
            source_b = batch["source_b"].to(self.device)
            ground_truth = batch["ground_truth"].to(self.device)

            batch_size = source_a.size(0)

            # Real and fake labels
            real_labels = torch.ones(batch_size, 1, 15, 15).to(
                self.device
            )  # PatchGAN output size
            fake_labels = torch.zeros(batch_size, 1, 15, 15).to(self.device)

            # ==========================================
            # Train Discriminator (exact paper procedure)
            # ==========================================
            self.optimizer_D.zero_grad()

            # Generate fused image
            with torch.no_grad():
                fused_image = self.model.generator(source_a, source_b)

            # Real loss (ground truth)
            real_pred = self.model.discriminator(ground_truth)
            d_real_loss = self.adversarial_loss(real_pred, real_labels)

            # Fake loss (generated)
            fake_pred = self.model.discriminator(fused_image.detach())
            d_fake_loss = self.adversarial_loss(fake_pred, fake_labels)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            self.optimizer_D.step()

            # ==========================================
            # Train Generator (exact paper procedure)
            # ==========================================
            self.optimizer_G.zero_grad()

            # Generate fused image
            fused_image = self.model.generator(source_a, source_b)

            # 1. L1 Loss (λ₁ = 1.0)
            l1_loss = self.l1_loss(fused_image, ground_truth)

            # 2. SSIM Loss (λ₂ = 0.5)
            ssim_loss = self.ssim_loss(fused_image, ground_truth)

            # 3. Perceptual Loss (λ₃ = 0.1)
            perceptual_loss = self.perceptual_loss(fused_image, ground_truth)

            # 4. Adversarial Loss (λ₄ = 0.01)
            fake_pred = self.model.discriminator(fused_image)
            adversarial_loss = self.adversarial_loss(fake_pred, real_labels)

            # Total generator loss (exact paper formulation)
            g_loss = (
                self.lambda_l1 * l1_loss
                + self.lambda_ssim * ssim_loss
                + self.lambda_perceptual * perceptual_loss
                + self.lambda_adversarial * adversarial_loss
            )

            g_loss.backward()
            self.optimizer_G.step()

            # Update metrics
            epoch_metrics["g_loss"] += g_loss.item()
            epoch_metrics["d_loss"] += d_loss.item()
            epoch_metrics["l1_loss"] += l1_loss.item()
            epoch_metrics["ssim_loss"] += ssim_loss.item()
            epoch_metrics["perceptual_loss"] += perceptual_loss.item()
            epoch_metrics["adversarial_loss"] += adversarial_loss.item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "G_Loss": f"{g_loss.item():.4f}",
                    "D_Loss": f"{d_loss.item():.4f}",
                    "L1": f"{l1_loss.item():.4f}",
                    "SSIM": f"{ssim_loss.item():.4f}",
                    "Perc": f"{perceptual_loss.item():.4f}",
                    "Adv": f"{adversarial_loss.item():.4f}",
                }
            )

        # Average epoch metrics
        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_metrics

    def validate_epoch(self, val_loader, epoch):
        """Validate with comprehensive paper metrics"""
        print("🔍 Running comprehensive validation...")

        # Comprehensive evaluation with all paper metrics
        metrics = evaluate_model_comprehensive(
            self.model, val_loader, self.device, num_samples=500  # Limit for speed
        )

        return metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "generator_state_dict": self.model.generator.state_dict(),
            "discriminator_state_dict": self.model.discriminator.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
            "scheduler_G_state_dict": self.scheduler_G.state_dict(),
            "scheduler_D_state_dict": self.scheduler_D.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config["checkpoint_dir"], f"checkpoint_epoch_{epoch+1}.pth"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.config["checkpoint_dir"], "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved: {best_path}")

    def train(self, train_loader, val_loader):
        """Main training loop with exact paper procedure"""
        print("🚀 Starting training with EXACT paper implementation...")
        print(f"📋 Hyperparameters:")
        print(f"   • Learning Rate: {self.config['learning_rate']}")
        print(f"   • Batch Size: {self.config['batch_size']}")
        print(f"   • Epochs: {self.config['num_epochs']}")
        print(
            f"   • Loss Weights: L1={self.lambda_l1}, SSIM={self.lambda_ssim}, Perc={self.lambda_perceptual}, Adv={self.lambda_adversarial}"
        )
        print(f"   • LR Decay: 0.88 every 2 epochs")

        for epoch in range(self.config["num_epochs"]):
            start_time = time.time()

            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validation phase (every 5 epochs for speed)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_metrics = self.validate_epoch(val_loader, epoch)
            else:
                val_metrics = {"qnmi": 0, "qg": 0, "qcb": 0, "qpiella": 0, "ssim": 0}

            # Learning rate scheduling
            self.scheduler_G.step()
            self.scheduler_D.step()

            # Check if best model (based on combined paper metrics)
            current_score = (
                val_metrics["qnmi"]
                + val_metrics["qg"]
                + val_metrics["qcb"]
                + val_metrics["qpiella"]
                + val_metrics["ssim"]
            ) / 5
            best_score = (
                self.best_metrics["qnmi"]
                + self.best_metrics["qg"]
                + self.best_metrics["qcb"]
                + self.best_metrics["qpiella"]
                + self.best_metrics["ssim"]
            ) / 5

            is_best = current_score > best_score
            if is_best:
                self.best_metrics = val_metrics.copy()

            # Save checkpoint
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)

            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\n📊 Epoch {epoch+1}/{self.config['num_epochs']} Summary:")
            print(f"   ⏱️  Time: {epoch_time:.2f}s")
            print(
                f"   🔥 G_Loss: {train_metrics['g_loss']:.4f}, D_Loss: {train_metrics['d_loss']:.4f}"
            )
            print(
                f"   📈 L1: {train_metrics['l1_loss']:.4f}, SSIM: {train_metrics['ssim_loss']:.4f}"
            )
            print(
                f"   🎨 Perceptual: {train_metrics['perceptual_loss']:.4f}, Adversarial: {train_metrics['adversarial_loss']:.4f}"
            )

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   📊 Paper Metrics:")
                print(f"      • QNMI: {val_metrics['qnmi']:.4f}")
                print(f"      • QG: {val_metrics['qg']:.4f}")
                print(f"      • QCB: {val_metrics['qcb']:.4f}")
                print(f"      • QPiella: {val_metrics['qpiella']:.4f}")
                print(f"      • SSIM: {val_metrics['ssim']:.4f}")
                print(f"      • PSNR: {val_metrics['psnr']:.2f} dB")

            print(
                f"   📚 LR: G={self.optimizer_G.param_groups[0]['lr']:.6f}, D={self.optimizer_D.param_groups[0]['lr']:.6f}"
            )

            if is_best:
                print("   🏆 NEW BEST MODEL!")

            print("-" * 80)

        print("🎉 Training completed!")
        print(f"🏆 Best metrics achieved:")
        for metric, value in self.best_metrics.items():
            print(f"   • {metric.upper()}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser(description="EXACT Paper Implementation Training")

    # Dataset paths
    parser.add_argument(
        "--source_a_dir",
        type=str,
        default="dataset/sourceA",
        help="Path to source A images",
    )
    parser.add_argument(
        "--source_b_dir",
        type=str,
        default="dataset/sourceB",
        help="Path to source B images",
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        default="dataset/groundtruth",
        help="Path to ground truth images",
    )

    # EXACT Paper Hyperparameters
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size (EXACT paper value)"
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Image size (EXACT paper value)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of epochs (EXACT paper value)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0005,
        help="Learning rate (EXACT paper value)",
    )

    # Training settings
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Limit dataset size for testing"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory"
    )

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Configuration
    config = {
        "source_a_dir": args.source_a_dir,
        "source_b_dir": args.source_b_dir,
        "ground_truth_dir": args.ground_truth_dir,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "max_samples": args.max_samples,
        "checkpoint_dir": args.checkpoint_dir,
    }

    print("🔬 EXACT Paper Implementation - Multi-Focus Image Fusion")
    print("=" * 80)
    print(f"📁 Dataset: {args.source_a_dir}")
    print(f"🎯 Batch Size: {args.batch_size} (paper exact)")
    print(f"📏 Image Size: {args.image_size}×{args.image_size} (paper exact)")
    print(f"🔄 Epochs: {args.num_epochs} (paper exact)")
    print(f"📈 Learning Rate: {args.learning_rate} (paper exact)")
    print(f"📉 LR Decay: 0.88 every 2 epochs (paper exact)")
    print(
        f"⚖️  Loss Weights: L1=1.0, SSIM=0.5, Perceptual=0.1, Adversarial=0.01 (paper exact)"
    )
    print("=" * 80)

    # Create data loaders
    print("📂 Loading dataset...")
    train_loader, val_loader = create_data_loaders(
        source_a_dir=config["source_a_dir"],
        source_b_dir=config["source_b_dir"],
        ground_truth_dir=config["ground_truth_dir"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        max_samples=config.get("max_samples", None),
    )

    print(f"✅ Training samples: {len(train_loader.dataset):,}")
    print(f"✅ Validation samples: {len(val_loader.dataset):,}")

    # Initialize trainer and start training
    trainer = PaperExactTrainer(config)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
