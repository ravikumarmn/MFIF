#!/usr/bin/env python3
"""
Comprehensive plotting utilities for Multi-Focus Image Fusion training
Saves all training metrics, losses, and scores as graphs to results folder
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import json

# Set style for better looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class TrainingPlotter:
    """Comprehensive training metrics plotter"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.plots_dir = os.path.join(results_dir, "training_plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        # Initialize metric storage
        self.training_history = {
            # Training losses
            "epochs": [],
            "g_loss": [],
            "d_loss": [],
            "l1_loss": [],
            "ssim_loss": [],
            "perceptual_loss": [],
            "adversarial_loss": [],
            "total_loss": [],
            # Validation metrics (paper metrics)
            "val_qnmi": [],
            "val_qg": [],
            "val_qcb": [],
            "val_qpiella": [],
            "val_ssim": [],
            "val_psnr": [],
            "val_combined_score": [],
            # Learning rates
            "lr_generator": [],
            "lr_discriminator": [],
            # Training time
            "epoch_time": [],
        }

        print(
            f"📊 TrainingPlotter initialized. Plots will be saved to: {self.plots_dir}"
        )

    def update_metrics(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        lr_g: float,
        lr_d: float,
        epoch_time: float,
    ):
        """Update training history with new metrics"""
        self.training_history["epochs"].append(epoch + 1)

        # Training losses
        self.training_history["g_loss"].append(train_metrics["g_loss"])
        self.training_history["d_loss"].append(train_metrics["d_loss"])
        self.training_history["l1_loss"].append(train_metrics["l1_loss"])
        self.training_history["ssim_loss"].append(train_metrics["ssim_loss"])
        self.training_history["perceptual_loss"].append(
            train_metrics["perceptual_loss"]
        )
        self.training_history["adversarial_loss"].append(
            train_metrics["adversarial_loss"]
        )
        self.training_history["total_loss"].append(train_metrics["g_loss"])

        # Validation metrics
        self.training_history["val_qnmi"].append(val_metrics.get("qnmi", 0))
        self.training_history["val_qg"].append(val_metrics.get("qg", 0))
        self.training_history["val_qcb"].append(val_metrics.get("qcb", 0))
        self.training_history["val_qpiella"].append(val_metrics.get("qpiella", 0))
        self.training_history["val_ssim"].append(val_metrics.get("ssim", 0))
        self.training_history["val_psnr"].append(val_metrics.get("psnr", 0))

        # Combined score
        combined_score = (
            val_metrics.get("qnmi", 0)
            + val_metrics.get("qg", 0)
            + val_metrics.get("qcb", 0)
            + val_metrics.get("qpiella", 0)
            + val_metrics.get("ssim", 0)
        ) / 5
        self.training_history["val_combined_score"].append(combined_score)

        # Learning rates
        self.training_history["lr_generator"].append(lr_g)
        self.training_history["lr_discriminator"].append(lr_d)

        # Training time
        self.training_history["epoch_time"].append(epoch_time)

    def plot_training_losses(self):
        """Plot all training losses"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Training Losses Over Time", fontsize=16, fontweight="bold")

        epochs = self.training_history["epochs"]

        # Generator vs Discriminator Loss
        axes[0, 0].plot(
            epochs, self.training_history["g_loss"], label="Generator Loss", linewidth=2
        )
        axes[0, 0].plot(
            epochs,
            self.training_history["d_loss"],
            label="Discriminator Loss",
            linewidth=2,
        )
        axes[0, 0].set_title("Generator vs Discriminator Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # L1 Loss
        axes[0, 1].plot(
            epochs, self.training_history["l1_loss"], color="red", linewidth=2
        )
        axes[0, 1].set_title("L1 Loss (Pixel-wise)")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("L1 Loss")
        axes[0, 1].grid(True, alpha=0.3)

        # SSIM Loss
        axes[0, 2].plot(
            epochs, self.training_history["ssim_loss"], color="green", linewidth=2
        )
        axes[0, 2].set_title("SSIM Loss (Structural)")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("SSIM Loss")
        axes[0, 2].grid(True, alpha=0.3)

        # Perceptual Loss
        axes[1, 0].plot(
            epochs,
            self.training_history["perceptual_loss"],
            color="purple",
            linewidth=2,
        )
        axes[1, 0].set_title("Perceptual Loss (VGG)")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Perceptual Loss")
        axes[1, 0].grid(True, alpha=0.3)

        # Adversarial Loss
        axes[1, 1].plot(
            epochs,
            self.training_history["adversarial_loss"],
            color="orange",
            linewidth=2,
        )
        axes[1, 1].set_title("Adversarial Loss")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Adversarial Loss")
        axes[1, 1].grid(True, alpha=0.3)

        # Combined Loss Components
        axes[1, 2].plot(epochs, self.training_history["l1_loss"], label="L1", alpha=0.7)
        axes[1, 2].plot(
            epochs, self.training_history["ssim_loss"], label="SSIM", alpha=0.7
        )
        axes[1, 2].plot(
            epochs,
            self.training_history["perceptual_loss"],
            label="Perceptual",
            alpha=0.7,
        )
        axes[1, 2].plot(
            epochs,
            self.training_history["adversarial_loss"],
            label="Adversarial",
            alpha=0.7,
        )
        axes[1, 2].set_title("All Loss Components")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Loss")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "training_losses.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_validation_metrics(self):
        """Plot all validation metrics (paper metrics)"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Validation Metrics (Paper Metrics) Over Time",
            fontsize=16,
            fontweight="bold",
        )

        epochs = self.training_history["epochs"]

        # QNMI (Normalized Mutual Information)
        axes[0, 0].plot(
            epochs,
            self.training_history["val_qnmi"],
            color="blue",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        axes[0, 0].set_title("QNMI (Information Preservation)")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("QNMI Score")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)

        # QG (Gradient-based)
        axes[0, 1].plot(
            epochs,
            self.training_history["val_qg"],
            color="red",
            linewidth=2,
            marker="s",
            markersize=4,
        )
        axes[0, 1].set_title("QG (Edge Preservation)")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("QG Score")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)

        # QCB (Correlation Coefficient-based)
        axes[0, 2].plot(
            epochs,
            self.training_history["val_qcb"],
            color="green",
            linewidth=2,
            marker="^",
            markersize=4,
        )
        axes[0, 2].set_title("QCB (Correlation)")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("QCB Score")
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1)

        # QPiella (Piella's metric)
        axes[1, 0].plot(
            epochs,
            self.training_history["val_qpiella"],
            color="purple",
            linewidth=2,
            marker="d",
            markersize=4,
        )
        axes[1, 0].set_title("QPiella (Structural Similarity)")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("QPiella Score")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)

        # SSIM and PSNR
        ax1 = axes[1, 1]
        ax2 = ax1.twinx()

        line1 = ax1.plot(
            epochs,
            self.training_history["val_ssim"],
            color="orange",
            linewidth=2,
            marker="o",
            markersize=4,
            label="SSIM",
        )
        line2 = ax2.plot(
            epochs,
            self.training_history["val_psnr"],
            color="brown",
            linewidth=2,
            marker="s",
            markersize=4,
            label="PSNR",
        )

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("SSIM Score", color="orange")
        ax2.set_ylabel("PSNR (dB)", color="brown")
        ax1.set_title("SSIM & PSNR")
        ax1.grid(True, alpha=0.3)

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left")

        # Combined Score
        axes[1, 2].plot(
            epochs,
            self.training_history["val_combined_score"],
            color="black",
            linewidth=3,
            marker="*",
            markersize=6,
        )
        axes[1, 2].set_title("Combined Paper Metrics Score")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Combined Score")
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "validation_metrics.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_learning_rates(self):
        """Plot learning rate schedules"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Learning Rate Schedules", fontsize=16, fontweight="bold")

        epochs = self.training_history["epochs"]

        # Generator and Discriminator LR
        ax1.plot(
            epochs,
            self.training_history["lr_generator"],
            label="Generator LR",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        ax1.plot(
            epochs,
            self.training_history["lr_discriminator"],
            label="Discriminator LR",
            linewidth=2,
            marker="s",
            markersize=4,
        )
        ax1.set_title("Learning Rates Over Time")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Learning Rate")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        # Training time per epoch
        ax2.plot(
            epochs,
            self.training_history["epoch_time"],
            color="red",
            linewidth=2,
            marker="^",
            markersize=4,
        )
        ax2.set_title("Training Time per Epoch")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Time (seconds)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "learning_rates_and_time.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_comprehensive_overview(self):
        """Create a comprehensive overview plot"""
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        epochs = self.training_history["epochs"]

        # Main losses (top row)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(
            epochs, self.training_history["g_loss"], label="Generator", linewidth=2
        )
        ax1.plot(
            epochs, self.training_history["d_loss"], label="Discriminator", linewidth=2
        )
        ax1.set_title("Main Losses")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.training_history["l1_loss"], color="red", linewidth=2)
        ax2.set_title("L1 Loss")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(epochs, self.training_history["ssim_loss"], color="green", linewidth=2)
        ax3.set_title("SSIM Loss")
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[0, 3])
        ax4.plot(
            epochs,
            self.training_history["val_combined_score"],
            color="black",
            linewidth=3,
        )
        ax4.set_title("Combined Score")
        ax4.grid(True, alpha=0.3)

        # Paper metrics (middle row)
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.plot(
            epochs,
            self.training_history["val_qnmi"],
            linewidth=2,
            marker="o",
            markersize=3,
        )
        ax5.set_title("QNMI")
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[1, 1])
        ax6.plot(
            epochs,
            self.training_history["val_qg"],
            linewidth=2,
            marker="s",
            markersize=3,
        )
        ax6.set_title("QG")
        ax6.grid(True, alpha=0.3)

        ax7 = fig.add_subplot(gs[1, 2])
        ax7.plot(
            epochs,
            self.training_history["val_qcb"],
            linewidth=2,
            marker="^",
            markersize=3,
        )
        ax7.set_title("QCB")
        ax7.grid(True, alpha=0.3)

        ax8 = fig.add_subplot(gs[1, 3])
        ax8.plot(
            epochs,
            self.training_history["val_qpiella"],
            linewidth=2,
            marker="d",
            markersize=3,
        )
        ax8.set_title("QPiella")
        ax8.grid(True, alpha=0.3)

        # Additional metrics (bottom row)
        ax9 = fig.add_subplot(gs[2, 0])
        ax9.plot(
            epochs,
            self.training_history["val_ssim"],
            linewidth=2,
            marker="o",
            markersize=3,
        )
        ax9.set_title("SSIM")
        ax9.grid(True, alpha=0.3)

        ax10 = fig.add_subplot(gs[2, 1])
        ax10.plot(
            epochs,
            self.training_history["val_psnr"],
            linewidth=2,
            marker="s",
            markersize=3,
        )
        ax10.set_title("PSNR (dB)")
        ax10.grid(True, alpha=0.3)

        ax11 = fig.add_subplot(gs[2, 2])
        ax11.plot(
            epochs, self.training_history["lr_generator"], linewidth=2, label="Gen"
        )
        ax11.plot(
            epochs, self.training_history["lr_discriminator"], linewidth=2, label="Disc"
        )
        ax11.set_title("Learning Rates")
        ax11.legend()
        ax11.grid(True, alpha=0.3)
        ax11.set_yscale("log")

        ax12 = fig.add_subplot(gs[2, 3])
        ax12.plot(epochs, self.training_history["epoch_time"], linewidth=2, color="red")
        ax12.set_title("Epoch Time (s)")
        ax12.grid(True, alpha=0.3)

        plt.suptitle(
            "Multi-Focus Image Fusion - Training Overview",
            fontsize=20,
            fontweight="bold",
        )
        plt.savefig(
            os.path.join(self.plots_dir, "comprehensive_overview.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def save_metrics_csv(self):
        """Save all metrics to CSV for further analysis"""
        df = pd.DataFrame(self.training_history)
        csv_path = os.path.join(self.results_dir, "training_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"📊 Metrics saved to CSV: {csv_path}")

    def save_metrics_json(self):
        """Save all metrics to JSON"""
        json_path = os.path.join(self.results_dir, "training_metrics.json")
        with open(json_path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        print(f"📊 Metrics saved to JSON: {json_path}")

    def generate_all_plots(self):
        """Generate all plots and save data"""
        print("📊 Generating comprehensive training plots...")

        if len(self.training_history["epochs"]) == 0:
            print("⚠️  No training data to plot!")
            return

        # Generate all plots
        self.plot_training_losses()
        self.plot_validation_metrics()
        self.plot_learning_rates()
        self.plot_comprehensive_overview()

        # Save data
        self.save_metrics_csv()
        self.save_metrics_json()

        print(f"✅ All plots and data saved to: {self.plots_dir}")
        print("📊 Generated plots:")
        print("   • training_losses.png - All training loss components")
        print("   • validation_metrics.png - Paper evaluation metrics")
        print("   • learning_rates_and_time.png - LR schedules and timing")
        print("   • comprehensive_overview.png - Complete training overview")
        print("   • training_metrics.csv - All metrics in CSV format")
        print("   • training_metrics.json - All metrics in JSON format")

    def plot_final_summary(self, best_metrics: Dict):
        """Create a final summary plot with best achieved metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Training Summary - Best Metrics Achieved", fontsize=16, fontweight="bold"
        )

        # Best metrics bar chart
        metrics_names = ["QNMI", "QG", "QCB", "QPiella", "SSIM"]
        metrics_values = [best_metrics.get(k.lower(), 0) for k in metrics_names]

        bars = ax1.bar(
            metrics_names,
            metrics_values,
            color=["blue", "red", "green", "purple", "orange"],
        )
        ax1.set_title("Best Paper Metrics Achieved")
        ax1.set_ylabel("Score")
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Training progress
        epochs = self.training_history["epochs"]
        ax2.plot(
            epochs, self.training_history["g_loss"], label="Generator Loss", linewidth=2
        )
        ax2.plot(
            epochs,
            self.training_history["val_combined_score"],
            label="Combined Score",
            linewidth=2,
        )
        ax2.set_title("Training Progress")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Loss components final values
        final_losses = {
            "L1": (
                self.training_history["l1_loss"][-1]
                if self.training_history["l1_loss"]
                else 0
            ),
            "SSIM": (
                self.training_history["ssim_loss"][-1]
                if self.training_history["ssim_loss"]
                else 0
            ),
            "Perceptual": (
                self.training_history["perceptual_loss"][-1]
                if self.training_history["perceptual_loss"]
                else 0
            ),
            "Adversarial": (
                self.training_history["adversarial_loss"][-1]
                if self.training_history["adversarial_loss"]
                else 0
            ),
        }

        ax3.bar(
            final_losses.keys(),
            final_losses.values(),
            color=["red", "green", "purple", "orange"],
        )
        ax3.set_title("Final Loss Components")
        ax3.set_ylabel("Loss Value")
        ax3.grid(True, alpha=0.3)

        # Training statistics
        if self.training_history["epoch_time"]:
            total_time = sum(self.training_history["epoch_time"])
            avg_time = np.mean(self.training_history["epoch_time"])

            stats_text = f"""Training Statistics:
            
Total Epochs: {len(epochs)}
Total Time: {total_time/3600:.2f} hours
Avg Time/Epoch: {avg_time:.2f} seconds

Best Combined Score: {max(self.training_history['val_combined_score']) if self.training_history['val_combined_score'] else 0:.4f}
Final Generator Loss: {self.training_history['g_loss'][-1] if self.training_history['g_loss'] else 0:.4f}
Final Discriminator Loss: {self.training_history['d_loss'][-1] if self.training_history['d_loss'] else 0:.4f}"""

            ax4.text(
                0.1,
                0.9,
                stats_text,
                transform=ax4.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            )
            ax4.set_title("Training Statistics")
            ax4.axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "training_summary.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("📊 Training summary plot saved!")
