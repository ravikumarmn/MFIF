#!/usr/bin/env python3
"""
Exact Paper Evaluation Script
Multi-Focus Image Fusion - Comprehensive Evaluation

This script implements the EXACT evaluation procedure as described in the paper:
- All paper metrics: QNMI, QG, QCB, QPiella, SSIM
- Qualitative analysis: edge sharpness, focus uniformity
- Performance benchmarking
- Comparison with baseline methods
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
import argparse
import time
from pathlib import Path

from models import SiameseMFIF, SiameseMFIFGAN
from data_loader import create_data_loaders
from metrics import (
    calculate_ssim,
    calculate_psnr,
    calculate_l1_loss,
    calculate_qnmi,
    calculate_qg,
    calculate_qcb,
    calculate_qpiella,
    evaluate_model_comprehensive,
)


def get_optimal_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class PaperExactEvaluator:
    """Exact paper evaluation implementation"""

    def __init__(self, config):
        self.config = config
        self.device = get_optimal_device()
        self.results = {}

        print(f"🔬 Evaluator initialized on device: {self.device}")

    def load_model(self, checkpoint_path, model_type="gan"):
        """Load trained model from checkpoint"""
        print(f"📂 Loading model from: {checkpoint_path}")

        if model_type == "gan":
            model = SiameseMFIFGAN(in_channels=3, out_channels=3).to(self.device)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if "generator_state_dict" in checkpoint:
                model.generator.load_state_dict(checkpoint["generator_state_dict"])
                model.discriminator.load_state_dict(
                    checkpoint["discriminator_state_dict"]
                )
            else:
                model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model = SiameseMFIF(in_channels=3, out_channels=3).to(self.device)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()
        print(f"✅ Model loaded successfully")
        return model

    def evaluate_comprehensive(self, model, data_loader, model_name="Model"):
        """Comprehensive evaluation with all paper metrics"""
        print(f"🔍 Evaluating {model_name} with comprehensive metrics...")

        start_time = time.time()

        # Get comprehensive metrics
        metrics = evaluate_model_comprehensive(
            model,
            data_loader,
            self.device,
            num_samples=self.config.get("eval_samples", None),
        )

        eval_time = time.time() - start_time

        # Add timing information
        metrics["evaluation_time"] = eval_time
        metrics["samples_per_second"] = len(data_loader.dataset) / eval_time

        # Store results
        self.results[model_name] = metrics

        print(f"✅ {model_name} evaluation completed in {eval_time:.2f}s")
        print(f"📊 {model_name} Results:")
        print(f"   • QNMI: {metrics['qnmi']:.4f}")
        print(f"   • QG: {metrics['qg']:.4f}")
        print(f"   • QCB: {metrics['qcb']:.4f}")
        print(f"   • QPiella: {metrics['qpiella']:.4f}")
        print(f"   • SSIM: {metrics['ssim']:.4f}")
        print(f"   • PSNR: {metrics['psnr']:.2f} dB")
        print(f"   • L1 Loss: {metrics['l1_loss']:.4f}")
        print(f"   • Speed: {metrics['samples_per_second']:.1f} samples/sec")

        return metrics

    def benchmark_performance(self, model, model_name="Model"):
        """Benchmark inference performance as per paper requirements"""
        print(f"⚡ Benchmarking {model_name} performance...")

        # Create dummy input (520x520 as mentioned in paper)
        dummy_input_a = torch.randn(1, 3, 520, 520).to(self.device)
        dummy_input_b = torch.randn(1, 3, 520, 520).to(self.device)

        model.eval()

        # Warmup runs
        with torch.no_grad():
            for _ in range(10):
                if hasattr(model, "generator"):
                    _ = model.generator(dummy_input_a, dummy_input_b)
                else:
                    _ = model(dummy_input_a, dummy_input_b)

        # Benchmark runs
        times = []
        with torch.no_grad():
            for _ in range(100):
                start_time = time.time()
                if hasattr(model, "generator"):
                    _ = model.generator(dummy_input_a, dummy_input_b)
                else:
                    _ = model(dummy_input_a, dummy_input_b)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                elif self.device.type == "mps":
                    torch.mps.synchronize()

                end_time = time.time()
                times.append(end_time - start_time)

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"📈 {model_name} Performance (520×520 images):")
        print(f"   • Average time: {avg_time:.4f}s")
        print(f"   • Std deviation: {std_time:.4f}s")
        print(f"   • Paper requirement: < 0.1s (RTX 3060 Ti)")
        print(
            f"   • Status: {'✅ PASS' if avg_time < 0.1 else '❌ FAIL (but device dependent)'}"
        )

        return {
            "avg_inference_time": avg_time,
            "std_inference_time": std_time,
            "meets_paper_requirement": avg_time < 0.1,
        }

    def qualitative_analysis(
        self, model, data_loader, model_name="Model", num_samples=5
    ):
        """Qualitative analysis: edge sharpness and focus uniformity"""
        print(f"🎨 Performing qualitative analysis for {model_name}...")

        model.eval()
        output_dir = (
            Path(self.config["output_dir"])
            / f"qualitative_{model_name.lower().replace(' ', '_')}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        sample_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if sample_count >= num_samples:
                    break

                source_a = batch["source_a"].to(self.device)
                source_b = batch["source_b"].to(self.device)
                ground_truth = batch["ground_truth"].to(self.device)

                # Generate fused image
                if hasattr(model, "generator"):
                    fused = model.generator(source_a, source_b)
                else:
                    fused = model(source_a, source_b)

                # Process first image in batch
                img_a = source_a[0].cpu().numpy().transpose(1, 2, 0)
                img_b = source_b[0].cpu().numpy().transpose(1, 2, 0)
                img_gt = ground_truth[0].cpu().numpy().transpose(1, 2, 0)
                img_fused = fused[0].cpu().numpy().transpose(1, 2, 0)

                # Normalize to [0, 1]
                img_a = (img_a + 1) / 2 if img_a.min() < 0 else img_a
                img_b = (img_b + 1) / 2 if img_b.min() < 0 else img_b
                img_gt = (img_gt + 1) / 2 if img_gt.min() < 0 else img_gt
                img_fused = (img_fused + 1) / 2 if img_fused.min() < 0 else img_fused

                # Create comparison figure
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                fig.suptitle(f"{model_name} - Sample {sample_count + 1}", fontsize=16)

                axes[0, 0].imshow(img_a)
                axes[0, 0].set_title("Source A (Near Focus)")
                axes[0, 0].axis("off")

                axes[0, 1].imshow(img_b)
                axes[0, 1].set_title("Source B (Far Focus)")
                axes[0, 1].axis("off")

                axes[1, 0].imshow(img_gt)
                axes[1, 0].set_title("Ground Truth")
                axes[1, 0].axis("off")

                axes[1, 1].imshow(img_fused)
                axes[1, 1].set_title(f"Fused ({model_name})")
                axes[1, 1].axis("off")

                plt.tight_layout()
                plt.savefig(
                    output_dir / f"sample_{sample_count + 1}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                # Calculate metrics for this sample
                sample_metrics = {
                    "qnmi": calculate_qnmi(fused[0:1], source_a[0:1], source_b[0:1]),
                    "qg": calculate_qg(fused[0:1], source_a[0:1], source_b[0:1]),
                    "qcb": calculate_qcb(fused[0:1], source_a[0:1], source_b[0:1]),
                    "qpiella": calculate_qpiella(
                        fused[0:1], source_a[0:1], source_b[0:1]
                    ),
                    "ssim": calculate_ssim(fused[0:1], ground_truth[0:1]),
                    "psnr": calculate_psnr(fused[0:1], ground_truth[0:1]).item(),
                }

                # Save individual metrics
                with open(
                    output_dir / f"sample_{sample_count + 1}_metrics.txt", "w"
                ) as f:
                    f.write(f"Sample {sample_count + 1} Metrics:\n")
                    f.write(f"QNMI: {sample_metrics['qnmi']:.4f}\n")
                    f.write(f"QG: {sample_metrics['qg']:.4f}\n")
                    f.write(f"QCB: {sample_metrics['qcb']:.4f}\n")
                    f.write(f"QPiella: {sample_metrics['qpiella']:.4f}\n")
                    f.write(f"SSIM: {sample_metrics['ssim']:.4f}\n")
                    f.write(f"PSNR: {sample_metrics['psnr']:.2f} dB\n")

                sample_count += 1

        print(f"✅ Qualitative analysis saved to: {output_dir}")
        return output_dir

    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        if len(self.results) < 1:
            print("❌ No results to compare")
            return

        print("📊 Generating comprehensive comparison report...")

        # Create results DataFrame
        df_results = pd.DataFrame(self.results).T

        # Save detailed results
        output_file = Path(self.config["output_dir"]) / "evaluation_results.csv"
        df_results.to_csv(output_file)

        # Create comparison plots
        self._create_comparison_plots(df_results)

        # Generate summary report
        self._generate_summary_report(df_results)

        print(f"✅ Comparison report saved to: {self.config['output_dir']}")

    def _create_comparison_plots(self, df_results):
        """Create comparison visualization plots"""
        output_dir = Path(self.config["output_dir"])

        # Paper metrics comparison
        paper_metrics = ["qnmi", "qg", "qcb", "qpiella", "ssim"]

        if len(df_results) > 1:
            # Multi-model comparison
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle("Paper Metrics Comparison", fontsize=16)

            for i, metric in enumerate(paper_metrics):
                ax = axes[i // 3, i % 3]
                df_results[metric].plot(kind="bar", ax=ax)
                ax.set_title(f"{metric.upper()}")
                ax.set_ylabel("Score")
                ax.tick_params(axis="x", rotation=45)

            # PSNR comparison
            ax = axes[1, 2]
            df_results["psnr"].plot(kind="bar", ax=ax)
            ax.set_title("PSNR (dB)")
            ax.set_ylabel("dB")
            ax.tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.savefig(
                output_dir / "metrics_comparison.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        # Radar chart for single model or best model
        best_model = df_results.loc[df_results["ssim"].idxmax()]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        # Normalize metrics to [0, 1] for radar chart
        metrics_normalized = []
        labels = []

        for metric in paper_metrics:
            value = best_model[metric]
            # Normalize based on typical ranges
            if metric == "qnmi":
                normalized = min(value / 1.0, 1.0)  # QNMI typically [0, 1]
            elif metric == "qg":
                normalized = min(value / 1.0, 1.0)  # QG typically [0, 1]
            elif metric == "qcb":
                normalized = min(value / 1.0, 1.0)  # QCB typically [0, 1]
            elif metric == "qpiella":
                normalized = min(value / 1.0, 1.0)  # QPiella typically [0, 1]
            elif metric == "ssim":
                normalized = min(value / 1.0, 1.0)  # SSIM [0, 1]

            metrics_normalized.append(normalized)
            labels.append(metric.upper())

        # Add PSNR (normalized to typical range)
        psnr_normalized = min(best_model["psnr"] / 40.0, 1.0)  # Assume max PSNR of 40dB
        metrics_normalized.append(psnr_normalized)
        labels.append("PSNR")

        # Close the radar chart
        metrics_normalized.append(metrics_normalized[0])
        labels.append(labels[0])

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)

        ax.plot(angles, metrics_normalized, "o-", linewidth=2, label=best_model.name)
        ax.fill(angles, metrics_normalized, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels[:-1])
        ax.set_ylim(0, 1)
        ax.set_title(f"Performance Radar Chart - {best_model.name}", size=16, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        plt.savefig(output_dir / "performance_radar.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_summary_report(self, df_results):
        """Generate text summary report"""
        output_dir = Path(self.config["output_dir"])

        with open(output_dir / "evaluation_summary.txt", "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MULTI-FOCUS IMAGE FUSION - EVALUATION SUMMARY\n")
            f.write("Exact Paper Implementation Results\n")
            f.write("=" * 80 + "\n\n")

            f.write("📊 QUANTITATIVE RESULTS:\n")
            f.write("-" * 40 + "\n")

            for model_name, results in self.results.items():
                f.write(f"\n🔬 {model_name}:\n")
                f.write(
                    f"   • QNMI (Normalized Mutual Information): {results['qnmi']:.4f}\n"
                )
                f.write(f"   • QG (Gradient-based): {results['qg']:.4f}\n")
                f.write(f"   • QCB (Correlation Coefficient): {results['qcb']:.4f}\n")
                f.write(f"   • QPiella (Piella's metric): {results['qpiella']:.4f}\n")
                f.write(f"   • SSIM (Structural Similarity): {results['ssim']:.4f}\n")
                f.write(
                    f"   • PSNR (Peak Signal-to-Noise Ratio): {results['psnr']:.2f} dB\n"
                )
                f.write(f"   • L1 Loss: {results['l1_loss']:.4f}\n")
                f.write(
                    f"   • Evaluation Speed: {results['samples_per_second']:.1f} samples/sec\n"
                )

            # Best model analysis
            if len(df_results) > 1:
                f.write(f"\n🏆 BEST PERFORMING MODEL:\n")
                f.write("-" * 40 + "\n")

                best_models = {}
                for metric in ["qnmi", "qg", "qcb", "qpiella", "ssim", "psnr"]:
                    best_model = df_results[metric].idxmax()
                    best_value = df_results.loc[best_model, metric]
                    best_models[metric] = (best_model, best_value)
                    f.write(
                        f"   • Best {metric.upper()}: {best_model} ({best_value:.4f})\n"
                    )

                # Overall best (average ranking)
                rankings = {}
                for model in df_results.index:
                    rank_sum = 0
                    for metric in ["qnmi", "qg", "qcb", "qpiella", "ssim"]:
                        rank = df_results[metric].rank(ascending=False)[model]
                        rank_sum += rank
                    rankings[model] = rank_sum / 5

                overall_best = min(rankings, key=rankings.get)
                f.write(
                    f"\n   🥇 Overall Best Model: {overall_best} (avg rank: {rankings[overall_best]:.2f})\n"
                )

            f.write(f"\n📈 PAPER REQUIREMENTS ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            f.write("   • Target Metrics: High QNMI, QG, QCB, QPiella, SSIM\n")
            f.write("   • Performance: < 0.1s per 520×520 image (RTX 3060 Ti)\n")
            f.write("   • Quality: Enhanced edge sharpness and texture fidelity\n")

            f.write(f"\n🎯 EVALUATION CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"   • Device: {self.device}\n")
            f.write(
                f"   • Evaluation Samples: {self.config.get('eval_samples', 'All')}\n"
            )
            f.write(f"   • Image Size: 256×256 (training), 520×520 (benchmark)\n")

            f.write(f"\n" + "=" * 80 + "\n")
            f.write("Report generated by MFIF Exact Paper Implementation\n")
            f.write("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="EXACT Paper Evaluation")

    # Model paths
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gan",
        choices=["gan", "basic"],
        help="Model type",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Paper_Exact_Model",
        help="Model name for reports",
    )

    # Comparison models (optional)
    parser.add_argument(
        "--compare_checkpoint", type=str, help="Path to comparison model checkpoint"
    )
    parser.add_argument(
        "--compare_model_type",
        type=str,
        default="basic",
        choices=["gan", "basic"],
        help="Comparison model type",
    )
    parser.add_argument(
        "--compare_model_name",
        type=str,
        default="Baseline_Model",
        help="Comparison model name",
    )

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

    # Evaluation settings
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--eval_samples", type=int, default=None, help="Limit evaluation samples"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--qualitative_samples",
        type=int,
        default=10,
        help="Number of qualitative samples",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Configuration
    config = {
        "source_a_dir": args.source_a_dir,
        "source_b_dir": args.source_b_dir,
        "ground_truth_dir": args.ground_truth_dir,
        "batch_size": args.batch_size,
        "eval_samples": args.eval_samples,
        "output_dir": args.output_dir,
    }

    print("🔬 EXACT Paper Evaluation - Multi-Focus Image Fusion")
    print("=" * 80)
    print(f"📁 Dataset: {args.source_a_dir}")
    print(f"🎯 Model: {args.checkpoint}")
    print(f"📊 Output: {args.output_dir}")
    print("=" * 80)

    # Create data loader
    print("📂 Loading dataset...")
    _, val_loader = create_data_loaders(
        source_a_dir=config["source_a_dir"],
        source_b_dir=config["source_b_dir"],
        ground_truth_dir=config["ground_truth_dir"],
        batch_size=config["batch_size"],
        image_size=256,
        max_samples=config.get("eval_samples", None),
    )

    print(f"✅ Evaluation samples: {len(val_loader.dataset):,}")

    # Initialize evaluator
    evaluator = PaperExactEvaluator(config)

    # Load and evaluate main model
    model = evaluator.load_model(args.checkpoint, args.model_type)
    evaluator.evaluate_comprehensive(model, val_loader, args.model_name)

    # Performance benchmark
    perf_results = evaluator.benchmark_performance(model, args.model_name)

    # Qualitative analysis
    evaluator.qualitative_analysis(
        model, val_loader, args.model_name, args.qualitative_samples
    )

    # Load and evaluate comparison model if provided
    if args.compare_checkpoint:
        print(f"\n🔄 Loading comparison model...")
        compare_model = evaluator.load_model(
            args.compare_checkpoint, args.compare_model_type
        )
        evaluator.evaluate_comprehensive(
            compare_model, val_loader, args.compare_model_name
        )
        evaluator.qualitative_analysis(
            compare_model, val_loader, args.compare_model_name, args.qualitative_samples
        )

    # Generate comprehensive report
    evaluator.generate_comparison_report()

    print("\n🎉 Evaluation completed!")
    print(f"📊 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
