import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

from models import SiameseMFIF


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SiameseMFIF(in_channels=3, out_channels=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, image_size=256):
    """Preprocess single image for inference"""
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension


def postprocess_image(tensor):
    """Convert tensor back to PIL image"""
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean

    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to PIL image
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor.permute(1, 2, 0)  # CHW to HWC
    tensor = (tensor * 255).byte()
    return Image.fromarray(tensor.cpu().numpy())


def fuse_images(model, source_a_path, source_b_path, device, image_size=256):
    """Fuse two images using the trained model"""
    # Load and preprocess images
    source_a = preprocess_image(source_a_path, image_size).to(device)
    source_b = preprocess_image(source_b_path, image_size).to(device)

    # Inference
    with torch.no_grad():
        fused_tensor = model(source_a, source_b)

    # Postprocess result
    fused_image = postprocess_image(fused_tensor)

    return fused_image


def visualize_results(source_a_path, source_b_path, fused_image, output_path=None):
    """Visualize input images and fusion result"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Load original images for display
    source_a = Image.open(source_a_path).convert("RGB")
    source_b = Image.open(source_b_path).convert("RGB")

    axes[0].imshow(source_a)
    axes[0].set_title("Source A (Near Focus)")
    axes[0].axis("off")

    axes[1].imshow(source_b)
    axes[1].set_title("Source B (Far Focus)")
    axes[1].axis("off")

    axes[2].imshow(fused_image)
    axes[2].set_title("Fused Result")
    axes[2].axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {output_path}")

    plt.show()


def batch_inference(
    model, source_a_dir, source_b_dir, output_dir, device, image_size=256
):
    """Perform batch inference on a directory of image pairs"""
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(source_a_dir) if f.endswith(".jpg")]
    image_files.sort()

    print(f"Processing {len(image_files)} image pairs...")

    for img_file in image_files:
        source_a_path = os.path.join(source_a_dir, img_file)
        source_b_path = os.path.join(source_b_dir, img_file)

        if not os.path.exists(source_b_path):
            print(f"Warning: {source_b_path} not found, skipping...")
            continue

        # Fuse images
        try:
            fused_image = fuse_images(
                model, source_a_path, source_b_path, device, image_size
            )

            # Save result
            output_path = os.path.join(output_dir, f"fused_{img_file}")
            fused_image.save(output_path)
            print(f"Processed: {img_file} -> {output_path}")

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")

    print("Batch inference completed!")


def main():
    parser = argparse.ArgumentParser(description="MFIF Inference")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--source_a", type=str, help="Path to source A image")
    parser.add_argument("--source_b", type=str, help="Path to source B image")
    parser.add_argument("--source_a_dir", type=str, help="Directory of source A images")
    parser.add_argument("--source_b_dir", type=str, help="Directory of source B images")
    parser.add_argument("--output", type=str, help="Output path for single image")
    parser.add_argument(
        "--output_dir", type=str, help="Output directory for batch processing"
    )
    parser.add_argument("--image_size", type=int, default=256, help="Input image size")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully!")

    if args.source_a and args.source_b:
        # Single image inference
        print("Performing single image inference...")
        fused_image = fuse_images(
            model, args.source_a, args.source_b, device, args.image_size
        )

        if args.output:
            fused_image.save(args.output)
            print(f"Fused image saved to: {args.output}")

        if args.visualize:
            visualize_results(
                args.source_a,
                args.source_b,
                fused_image,
                (
                    args.output.replace(".jpg", "_visualization.png")
                    if args.output
                    else None
                ),
            )

    elif args.source_a_dir and args.source_b_dir and args.output_dir:
        # Batch inference
        print("Performing batch inference...")
        batch_inference(
            model,
            args.source_a_dir,
            args.source_b_dir,
            args.output_dir,
            device,
            args.image_size,
        )

    else:
        print(
            "Please provide either single image paths (--source_a, --source_b) or "
            "directory paths (--source_a_dir, --source_b_dir, --output_dir)"
        )


if __name__ == "__main__":
    main()
