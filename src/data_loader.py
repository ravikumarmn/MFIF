import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class MFIFDataset(Dataset):
    """Multi-Focus Image Fusion Dataset"""

    def __init__(
        self,
        source_a_dir,
        source_b_dir,
        ground_truth_dir,
        transform=None,
        image_size=256,
    ):
        self.source_a_dir = source_a_dir
        self.source_b_dir = source_b_dir
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform
        self.image_size = image_size

        # Get all image filenames
        self.image_names = [f for f in os.listdir(source_a_dir) if f.endswith(".jpg")]
        self.image_names.sort()

        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        # Load images
        source_a_path = os.path.join(self.source_a_dir, img_name)
        source_b_path = os.path.join(self.source_b_dir, img_name)
        gt_path = os.path.join(self.ground_truth_dir, img_name)

        source_a = Image.open(source_a_path).convert("RGB")
        source_b = Image.open(source_b_path).convert("RGB")
        ground_truth = Image.open(gt_path).convert("RGB")

        # Apply transforms
        source_a = self.transform(source_a)
        source_b = self.transform(source_b)
        ground_truth = self.transform(ground_truth)

        return {
            "source_a": source_a,
            "source_b": source_b,
            "ground_truth": ground_truth,
            "filename": img_name,
        }


def create_data_loaders(
    source_a_dir,
    source_b_dir,
    ground_truth_dir,
    batch_size=8,
    image_size=256,
    num_workers=4,
    max_samples=None,
):

    dataset = MFIFDataset(
        source_a_dir, source_b_dir, ground_truth_dir, image_size=image_size
    )

    # Limit dataset size if specified
    if max_samples is not None and max_samples < len(dataset):
        # Use a subset of the dataset
        indices = torch.randperm(len(dataset))[:max_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
        print(
            f"Using subset of {max_samples} samples from {len(dataset.dataset)} total samples"
        )

    # Split dataset (80% train, 20% val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
