#!/usr/bin/env python3
"""
Image quality evaluation script.
Evaluates SSIM, MSE, PSNR, L2 norm between clean and steganography images.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from src.evaluate import (
        compute_image_quality_metrics as eval_compute_image_quality_metrics,
    )
    _HAS_EVAL = True
except Exception as exc:
    print(f"Warning: src.evaluate not available ({exc}). Using fallback implementation.")
    _HAS_EVAL = False
    eval_compute_image_quality_metrics = None
    from skimage.metrics import structural_similarity as ssim

def _tensor_to_uint8_np(image: torch.Tensor) -> np.ndarray:
    """Convert tensor to uint8 numpy array.
    
    Args:
        image: Tensor of shape [C, H, W] or [H, W, C] with values in [0, 1]
    
    Returns:
        Numpy array of shape [H, W, C] with values in [0, 255] as uint8
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().float().cpu().clamp(0, 1)
        # Handle different tensor shapes
        if len(image.shape) == 3:
            if image.shape[0] == 3 or image.shape[0] == 1:  # [C, H, W]
                image = image.permute(1, 2, 0)
            # Now should be [H, W, C]
        image_np = (image.numpy() * 255.0).astype(np.uint8)
        # Ensure 3 channels
        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[2] == 1:
            image_np = np.repeat(image_np, 3, axis=2)
        return image_np
    return image


def _compute_quality_metrics(
    clean_tensor: torch.Tensor,
    stego_tensor: torch.Tensor,
) -> Dict[str, float]:
    """Compute quality metrics between clean and stego images.
    
    Args:
        clean_tensor: Clean image tensor [C, H, W] with values in [0, 1]
        stego_tensor: Stego image tensor [C, H, W] with values in [0, 1]
    Returns:
        Dictionary of quality metrics
    """
    # Convert to numpy uint8 arrays
    clean_np = _tensor_to_uint8_np(clean_tensor)
    stego_np = _tensor_to_uint8_np(stego_tensor)
    
    # Ensure shapes match
    if clean_np.shape != stego_np.shape:
        # Resize stego to match clean if needed
        from skimage.transform import resize
        stego_np = resize(stego_np, clean_np.shape, preserve_range=True).astype(np.uint8)

    if _HAS_EVAL and eval_compute_image_quality_metrics is not None:
        try:
            result = eval_compute_image_quality_metrics(
                clean_np,
                stego_np,
            )
            mse, ssim_value, psnr_value, l2_norm = result
        except Exception as e:
            print(f"Warning: Error using src.evaluate metrics ({e}), falling back to skimage")
            # Fallback to skimage implementation
            ssim_value = ssim(
                clean_np,
                stego_np,
                channel_axis=2,
                data_range=255,
            )
            mse = float(np.mean((clean_np.astype(np.float32) - stego_np.astype(np.float32)) ** 2))
            psnr_value = float("inf") if mse == 0 else -10 * np.log10(mse / (255.0 ** 2))
            l2_norm = float(np.linalg.norm(clean_np.astype(np.float32) - stego_np.astype(np.float32)))
    else:
        ssim_value = ssim(
            clean_np,
            stego_np,
            channel_axis=2,
            data_range=255,
        )
        mse = float(np.mean((clean_np.astype(np.float32) - stego_np.astype(np.float32)) ** 2))
        psnr_value = float("inf") if mse == 0 else -10 * np.log10(mse / (255.0 ** 2))
        l2_norm = float(np.linalg.norm(clean_np.astype(np.float32) - stego_np.astype(np.float32)))

    metrics = {
        "SSIM": float(ssim_value),
        "MSE": float(mse),
        "PSNR": float(psnr_value),
        "L2": float(l2_norm),
    }

    return metrics


class ImagePairDataset(Dataset):
    """Dataset for paired clean and steganography images."""

    def __init__(
        self,
        clean_image_dir: str,
        stego_image_dir: str,
        transform=None,
        image_extensions: tuple = (".jpg", ".jpeg", ".png"),
        image_list: Optional[Set[str]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            clean_image_dir: Directory containing clean images
            stego_image_dir: Directory containing steganography images
            transform: Image transformation function
            image_extensions: Allowed image extensions
            image_list: Optional set of image filenames to include
        """
        self.clean_image_dir = Path(clean_image_dir)
        self.stego_image_dir = Path(stego_image_dir)
        self.transform = transform
        self.image_extensions = image_extensions

        # Get all clean images
        clean_images = []
        for ext in self.image_extensions:
            clean_images.extend(list(self.clean_image_dir.glob(f"*{ext}")))
            clean_images.extend(list(self.clean_image_dir.glob(f"*{ext.upper()}")))

        # Match with stego images
        allowed_names = set(image_list) if image_list else None
        allowed_stems = (
            {Path(name).stem for name in allowed_names} if allowed_names else None
        )
        self.image_pairs = []
        for clean_img in sorted(clean_images):
            if allowed_names:
                if clean_img.name not in allowed_names and clean_img.stem not in allowed_stems:
                    continue
            stego_img = self.stego_image_dir / clean_img.name
            if stego_img.exists():
                self.image_pairs.append((clean_img, stego_img))

        print(f"Found {len(self.image_pairs)} matching image pairs")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_pairs)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get an item from the dataset.

        Args:
            idx: Index

        Returns:
            Tuple of (clean_image, stego_image, image_name)
        """
        clean_path, stego_path = self.image_pairs[idx]

        # Load images
        try:
            clean_image = Image.open(clean_path).convert("RGB")
            stego_image = Image.open(stego_path).convert("RGB")
        except Exception as e:
            print(f"Error loading images {clean_path}, {stego_path}: {e}")
            clean_image = Image.new("RGB", (224, 224))
            stego_image = Image.new("RGB", (224, 224))

        if self.transform:
            clean_image = self.transform(clean_image)
            stego_image = self.transform(stego_image)

        return clean_image, stego_image, clean_path.name


def load_image_list(image_list_path: Optional[str]) -> Optional[Set[str]]:
    """Load a list of image filenames to include."""
    if not image_list_path:
        return None
    image_list = set()
    with open(image_list_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                image_list.add(name)
    return image_list


def evaluate_image_quality(
    clean_image_dir: str,
    stego_image_dir: str,
    transform=None,
    batch_size: int = 32,
    image_list: Optional[Set[str]] = None,
) -> Dict:
    """
    Evaluate image quality metrics between clean and steganography images.

    Args:
        clean_image_dir: Directory containing clean images
        stego_image_dir: Directory containing steganography images
        transform: Image transformation function
        batch_size: Batch size for processing

    Returns:
        Dictionary of average metrics
    """
    dataset = ImagePairDataset(
        clean_image_dir,
        stego_image_dir,
        transform=transform,
        image_list=image_list,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    all_metrics = {
        "SSIM": [],
        "MSE": [],
        "PSNR": [],
        "L2": [],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for clean_images, stego_images, _ in tqdm(
        dataloader, desc="Evaluating image quality"
    ):
        clean_images = clean_images.to(device)
        stego_images = stego_images.to(device)

        for i in range(len(clean_images)):
            metrics = _compute_quality_metrics(
                clean_images[i], stego_images[i]
            )
            for key in all_metrics:
                if key in metrics and metrics[key] is not None:
                    all_metrics[key].append(metrics[key])

    # Compute averages
    avg_metrics = {}
    std_metrics = {}
    for key, values in all_metrics.items():
        # Filter out None values
        valid_values = [v for v in values if v is not None]
        if valid_values:
            avg_metrics[key] = float(np.mean(valid_values))
            std_metrics[key] = float(np.std(valid_values))
        else:
            avg_metrics[key] = None
            std_metrics[key] = None

    return {
        "mean": avg_metrics,
        "std": std_metrics,
        "num_samples": len(dataset),
    }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Image Quality Evaluation")
    parser.add_argument(
        "--clean_image_dir",
        type=str,
        required=True,
        help="Directory containing clean images",
    )
    parser.add_argument(
        "--stego_image_dir",
        type=str,
        required=True,
        help="Directory containing steganography images",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/quality",
        help="Output directory for results",
    )
    parser.add_argument(
        "--image_list",
        type=str,
        default=None,
        help="Optional file with image names to include (e.g., Flickr_8k.testImages.txt)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag to append to output filenames",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Simple transform (normalize to [0, 1])
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Evaluate image quality
    print("Evaluating image quality metrics...")
    image_list = load_image_list(args.image_list)
    metrics = evaluate_image_quality(
        args.clean_image_dir,
        args.stego_image_dir,
        transform=transform,
        image_list=image_list,
    )

    # Save results
    tag_suffix = f"_{args.tag}" if args.tag else ""
    output_file = os.path.join(
        args.output_dir, f"image_quality_evaluation{tag_suffix}.json"
    )
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print("\nImage Quality Metrics (Mean ± Std):")
    for key in ["SSIM", "MSE", "PSNR", "L2"]:
        if key in metrics["mean"]:
            mean_val = metrics["mean"][key]
            std_val = metrics["std"][key]
            if mean_val is None or std_val is None:
                print(f"  {key}: n/a")
            else:
                print(f"  {key}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    main()
