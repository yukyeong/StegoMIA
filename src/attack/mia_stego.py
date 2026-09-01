#!/usr/bin/env python3
"""
Membership Inference Attack (MIA) evaluation using stego-amplified membership inference method.
This script evaluates MIA performance using stego-clean image similarity difference.
"""

import argparse
import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_curve, auc, accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import open_clip
    from open_clip import tokenize
except ImportError:
    print("Error: open_clip not found. Please install it.")
    sys.exit(1)

# Import evaluation metrics
try:
    from src.evaluate import (
        compute_image_quality_metrics,
        load_image,
        compute_mia_metrics,
    )
except ImportError as e:
    print(f"Warning: Could not import some evaluation functions: {e}")
    # Define fallback functions
    def load_image(path):
        """Load image using PIL as fallback."""
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    def compute_mse(img1, img2):
        diff = img1.astype(np.float32) - img2.astype(np.float32)
        return float(np.mean(np.square(diff)))
    def compute_ssim_skimage(img1, img2, win_size=9, multichannel=True):
        try:
            from skimage.metrics import structural_similarity as ssim_func
        except ImportError:
            from skimage.measure import compare_ssim as ssim_func
        if multichannel and len(img1.shape) == 3:
            return float(ssim_func(img1, img2, win_size=win_size, channel_axis=-1))
        return float(ssim_func(img1, img2, win_size=win_size, multichannel=multichannel))
    def compute_image_quality_metrics(ref_img, pred_img, use_skimage_ssim=False):
        """Fallback function for computing image quality metrics."""
        if ref_img.shape != pred_img.shape:
            raise ValueError(f"Image shapes do not match: {ref_img.shape} vs {pred_img.shape}")
        mse_value = compute_mse(ref_img, pred_img)
        ssim_value = compute_ssim_skimage(ref_img, pred_img)
        psnr_value = float(calc_psnr(ref_img, pred_img))
        l2_norm_value = float(np.sqrt(np.sum((ref_img.astype(np.float32) - pred_img.astype(np.float32)) ** 2)))
        return mse_value, ssim_value, psnr_value, l2_norm_value

    def compute_mia_metrics(y_true, y_scores, threshold):
        y_pred = (y_scores > threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        fpr_below_1pct = np.where(fpr < 0.01)[0]
        if len(fpr_below_1pct) > 0:
            tpr_at_1pct_fpr = float(tpr[fpr_below_1pct[-1]])
        else:
            tpr_at_1pct_fpr = float(tpr[0]) if len(tpr) > 0 else 0.0
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        asr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        clean_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics = {
            "accuracy": accuracy,
            "auc": roc_auc,
            "tpr_at_1pct_fpr": tpr_at_1pct_fpr,
            "asr": asr,
            "clean_accuracy": clean_accuracy,
            "threshold": float(threshold),
        }
        return metrics, fpr, tpr

def calc_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def resolve_device(device: str) -> str:
    """Resolve device string to an available device."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return device


def write_command_script(output_dir: Path, timestamp: str, prefix: str = "Flicker8k_stgeo_") -> Path:
    """Write a reproducible command script to the output directory."""
    script_path = output_dir / f"{prefix}stego_mia_command_{timestamp}.sh"
    cmd = ["python", str(Path(__file__).resolve())] + sys.argv[1:]
    script_path.write_text("#!/bin/bash\n" + " ".join(shlex.quote(arg) for arg in cmd) + "\n")
    return script_path


class ImageTextDataset(Dataset):
    """Dataset for image-text pairs."""
    
    def __init__(self, image_paths: List[str], texts: List[str], preprocess, max_samples: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of image file paths.
            texts: List of corresponding text captions.
            preprocess: Image preprocessing function.
            max_samples: Maximum number of samples to use. If None, use all.
        """
        self.image_paths = image_paths[:max_samples] if max_samples else image_paths
        self.texts = texts[:max_samples] if max_samples else texts
        self.preprocess = preprocess
        
        assert len(self.image_paths) == len(self.texts), \
            f"Number of images ({len(self.image_paths)}) != number of texts ({len(self.texts)})"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        text = self.texts[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.preprocess(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = torch.zeros((3, 224, 224))
        
        return image, text, image_path


def load_flickr8k_dataset(image_dir: str, token_file: str, max_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Load Flickr8k dataset.
    
    Args:
        image_dir: Directory containing images.
        token_file: Path to Flickr8k.token.txt or Annotations.txt file.
        max_samples: Maximum number of samples to load.
    
    Returns:
        Tuple of (image_paths, texts) lists.
    """
    print(f"Loading Flickr8k dataset from {image_dir}...")
    
    # Load captions
    image_to_captions = {}
    image_dir_path = Path(image_dir)
    
    with open(token_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse different formats:
            # Format 1: image.jpg#0	caption (Flickr8k.token.txt)
            # Format 2: image.jpg	caption (Annotations.txt)
            if '#' in line and '\t' in line:
                # Format 1: image.jpg#0	caption
                hash_idx = line.index('#')
                tab_idx = line.index('\t')
                img_name = line[:hash_idx]
                caption = line[tab_idx + 1:].strip()
            elif '\t' in line:
                # Format 2: image.jpg	caption
                parts = line.split('\t', 1)
                if len(parts) >= 2:
                    img_name = parts[0].strip()
                    caption = parts[1].strip()
                else:
                    continue
            else:
                continue
            
            # Store captions (use first caption for each image)
            if img_name not in image_to_captions:
                image_to_captions[img_name] = []
            image_to_captions[img_name].append(caption)
    
    # Get image paths and corresponding captions
    image_paths = []
    texts = []
    
    # First, try to match images from annotation file
    for img_name, captions in image_to_captions.items():
        # Try different extensions
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = image_dir_path / (img_name + ext)
            if candidate.exists():
                img_path = candidate
                break
        
        # If no extension match, try direct path
        if img_path is None:
            img_path = image_dir_path / img_name
            if not img_path.exists():
                continue
        
        if img_path.exists():
            # Use the first caption for each image
            image_paths.append(str(img_path))
            texts.append(captions[0])
    
    # If no matches found, try reverse matching: scan image directory and match with annotations
    # This handles cases where annotation file uses different naming (e.g., s0000000.jpg)
    # but actual images use Flickr8k standard naming (e.g., 1000268201_693b08cb0e.jpg)
    if len(image_paths) == 0:
        print(f"Warning: No images matched from annotation file. Trying reverse matching by scanning image directory...")
        # Get all images in directory
        all_image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            all_image_files.extend(list(image_dir_path.glob(f'*{ext}')))
        
        # Try to match by filename (with and without extension)
        matched_count = 0
        for img_path in all_image_files:
            img_name = img_path.name
            img_stem = img_path.stem
            
            # Try exact match
            if img_name in image_to_captions:
                image_paths.append(str(img_path))
                texts.append(image_to_captions[img_name][0])
                matched_count += 1
            # Try stem match (without extension)
            elif img_stem in image_to_captions:
                image_paths.append(str(img_path))
                texts.append(image_to_captions[img_stem][0])
                matched_count += 1
        
        if matched_count > 0:
            print(f"Reverse matching found {matched_count} image-text pairs")
        else:
            print(f"Warning: Reverse matching also found 0 matches. Image names in directory may not match annotation file format.")
            print(f"  Annotation file format example: {list(image_to_captions.keys())[:3] if image_to_captions else 'N/A'}")
            print(f"  Image directory example: {[f.name for f in all_image_files[:3]] if all_image_files else 'N/A'}")
    
    if max_samples:
        image_paths = image_paths[:max_samples]
        texts = texts[:max_samples]
    
    print(f"Loaded {len(image_paths)} image-text pairs from Flickr8k")
    return image_paths, texts


def load_animal_image_caption_dataset(image_dir: str, csv_file: str, max_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Load Animal-Image-Caption dataset.
    
    Args:
        image_dir: Directory containing images.
        csv_file: Path to captions.csv file.
        max_samples: Maximum number of samples to load.
    
    Returns:
        Tuple of (image_paths, texts) lists.
    """
    print(f"Loading Animal-Image-Caption dataset from {image_dir}...")
    
    df = pd.read_csv(csv_file)
    image_dir_path = Path(image_dir)
    
    image_paths = []
    texts = []
    
    # Group by image name and use first caption for each image
    image_to_caption = {}
    for _, row in df.iterrows():
        img_name = row['image']
        caption = row['caption']
        # Use first caption for each image
        if img_name not in image_to_caption:
            image_to_caption[img_name] = caption
    
    for img_name, caption in image_to_caption.items():
        img_path = image_dir_path / img_name
        
        if img_path.exists():
            image_paths.append(str(img_path))
            texts.append(caption)
    
    if max_samples:
        image_paths = image_paths[:max_samples]
        texts = texts[:max_samples]
    
    print(f"Loaded {len(image_paths)} image-text pairs from Animal-Image-Caption")
    return image_paths, texts


def load_mscoco_dataset(image_dir: str, json_file: str, max_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Load MSCOCO2017 dataset.
    
    Args:
        image_dir: Directory containing images.
        json_file: Path to captions_train2017.json file.
        max_samples: Maximum number of samples to load.
    
    Returns:
        Tuple of (image_paths, texts) lists.
    """
    print(f"Loading MSCOCO2017 dataset from {image_dir}...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create mapping from image_id to image info
    image_id_to_info = {img['id']: img for img in data['images']}
    
    # Create mapping from image_id to captions
    image_id_to_captions = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_captions:
            image_id_to_captions[img_id] = []
        image_id_to_captions[img_id].append(ann['caption'])
    
    image_dir_path = Path(image_dir)
    image_paths = []
    texts = []
    
    # Get all image files in directory for reverse matching
    # Map stem (filename without extension) to full path
    all_image_files = {}
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        for img_file in image_dir_path.glob(f'*{ext}'):
            img_stem = img_file.stem  # filename without extension
            # Store the first match (prefer .png if both exist, but usually only one exists)
            if img_stem not in all_image_files:
                all_image_files[img_stem] = str(img_file)
    
    matched_count = 0
    skipped_count = 0
    
    for img_id, img_info in image_id_to_info.items():
        if img_id in image_id_to_captions:
            img_name = img_info['file_name']
            img_stem = Path(img_name).stem  # filename without extension (e.g., "000000000025")
            
            # Try exact match first (with .jpg extension from JSON)
            img_path = image_dir_path / img_name
            if not img_path.exists():
                # Try matching by stem (filename without extension) with different extensions
                if img_stem in all_image_files:
                    img_path = Path(all_image_files[img_stem])
                else:
                    skipped_count += 1
                    continue
            
            if img_path.exists():
                # Use the first caption for each image
                image_paths.append(str(img_path))
                texts.append(image_id_to_captions[img_id][0])
                matched_count += 1
    
    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} images that were not found in directory")
    if matched_count == 0 and len(all_image_files) > 0:
        print(f"Warning: No images matched. Directory contains {len(all_image_files)} images.")
        print(f"  Sample directory files: {list(all_image_files.keys())[:5]}")
        print(f"  Sample JSON filenames: {[Path(img['file_name']).stem for img in list(image_id_to_info.values())[:5]]}")
    
    if max_samples:
        image_paths = image_paths[:max_samples]
        texts = texts[:max_samples]
    
    print(f"Loaded {len(image_paths)} image-text pairs from MSCOCO2017")
    return image_paths, texts


def load_image_caption_matching_dataset(image_dir: str, captions_file: str, max_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Load Image-Caption-Matching dataset.
    
    Args:
        image_dir: Directory containing images.
        captions_file: Path to captions.txt file (CSV format: image,caption).
        max_samples: Maximum number of samples to load.
    
    Returns:
        Tuple of (image_paths, texts) lists.
    """
    print(f"Loading Image-Caption-Matching dataset from {image_dir}...")
    
    df = pd.read_csv(captions_file)
    image_dir_path = Path(image_dir)
    
    # Group by image name and use first caption
    image_to_caption = {}
    for _, row in df.iterrows():
        img_name = row['image']
        caption = row['caption']
        # Use first caption for each image
        if img_name not in image_to_caption:
            image_to_caption[img_name] = caption
    
    image_paths = []
    texts = []
    
    for img_name, caption in image_to_caption.items():
        img_path = image_dir_path / img_name
        
        if img_path.exists():
            image_paths.append(str(img_path))
            texts.append(caption)
    
    if max_samples:
        image_paths = image_paths[:max_samples]
        texts = texts[:max_samples]
    
    print(f"Loaded {len(image_paths)} image-text pairs from Image-Caption-Matching")
    return image_paths, texts


def load_imagenet1k_dataset(image_dir: str, json_file: str, max_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Load ImageNet-1k dataset.
    
    Args:
        image_dir: Directory containing images.
        json_file: Path to captions_train.json or captions_val.json file.
        max_samples: Maximum number of samples to load.
    
    Returns:
        Tuple of (image_paths, texts) lists.
    """
    print(f"Loading ImageNet-1k dataset from {image_dir}...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_dir_path = Path(image_dir)
    image_paths = []
    texts = []
    
    # Data format: {"image_name": ["caption1", "caption2", ...], ...}
    for img_name, captions in data.items():
        if not captions or len(captions) == 0:
            continue
        
        # Try different extensions and paths
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = image_dir_path / (img_name + ext)
            if candidate.exists():
                img_path = candidate
                break
        
        # If no extension match, try direct path
        if img_path is None:
            img_path = image_dir_path / img_name
            if not img_path.exists():
                continue
        
        if img_path.exists():
            # Use the first caption for each image
            image_paths.append(str(img_path))
            texts.append(captions[0])
    
    if max_samples:
        image_paths = image_paths[:max_samples]
        texts = texts[:max_samples]
    
    print(f"Loaded {len(image_paths)} image-text pairs from ImageNet-1k")
    return image_paths, texts


def load_model(model_path: str, pretrained_model_path: Optional[str], model_name: str = "ViT-B-16", device: str = "cuda"):
    """
    Load CLIP model from checkpoint.
    
    Args:
        model_path: Path to fine-tuned model checkpoint.
        pretrained_model_path: Path to pretrained model checkpoint.
        model_name: CLIP model name.
        device: Device to load model on.
    
    Returns:
        Tuple of (model, preprocess) functions.
    """
    print(f"Loading model from {model_path}...")
    
    # Create model
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=None
        )
    except RuntimeError as e:
        print(f"Error creating model: {e}")
        raise
    
    # Load pretrained weights if specified
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Loading pretrained weights from {pretrained_model_path}...")
        pretrained_checkpoint = torch.load(
            pretrained_model_path, map_location="cpu", weights_only=False
        )
        sd = pretrained_checkpoint
        if "state_dict" in sd:
            sd = pretrained_checkpoint["state_dict"]
        elif "model_state_dict" in sd:
            sd = pretrained_checkpoint["model_state_dict"]
        if sd and next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        print("Pretrained model loaded successfully")
    
    # Load fine-tuned checkpoint
    if model_path and os.path.exists(model_path):
        print(f"Loading fine-tuned checkpoint from {model_path}...")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        sd = checkpoint
        if "state_dict" in sd:
            sd = checkpoint["state_dict"]
        elif "model_state_dict" in sd:
            sd = checkpoint["model_state_dict"]
        if sd and next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        try:
            incompatible = model.load_state_dict(sd, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                print("Warning: Some keys were not loaded:")
                if incompatible.missing_keys:
                    print(f"  Missing keys: {len(incompatible.missing_keys)}")
                if incompatible.unexpected_keys:
                    print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")
            print("Fine-tuned model checkpoint loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load fine-tuned checkpoint: {e}")
            print("Continuing with pretrained model only.")
    
    model = model.to(device)
    model.eval()
    
    return model, preprocess


def compute_cosine_similarities(model, dataloader, device: str = "cuda") -> np.ndarray:
    """
    Compute cosine similarities between image and text embeddings.
    
    Args:
        model: CLIP model.
        dataloader: DataLoader for image-text pairs.
        device: Device to run computation on.
    
    Returns:
        Array of cosine similarity scores.
    """
    similarities = []
    
    with torch.no_grad():
        for images, texts, _ in tqdm(dataloader, desc="Computing similarities"):
            images = images.to(device)
            text_tokens = tokenize(texts).to(device)
            
            # Get embeddings
            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)
            
            # Normalize
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute cosine similarity
            similarity = (image_features * text_features).sum(dim=-1)
            similarities.extend(similarity.cpu().numpy())
    
    return np.array(similarities)


def compute_similarity_difference(
    model,
    stego_image_paths: List[str],
    clean_image_paths: List[str],
    texts: List[str],
    preprocess,
    batch_size: int,
    device: str = "cuda"
) -> Tuple[np.ndarray, float]:
    """
    Compute similarity difference: stego_image-text similarity - clean_image-text similarity.
    
    Args:
        model: CLIP model.
        stego_image_paths: List of stego image paths.
        clean_image_paths: List of corresponding clean image paths.
        texts: List of corresponding text captions.
        preprocess: Image preprocessing function.
        batch_size: Batch size for processing.
        device: Device to run computation on.
    
    Returns:
        Tuple of (differences array, mean difference).
    """
    print("Computing similarity differences (stego - clean)...")
    
    differences = []
    
    # Create datasets
    stego_dataset = ImageTextDataset(stego_image_paths, texts, preprocess)
    clean_dataset = ImageTextDataset(clean_image_paths, texts, preprocess)
    
    stego_dataloader = DataLoader(stego_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    clean_dataloader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    with torch.no_grad():
        for (stego_images, stego_texts, _), (clean_images, clean_texts, _) in zip(
            tqdm(stego_dataloader, desc="Computing differences"),
            clean_dataloader
        ):
            stego_images = stego_images.to(device)
            clean_images = clean_images.to(device)
            text_tokens = tokenize(stego_texts).to(device)
            
            # Get embeddings
            stego_image_features = model.encode_image(stego_images)
            clean_image_features = model.encode_image(clean_images)
            text_features = model.encode_text(text_tokens)
            
            # Normalize
            stego_image_features = F.normalize(stego_image_features, dim=-1)
            clean_image_features = F.normalize(clean_image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute cosine similarities
            stego_similarity = (stego_image_features * text_features).sum(dim=-1)
            clean_similarity = (clean_image_features * text_features).sum(dim=-1)
            
            # Compute difference
            diff = stego_similarity - clean_similarity
            differences.extend(diff.cpu().numpy())
    
    differences = np.array(differences)
    mean_difference = float(np.mean(differences))
    
    print(f"Mean similarity difference: {mean_difference:.4e}")
    print(f"Difference stats: mean={np.mean(differences):.4e}, std={np.std(differences):.4e}")
    
    return differences, mean_difference


def evaluate_mia_stego(
    model,
    member_dataloader,
    non_member_dataloader,
    mean_difference: float,
    hyper_lambda: float,
    device: str = "cuda",
    member_similarities: Optional[np.ndarray] = None,
    non_member_similarities: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate membership inference using frequency-domain stego amplification.
    
    Args:
        model: CLIP model.
        member_dataloader: DataLoader for member samples.
        non_member_dataloader: DataLoader for non-member samples.
        mean_difference: Mean similarity difference from stego-clean pairs.
        hyper_lambda: Hyperparameter for threshold calculation.
        device: Device to run computation on.
        member_similarities: Pre-computed member similarities (optional).
        non_member_similarities: Pre-computed non-member similarities (optional).
    
    Returns:
        Tuple of (metrics, fpr, tpr).
    """
    if member_similarities is None:
        print("Computing similarities for member samples...")
        member_similarities = compute_cosine_similarities(model, member_dataloader, device)
    if non_member_similarities is None:
        print("Computing similarities for non-member samples...")
        non_member_similarities = compute_cosine_similarities(model, non_member_dataloader, device)
    
    # Check for NaN in mean_difference
    if np.isnan(mean_difference) or np.isinf(mean_difference):
        print(f"Warning: mean_difference is NaN or Inf. Setting to 0.0")
        mean_difference = 0.0
    
    # Add mean difference to member similarities (frequency-domain stego MIA)
    enhanced_member_similarities = member_similarities + mean_difference
    
    # Check for NaN in enhanced similarities
    if np.any(np.isnan(enhanced_member_similarities)) or np.any(np.isinf(enhanced_member_similarities)):
        print(f"Warning: Enhanced member similarities contain NaN or Inf. Replacing with original similarities.")
        enhanced_member_similarities = member_similarities.copy()
    
    # Compute threshold: mean(non_member) + hyper_lambda * std(non_member)
    train_threshold = float(np.mean(non_member_similarities) + hyper_lambda * np.std(non_member_similarities))
    
    # Check for NaN in threshold
    if np.isnan(train_threshold) or np.isinf(train_threshold):
        print(f"Warning: train_threshold is NaN or Inf. Using mean of non_member_similarities.")
        train_threshold = float(np.mean(non_member_similarities))
    
    print(f"Train threshold: {train_threshold:.4e} (mean={np.mean(non_member_similarities):.4e}, std={np.std(non_member_similarities):.4e}, lambda={hyper_lambda:.4e})")
    
    # Create labels: 1 for members, 0 for non-members
    y_true = np.concatenate([
        np.ones(len(enhanced_member_similarities)),
        np.zeros(len(non_member_similarities))
    ])
    y_scores = np.concatenate([enhanced_member_similarities, non_member_similarities])
    
    # Check for NaN in y_scores before computing metrics
    if np.any(np.isnan(y_scores)) or np.any(np.isinf(y_scores)):
        print(f"Warning: y_scores contain NaN or Inf. Replacing NaN with 0.0 and Inf with finite max/min.")
        y_scores = np.nan_to_num(y_scores, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    
    metrics, fpr, tpr = compute_mia_metrics(y_true, y_scores, train_threshold)
    metrics.update({
        'mean_difference': mean_difference,
        'hyper_lambda': hyper_lambda,
        'member_mean_similarity': float(np.mean(member_similarities)),
        'enhanced_member_mean_similarity': float(np.mean(enhanced_member_similarities)),
        'non_member_mean_similarity': float(np.mean(non_member_similarities)),
        'member_std_similarity': float(np.std(member_similarities)),
        'non_member_std_similarity': float(np.std(non_member_similarities)),
    })
    
    return metrics, fpr, tpr


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate membership inference via frequency-domain image steganography",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B-16",
        help="CLIP model name",
    )
    parser.add_argument(
        "--member_image_dir",
        type=str,
        required=True,
        help="Directory containing member images",
    )
    parser.add_argument(
        "--member_token_file",
        type=str,
        default=None,
        help="Path to Flickr8k token file or MSCOCO JSON file",
    )
    parser.add_argument(
        "--member_is_mscoco",
        action="store_true",
        help="If set, member dataset is MSCOCO (uses JSON file instead of token file)",
    )
    parser.add_argument(
        "--stego_image_dir",
        type=str,
        default=None,
        help="Directory containing stego images",
    )
    parser.add_argument(
        "--clean_image_dir",
        type=str,
        default=None,
        help="Directory containing clean images corresponding to stego images",
    )
    parser.add_argument(
        "--non_member_animal_dir",
        type=str,
        default=None,
        help="Directory containing Animal-Image-Caption images",
    )
    parser.add_argument(
        "--non_member_animal_csv",
        type=str,
        default=None,
        help="Path to Animal-Image-Caption captions CSV",
    )
    parser.add_argument(
        "--non_member_mscoco_dir",
        type=str,
        default=None,
        help="Directory containing MSCOCO2017 images",
    )
    parser.add_argument(
        "--non_member_mscoco_json",
        type=str,
        default=None,
        help="Path to MSCOCO2017 captions JSON",
    )
    parser.add_argument(
        "--non_member_flickr30k_dir",
        type=str,
        default=None,
        help="Directory containing Flickr30k_Captains images",
    )
    parser.add_argument(
        "--non_member_flickr30k_token",
        type=str,
        default=None,
        help="Path to Flickr30k_Captains token file",
    )
    parser.add_argument(
        "--non_member_image_caption_matching_dir",
        type=str,
        default=None,
        help="Directory containing Image-Caption-Matching images",
    )
    parser.add_argument(
        "--non_member_image_caption_matching_txt",
        type=str,
        default=None,
        help="Path to Image-Caption-Matching captions.txt file",
    )
    parser.add_argument(
        "--member_is_imagenet1k",
        action="store_true",
        help="If set, member dataset is ImageNet-1k (uses JSON file instead of token file)",
    )
    parser.add_argument(
        "--reference_image_dir",
        type=str,
        default=None,
        help="Directory containing reference images for quality metrics",
    )
    parser.add_argument(
        "--stego_image_dirs",
        type=str,
        nargs="+",
        default=[
            "./datasets/Flickr8k_Captains/stego_Flicker8k_Dataset",
            "./datasets/Flickr8k_Captains/subject_stego_Flicker8k_Dataset",
        ],
        help="Directories containing stego images for quality metrics comparison",
    )
    parser.add_argument(
        "--hyper_lambda",
        type=float,
        default=0.5,
        help="Hyperparameter lambda for threshold calculation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_member_samples",
        type=int,
        default=None,
        help="Maximum number of member samples to use",
    )
    parser.add_argument(
        "--max_non_member_samples",
        type=int,
        default=None,
        help="Maximum number of non-member samples to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--max_quality_samples",
        type=int,
        default=None,
        help="Maximum number of images per stego dataset for quality metrics",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix for output filenames (default: 'Flicker8k_stgeo_')",
    )
    
    args = parser.parse_args()
    
    device = resolve_device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.prefix if args.prefix is not None else "Flicker8k_stgeo_"
    command_script = write_command_script(output_dir, timestamp, prefix)
    print(f"Command script saved to {command_script}")
    
    print("=" * 80)
    print("Membership inference via frequency-domain image steganography")
    print("=" * 80)
    
    # Load model
    model, preprocess = load_model(
        args.model_path,
        args.pretrained_model_path,
        args.model_name,
        device
    )
    
    # Step 1: Compute similarity difference from stego-clean pairs
    print("\n" + "=" * 80)
    print("Step 1: Computing similarity difference from stego-clean image pairs")
    print("=" * 80)
    
    # Load stego and clean datasets
    if args.member_is_mscoco:
        # Use MSCOCO format for stego/clean datasets
        print("Loading stego and clean datasets using MSCOCO format")
        stego_image_paths, stego_texts = load_mscoco_dataset(
            args.stego_image_dir,
            args.member_token_file,
            args.max_member_samples
        )
        clean_image_paths, clean_texts = load_mscoco_dataset(
            args.clean_image_dir,
            args.member_token_file,
            args.max_member_samples
        )
    elif args.member_is_imagenet1k:
        # Use ImageNet-1k format for stego/clean datasets
        print("Loading stego and clean datasets using ImageNet-1k format")
        stego_image_paths, stego_texts = load_imagenet1k_dataset(
            args.stego_image_dir,
            args.member_token_file,
            args.max_member_samples
        )
        clean_image_paths, clean_texts = load_imagenet1k_dataset(
            args.clean_image_dir,
            args.member_token_file,
            args.max_member_samples
        )
    else:
        # Use Flickr8k.token.txt for stego/clean datasets if Annotations.txt doesn't match
        stego_clean_token_file = args.member_token_file
        if 'Annotations.txt' in stego_clean_token_file:
            flickr8k_token_file = stego_clean_token_file.replace('Annotations.txt', 'Flickr8k.token.txt')
            if os.path.exists(flickr8k_token_file):
                print(f"Using Flickr8k.token.txt for stego/clean datasets (Annotations.txt format may not match)")
                stego_clean_token_file = flickr8k_token_file
        
        stego_image_paths, stego_texts = load_flickr8k_dataset(
            args.stego_image_dir,
            stego_clean_token_file,
            args.max_member_samples
        )
        
        clean_image_paths, clean_texts = load_flickr8k_dataset(
            args.clean_image_dir,
            stego_clean_token_file,
            args.max_member_samples
        )
    
    # Ensure one-to-one correspondence
    if len(stego_image_paths) != len(clean_image_paths):
        print(f"Warning: Mismatch in dataset sizes. Stego: {len(stego_image_paths)}, Clean: {len(clean_image_paths)}")
        # Match by filename
        stego_dict = {Path(p).name: (p, t) for p, t in zip(stego_image_paths, stego_texts)}
        clean_dict = {Path(p).name: (p, t) for p, t in zip(clean_image_paths, clean_texts)}
        
        common_names = set(stego_dict.keys()) & set(clean_dict.keys())
        stego_image_paths = [stego_dict[n][0] for n in sorted(common_names)]
        stego_texts = [stego_dict[n][1] for n in sorted(common_names)]
        clean_image_paths = [clean_dict[n][0] for n in sorted(common_names)]
        clean_texts = [clean_dict[n][1] for n in sorted(common_names)]
        
        print(f"Matched {len(stego_image_paths)} image-text pairs")
    
    # Compute similarity difference
    if len(stego_image_paths) > 0 and len(clean_image_paths) > 0:
        differences, mean_difference = compute_similarity_difference(
            model,
            stego_image_paths,
            clean_image_paths,
            stego_texts,
            preprocess,
            args.batch_size,
            device
        )
        
        # Check for NaN
        if np.isnan(mean_difference) or np.isinf(mean_difference):
            print(f"Warning: mean_difference is NaN or Inf. Setting to 0.0")
            mean_difference = 0.0
    else:
        print(f"Warning: No matching stego-clean pairs found. Setting mean_difference to 0.0")
        mean_difference = 0.0
        differences = np.array([])
    
    print(f"\nMean similarity difference (to be added to member similarities): {mean_difference:.4f}")
    
    # Load member dataset
    if args.member_is_mscoco:
        print("Loading MSCOCO2017 as member dataset")
        member_image_paths, member_texts = load_mscoco_dataset(
            args.member_image_dir,
            args.member_token_file,
            args.max_member_samples
        )
    elif args.member_is_imagenet1k:
        print("Loading ImageNet-1k as member dataset")
        member_image_paths, member_texts = load_imagenet1k_dataset(
            args.member_image_dir,
            args.member_token_file,
            args.max_member_samples
        )
    else:
        print("Loading Flickr8k as member dataset")
        member_image_paths, member_texts = load_flickr8k_dataset(
            args.member_image_dir,
            args.member_token_file,
            args.max_member_samples
        )
    
    member_dataset = ImageTextDataset(member_image_paths, member_texts, preprocess)
    member_dataloader = DataLoader(
        member_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load non-member datasets
    # First non-member dataset: MSCOCO2017
    mscoco_image_paths, mscoco_texts = load_mscoco_dataset(
        args.non_member_mscoco_dir,
        args.non_member_mscoco_json,
        args.max_non_member_samples // 2 if args.max_non_member_samples else None
    )
    
    # Second non-member dataset: Image-Caption-Matching or Flickr30k_Captains or Animal-Image-Caption
    if args.non_member_image_caption_matching_dir and args.non_member_image_caption_matching_txt:
        print("Using Image-Caption-Matching as second non-member dataset")
        icm_image_paths, icm_texts = load_image_caption_matching_dataset(
            args.non_member_image_caption_matching_dir,
            args.non_member_image_caption_matching_txt,
            args.max_non_member_samples // 2 if args.max_non_member_samples else None
        )
        # Combine non-member datasets
        non_member_image_paths = mscoco_image_paths + icm_image_paths
        non_member_texts = mscoco_texts + icm_texts
    elif args.non_member_flickr30k_dir and args.non_member_flickr30k_token:
        print("Using Flickr30k_Captains as second non-member dataset")
        flickr30k_image_paths, flickr30k_texts = load_flickr8k_dataset(
            args.non_member_flickr30k_dir,
            args.non_member_flickr30k_token,
            args.max_non_member_samples // 2 if args.max_non_member_samples else None
        )
        # Combine non-member datasets
        non_member_image_paths = mscoco_image_paths + flickr30k_image_paths
        non_member_texts = mscoco_texts + flickr30k_texts
    else:
        print("Using Animal-Image-Caption as second non-member dataset")
        animal_image_paths, animal_texts = load_animal_image_caption_dataset(
            args.non_member_animal_dir,
            args.non_member_animal_csv,
            args.max_non_member_samples // 2 if args.max_non_member_samples else None
        )
        # Combine non-member datasets
        non_member_image_paths = mscoco_image_paths + animal_image_paths
        non_member_texts = mscoco_texts + animal_texts
    
    non_member_dataset = ImageTextDataset(non_member_image_paths, non_member_texts, preprocess)
    non_member_dataloader = DataLoader(
        non_member_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nMember samples: {len(member_dataset)}")
    print(f"Non-member samples: {len(non_member_dataset)}")
    
    # Step 1: Compute threshold from member dataset
    print("\n" + "=" * 80)
    print("Step 1: Computing threshold from member dataset")
    print("=" * 80)
    
    # Compute member similarities for threshold calculation
    print("Computing member similarities for threshold calculation...")
    member_similarities_for_threshold = compute_cosine_similarities(model, member_dataloader, device)
    threshold_tau = float(np.mean(member_similarities_for_threshold))
    member_mean = float(np.mean(member_similarities_for_threshold))
    member_std = float(np.std(member_similarities_for_threshold))
    
    # Format metrics as mean ± std (LaTeX format)
    def format_metric_latex(mean_val, std_val=0.00, decimals=2):
        r"""Format metric as LaTeX: $mean \pm std$"""
        return f"${mean_val:.{decimals}f} \\pm {std_val:.{decimals}f}$"
    
    print(f"Average cosine similarity (threshold τ): {format_metric_latex(threshold_tau, decimals=4)}")
    print(f"Member similarity stats: {format_metric_latex(member_mean, member_std, decimals=4)}")
    
    # Step 2: Evaluate MIA using frequency-domain stego scores
    print("\n" + "=" * 80)
    print("Step 2: Evaluating membership inference (frequency-domain stego)")
    print("=" * 80)
    
    # 2.1: Evaluate MIA on clean member dataset
    # Extract dataset name from member_image_dir
    member_dataset_name = Path(args.member_image_dir).name
    print(f"\n2.1: Evaluating MIA on clean member dataset ({member_dataset_name})")
    print("-" * 80)
    mia_metrics_clean, fpr_clean, tpr_clean = evaluate_mia_stego(
        model,
        member_dataloader,
        non_member_dataloader,
        mean_difference,
        args.hyper_lambda,
        device,
    )
    
    print("\nMIA Evaluation Results (Clean Dataset with Stego Amplification):")
    print(f"  Accuracy: {format_metric_latex(mia_metrics_clean['accuracy'], decimals=4)}")
    print(f"  AUC: {format_metric_latex(mia_metrics_clean['auc'], decimals=4)}")
    print(f"  TPR@1%FPR: {format_metric_latex(mia_metrics_clean['tpr_at_1pct_fpr'], decimals=4)}")
    print(f"  ASR (Attack Success Rate): {format_metric_latex(mia_metrics_clean['asr'], decimals=4)}")
    print(f"  Clean Accuracy: {format_metric_latex(mia_metrics_clean['clean_accuracy'], decimals=4)}")
    print(f"  Threshold: {format_metric_latex(mia_metrics_clean['threshold'], decimals=4)}")
    print(f"  Mean Difference: {format_metric_latex(mia_metrics_clean['mean_difference'], decimals=4)}")
    
    # 2.2: Evaluate MIA on stego member dataset
    # Extract dataset name from stego_image_dir
    stego_dataset_name = Path(args.stego_image_dir).name
    print("\n" + "-" * 80)
    print(f"2.2: Evaluating MIA on stego member dataset ({stego_dataset_name})")
    print("-" * 80)
    
    # Load subject_stego dataset
    sstgeo_image_dir = args.stego_image_dir
    if os.path.exists(sstgeo_image_dir):
        print(f"Loading subject_stego dataset from {sstgeo_image_dir}...")
        
        # Load subject_stego dataset
        if args.member_is_mscoco:
            print("Loading subject_stego dataset using MSCOCO format")
            sstgeo_image_paths, sstgeo_texts = load_mscoco_dataset(
                sstgeo_image_dir,
                args.member_token_file,
                args.max_member_samples
            )
        elif args.member_is_imagenet1k:
            print("Loading subject_stego dataset using ImageNet-1k format")
            sstgeo_image_paths, sstgeo_texts = load_imagenet1k_dataset(
                sstgeo_image_dir,
                args.member_token_file,
                args.max_member_samples
            )
        else:
            # Use Flickr8k.token.txt for subject_stego dataset if Annotations.txt doesn't match
            # Check if we should use Flickr8k.token.txt instead
            sstgeo_token_file = args.member_token_file
            # If member_token_file is Annotations.txt, try Flickr8k.token.txt for subject_stego
            if 'Annotations.txt' in sstgeo_token_file:
                flickr8k_token_file = sstgeo_token_file.replace('Annotations.txt', 'Flickr8k.token.txt')
                if os.path.exists(flickr8k_token_file):
                    print(f"Using Flickr8k.token.txt for subject_stego dataset (Annotations.txt format may not match)")
                    sstgeo_token_file = flickr8k_token_file
            
            sstgeo_image_paths, sstgeo_texts = load_flickr8k_dataset(
                sstgeo_image_dir,
                sstgeo_token_file,
                args.max_member_samples
            )
        
        if len(sstgeo_image_paths) > 0:
            sstgeo_dataset = ImageTextDataset(sstgeo_image_paths, sstgeo_texts, preprocess)
            sstgeo_dataloader = DataLoader(
                sstgeo_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            print(f"Loaded {len(sstgeo_dataset)} image-text pairs from subject_stego dataset")
            
            # Compute similarities for subject_stego dataset
            print("Computing similarities for subject_stego member dataset...")
            sstgeo_member_similarities = compute_cosine_similarities(model, sstgeo_dataloader, device)
            
            # Evaluate MIA on subject_stego dataset
            mia_metrics_sstgeo, fpr_sstgeo, tpr_sstgeo = evaluate_mia_stego(
                model,
                sstgeo_dataloader,
                non_member_dataloader,
                mean_difference,
                args.hyper_lambda,
                device,
                member_similarities=sstgeo_member_similarities,
            )
            
            print("\nMIA Evaluation Results (subject_stego Dataset):")
            print(f"  Accuracy: {format_metric_latex(mia_metrics_sstgeo['accuracy'], decimals=4)}")
            print(f"  AUC: {format_metric_latex(mia_metrics_sstgeo['auc'], decimals=4)}")
            print(f"  TPR@1%FPR: {format_metric_latex(mia_metrics_sstgeo['tpr_at_1pct_fpr'], decimals=4)}")
            print(f"  ASR (Attack Success Rate): {format_metric_latex(mia_metrics_sstgeo['asr'], decimals=4)}")
            print(f"  Clean Accuracy: {format_metric_latex(mia_metrics_sstgeo['clean_accuracy'], decimals=4)}")
            print(f"  Threshold: {format_metric_latex(mia_metrics_sstgeo['threshold'], decimals=4)}")
            print(f"  Mean Difference: {format_metric_latex(mia_metrics_sstgeo['mean_difference'], decimals=4)}")
            
            # Store subject_stego metrics with prefix
            mia_metrics_sstgeo_prefixed = {
                f'sstgeo_{k}': v for k, v in mia_metrics_sstgeo.items()
            }
            mia_metrics_sstgeo_prefixed['sstgeo_fpr'] = fpr_sstgeo.tolist()
            mia_metrics_sstgeo_prefixed['sstgeo_tpr'] = tpr_sstgeo.tolist()
        else:
            print(f"Warning: No images found in subject_stego dataset. Skipping subject_stego MIA evaluation.")
            mia_metrics_sstgeo_prefixed = {}
    else:
        print(f"Warning: subject_stego dataset directory {sstgeo_image_dir} does not exist. Skipping subject_stego MIA evaluation.")
        mia_metrics_sstgeo_prefixed = {}
    
    # Combine all MIA metrics
    all_mia_metrics = {
        **{f'clean_{k}': v for k, v in mia_metrics_clean.items()},
        **mia_metrics_sstgeo_prefixed,
    }
    
    # Store ROC data
    all_mia_metrics['clean_fpr'] = fpr_clean.tolist()
    all_mia_metrics['clean_tpr'] = tpr_clean.tolist()
    
    # Use clean metrics as primary for backward compatibility
    mia_metrics = mia_metrics_clean
    fpr = fpr_clean
    tpr = tpr_clean
    
    # Step 3: Compute image quality metrics for stego datasets
    print("\n" + "=" * 80)
    print("Step 3: Computing image quality metrics for stego datasets")
    print("=" * 80)
    
    # Determine reference directory (clean images)
    ref_dir = Path(args.reference_image_dir) if args.reference_image_dir else Path(args.member_image_dir)
    if not ref_dir.exists():
        print(f"Warning: Reference directory {ref_dir} does not exist. Using member image directory.")
        ref_dir = Path(args.member_image_dir)
    
    print(f"Reference images (clean): {ref_dir}")
    
    # Compute quality metrics for each stego dataset
    all_quality_metrics = {}
    
    for stego_dir_str in args.stego_image_dirs:
        stego_dir = Path(stego_dir_str)
        if not stego_dir.exists():
            print(f"Warning: Stego directory {stego_dir} does not exist. Skipping.")
            continue
        
        print(f"\nComputing quality metrics for: {stego_dir.name}")
        print(f"  Stego images directory: {stego_dir}")
        
        # Get all images in stego directory
        stego_image_files = list(stego_dir.glob("*.jpg")) + list(stego_dir.glob("*.jpeg")) + list(stego_dir.glob("*.png"))
        if args.max_quality_samples is not None:
            stego_image_files = stego_image_files[:args.max_quality_samples]
        
        if not stego_image_files:
            print(f"  No images found in {stego_dir}. Skipping.")
            continue
        
        print(f"  Found {len(stego_image_files)} stego images")
        
        # Compute metrics for each stego image compared to reference
        mse_values = []
        ssim_values = []
        psnr_values = []
        l2_norm_values = []
        
        matched_count = 0
        for stego_img_path in tqdm(stego_image_files, desc=f"  Processing {stego_dir.name}"):
            ref_img_path = ref_dir / stego_img_path.name
            
            if not ref_img_path.exists():
                # Try to find matching image with different extension
                ref_img_path = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF']:
                    candidate = ref_dir / (stego_img_path.stem + ext)
                    if candidate.exists():
                        ref_img_path = candidate
                        break
                
                if ref_img_path is None:
                    continue
            
            try:
                ref_img = load_image(ref_img_path)
                stego_img = load_image(stego_img_path)
                
                # Ensure same shape
                if ref_img.shape != stego_img.shape:
                    from PIL import Image
                    ref_pil = Image.fromarray(ref_img)
                    stego_pil = Image.fromarray(stego_img)
                    # Resize to match
                    target_size = (stego_img.shape[1], stego_img.shape[0])
                    ref_pil = ref_pil.resize(target_size, Image.Resampling.LANCZOS)
                    ref_img = np.array(ref_pil)
                
                # Compute metrics using src.evaluate helpers
                mse_value, ssim_value, psnr_value, l2_norm_value = compute_image_quality_metrics(
                    ref_img,
                    stego_img,
                )
                
                mse_values.append(mse_value)
                ssim_values.append(ssim_value)
                psnr_values.append(psnr_value)
                l2_norm_values.append(l2_norm_value)
                
                matched_count += 1
            except Exception as e:
                print(f"  Error processing {stego_img_path.name}: {e}")
                continue
        
        if matched_count == 0:
            print(f"  No matching images found. Skipping quality metrics for {stego_dir.name}.")
            continue
        
        # Compute average metrics with standard deviation
        stego_metrics = {
            'mse_mean': float(np.mean(mse_values)) if mse_values else 0.0,
            'mse_std': float(np.std(mse_values)) if mse_values else 0.0,
            'ssim_mean': float(np.mean(ssim_values)) if ssim_values else 0.0,
            'ssim_std': float(np.std(ssim_values)) if ssim_values else 0.0,
            'psnr_mean': float(np.mean(psnr_values)) if psnr_values else 0.0,
            'psnr_std': float(np.std(psnr_values)) if psnr_values else 0.0,
            'l2_norm_mean': float(np.mean(l2_norm_values)) if l2_norm_values else 0.0,
            'l2_norm_std': float(np.std(l2_norm_values)) if l2_norm_values else 0.0,
            'matched_images': matched_count,
        }
        
        # Format metrics as mean ± std
        def format_metric(mean_val, std_val, decimals=2):
            """Format metric as: mean ± std"""
            return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"
        
        stego_metrics['mse'] = format_metric(stego_metrics['mse_mean'], stego_metrics['mse_std'], decimals=4)
        stego_metrics['ssim'] = format_metric(stego_metrics['ssim_mean'], stego_metrics['ssim_std'], decimals=4)
        stego_metrics['psnr'] = format_metric(stego_metrics['psnr_mean'], stego_metrics['psnr_std'], decimals=2)
        
        all_quality_metrics[stego_dir.name] = stego_metrics
        
        print(f"\n  Image Quality Metrics for {stego_dir.name} ({matched_count} images):")
        print(f"    MSE: {stego_metrics['mse']}")
        print(f"    SSIM: {stego_metrics['ssim']}")
        print(f"    PSNR: {stego_metrics['psnr']}")
    
    # Use the first stego dataset metrics for overall reporting (or average if multiple)
    if all_quality_metrics:
        # Average across all stego datasets (weighted by number of images)
        total_images = sum(m['matched_images'] for m in all_quality_metrics.values())
        
        # Calculate weighted mean and pooled standard deviation
        mse_weighted_mean = sum(m['mse_mean'] * m['matched_images'] for m in all_quality_metrics.values()) / total_images if total_images > 0 else 0.0
        ssim_weighted_mean = sum(m['ssim_mean'] * m['matched_images'] for m in all_quality_metrics.values()) / total_images if total_images > 0 else 0.0
        psnr_weighted_mean = sum(m['psnr_mean'] * m['matched_images'] for m in all_quality_metrics.values()) / total_images if total_images > 0 else 0.0
        
        # Calculate pooled standard deviation
        mse_pooled_std = np.sqrt(sum((m['mse_std']**2 + (m['mse_mean'] - mse_weighted_mean)**2) * m['matched_images'] 
                                     for m in all_quality_metrics.values()) / total_images) if total_images > 0 else 0.0
        ssim_pooled_std = np.sqrt(sum((m['ssim_std']**2 + (m['ssim_mean'] - ssim_weighted_mean)**2) * m['matched_images'] 
                                     for m in all_quality_metrics.values()) / total_images) if total_images > 0 else 0.0
        psnr_pooled_std = np.sqrt(sum((m['psnr_std']**2 + (m['psnr_mean'] - psnr_weighted_mean)**2) * m['matched_images'] 
                                     for m in all_quality_metrics.values()) / total_images) if total_images > 0 else 0.0
        
        quality_metrics = {
            'mse_mean': float(mse_weighted_mean),
            'mse_std': float(mse_pooled_std),
            'ssim_mean': float(ssim_weighted_mean),
            'ssim_std': float(ssim_pooled_std),
            'psnr_mean': float(psnr_weighted_mean),
            'psnr_std': float(psnr_pooled_std),
        }
        
        # Format as mean ± std
        def format_metric(mean_val, std_val, decimals=2):
            """Format metric as: mean ± std"""
            return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"
        
        quality_metrics['mse'] = format_metric(quality_metrics['mse_mean'], quality_metrics['mse_std'], decimals=4)
        quality_metrics['ssim'] = format_metric(quality_metrics['ssim_mean'], quality_metrics['ssim_std'], decimals=4)
        quality_metrics['psnr'] = format_metric(quality_metrics['psnr_mean'], quality_metrics['psnr_std'], decimals=2)
    else:
        quality_metrics = {
            'mse_mean': 0.0,
            'mse_std': 0.0,
            'ssim_mean': 0.0,
            'ssim_std': 0.0,
            'psnr_mean': 0.0,
            'psnr_std': 0.0,
            'mse': "0.0000 ± 0.0000",
            'ssim': "0.0000 ± 0.0000",
            'psnr': "0.00 ± 0.00",
        }
    
    print("\nOverall Image Quality Metrics (averaged across stego datasets):")
    print(f"  MSE: {quality_metrics['mse']}")
    print(f"  SSIM: {quality_metrics['ssim']}")
    print(f"  PSNR: {quality_metrics['psnr']}")
    
    # Combine all metrics
    all_metrics = {
        **mia_metrics,  # Primary metrics (clean dataset) for backward compatibility
        **all_mia_metrics,  # All MIA metrics including clean and sstgeo
        **quality_metrics,
        'quality_metrics_by_dataset': all_quality_metrics,
        'timestamp': timestamp,
        'model_path': args.model_path,
        'member_samples': len(member_dataset),
        'non_member_samples': len(non_member_dataset),
        'reference_image_dir': str(ref_dir),
        'stego_image_dirs': [str(Path(d).name) for d in args.stego_image_dirs],
    }
    
    # Save results
    results_file = output_dir / f"{prefix}stego_mia_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Save ROC curve data
    roc_data = {
        'clean': {
            'fpr': fpr_clean.tolist(),
            'tpr': tpr_clean.tolist(),
            'auc': mia_metrics_clean['auc'],
            'threshold': mia_metrics_clean['threshold'],
        }
    }
    
    # Add subject_stego ROC data if available
    if mia_metrics_sstgeo_prefixed and 'sstgeo_fpr' in mia_metrics_sstgeo_prefixed:
        roc_data['sstgeo'] = {
            'fpr': mia_metrics_sstgeo_prefixed['sstgeo_fpr'],
            'tpr': mia_metrics_sstgeo_prefixed['sstgeo_tpr'],
            'auc': mia_metrics_sstgeo_prefixed.get('sstgeo_auc', 0.0),
            'threshold': mia_metrics_sstgeo_prefixed.get('sstgeo_threshold', 0.0),
        }
    
    roc_file = output_dir / f"{prefix}stego_mia_roc_{timestamp}.json"
    with open(roc_file, 'w') as f:
        json.dump(roc_data, f, indent=2)
    print(f"ROC curve data saved to {roc_file}")
    
    # Save detailed results to CSV
    results_df = pd.DataFrame([all_metrics])
    csv_file = output_dir / f"{prefix}stego_mia_results_{timestamp}.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"Results CSV saved to {csv_file}")
    
    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
