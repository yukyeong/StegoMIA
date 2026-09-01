#!/usr/bin/env python3
"""
Membership Inference Attack (MIA) evaluation using cosine-similarity membership inference.
This script evaluates MIA performance on CLIP models using cosine similarity threshold.
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

# LPIPS removed from evaluation metrics


def resolve_device(device: str) -> str:
    """Resolve device string to an available device."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return device


def write_command_script(output_dir: Path, timestamp: str) -> Path:
    """Write a reproducible command script to the output directory."""
    script_path = output_dir / f"mia_csa_command_{timestamp}.sh"
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
        token_file: Path to Flickr8k.token.txt file.
        max_samples: Maximum number of samples to load.
    
    Returns:
        Tuple of (image_paths, texts) lists.
    """
    print(f"Loading Flickr8k dataset from {image_dir}...")
    
    # Load captions
    image_to_captions = {}
    with open(token_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_name = parts[0].split('#')[0]
                caption = parts[1].strip()
                if img_name not in image_to_captions:
                    image_to_captions[img_name] = []
                image_to_captions[img_name].append(caption)
    
    # Get image paths and corresponding captions
    image_paths = []
    texts = []
    image_dir_path = Path(image_dir)
    
    for img_name, captions in image_to_captions.items():
        img_path = image_dir_path / img_name
        if img_path.exists():
            # Use the first caption for each image
            image_paths.append(str(img_path))
            texts.append(captions[0])
    
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
    
    for _, row in df.iterrows():
        img_name = row['image']
        caption = row['caption']
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
    
    for img_id, img_info in image_id_to_info.items():
        if img_id in image_id_to_captions:
            img_name = img_info['file_name']
            img_path = image_dir_path / img_name
            
            if img_path.exists():
                # Use the first caption for each image
                image_paths.append(str(img_path))
                texts.append(image_id_to_captions[img_id][0])
    
    if max_samples:
        image_paths = image_paths[:max_samples]
        texts = texts[:max_samples]
    
    print(f"Loaded {len(image_paths)} image-text pairs from MSCOCO2017")
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


def evaluate_mia(
    model,
    member_dataloader,
    non_member_dataloader,
    threshold: float,
    device: str = "cuda",
    member_similarities: Optional[np.ndarray] = None,
    non_member_similarities: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate Membership Inference Attack performance.
    
    Args:
        model: CLIP model.
        member_dataloader: DataLoader for member samples.
        non_member_dataloader: DataLoader for non-member samples.
        threshold: Cosine similarity threshold.
        device: Device to run computation on.
    
    Returns:
        Tuple of (metrics, fpr, tpr). clean_accuracy is true-negative-rate on non-members.
    """
    if member_similarities is None:
        print("Computing similarities for member samples...")
        member_similarities = compute_cosine_similarities(model, member_dataloader, device)
    if non_member_similarities is None:
        print("Computing similarities for non-member samples...")
        non_member_similarities = compute_cosine_similarities(model, non_member_dataloader, device)
    
    # Create labels: 1 for members, 0 for non-members
    y_true = np.concatenate([
        np.ones(len(member_similarities)),
        np.zeros(len(non_member_similarities))
    ])
    y_scores = np.concatenate([member_similarities, non_member_similarities])
    
    metrics, fpr, tpr = compute_mia_metrics(y_true, y_scores, threshold)
    metrics.update({
        'member_mean_similarity': float(np.mean(member_similarities)),
        'non_member_mean_similarity': float(np.mean(non_member_similarities)),
        'member_std_similarity': float(np.std(member_similarities)),
        'non_member_std_similarity': float(np.std(non_member_similarities)),
    })
    
    return metrics, fpr, tpr


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate Membership Inference Attack using Cosine Similarity"
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
        help="Path to Flickr8k token file",
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
        "--reference_image_dir",
        type=str,
        default=None,
        help="Directory containing reference images for quality metrics (clean images)",
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
        "--max_quality_samples",
        type=int,
        default=None,
        help="Maximum number of images per stego dataset for quality metrics (default: all)",
    )
    parser.add_argument(
        "--hyper_lambda",
        type=float,
        default=0.5,
        help="Attack intensity for threshold = mean(nonmember) + lambda * std(nonmember)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="Flicker8k_csa_",
        help="Prefix for output filenames",
    )
    parser.add_argument(
        "--member_is_mscoco",
        action="store_true",
        help="Member dataset is MSCOCO JSON format",
    )
    parser.add_argument(
        "--member_is_imagenet1k",
        action="store_true",
        help="Member dataset is ImageNet JSON format",
    )
    parser.add_argument(
        "--stego_image_dir",
        type=str,
        default=None,
        help="Stego member directory for subject_stego MIA evaluation",
    )
    
    args = parser.parse_args()

    device = resolve_device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    command_script = write_command_script(output_dir, timestamp)
    print(f"Command script saved to {command_script}")
    
    print("=" * 80)
    print("Membership Inference Attack Evaluation using Cosine Similarity")
    print("=" * 80)
    
    # Load model
    model, preprocess = load_model(
        args.model_path,
        args.pretrained_model_path,
        args.model_name,
        device
    )
    
    # Load member dataset
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
    animal_image_paths, animal_texts = load_animal_image_caption_dataset(
        args.non_member_animal_dir,
        args.non_member_animal_csv,
        args.max_non_member_samples // 2 if args.max_non_member_samples else None
    )
    
    mscoco_image_paths, mscoco_texts = load_mscoco_dataset(
        args.non_member_mscoco_dir,
        args.non_member_mscoco_json,
        args.max_non_member_samples // 2 if args.max_non_member_samples else None
    )
    
    # Combine non-member datasets
    non_member_image_paths = animal_image_paths + mscoco_image_paths
    non_member_texts = animal_texts + mscoco_texts
    
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
    
    # Step 1: Compute similarities and threshold from non-members
    # threshold = mean(nonmember) + hyper_lambda * std(nonmember)
    # Note: AUC / TPR@1%FPR depend only on y_scores ranking, so they stay constant
    # across hyper_lambda; Accuracy / ASR / Clean ACC / Threshold change with λ.
    print("\n" + "=" * 80)
    print("Step 1: Computing cosine similarities and cosine-similarity threshold")
    print("=" * 80)
    member_similarities = compute_cosine_similarities(model, member_dataloader, device)
    non_member_similarities = compute_cosine_similarities(model, non_member_dataloader, device)
    threshold = float(np.mean(non_member_similarities) + args.hyper_lambda * np.std(non_member_similarities))
    print(f"hyper_lambda (attack intensity): {args.hyper_lambda}")
    print(f"Non-member mean={np.mean(non_member_similarities):.4f}, std={np.std(non_member_similarities):.4f}")
    print(f"Threshold τ = mean + λ*std = {threshold:.4f}")
    print(f"Member similarity stats: mean={np.mean(member_similarities):.4f}, std={np.std(member_similarities):.4f}")
    
    # Step 2: Evaluate MIA
    print("\n" + "=" * 80)
    print("Step 2: Evaluating cosine-similarity membership inference")
    print("=" * 80)
    
    # 2.1: Evaluate MIA on clean member dataset
    print("\n2.1: Evaluating MIA on clean member dataset")
    print("-" * 80)
    mia_metrics_clean, fpr_clean, tpr_clean = evaluate_mia(
        model,
        member_dataloader,
        non_member_dataloader,
        threshold,
        device,
        member_similarities=member_similarities,
        non_member_similarities=non_member_similarities,
    )
    
    print("\nMIA Evaluation Results (Clean Dataset):")
    print(f"  Accuracy: {mia_metrics_clean['accuracy']:.4f}")
    print(f"  AUC: {mia_metrics_clean['auc']:.4f}")
    print(f"  TPR@1%FPR: {mia_metrics_clean['tpr_at_1pct_fpr']:.4f}")
    print(f"  ASR (Attack Success Rate): {mia_metrics_clean['asr']:.4f}")
    print(f"  Clean Accuracy: {mia_metrics_clean['clean_accuracy']:.4f}")
    print(f"  Threshold: {mia_metrics_clean['threshold']:.4f}")
    
    # 2.2: Evaluate MIA on stego member dataset (subject_stego_Flicker8k_Dataset)
    print("\n" + "-" * 80)
    print("2.2: Evaluating MIA on stego member dataset (subject_stego_Flicker8k_Dataset)")
    print("-" * 80)
    
    # Load subject_stego dataset
    sstgeo_image_dir = args.stego_image_dir
    if os.path.exists(sstgeo_image_dir):
        print(f"Loading subject_stego dataset from {sstgeo_image_dir}...")
        
        # Load images and texts for subject_stego dataset (using same token file)
        sstgeo_image_paths, sstgeo_texts = load_flickr8k_dataset(
            sstgeo_image_dir,
            args.member_token_file,
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
            
            # Same threshold rule as clean split: mean(nonmember) + λ * std
            print("Computing subject_stego member similarities...")
            sstgeo_member_similarities = compute_cosine_similarities(model, sstgeo_dataloader, device)
            sstgeo_threshold = threshold
            print(f"Using shared cosine-similarity threshold τ={sstgeo_threshold:.4f} (λ={args.hyper_lambda})")
            
            # Evaluate MIA on subject_stego dataset
            mia_metrics_sstgeo, fpr_sstgeo, tpr_sstgeo = evaluate_mia(
                model,
                sstgeo_dataloader,
                non_member_dataloader,
                sstgeo_threshold,
                device,
                member_similarities=sstgeo_member_similarities,
                non_member_similarities=non_member_similarities,
            )
            
            print("\nMIA Evaluation Results (subject_stego Dataset):")
            print(f"  Accuracy: {mia_metrics_sstgeo['accuracy']:.4f}")
            print(f"  AUC: {mia_metrics_sstgeo['auc']:.4f}")
            print(f"  TPR@1%FPR: {mia_metrics_sstgeo['tpr_at_1pct_fpr']:.4f}")
            print(f"  ASR (Attack Success Rate): {mia_metrics_sstgeo['asr']:.4f}")
            print(f"  Clean Accuracy: {mia_metrics_sstgeo['clean_accuracy']:.4f}")
            print(f"  Threshold: {mia_metrics_sstgeo['threshold']:.4f}")
            
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
        
        # Get all images in stego directory (case-insensitive extensions)
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        stego_image_files = sorted(
            [
                path
                for path in stego_dir.iterdir()
                if path.is_file() and path.suffix.lower() in supported_exts
            ]
        )
        if args.max_quality_samples is not None:
            stego_image_files = stego_image_files[:args.max_quality_samples]
            print(f"  Limiting to {args.max_quality_samples} images for quality metrics")
        
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
        'hyper_lambda': args.hyper_lambda,
        'member_samples': len(member_dataset),
        'non_member_samples': len(non_member_dataset),
        'reference_image_dir': str(ref_dir),
        'stego_image_dirs': [str(Path(d).name) for d in args.stego_image_dirs],
    }
    
    # Save results
    results_file = output_dir / f"{args.prefix}results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Save ROC curve data
    roc_data = {
        'clean': {
            'fpr': fpr_clean.tolist(),
            'tpr': tpr_clean.tolist(),
            'auc': mia_metrics_clean['auc'],
            'threshold': threshold,
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
    
    roc_file = output_dir / f"{args.prefix}roc_{timestamp}.json"
    with open(roc_file, 'w') as f:
        json.dump(roc_data, f, indent=2)
    print(f"ROC curve data saved to {roc_file}")
    
    # Save detailed results to CSV
    results_df = pd.DataFrame([all_metrics])
    csv_file = output_dir / f"{args.prefix}results_{timestamp}.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"Results CSV saved to {csv_file}")
    
    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
