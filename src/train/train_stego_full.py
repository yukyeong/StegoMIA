#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train CLIP model with full steganography dataset (517 images).

This script trains CLIP model using all 517 stego images with their captions,
using contrastive loss + CLIP Consistency + pixel fidelity loss.
"""

import os
import sys
import math
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Use the installed open_clip package.
from open_clip import create_model_and_transforms, get_tokenizer, ClipLoss
# Optional open_clip training helpers
try:
    from training.precision import get_autocast
except ImportError:
    from contextlib import suppress
    def get_autocast(precision):
        if precision == 'amp':
            return torch.cuda.amp.autocast
        return suppress

try:
    from training.distributed import is_master, init_distributed_device, world_info_from_env
except ImportError:
    def is_master(args, local=False):
        return True
    def init_distributed_device(args):
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def world_info_from_env():
        return 0, 0, 1

try:
    from training.logger import setup_logging
except ImportError:
    def setup_logging(log_file, level, include_host=False):
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
        logging.root.setLevel(level)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logging.root.addHandler(stream_handler)
        if log_file:
            file_handler = logging.FileHandler(filename=log_file)
            file_handler.setFormatter(formatter)
            logging.root.addHandler(file_handler)

try:
    from training.scheduler import cosine_lr
except ImportError:
    def assign_learning_rate(optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
    def cosine_lr(optimizer, base_lr, warmup_length, steps):
        def _lr_adjuster(step):
            if step < warmup_length:
                lr = base_lr * (step + 1) / warmup_length
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(optimizer, lr)
            return lr
        return _lr_adjuster


class StegoFullDataset(Dataset):
    """Dataset for loading all stego images with their captions."""
    
    def __init__(self, stego_image_dir, clean_image_dir, caption_file, transforms, tokenizer,
                 stego_sample_ratio=None, max_samples=None, use_imagenet_format=False, use_coco_format=False):
        """
        Initialize dataset.
        
        Args:
            stego_image_dir: Directory containing stego images
            clean_image_dir: Directory containing clean images
            caption_file: Path to caption file (Flickr8k.lemma.token.txt, ImageNet JSON, or COCO JSON)
            transforms: Image transforms
            tokenizer: Text tokenizer
            stego_sample_ratio: Ratio to sample from stego images (e.g., 0.05 for 5%)
            max_samples: Optional cap on total number of pairs (for quick verification)
            use_imagenet_format: Whether to use ImageNet JSON format for captions
            use_coco_format: Whether to use COCO JSON format for captions
        """
        self.stego_image_dir = Path(stego_image_dir)
        self.clean_image_dir = Path(clean_image_dir)
        self.transforms = transforms
        self.tokenize = tokenizer
        self.use_imagenet_format = use_imagenet_format
        self.use_coco_format = use_coco_format
        
        # Get all stego image files (support multiple extensions)
        stego_files_all = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            stego_files_all.extend(list(self.stego_image_dir.glob(ext)))
        
        # Get stego file names (stem for matching)
        stego_file_stems = {f.stem: f.name for f in stego_files_all}
        stego_file_names = set(f.name for f in stego_files_all)
        
        # Load captions first to determine total pairs
        self.pairs = []
        
        if use_coco_format:
            # Load COCO JSON format: {"images": [...], "annotations": [...]}
            import json
            with open(caption_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            # Build image_id to file_name mapping
            image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
            
            # Build image_id to captions mapping (use first caption only)
            image_id_to_caption = {}
            for ann in coco_data['annotations']:
                image_id = ann['image_id']
                if image_id not in image_id_to_caption:
                    image_id_to_caption[image_id] = ann['caption']
            
            # Build filename to caption mapping
            filename_to_caption = {}
            for image_id, filename in image_id_to_filename.items():
                if image_id in image_id_to_caption:
                    filename_to_caption[filename] = image_id_to_caption[image_id]
            
            # Build stem to caption mapping for flexible matching (handle .jpg vs .png)
            stem_to_caption = {}
            for filename, caption in filename_to_caption.items():
                stem = Path(filename).stem
                stem_to_caption[stem] = caption
            
            # Calculate total image-caption pairs count
            total_pairs_count = len(filename_to_caption)
            
            # Sample stego images based on pairs ratio
            if stego_sample_ratio is not None and stego_sample_ratio > 0:
                # Calculate X = 5% of total pairs
                target_stego_count = int(total_pairs_count * stego_sample_ratio)
                logging.info(f"Target stego pairs: {target_stego_count} ({stego_sample_ratio*100:.1f}% of {total_pairs_count} total pairs)")
                
                # Sample X images from stego directory
                sorted_stego_files = sorted(stego_file_names)
                sampled_stego_count = min(target_stego_count, len(sorted_stego_files))
                sampled_stego_files = sorted_stego_files[:sampled_stego_count]
                logging.info(f"Sampled {sampled_stego_count} stego images from {len(sorted_stego_files)} total stego images")
                stego_file_names = set(sampled_stego_files)
                # Update stego_file_stems to only include sampled files
                stego_file_stems = {stem: name for stem, name in stego_file_stems.items() if name in stego_file_names}
            
            # Process stego images with captions (match by stem to handle different extensions)
            for img_name in stego_file_names:
                img_stem = Path(img_name).stem
                if img_stem in stem_to_caption:
                    self.pairs.append((img_name, stem_to_caption[img_stem], True))  # True indicates stego image
            
            # Process clean images (not in stego set)
            clean_image_dir_path = Path(clean_image_dir)
            clean_files_all = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                clean_files_all.extend(list(clean_image_dir_path.rglob(ext)))
            
            clean_file_stems = {f.stem: f.name for f in clean_files_all}
            
            for clean_file in clean_files_all:
                clean_stem = clean_file.stem
                clean_name = clean_file.name
                
                # Skip if this image is in stego set
                if clean_stem in stego_file_stems:
                    continue
                
                # Get caption if available (match by stem)
                if clean_stem in stem_to_caption:
                    self.pairs.append((clean_name, stem_to_caption[clean_stem], False))  # False indicates clean image
            
            logging.info(f"Loaded {len(self.pairs)} image-caption pairs from COCO JSON")
            stego_count = sum(1 for _, _, is_stego in self.pairs if is_stego)
            clean_count = sum(1 for _, _, is_stego in self.pairs if not is_stego)
            logging.info(f"  Stego images: {stego_count}")
            logging.info(f"  Clean images: {clean_count}")
        elif use_imagenet_format:
            # Load ImageNet JSON format: {"image_name": ["caption1", "caption2", ...]}
            import json
            with open(caption_file, 'r', encoding='utf-8') as f:
                captions_dict = json.load(f)
            
            # Calculate total image-caption pairs count
            total_pairs_count = len(captions_dict)
            
            # Sample stego images based on pairs ratio
            if stego_sample_ratio is not None and stego_sample_ratio > 0:
                # Calculate X = 5% of total pairs
                target_stego_count = int(total_pairs_count * stego_sample_ratio)
                logging.info(f"Target stego pairs: {target_stego_count} ({stego_sample_ratio*100:.1f}% of {total_pairs_count} total pairs)")
                
                # Sample X images from stego directory
                sorted_stego_files = sorted(stego_file_names)
                sampled_stego_count = min(target_stego_count, len(sorted_stego_files))
                sampled_stego_files = sorted_stego_files[:sampled_stego_count]
                logging.info(f"Sampled {sampled_stego_count} stego images from {len(sorted_stego_files)} total stego images")
                stego_file_names = set(sampled_stego_files)
                # Update stego_file_stems to only include sampled files
                stego_file_stems = {stem: name for stem, name in stego_file_stems.items() if name in stego_file_names}
            
            # Process stego images with captions (match by stem to handle different extensions)
            for img_name in stego_file_names:
                img_stem = Path(img_name).stem
                if img_stem in captions_dict:
                    captions = captions_dict[img_stem]
                    if captions and len(captions) > 0:
                        # Use first caption only
                        self.pairs.append((img_name, captions[0], True))  # True indicates stego image
            
            # Process clean images (not in stego set)
            clean_image_dir_path = Path(clean_image_dir)
            clean_files_all = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                clean_files_all.extend(list(clean_image_dir_path.rglob(ext)))
            
            clean_file_stems = {f.stem: f.name for f in clean_files_all}
            
            for clean_file in clean_files_all:
                clean_stem = clean_file.stem
                clean_name = clean_file.name
                
                # Skip if this image is in stego set
                if clean_stem in stego_file_stems:
                    continue
                
                # Get caption if available (match by stem)
                if clean_stem in captions_dict:
                    captions = captions_dict[clean_stem]
                    if captions and len(captions) > 0:
                        # Use first caption only
                        self.pairs.append((clean_name, captions[0], False))  # False indicates clean image
            
            logging.info(f"Loaded {len(self.pairs)} image-caption pairs from ImageNet JSON")
            stego_count = sum(1 for _, _, is_stego in self.pairs if is_stego)
            clean_count = sum(1 for _, _, is_stego in self.pairs if not is_stego)
            logging.info(f"  Stego images: {stego_count}")
            logging.info(f"  Clean images: {clean_count}")
        else:
            # Load Flickr8k format: image.jpg#0	caption
            # Build filename to captions mapping (use first caption only)
            filename_to_caption = {}
            for line in open(caption_file, 'r', encoding='utf-8'):
                line = line.strip()
                if not line:
                    continue
                
                # Parse format: image.jpg#0	caption
                if '#' in line and '\t' in line:
                    hash_idx = line.index('#')
                    tab_idx = line.index('\t')
                    img_name = line[:hash_idx]
                    caption = line[tab_idx + 1:].strip()
                    
                    # Store first caption for each image
                    if img_name not in filename_to_caption:
                        filename_to_caption[img_name] = caption
            
            # Calculate total image-caption pairs count
            total_pairs_count = len(filename_to_caption)
            
            # Sample stego images based on pairs ratio
            if stego_sample_ratio is not None and stego_sample_ratio > 0:
                # Calculate X = 5% of total pairs
                target_stego_count = int(total_pairs_count * stego_sample_ratio)
                logging.info(f"Target stego pairs: {target_stego_count} ({stego_sample_ratio*100:.1f}% of {total_pairs_count} total pairs)")
                
                # Sample X images from stego directory
                sorted_stego_files = sorted(stego_file_names)
                sampled_stego_count = min(target_stego_count, len(sorted_stego_files))
                sampled_stego_files = sorted_stego_files[:sampled_stego_count]
                logging.info(f"Sampled {sampled_stego_count} stego images from {len(sorted_stego_files)} total stego images")
                stego_file_names = set(sampled_stego_files)
                # Update stego_file_stems to only include sampled files
                stego_file_stems = {stem: name for stem, name in stego_file_stems.items() if name in stego_file_names}
            
            # Process stego images with captions
            for img_name in stego_file_names:
                if img_name in filename_to_caption:
                    self.pairs.append((img_name, filename_to_caption[img_name], True))  # True indicates stego image
            
            # Process clean images (not in stego set)
            clean_image_dir_path = Path(clean_image_dir)
            clean_files_all = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                clean_files_all.extend(list(clean_image_dir_path.rglob(ext)))
            
            clean_file_stems = {f.stem: f.name for f in clean_files_all}
            
            for clean_file in clean_files_all:
                clean_stem = clean_file.stem
                clean_name = clean_file.name
                
                # Skip if this image is in stego set
                if clean_stem in stego_file_stems:
                    continue
                
                # Get caption if available
                if clean_name in filename_to_caption:
                    self.pairs.append((clean_name, filename_to_caption[clean_name], False))  # False indicates clean image
            
            logging.info(f"Loaded {len(self.pairs)} image-caption pairs from Flickr8k format")
            stego_count = sum(1 for _, _, is_stego in self.pairs if is_stego)
            clean_count = sum(1 for _, _, is_stego in self.pairs if not is_stego)
            logging.info(f"  Stego images: {stego_count}")
            logging.info(f"  Clean images: {clean_count}")

        if max_samples is not None and max_samples > 0 and len(self.pairs) > max_samples:
            logging.info(f"Limiting dataset to {max_samples} samples for verification")
            self.pairs = self.pairs[:max_samples]
            stego_count = sum(1 for pair in self.pairs if len(pair) < 3 or pair[2])
            clean_count = len(self.pairs) - stego_count
            logging.info(f"After limiting: {len(self.pairs)} samples (stego {stego_count}, clean {clean_count})")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        if len(self.pairs[idx]) == 3:
            img_name, caption, is_stego = self.pairs[idx]
        else:
            # Backward compatibility
            img_name, caption = self.pairs[idx]
            is_stego = True
        
        if is_stego:
            # Load stego image
            stego_image_path = self.stego_image_dir / img_name
            stego_image = Image.open(stego_image_path).convert('RGB')
            stego_image_tensor = self.transforms(stego_image)
            
            # Load clean image (same name from clean directory)
            clean_image_path = self.clean_image_dir / img_name
            if not clean_image_path.exists():
                # Try to find by stem (for ImageNet nested structure)
                img_stem = Path(img_name).stem
                clean_image_path = None
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    potential_path = self.clean_image_dir / f"{img_stem}{ext}"
                    if potential_path.exists():
                        clean_image_path = potential_path
                        break
                
                if clean_image_path is None:
                    # If clean image not found, use stego image as clean
                    clean_image = stego_image.copy()
                else:
                    clean_image = Image.open(clean_image_path).convert('RGB')
            else:
                clean_image = Image.open(clean_image_path).convert('RGB')
            
            clean_image_tensor = self.transforms(clean_image)
        else:
            # Clean image only (no stego version)
            clean_image_path = self.clean_image_dir / img_name
            if not clean_image_path.exists():
                # Try to find by stem (for ImageNet nested structure)
                img_stem = Path(img_name).stem
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    potential_path = self.clean_image_dir / f"{img_stem}{ext}"
                    if potential_path.exists():
                        clean_image_path = potential_path
                        break
                
                if not clean_image_path.exists():
                    raise FileNotFoundError(f"Clean image not found: {img_name}")
            
            clean_image = Image.open(clean_image_path).convert('RGB')
            clean_image_tensor = self.transforms(clean_image)
            # For clean-only images, use clean image as both stego and clean
            stego_image_tensor = clean_image_tensor
        
        # Tokenize text
        text_tensor = self.tokenize([caption])[0]
        
        return {
            'stego_image': stego_image_tensor,
            'clean_image': clean_image_tensor,
            'text': text_tensor,
            'image_name': img_name
        }


def collate_fn(batch):
    """Custom collate function for batching."""
    stego_images = torch.stack([item['stego_image'] for item in batch])
    clean_images = torch.stack([item['clean_image'] for item in batch])
    texts = torch.stack([item['text'] for item in batch])
    
    return {
        'stego_images': stego_images,
        'clean_images': clean_images,
        'texts': texts
    }


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_clip_consistency_loss(stego_features, clean_features):
    """
    Compute CLIP Consistency loss using cosine similarity.
    
    L_latent = 1 - (E_img(I) · E_img(I')) / (||E_img(I)|| ||E_img(I')||)
    
    Args:
        stego_features: Features from stego images [B, D]
        clean_features: Features from clean images [B, D]
    
    Returns:
        Consistency loss scalar
    """
    # Normalize features
    stego_features_norm = F.normalize(stego_features, p=2, dim=1)
    clean_features_norm = F.normalize(clean_features, p=2, dim=1)
    
    # Compute cosine similarity
    cosine_sim = (stego_features_norm * clean_features_norm).sum(dim=1)
    
    # Loss: 1 - cosine_similarity (to maximize similarity)
    consistency_loss = (1 - cosine_sim).mean()
    
    return consistency_loss


def compute_pixel_fidelity_loss(stego_images, clean_images):
    """
    Compute pixel fidelity loss using MSE.
    
    L_pixel = ||I - I'||_2^2
    
    Args:
        stego_images: Stego images [B, C, H, W]
        clean_images: Clean images [B, C, H, W]
    
    Returns:
        Pixel fidelity loss scalar
    """
    mse_loss = F.mse_loss(stego_images, clean_images)
    return mse_loss


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scaler,
    scheduler,
    epoch,
    device,
    args
):
    """Train for one epoch. Returns average loss."""
    model.train()
    autocast_fn = get_autocast(args.precision)
    if args.precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif args.precision == 'fp16':
        cast_dtype = torch.float16
    else:
        cast_dtype = torch.float32
    
    # Contrastive loss
    contrastive_loss_fn = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod
    )
    
    loss_m = AverageMeter()
    contrastive_loss_m = AverageMeter()
    consistency_loss_m = AverageMeter()
    pixel_loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    
    # Create progress bar
    if is_master(args):
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch+1}/{args.epochs}",
            bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
            ncols=120,
            leave=True
        )
    else:
        pbar = enumerate(dataloader)
    
    end = time.time()
    for i, batch in pbar:
        step = len(dataloader) * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)
        
        stego_images = batch['stego_images'].to(device=device, dtype=cast_dtype, non_blocking=True)
        clean_images = batch['clean_images'].to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = batch['texts'].to(device=device, non_blocking=True)
        
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        
        with autocast_fn():
            # Forward pass with stego images
            stego_image_features, text_features, logit_scale = model(stego_images, texts)
            
            # Contrastive loss (using stego images)
            contrastive_loss = contrastive_loss_fn(stego_image_features, text_features, logit_scale)
            
            # CLIP Consistency loss
            with torch.no_grad():
                clean_image_features, _, _ = model(clean_images, texts)
            
            consistency_loss = compute_clip_consistency_loss(stego_image_features, clean_image_features)
            
            # Pixel fidelity loss
            pixel_loss = compute_pixel_fidelity_loss(stego_images, clean_images)
            
            # Total loss
            total_loss = (
                contrastive_loss +
                args.consistency_weight * consistency_loss +
                args.pixel_weight * pixel_loss
            )
        
        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()
        
        # Clamp logit scale
        with torch.no_grad():
            model.logit_scale.clamp_(0, math.log(100))
        
        batch_time_m.update(time.time() - end)
        end = time.time()
        
        # Update meters
        batch_size = len(stego_images)
        loss_m.update(total_loss.item(), batch_size)
        contrastive_loss_m.update(contrastive_loss.item(), batch_size)
        consistency_loss_m.update(consistency_loss.item(), batch_size)
        pixel_loss_m.update(pixel_loss.item(), batch_size)
        
        # Update progress bar
        if is_master(args):
            pbar.set_postfix({
                'Loss': f'{loss_m.avg:.4f}',
                'Contrastive': f'{contrastive_loss_m.avg:.4f}',
                'Consistency': f'{consistency_loss_m.avg:.4f}',
                'Pixel': f'{pixel_loss_m.avg:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
                'Time': f'{batch_time_m.avg:.2f}s'
            })
        
        # Logging (detailed logging to file)
        if is_master(args) and (i % args.log_every_n_steps == 0 or i == len(dataloader) - 1):
            logging.info(
                f"Epoch {epoch+1} [{i}/{len(dataloader)}] "
                f"Loss: {loss_m.val:.4f} ({loss_m.avg:.4f}) "
                f"Contrastive: {contrastive_loss_m.val:.4f} "
                f"Consistency: {consistency_loss_m.val:.4f} "
                f"Pixel: {pixel_loss_m.val:.4f} "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
    
    return loss_m.avg


def main():
    parser = argparse.ArgumentParser(description='Train CLIP with full steganography dataset')
    
    # Data arguments
    parser.add_argument('--stego_image_dir', type=str, required=True,
                       help='Directory containing stego images')
    parser.add_argument('--clean_image_dir', type=str, required=True,
                       help='Directory containing clean images')
    parser.add_argument('--caption_file', type=str, required=True,
                       help='Path to caption file')
    parser.add_argument('--stego_sample_ratio', type=float, default=None,
                       help='Ratio to sample from stego images (e.g., 0.05 for 5%%)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit total number of samples (for quick verification)')
    parser.add_argument('--use_imagenet_format', action='store_true',
                       help='Use ImageNet JSON format for captions')
    parser.add_argument('--use_coco_format', action='store_true',
                       help='Use COCO JSON format for captions')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='ViT-B-16',
                       help='Model architecture')
    parser.add_argument('--pretrained_model_path', type=str,
                       default=None,
                       help='Path to pretrained model checkpoint')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.1,
                       help='Weight decay')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--warmup', type=int, default=1000,
                       help='Warmup steps')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Loss weights
    parser.add_argument('--consistency_weight', type=float, default=0.1,
                       help='Weight for CLIP consistency loss')
    parser.add_argument('--pixel_weight', type=float, default=0.01,
                       help='Weight for pixel fidelity loss')
    
    # Other arguments
    parser.add_argument('--precision', type=str, default='amp',
                       choices=['amp', 'fp16', 'fp32'],
                       help='Precision mode')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                       help='Gradient clipping norm')
    parser.add_argument('--log_every_n_steps', type=int, default=10,
                       help='Log every N steps')
    parser.add_argument('--verify_samples', type=int, default=0,
                       help='Load N samples after building the dataset to verify data loading')
    parser.add_argument('--verify_only', action='store_true',
                       help='Exit after verification without training')
    parser.add_argument('--save_every_n_epochs', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--output_dir', type=str, default='./outputs/logs',
                       help='Output directory for checkpoints')
    parser.add_argument('--model_save_dir', type=str, default='./outputs/checkpoints',
                       help='Directory to save final models')
    parser.add_argument('--name', type=str, default='flickr8k_stego_full',
                       help='Experiment name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--local_loss', action='store_true',
                       help='Use local loss')
    parser.add_argument('--gather_with_grad', action='store_true',
                       help='Gather with gradient')
    parser.add_argument('--skip_scheduler', action='store_true',
                       help='Skip scheduler')
    
    # Resume training arguments
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--resume_epoch', type=int, default=None,
                       help='Epoch to resume from (if not specified, will try to infer from checkpoint)')
    
    # Device arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). Auto-detect if None.')
    
    # Distributed arguments
    parser.add_argument('--distributed', action='store_true',
                       help='Use distributed training')
    parser.add_argument('--rank', type=int, default=0,
                       help='Process rank')
    parser.add_argument('--world_size', type=int, default=1,
                       help='World size')
    parser.add_argument('--horovod', action='store_true',
                       help='Use horovod')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize distributed and device
    if args.distributed:
        args.local_rank, args.rank, args.world_size = world_info_from_env()
        init_distributed_device(args)
    else:
        if args.device is None:
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            args.device = torch.device(args.device)
        args.rank = 0
        args.world_size = 1
    
    # Log device info
    if is_master(args):
        logging.info(f"Using device: {args.device}")
        if args.device.type == 'cuda':
            logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logging.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Setup logging and experiment name
    log_dir = Path(args.output_dir) / args.name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if is_master(args):
        log_file = log_dir / 'train.log'
        setup_logging(log_file, logging.INFO)
        logging.info(f"Starting training: {args.name}")
        logging.info(f"Arguments: {args}")
    
    # Load model
    logging.info(f"Loading model: {args.model}")
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        pretrained=None,  # We'll load from checkpoint
        precision=args.precision,
        device=args.device,
    )
    
    # Load pretrained checkpoint
    if not args.verify_only:
        if os.path.exists(args.pretrained_model_path):
            logging.info(f"Loading pretrained model from {args.pretrained_model_path}")
            try:
                checkpoint = torch.load(
                    args.pretrained_model_path,
                    map_location='cpu',
                    weights_only=False,
                )
            except TypeError:
                # Older PyTorch versions do not support weights_only
                checkpoint = torch.load(args.pretrained_model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load state dict
            try:
                model.load_state_dict(state_dict, strict=False)
                logging.info("Pretrained model loaded successfully")
            except Exception as e:
                logging.warning(f"Error loading pretrained model: {e}")
                logging.info("Continuing with randomly initialized model")
        else:
            logging.warning(f"Pretrained model not found at {args.pretrained_model_path}")
            logging.info("Using randomly initialized model")
    
    # Create dataset and dataloader
    tokenizer = get_tokenizer(args.model)
    dataset = StegoFullDataset(
        args.stego_image_dir,
        args.clean_image_dir,
        args.caption_file,
        preprocess_train,
        tokenizer,
        stego_sample_ratio=args.stego_sample_ratio,
        max_samples=args.max_samples,
        use_imagenet_format=args.use_imagenet_format,
        use_coco_format=args.use_coco_format
    )

    if is_master(args) and (args.verify_samples > 0 or args.verify_only):
        logging.info(f"Total samples: {len(dataset)}")

    if args.verify_samples > 0 and is_master(args):
        verify_count = min(args.verify_samples, len(dataset))
        logging.info(f"Verifying {verify_count} samples by loading images")
        for i in range(verify_count):
            item = dataset[i]
            pair = dataset.pairs[i]
            is_stego = True if len(pair) < 3 else pair[2]
            logging.info(
                f"Verify sample {i}: {item['image_name']} "
                f"(stego={is_stego}) "
                f"stego_shape={tuple(item['stego_image'].shape)} "
                f"clean_shape={tuple(item['clean_image'].shape)}"
            )

    if args.verify_only:
        if is_master(args):
            logging.info("Verify-only mode enabled; exiting before training.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Create optimizer
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)
    
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Create scheduler
    total_steps = len(dataloader) * args.epochs
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    
    # Create scaler
    scaler = GradScaler() if args.precision == 'amp' else None
    
    # Resume training if checkpoint provided
    start_epoch = args.resume_epoch if args.resume_epoch is not None else 0
    best_loss = float('inf')
    best_epoch = 0
    
    if args.resume and os.path.exists(args.resume):
        logging.info(f"Resuming training from checkpoint: {args.resume}")
        try:
            try:
                checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            except TypeError:
                checkpoint = torch.load(args.resume, map_location='cpu')
            
            # Load model state
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    model.load_state_dict(state_dict, strict=False)
                
                # Load optimizer state if available
                if 'optimizer_state_dict' in checkpoint and optimizer is not None:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logging.info("Loaded optimizer state from checkpoint")
                
                # Load scaler state if available
                if 'scaler_state_dict' in checkpoint and scaler is not None:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    logging.info("Loaded scaler state from checkpoint")
                
                # Determine start epoch
                if args.resume_epoch is not None:
                    start_epoch = args.resume_epoch
                elif 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                elif 'best_epoch' in checkpoint:
                    start_epoch = checkpoint['best_epoch']
                
                if 'best_loss' in checkpoint:
                    best_loss = checkpoint['best_loss']
                if 'best_epoch' in checkpoint:
                    best_epoch = checkpoint['best_epoch']
                
                logging.info(f"Resumed from epoch {start_epoch}, best loss: {best_loss:.4f} (epoch {best_epoch})")
        except Exception as e:
            logging.warning(f"Error loading checkpoint: {e}")
            logging.info("Starting training from scratch")
            start_epoch = args.resume_epoch if args.resume_epoch is not None else 0
    
    # If resume_epoch is specified but no resume checkpoint, start from that epoch
    if args.resume_epoch is not None and (not args.resume or not os.path.exists(args.resume)):
        start_epoch = args.resume_epoch
        logging.info(f"Starting training from epoch {start_epoch} (using pretrained model as starting point)")
    
    # Training loop
    logging.info(f"Starting training for {args.epochs} epochs (from epoch {start_epoch})")
    logging.info(f"Total samples: {len(dataset)}")
    logging.info(f"Total batches per epoch: {len(dataloader)}")
    
    checkpoint_dir = log_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    model_save_dir = Path(args.model_save_dir)
    model_save_dir.mkdir(exist_ok=True, parents=True)
    
    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            scaler,
            scheduler,
            epoch,
            args.device,
            args
        )
        
        # Save checkpoint every epoch
        if is_master(args) and (epoch + 1) % args.save_every_n_epochs == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'best_epoch': best_epoch,
            }
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_master(args) and avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            
            # Generate model name based on model architecture and dataset
            model_arch = args.model.replace('-', '_').replace(' ', '_')
            dataset_name = args.name if args.name else 'imagenet1k'
            best_model_name = f"{dataset_name}_{model_arch}_cc3m_12m_ep{args.epochs}.pt"
            best_model_path = model_save_dir / best_model_name
            
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Saved best model (epoch {best_epoch}, loss {best_loss:.4f}) to {best_model_path}")
    
    # Save final model
    if is_master(args):
        # Generate model name based on model architecture and dataset
        model_arch = args.model.replace('-', '_').replace(' ', '_')
        dataset_name = args.name if args.name else 'imagenet1k'
        final_model_name = f"{dataset_name}_{model_arch}_cc3m_12m_ep{args.epochs}.pt"
        final_model_path = model_save_dir / final_model_name
        
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Saved final model (epoch {args.epochs}, loss {avg_loss:.4f}) to {final_model_path}")
        logging.info(f"Best model was at epoch {best_epoch} with loss {best_loss:.4f}")
    
    logging.info("Training completed!")


if __name__ == "__main__":
    main()
