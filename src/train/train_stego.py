#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a CLIP model on a mixture of clean and stego image-text pairs.

This script trains CLIP models with different stego ratios (1%, 2%, 5%),
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

# Use the installed open_clip package.
# open_clip is installed via pip
from open_clip import create_model_and_transforms, get_tokenizer, ClipLoss
from open_clip.factory import get_model_config
from open_clip.tokenizer import HFTokenizer
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


class StegoImageTextDataset(Dataset):
    """Dataset for loading images and texts with stego/clean labels."""
    
    def __init__(self, csv_file, clean_image_dir, stego_image_dir, transforms, tokenizer):
        """
        Initialize dataset.
        
        Args:
            csv_file: Path to CSV file with training data
            clean_image_dir: Directory containing clean images
            stego_image_dir: Directory containing stego images
            transforms: Image transforms
            tokenizer: Text tokenizer
        """
        df = pd.read_csv(csv_file, sep='\t')
        self.samples = list(
            df[['filepath', 'caption', 'is_stego', 'clean_filepath']]
            .itertuples(index=False, name=None)
        )
        self.clean_image_dir = Path(clean_image_dir)
        self.stego_image_dir = Path(stego_image_dir)
        self.transforms = transforms
        self.tokenize = tokenizer
        
        stego_count = int(df['is_stego'].sum())
        logging.info(f"Loaded {len(self.samples)} samples from {csv_file}")
        logging.info(f"  Stego samples: {stego_count}")
        logging.info(f"  Clean samples: {len(self.samples) - stego_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, caption, is_stego, clean_filepath = self.samples[idx]
        is_stego = bool(is_stego)
        
        # Load image (stego or clean)
        if is_stego:
            image_path = self.stego_image_dir / filepath
        else:
            image_path = self.clean_image_dir / filepath
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transforms(image)
        
        # Tokenize text
        text_tensor = self.tokenize([caption])[0]
        
        # Load clean image for consistency loss (if stego)
        clean_image_tensor = None
        if is_stego:
            clean_image_path = self.clean_image_dir / clean_filepath
            clean_image = Image.open(clean_image_path).convert('RGB')
            clean_image_tensor = self.transforms(clean_image)
        
        return {
            'image': image_tensor,
            'text': text_tensor,
            'is_stego': is_stego,
            'clean_image': clean_image_tensor if is_stego else image_tensor,
            'filepath': filepath
        }


def collate_fn(batch):
    """Custom collate function for batching."""
    images = torch.stack([item['image'] for item in batch])
    texts = torch.stack([item['text'] for item in batch])
    is_stego = torch.tensor([item['is_stego'] for item in batch], dtype=torch.bool)
    clean_images = torch.stack([item['clean_image'] for item in batch])
    
    return {
        'images': images,
        'texts': texts,
        'is_stego': is_stego,
        'clean_images': clean_images
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


def parse_stego_ratio_label(csv_name: str) -> str:
    """Extract stego ratio label from training CSV filename."""
    name = csv_name.lower()
    if "0.5pct" in name:
        return "0.5"
    if "1pct" in name:
        return "1"
    if "2pct" in name:
        return "2"
    if "5pct" in name:
        return "5"
    if "10pct" in name:
        return "10"
    if "20pct" in name:
        return "20"
    return "unknown"


def build_model_filename(dataset_prefix: str, model_tag: str, ratio: str, best: bool = False) -> str:
    """Build checkpoint filename following the project naming convention."""
    suffix = "_best" if best else ""
    return f"{dataset_prefix}_{model_tag}_{ratio}%{suffix}.pt"


def infer_model_tag(pretrained_model_path: str) -> str:
    """Infer model tag from pretrained checkpoint filename."""
    stem = Path(pretrained_model_path).stem
    for token in ("_ep32", "_ep30", "_e32"):
        stem = stem.replace(token, "")
    return stem


def unpack_model_outputs(outputs):
    """Normalize open_clip forward outputs (CLIP: 3-tuple, SigLIP: 4-tuple)."""
    if not isinstance(outputs, (tuple, list)):
        raise TypeError(f"Unexpected model output type: {type(outputs)}")
    if len(outputs) < 3:
        raise ValueError(f"Unexpected model output length: {len(outputs)}")
    image_features, text_features, logit_scale = outputs[0], outputs[1], outputs[2]
    return image_features, text_features, logit_scale


def build_tokenizer(model_name: str, hf_tokenizer: str | None = None):
    """Build tokenizer; optionally force a local HF tokenizer path (offline SigLIP2)."""
    if hf_tokenizer:
        cfg = get_model_config(model_name) or {}
        text_cfg = cfg.get("text_cfg", {}) if isinstance(cfg, dict) else {}
        context_length = text_cfg.get("context_length", 77)
        tokenizer_kwargs = dict(text_cfg.get("tokenizer_kwargs") or {})
        tokenizer_mode = text_cfg.get("tokenizer_mode", None)
        logging.info(
            f"Using local HFTokenizer from '{hf_tokenizer}' "
            f"(context_length={context_length})"
        )
        return HFTokenizer(
            hf_tokenizer,
            context_length=context_length,
            tokenizer_mode=tokenizer_mode,
            **tokenizer_kwargs,
        )
    return get_tokenizer(model_name)


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
    
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = len(dataloader) * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)
        
        images = batch['images'].to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = batch['texts'].to(device=device, non_blocking=True)
        is_stego = batch['is_stego'].to(device=device)
        clean_images = batch['clean_images'].to(device=device, dtype=cast_dtype, non_blocking=True)
        
        data_time_m.update(time.time() - end)
        optimizer.zero_grad(set_to_none=True)
        
        with autocast_fn():
            # Forward pass (SigLIP returns optional logit_bias as 4th value)
            image_features, text_features, logit_scale = unpack_model_outputs(
                model(images, texts)
            )
            
            # Contrastive loss
            contrastive_loss = contrastive_loss_fn(image_features, text_features, logit_scale)
            
            # CLIP Consistency loss (only for stego samples)
            consistency_loss = torch.tensor(0.0, device=device)
            if is_stego.any():
                stego_mask = is_stego
                if stego_mask.sum() > 0:
                    # Get features for clean images (for stego samples)
                    with torch.no_grad():
                        clean_image_features = model.encode_image(
                            clean_images[stego_mask], normalize=True
                        )
                    
                    stego_features = image_features[stego_mask]
                    consistency_loss = compute_clip_consistency_loss(stego_features, clean_image_features)
            
            # Pixel fidelity loss (only for stego samples)
            pixel_loss = torch.tensor(0.0, device=device)
            if is_stego.any():
                stego_mask = is_stego
                if stego_mask.sum() > 0:
                    pixel_loss = compute_pixel_fidelity_loss(
                        images[stego_mask],
                        clean_images[stego_mask]
                    )
            
            # Total loss
            total_loss = (
                contrastive_loss +
                args.consistency_weight * consistency_loss +
                args.pixel_weight * pixel_loss
            )

        if not torch.isfinite(total_loss):
            message = (
                f"Non-finite loss at epoch={epoch}, step={i}, "
                f"lr={optimizer.param_groups[0]['lr']:.3e}; aborting to avoid "
                "wasting GPU time and writing an invalid checkpoint"
            )
            if args.fail_on_non_finite:
                raise FloatingPointError(message)
            logging.error(message)

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
        batch_size = len(images)
        loss_m.update(total_loss.item(), batch_size)
        contrastive_loss_m.update(contrastive_loss.item(), batch_size)
        consistency_loss_m.update(consistency_loss.item(), batch_size)
        pixel_loss_m.update(pixel_loss.item(), batch_size)
        
        # Logging
        if is_master(args) and (i % args.log_every_n_steps == 0 or i == len(dataloader) - 1):
            logging.info(
                f"Epoch {epoch} [{i}/{len(dataloader)}] "
                f"Loss: {loss_m.val:.4f} ({loss_m.avg:.4f}) "
                f"Contrastive: {contrastive_loss_m.val:.4f} "
                f"Consistency: {consistency_loss_m.val:.4f} "
                f"Pixel: {pixel_loss_m.val:.4f} "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} "
                f"Data: {data_time_m.avg:.3f}s "
                f"Rate: {batch_size / max(batch_time_m.val, 1e-9):.1f} samples/s"
            )
    
    return loss_m.avg




def main():
    parser = argparse.ArgumentParser(description='Train CLIP with steganography-based membership inference')
    
    # Data arguments
    parser.add_argument('--train_csv', type=str, required=True,
                       help='Path to training CSV file')
    parser.add_argument('--clean_image_dir', type=str, required=True,
                       help='Directory containing clean images')
    parser.add_argument('--stego_image_dir', type=str, required=True,
                       help='Directory containing stego images')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='ViT-B-16',
                       help='Model architecture')
    parser.add_argument('--hf_tokenizer', type=str, default=None,
                       help='Local HF tokenizer path/dir (offline SigLIP2); skips Hub download')
    parser.add_argument('--pretrained_model_path', type=str,
                       default=None,
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume training from checkpoint (loads model weights)')
    parser.add_argument('--start_epoch', type=int, default=0,
                       help='0-indexed epoch to start/resume from')
    parser.add_argument('--initial_best_loss', type=float, default=None,
                       help='Best loss from prior run when resuming')
    parser.add_argument('--initial_best_epoch', type=int, default=None,
                       help='Best epoch (1-indexed) from prior run when resuming')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.1,
                       help='Weight decay')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--warmup', type=int, default=1000,
                       help='Warmup steps')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                       help='Batches prefetched by each data loader worker')
    
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
    parser.add_argument('--log_every_n_steps', type=int, default=100,
                       help='Log every N steps')
    parser.add_argument('--save_every_n_epochs', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--fused_optimizer', action=argparse.BooleanOptionalAction,
                       default=True, help='Use fused AdamW on CUDA when supported')
    parser.add_argument('--fail_on_non_finite', action=argparse.BooleanOptionalAction,
                       default=True, help='Abort immediately when loss becomes NaN/Inf')
    parser.add_argument('--output_dir', type=str, default='./outputs/logs',
                       help='Output directory for checkpoints')
    parser.add_argument('--model_save_dir', type=str, default='./outputs/checkpoints',
                       help='Directory to save final models')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--dataset_prefix', type=str, default='Flickr8k',
                       help='Dataset prefix for saved model filenames')
    parser.add_argument('--model_tag', type=str, default=None,
                       help='Model tag for saved filenames (auto from pretrained path if omitted)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--local_loss', action='store_true',
                       help='Use local loss')
    parser.add_argument('--gather_with_grad', action='store_true',
                       help='Gather with gradient')
    parser.add_argument('--skip_scheduler', action='store_true',
                       help='Skip scheduler')
    
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

    if args.device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    
    # Setup logging and experiment name
    csv_name = Path(args.train_csv).stem
    ratio_label = parse_stego_ratio_label(csv_name)
    if args.model_tag is None:
        args.model_tag = infer_model_tag(args.pretrained_model_path)

    if args.name is None:
        timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        lr_str = f"{args.lr:.4f}".rstrip('0').rstrip('.')
        args.name = (
            f"{timestamp}-{args.dataset_prefix}_{args.model_tag}_{ratio_label}%"
            f"-seed_{args.seed}-lr_{lr_str}-b_{args.batch_size}-epochs_{args.epochs}"
        )
    
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
    
    def load_weights_from_checkpoint(checkpoint_path, label):
        if not os.path.exists(checkpoint_path):
            logging.warning(f"{label} not found at {checkpoint_path}")
            return False
        logging.info(f"Loading {label} from {checkpoint_path}")
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location='cpu',
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(state_dict, strict=False)
            logging.info(f"{label} loaded successfully")
            return True
        except Exception as e:
            logging.warning(f"Error loading {label}: {e}")
            return False

    # Load pretrained checkpoint (skip if resuming from a training checkpoint)
    if args.resume_from:
        if not load_weights_from_checkpoint(args.resume_from, "resume checkpoint"):
            logging.info("Falling back to pretrained model")
            load_weights_from_checkpoint(args.pretrained_model_path, "pretrained model")
    elif os.path.exists(args.pretrained_model_path):
        load_weights_from_checkpoint(args.pretrained_model_path, "pretrained model")
    else:
        logging.warning(f"Pretrained model not found at {args.pretrained_model_path}")
        logging.info("Using randomly initialized model")
    
    # Create dataset and dataloader
    tokenizer = build_tokenizer(args.model, args.hf_tokenizer)
    dataset = StegoImageTextDataset(
        args.train_csv,
        args.clean_image_dir,
        args.stego_image_dir,
        preprocess_train,
        tokenizer
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        collate_fn=collate_fn,
    )
    
    # Create optimizer
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)
    
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    
    optimizer_kwargs = {
        "lr": args.lr,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }
    if args.device.type == 'cuda' and args.fused_optimizer:
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            **optimizer_kwargs,
        )
    except (RuntimeError, TypeError):
        optimizer_kwargs.pop("fused", None)
        logging.warning("Fused AdamW is unavailable; falling back to standard AdamW")
        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            **optimizer_kwargs,
        )
    
    # Create scheduler
    total_steps = len(dataloader) * args.epochs
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    
    # Create scaler
    scaler = GradScaler() if args.precision == 'amp' else None
    
    # Training loop
    if args.start_epoch >= args.epochs:
        raise ValueError(f"start_epoch ({args.start_epoch}) must be < epochs ({args.epochs})")

    if args.start_epoch > 0:
        logging.info(f"Resuming training from epoch {args.start_epoch} (1-indexed: {args.start_epoch + 1})")
    logging.info(f"Training epochs {args.start_epoch}..{args.epochs - 1} (total target: {args.epochs})")

    best_loss = args.initial_best_loss if args.initial_best_loss is not None else float('inf')
    best_epoch = args.initial_best_epoch if args.initial_best_epoch is not None else 0
    if args.initial_best_loss is not None:
        logging.info(f"Restored best checkpoint tracking: epoch {best_epoch}, loss {best_loss:.4f}")

    checkpoint_dir = log_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def save_model_once(primary_path, mirror_path):
        """Serialize once, then mirror with a same-filesystem hard link."""
        torch.save(model.state_dict(), primary_path)
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        if mirror_path.exists():
            mirror_path.unlink()
        try:
            os.link(primary_path, mirror_path)
        except OSError:
            shutil.copy2(primary_path, mirror_path)

    for epoch in range(args.start_epoch, args.epochs):
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
        
        # Save best model
        if is_master(args) and avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1

            best_model_name = build_model_filename(args.dataset_prefix, args.model_tag, ratio_label, best=True)
            best_model_path = checkpoint_dir / best_model_name
            model_save_dir = Path(args.model_save_dir)
            save_model_once(best_model_path, model_save_dir / best_model_name)
            logging.info(f"Saved best model (epoch {best_epoch}, loss {best_loss:.4f}) to {best_model_path}")
    
    # Save final model
    if is_master(args):
        final_model_name = build_model_filename(args.dataset_prefix, args.model_tag, ratio_label, best=False)
        final_model_path = checkpoint_dir / final_model_name
        model_save_dir = Path(args.model_save_dir)
        final_save_path = model_save_dir / final_model_name
        save_model_once(final_model_path, final_save_path)
        logging.info(f"Saved final model (epoch {args.epochs}, loss {avg_loss:.4f}) to {final_model_path}")
        logging.info(f"Copied final model to {final_save_path}")
        logging.info(f"Best model was at epoch {best_epoch} with loss {best_loss:.4f}")
    
    logging.info("Training completed!")


if __name__ == "__main__":
    main()
