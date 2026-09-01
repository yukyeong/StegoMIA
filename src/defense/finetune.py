"""
Fine-tuning defense against steganography-based membership inference.
"""
import os
import sys
current_directory = os.getcwd()
sys.path.insert(1, current_directory)
import csv
import json
import time
import wandb
import torch
import logging
import warnings
import numpy as np
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from pkgs.openai.clip import load as load_model
from src.clip_loop import train, get_loss
from src.parser import parse_args
from src.scheduler import cosine_scheduler
from src.logger import get_logger, set_logger
from src.evaluate import (
    compute_image_quality_metrics,
    build_image_index,
)
from utils.augment_text import _augment_text
from utils.augment_image import _augment_image

mp.set_start_method("spawn", force=True)
warnings.filterwarnings("ignore")

def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    """Extract a state_dict from a checkpoint or state_dict-like object."""
    if isinstance(checkpoint, dict):
        # Try common checkpoint keys
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        for key in ("model_state_dict", "model", "model_state", "state_dict_ema"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        # If checkpoint itself is a state_dict (all values are tensors)
        if checkpoint and len(checkpoint) > 0:
            first_value = next(iter(checkpoint.values()))
            if torch.is_tensor(first_value):
                return checkpoint
            # Check if it's a nested structure
            if isinstance(first_value, dict) and len(first_value) > 0:
                nested_value = next(iter(first_value.values()))
                if torch.is_tensor(nested_value):
                    # Flatten nested structure
                    result = {}
                    for outer_key, inner_dict in checkpoint.items():
                        if isinstance(inner_dict, dict):
                            for inner_key, tensor in inner_dict.items():
                                result[f"{outer_key}.{inner_key}" if outer_key else inner_key] = tensor
                        else:
                            result[outer_key] = inner_dict
                    return result
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    # Last resort: return checkpoint if it's already a dict
    if isinstance(checkpoint, dict):
        return checkpoint
    raise KeyError("No compatible state_dict found in checkpoint")


def _normalize_state_dict(state_dict: Dict[str, torch.Tensor], distributed: bool) -> Dict[str, torch.Tensor]:
    """Normalize state_dict keys for distributed or non-distributed loading."""
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict))
    if not distributed and first_key.startswith("module."):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    if distributed and not first_key.startswith("module."):
        return {f"module.{key}": value for key, value in state_dict.items()}
    return state_dict


class TokenImageCaptionDataset(Dataset):
    """Dataset for image-text pairs using a token file (image#idx\\tcaption)."""

    def __init__(
        self,
        image_dir: str,
        token_file: str,
        processor,
        inmodal: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.inmodal = inmodal

        image_captions: Dict[str, str] = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                image_key, caption = parts
                image_name = image_key.split("#", 1)[0]
                if image_name not in image_captions:
                    image_captions[image_name] = caption.strip()

        self.images: List[str] = []
        self.captions_text: List[str] = []
        for image_name, caption in image_captions.items():
            image_path = self.image_dir / image_name
            if image_path.exists():
                self.images.append(str(image_path))
                self.captions_text.append(caption)

        if max_samples:
            self.images = self.images[:max_samples]
            self.captions_text = self.captions_text[:max_samples]

        self.captions = processor.process_text(self.captions_text)
        if self.inmodal:
            self.augment_captions = processor.process_text(
                [_augment_text(caption) for caption in self.captions_text]
            )

        logging.info(
            f"Loaded {len(self.images)} image-text pairs from {image_dir} using {token_file}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        image_path = self.images[idx]
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        item["image_path"] = image_path
        item["is_backdoor"] = "backdoor" in image_path
        item["caption"] = self.captions_text[idx]

        if self.inmodal:
            item["input_ids"] = (
                self.captions["input_ids"][idx],
                self.augment_captions["input_ids"][idx],
            )
            item["attention_mask"] = (
                self.captions["attention_mask"][idx],
                self.augment_captions["attention_mask"][idx],
            )
            item["pixel_values"] = (
                self.processor.process_image(image),
                self.processor.process_image(_augment_image(image_path)),
            )
        else:
            item["input_ids"] = self.captions["input_ids"][idx]
            item["attention_mask"] = self.captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(image)

        return item


class MSCOCOImageCaptionDataset(Dataset):
    """Dataset for image-text pairs using MSCOCO JSON format."""

    def __init__(
        self,
        image_dir: str,
        json_file: str,
        processor,
        inmodal: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.inmodal = inmodal

        # Load MSCOCO JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create mapping from image_id to image info
        image_id_to_info = {img['id']: img for img in data['images']}
        
        # Create mapping from image_id to captions (use first caption)
        image_id_to_caption = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_id_to_caption:
                image_id_to_caption[img_id] = ann['caption']
        
        # Get all image files in directory for matching
        all_image_files = {}
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            for img_file in self.image_dir.glob(f'*{ext}'):
                img_stem = img_file.stem
                if img_stem not in all_image_files:
                    all_image_files[img_stem] = str(img_file)
        
        self.images: List[str] = []
        self.captions_text: List[str] = []
        
        for img_id, img_info in image_id_to_info.items():
            if img_id in image_id_to_caption:
                img_name = img_info['file_name']
                img_stem = Path(img_name).stem
                
                # Try to find matching image file
                img_path = None
                if img_stem in all_image_files:
                    img_path = Path(all_image_files[img_stem])
                else:
                    # Try direct path
                    candidate = self.image_dir / img_name
                    if candidate.exists():
                        img_path = candidate
                
                if img_path and img_path.exists():
                    self.images.append(str(img_path))
                    self.captions_text.append(image_id_to_caption[img_id])
        
        if max_samples:
            self.images = self.images[:max_samples]
            self.captions_text = self.captions_text[:max_samples]

        self.captions = processor.process_text(self.captions_text)
        if self.inmodal:
            self.augment_captions = processor.process_text(
                [_augment_text(caption) for caption in self.captions_text]
            )

        logging.info(
            f"Loaded {len(self.images)} image-text pairs from {image_dir} using {json_file}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        image_path = self.images[idx]
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        item["image_path"] = image_path
        item["is_backdoor"] = "backdoor" in image_path
        item["caption"] = self.captions_text[idx]

        if self.inmodal:
            item["input_ids"] = (
                self.captions["input_ids"][idx],
                self.augment_captions["input_ids"][idx],
            )
            item["attention_mask"] = (
                self.captions["attention_mask"][idx],
                self.augment_captions["attention_mask"][idx],
            )
            item["pixel_values"] = (
                self.processor.process_image(image),
                self.processor.process_image(_augment_image(image_path)),
            )
        else:
            item["input_ids"] = self.captions["input_ids"][idx]
            item["attention_mask"] = self.captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(image)

        return item


class ImageNet1kImageCaptionDataset(Dataset):
    """Dataset for image-text pairs using ImageNet-1k JSON format (dict: filename -> captions list)."""

    def __init__(
        self,
        image_dir: str,
        json_file: str,
        processor,
        inmodal: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.inmodal = inmodal

        # Load ImageNet-1k JSON file (dict format: filename -> captions list)
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get all image files in directory for matching
        all_image_files = {}
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.JPEG']:
            for img_file in self.image_dir.rglob(f'*{ext}'):
                img_stem = img_file.stem
                if img_stem not in all_image_files:
                    all_image_files[img_stem] = str(img_file)
        
        self.images: List[str] = []
        self.captions_text: List[str] = []
        
        # Process each entry in the JSON dict
        for img_stem, captions in data.items():
            if isinstance(captions, list) and len(captions) > 0:
                # Use first caption if multiple captions exist
                caption = captions[0]
            elif isinstance(captions, str):
                caption = captions
            else:
                continue
            
            # Skip empty or invalid captions
            if not caption or not isinstance(caption, str) or len(caption.strip()) == 0:
                continue
            
            # Try to find matching image file
            img_path = None
            if img_stem in all_image_files:
                img_path = Path(all_image_files[img_stem])
            else:
                # Try direct path with common extensions
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    candidate = self.image_dir / f"{img_stem}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break
            
            if img_path and img_path.exists():
                self.images.append(str(img_path))
                self.captions_text.append(caption.strip())
        
        if max_samples:
            self.images = self.images[:max_samples]
            self.captions_text = self.captions_text[:max_samples]

        self.captions = processor.process_text(self.captions_text)
        if self.inmodal:
            self.augment_captions = processor.process_text(
                [_augment_text(caption) for caption in self.captions_text]
            )

        logging.info(
            f"Loaded {len(self.images)} image-text pairs from {image_dir} using {json_file}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        image_path = self.images[idx]
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        item["image_path"] = image_path
        item["is_backdoor"] = "backdoor" in image_path
        item["caption"] = self.captions_text[idx]

        if self.inmodal:
            item["input_ids"] = (
                self.captions["input_ids"][idx],
                self.augment_captions["input_ids"][idx],
            )
            item["attention_mask"] = (
                self.captions["attention_mask"][idx],
                self.augment_captions["attention_mask"][idx],
            )
            item["pixel_values"] = (
                self.processor.process_image(image),
                self.processor.process_image(_augment_image(image_path)),
            )
        else:
            item["input_ids"] = self.captions["input_ids"][idx]
            item["attention_mask"] = self.captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(image)

        return item


def get_token_train_dataloader(options, processor):
    if options.train_data is None:
        return None
    if not os.path.isdir(options.train_data):
        raise ValueError("train_data must be an image directory when using token files")
    if not options.train_token_file or not os.path.exists(options.train_token_file):
        raise ValueError(
            "train_token_file is required and must exist when train_data is a directory"
        )

    # Check dataset type flags first, then file extension
    if options.train_is_imagenet1k:
        # Use ImageNet-1k dataset loader
        dataset = ImageNet1kImageCaptionDataset(
            options.train_data,
            options.train_token_file,
            processor,
            inmodal=options.inmodal,
        )
    elif options.train_is_mscoco:
        # Use MSCOCO dataset loader
        dataset = MSCOCOImageCaptionDataset(
            options.train_data,
            options.train_token_file,
            processor,
            inmodal=options.inmodal,
        )
    else:
        # Check file extension as fallback
        train_token_file_lower = options.train_token_file.lower()
        if train_token_file_lower.endswith('.json'):
            # Use MSCOCO dataset loader (default for JSON files)
            dataset = MSCOCOImageCaptionDataset(
                options.train_data,
                options.train_token_file,
                processor,
                inmodal=options.inmodal,
            )
        else:
            # Use token file dataset loader (Flickr8k format)
            dataset = TokenImageCaptionDataset(
                options.train_data,
                options.train_token_file,
                processor,
                inmodal=options.inmodal,
            )
    
    if len(dataset) == 0:
        raise ValueError(
            "No training samples found. Check train_data and train_token_file alignment."
        )
    sampler = DistributedSampler(dataset) if options.distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=(sampler is None),
        num_workers=options.num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
    dataloader.num_samples = len(dataloader) * options.batch_size
    dataloader.num_batches = len(dataloader)
    return dataloader


def create_flickr8k_dataset_csv(
    image_dir: str,
    annotations_file: str,
    output_csv: str,
    image_key: str = "image",
    caption_key: str = "caption"
) -> str:
    """
    Create a CSV file for Flickr8k dataset from image directory and annotations file.
    
    Args:
        image_dir: Directory containing images.
        annotations_file: Path to annotations file (tab-separated: image caption).
        output_csv: Output CSV file path.
        image_key: Column name for image paths.
        caption_key: Column name for captions.
    
    Returns:
        Path to created CSV file.
    """
    image_dir_path = Path(image_dir)
    annotations_path = Path(annotations_file)
    
    # Read annotations
    image_captions = {}
    with open(annotations_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                image_name, caption = parts
                if '#' in image_name:
                    image_name = image_name.split('#', 1)[0]
                if image_name not in image_captions:
                    image_captions[image_name] = []
                image_captions[image_name].append(caption.strip())
    
    # Create CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([image_key, caption_key])
        
        # Get all images from directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(image_dir_path.glob(f'*{ext}')))
            image_files.extend(list(image_dir_path.glob(f'*{ext.upper()}')))
        
        # Get parent directory for relative paths
        parent_dir = os.path.dirname(image_dir.rstrip('/'))
        
        matched_count = 0
        for image_path in sorted(image_files):
            image_name = image_path.name
            if image_name in image_captions:
                # Use first caption for each image
                caption = image_captions[image_name][0]
                # Use relative path from parent directory (where CSV will be located)
                # This ensures ImageCaptionDataset can find images correctly
                rel_path = os.path.relpath(str(image_path), parent_dir)
                # Normalize path separators
                rel_path = rel_path.replace('\\', '/')
                writer.writerow([rel_path, caption])
                matched_count += 1
    
    logging.info(
        f"Created dataset CSV: {output_csv} with {matched_count} matched images "
        f"(found {len(image_files)} files)"
    )
    return output_csv


def _csv_has_samples(csv_path: str) -> bool:
    """Return True if CSV contains at least one data row."""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            next(f, None)
            for line in f:
                if line.strip():
                    return True
    except OSError as exc:
        logging.warning(f"Failed to read CSV {csv_path}: {exc}")
    return False


def validate_with_image_quality_metrics(
    model,
    processor,
    test_image_dir: str,
    annotations_file: str,
    options,
    reference_image_dir: Optional[str] = None,
    epoch: int = 0
) -> Dict[str, float]:
    """
    Validate model on test images and compute image quality metrics (MSE, SSIM, PSNR).
    
    Args:
        model: CLIP model to validate.
        processor: Image/text processor.
        test_image_dir: Directory containing test images.
        annotations_file: Path to annotations file.
        options: Training options.
        reference_image_dir: Optional directory of reference images for quality metrics.
        epoch: Current epoch number.
    
    Returns:
        Dictionary containing validation metrics.
    """
    model.eval()
    umodel = model.module if options.distributed else model
    
    test_image_dir_path = Path(test_image_dir)
    reference_dir_path = Path(reference_image_dir) if reference_image_dir else None
    annotations_path = Path(annotations_file)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    ref_index = None
    if reference_dir_path:
        if reference_dir_path.exists():
            ref_index = build_image_index(reference_dir_path, image_extensions)
            if not ref_index:
                logging.warning(f"No reference images found in {reference_dir_path}")
        else:
            logging.warning(f"Reference image directory not found: {reference_dir_path}")
            reference_dir_path = None
    
    # Read annotations
    image_captions = {}
    with open(annotations_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                image_name, caption = parts
                if image_name not in image_captions:
                    image_captions[image_name] = []
                image_captions[image_name].append(caption.strip())
    
    # Get test images
    test_images = []
    for ext in image_extensions:
        test_images.extend(list(test_image_dir_path.glob(f'*{ext}')))
        test_images.extend(list(test_image_dir_path.glob(f'*{ext.upper()}')))
    
    test_images = sorted(test_images)
    
    mse_values = []
    ssim_values = []
    psnr_values = []
    similarities = []
    
    logging.info(f"Validating on {len(test_images)} test images...")
    
    missing_references = 0

    with torch.no_grad():
        for image_path in tqdm(test_images, desc="Validating"):
            image_name = image_path.name
            
            if image_name not in image_captions:
                continue
            
            # Load test image
            test_image = Image.open(image_path).convert('RGB')
            test_array = np.array(test_image)
            
            # Process image through model
            processed_image = processor.process_image(test_image)
            processed_image_tensor = processed_image.unsqueeze(0).to(options.device)
            
            # Get image embedding
            image_embedding = umodel.get_image_features(processed_image_tensor)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            
            # Process text captions
            captions = image_captions[image_name]
            if len(captions) == 0:
                continue
            
            # Use first caption
            caption = captions[0]
            text_tokens = processor.process_text([caption])
            text_input_ids = text_tokens["input_ids"].to(options.device)
            text_attention_mask = text_tokens["attention_mask"].to(options.device)
            
            # Get text embedding
            text_embedding = umodel.get_text_features(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
            )
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (image_embedding @ text_embedding.t()).item()
            similarities.append(similarity)
            
            # Compute image quality metrics aligned with reference images when provided
            try:
                if reference_dir_path and ref_index is not None:
                    ref_path = ref_index.get(image_path.stem)
                    if ref_path is None:
                        candidate = reference_dir_path / image_name
                        if candidate.exists():
                            ref_path = candidate
                    if ref_path is None:
                        missing_references += 1
                        continue

                    ref_image = Image.open(ref_path).convert('RGB')
                    ref_array = np.array(ref_image)
                else:
                    # Fallback to preprocessing baseline when no reference directory is provided
                    ref_image = test_image
                    ref_array = test_array

                if ref_array.shape != test_array.shape:
                    ref_image = Image.fromarray(ref_array).resize(
                        (test_array.shape[1], test_array.shape[0]),
                        Image.LANCZOS
                    )
                    ref_array = np.array(ref_image)

                mse_value, ssim_value, psnr_value, _ = compute_image_quality_metrics(
                    ref_array,
                    test_array,
                )

                mse_values.append(mse_value)
                ssim_values.append(ssim_value)
                psnr_values.append(psnr_value)
            except Exception as e:
                logging.warning(f"Error computing metrics for {image_name}: {e}")
                continue

    metrics = {}
    
    # Only compute metrics if we have valid data
    if mse_values:
        metrics['mse'] = float(np.mean(mse_values))
        metrics['mse_std'] = float(np.std(mse_values))
    
    if ssim_values:
        metrics['ssim'] = float(np.mean(ssim_values))
        metrics['ssim_std'] = float(np.std(ssim_values))
    
    if psnr_values:
        metrics['psnr'] = float(np.mean(psnr_values))
        metrics['psnr_std'] = float(np.std(psnr_values))
    
    if similarities:
        metrics['avg_similarity'] = float(np.mean(similarities))
    
    if missing_references > 0:
        logging.info(f"Skipped {missing_references} images without reference matches")
    
    return metrics


def _latex_mean_std(mean_val: float, std_val: float, decimals: int = 2) -> str:
    return f"${mean_val:.{decimals}f} \\pm {std_val:.{decimals}f}$"


def compute_paired_stego_quality(
    clean_dir: str,
    stego_dir: str,
    max_pairs: int = 500,
) -> Dict[str, float]:
    """Compute MSE/SSIM/PSNR between paired clean and stego images (no LPIPS)."""
    clean_path = Path(clean_dir)
    stego_path = Path(stego_dir)
    if not clean_path.exists() or not stego_path.exists():
        return {}

    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    clean_index = build_image_index(clean_path, exts)
    stego_files = []
    for path in stego_path.iterdir():
        if path.is_file() and path.suffix.lower() in exts:
            stego_files.append(path)
    stego_files = sorted(stego_files)[:max_pairs]

    mse_values, ssim_values, psnr_values = [], [], []
    for stego_file in stego_files:
        ref = clean_index.get(stego_file.stem)
        if ref is None:
            continue
        try:
            ref_arr = np.array(Image.open(ref).convert("RGB"))
            stego_arr = np.array(Image.open(stego_file).convert("RGB"))
            if ref_arr.shape != stego_arr.shape:
                stego_arr = np.array(
                    Image.fromarray(stego_arr).resize(
                        (ref_arr.shape[1], ref_arr.shape[0]), Image.LANCZOS
                    )
                )
            mse_v, ssim_v, psnr_v, _ = compute_image_quality_metrics(
                ref_arr, stego_arr, use_skimage_ssim=True
            )
            mse_values.append(mse_v)
            ssim_values.append(ssim_v)
            psnr_values.append(psnr_v)
        except Exception as exc:
            logging.debug("Skip quality pair %s: %s", stego_file.name, exc)
            continue

    if not mse_values:
        return {}
    return {
        "mse": float(np.mean(mse_values)),
        "mse_std": float(np.std(mse_values)),
        "ssim": float(np.mean(ssim_values)),
        "ssim_std": float(np.std(ssim_values)),
        "psnr": float(np.mean(psnr_values)),
        "psnr_std": float(np.std(psnr_values)),
        "n_pairs": float(len(mse_values)),
    }


def enrich_validation_with_stego_quality(metrics: Dict[str, float], options) -> Dict[str, float]:
    """Append subject-stego / full-stego quality metrics (MSE/SSIM/PSNR)."""
    out = dict(metrics or {})
    clean_dir = getattr(options, "reference_image_dir", None) or os.environ.get("STEGOMIA_COVER_DIR")
    pair_clean = os.environ.get("STEGOMIA_SUBJECT_COVER_DIR")
    if pair_clean and Path(pair_clean).exists():
        clean_for_subject = pair_clean
    else:
        clean_for_subject = clean_dir

    subject_stego_dir = os.environ.get("STEGOMIA_SUBJECT_STEGO_DIR")
    stego_dir = os.environ.get("STEGOMIA_STEGO_DIR")
    if not clean_dir or not subject_stego_dir or not stego_dir:
        return out
    sstego = compute_paired_stego_quality(clean_for_subject, subject_stego_dir)
    stego = compute_paired_stego_quality(clean_dir, stego_dir)
    for key, value in sstego.items():
        out[f"sstego_{key}"] = value
    for key, value in stego.items():
        out[f"stego_{key}"] = value
    return out


def save_validation_results(
    metrics: Dict[str, float],
    epoch: int,
    output_file: str
):
    """
    Save validation results to file.
    MSE/SSIM/PSNR use LaTeX `$mean \\pm std$`. LPIPS is intentionally omitted.
    """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    mode = "a" if os.path.exists(output_file) else "w"
    with open(output_file, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("# Defense fine-tune validation log (Flickr8k)\n")
            f.write("# No LPIPS. Image quality shown as $mean \\pm std$.\n")
            f.write(
                "Epoch\tAvg_Similarity\tMSE\tSSIM\tPSNR\t"
                "subject_stego_MSE\tsubject_stego_SSIM\tsubject_stego_PSNR\t"
                "Stego_MSE\tStego_SSIM\tStego_PSNR\n"
            )

        def fmt_group(prefix: str) -> str:
            if f"{prefix}mse" not in metrics:
                return "NA\tNA\tNA"
            return "\t".join([
                _latex_mean_std(metrics[f"{prefix}mse"], metrics.get(f"{prefix}mse_std", 0.0), 4),
                _latex_mean_std(metrics[f"{prefix}ssim"], metrics.get(f"{prefix}ssim_std", 0.0), 4),
                _latex_mean_std(metrics[f"{prefix}psnr"], metrics.get(f"{prefix}psnr_std", 0.0), 2),
            ])

        avg_sim = metrics.get("avg_similarity")
        avg_sim_s = f"{avg_sim:.6f}" if avg_sim is not None else "NA"
        f.write(f"{epoch}\t{avg_sim_s}\t{fmt_group('')}\t{fmt_group('sstego_')}\t{fmt_group('stego_')}\n")

        f.write(f"\n=== Epoch {epoch} ===\n")
        if avg_sim is not None:
            f.write(f"Avg image-text cosine similarity: {avg_sim:.6f}\n")
        if "mse" in metrics:
            f.write(
                f"Test/ref MSE/SSIM/PSNR: "
                f"{_latex_mean_std(metrics['mse'], metrics.get('mse_std', 0), 4)} / "
                f"{_latex_mean_std(metrics['ssim'], metrics.get('ssim_std', 0), 4)} / "
                f"{_latex_mean_std(metrics['psnr'], metrics.get('psnr_std', 0), 2)}\n"
            )
        if "sstego_mse" in metrics:
            f.write(
                f"subject_stego vs clean MSE/SSIM/PSNR: "
                f"{_latex_mean_std(metrics['sstego_mse'], metrics.get('sstego_mse_std', 0), 4)} / "
                f"{_latex_mean_std(metrics['sstego_ssim'], metrics.get('sstego_ssim_std', 0), 4)} / "
                f"{_latex_mean_std(metrics['sstego_psnr'], metrics.get('sstego_psnr_std', 0), 2)}\n"
            )
        if "stego_mse" in metrics:
            f.write(
                f"stego vs clean MSE/SSIM/PSNR: "
                f"{_latex_mean_std(metrics['stego_mse'], metrics.get('stego_mse_std', 0), 4)} / "
                f"{_latex_mean_std(metrics['stego_ssim'], metrics.get('stego_ssim_std', 0), 4)} / "
                f"{_latex_mean_std(metrics['stego_psnr'], metrics.get('stego_psnr_std', 0), 2)}\n"
            )
        f.write("\n")
        f.flush()


def worker(rank, options, logger):
    """Worker function for distributed training."""
    options.rank = rank
    options.master = rank == 0
    
    set_logger(rank=rank, logger=logger, distributed=options.distributed)
    
    # Set device properly (consistent with main.py)
    if options.device == "cuda":
        options.device += ":" + str(options.device_ids[options.rank] if options.distributed else options.device_id)
    
    logging.info(f"Using {options.device} device")
    
    if options.master:
        logging.info("Params:")
        with open(os.path.join(options.log_dir_path, "params.txt"), "w") as file:
            for key in sorted(vars(options)):
                value = getattr(options, key)
                logging.info(f"{key}: {value}")
                file.write(f"{key}: {value}\n")
    
    if options.distributed:
        dist.init_process_group(
            backend=options.distributed_backend,
            init_method=options.distributed_init_method,
            world_size=options.num_devices,
            rank=options.rank
        )
    
    options.batch_size = options.batch_size // options.num_devices
    
    # Load model
    model, processor = load_model(name=options.model_name, pretrained=options.pretrained)
    
    if options.device == "cpu":
        model.float()
    else:
        torch.cuda.set_device(options.device_ids[options.rank] if options.distributed else options.device_id)
        model.to(options.device)
        if options.distributed:
            model = DDP(model, device_ids=[options.device_ids[options.rank]])
    
    # Load checkpoint
    if options.checkpoint is not None:
        if os.path.isfile(options.checkpoint):
            checkpoint = None
            try:
                checkpoint = torch.load(options.checkpoint, map_location=options.device)
                state_dict = _extract_state_dict(checkpoint)
                state_dict = _normalize_state_dict(state_dict, options.distributed)
                
                # Try to load state dict with strict=False to handle missing keys
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    logging.warning(f"Missing keys when loading checkpoint: {missing_keys[:5]}...")
                if unexpected_keys:
                    logging.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys[:5]}...")
                logging.info(f"Loaded checkpoint '{options.checkpoint}'")
            except Exception as exc:
                keys = []
                if checkpoint is not None and isinstance(checkpoint, dict):
                    keys = list(checkpoint.keys())[:10]
                logging.error(f"Error loading checkpoint '{options.checkpoint}': {exc}")
                if keys:
                    logging.error(f"Checkpoint keys: {keys}")
                raise exc
        else:
            logging.warning(f"No checkpoint found at {options.checkpoint}")
    
    # Load data
    data = {"train": None}
    if options.train_data:
        data["train"] = get_token_train_dataloader(options, processor)
    
    # Setup optimizer
    optimizer = None
    scheduler = None
    if data["train"] is not None:
        weight_decay_parameters = []
        no_weight_decay_parameters = []
        
        for name, parameter in model.named_parameters():
            if all(key not in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad:
                weight_decay_parameters.append(parameter)
            if any(key in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad:
                no_weight_decay_parameters.append(parameter)
        
        optimizer = optim.AdamW(
            [
                {"params": no_weight_decay_parameters, "weight_decay": 0},
                {"params": weight_decay_parameters, "weight_decay": options.weight_decay}
            ],
            lr=options.lr,
            betas=(options.beta1, options.beta2),
            eps=options.eps
        )
        scheduler = cosine_scheduler(
            optimizer,
            options.lr,
            options.num_warmup_steps,
            data["train"].num_batches * options.epochs
        )
    
    cudnn.benchmark = True
    cudnn.deterministic = False
    
    if options.wandb and options.master:
        logging.debug("Starting wandb")
        wandb.init(
            project="stegomia-defense",
            notes=options.notes,
            tags=[],
            config=vars(options),
            entity=None
        )
        wandb.run.name = options.name
        wandb.save(os.path.join(options.log_dir_path, "params.txt"))
    
    # Validation before training
    if options.master and options.test_image_dir and options.annotations_file:
        logging.info("Initial validation (epoch 0)")
        metrics = validate_with_image_quality_metrics(
            model,
            processor,
            options.test_image_dir,
            options.annotations_file,
            options,
            reference_image_dir=options.reference_image_dir,
            epoch=0
        )
        metrics = enrich_validation_with_stego_quality(metrics or {}, options)
        if metrics:
            logging.info(f"Initial metrics: {metrics}")
        else:
            logging.info("Initial validation: No metrics computed (no valid image pairs found)")
        save_validation_results(metrics, 0, options.output_results_file)
    
    # Training loop
    if data["train"] is not None:
        if not getattr(options, "checkpoints_dir_path", None):
            options.checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
        os.makedirs(options.checkpoints_dir_path, exist_ok=True)
        
        scaler = GradScaler()
        
        for epoch in range(1, options.epochs + 1):
            if options.master:
                logging.info(f"Starting Epoch {epoch}")
            
            start = time.time()
            train(epoch, model, data, optimizer, scheduler, scaler, options)
            end = time.time()
            
            if options.master:
                logging.info(f"Finished Epoch {epoch}, Time Taken: {end - start:.3f}")
                
                # Validation after each epoch
                if options.test_image_dir and options.annotations_file:
                    metrics = validate_with_image_quality_metrics(
                        model,
                        processor,
                        options.test_image_dir,
                        options.annotations_file,
                        options,
                        reference_image_dir=options.reference_image_dir,
                        epoch=epoch
                    )
                    metrics = enrich_validation_with_stego_quality(metrics or {}, options)
                    if metrics:
                        logging.info(f"Epoch {epoch} validation metrics:")
                        for key, value in metrics.items():
                            if key.endswith('_std') or key.endswith('n_pairs'):
                                continue
                            std_key = f"{key}_std"
                            if std_key in metrics:
                                logging.info(
                                    f"  {key}: {_latex_mean_std(value, metrics[std_key], 4 if 'psnr' not in key else 2)}"
                                )
                            else:
                                logging.info(f"  {key}: {value:.6f}")
                    else:
                        logging.info(f"Epoch {epoch} validation: No metrics computed (no valid image pairs found)")
                    
                    save_validation_results(metrics, epoch, options.output_results_file)
                    
                    if options.wandb:
                        for key, value in metrics.items():
                            wandb.log({f"validation/{key}": value, "epoch": epoch})
                
                # Save checkpoint
                checkpoint = {
                    "epoch": epoch,
                    "name": options.name,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict() if optimizer else None
                }
                torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch_{epoch}.pt"))
                
                # Save final model
                if epoch == options.epochs:
                    final_model_path = os.path.join(
                        options.checkpoints_dir_path,
                        options.final_model_name
                    )
                    torch.save(checkpoint, final_model_path)
                    logging.info(f"Saved final model to {final_model_path}")
    
    if options.distributed:
        dist.destroy_process_group()
    
    if options.wandb and options.master:
        wandb.finish()


def main():
    """Main function."""
    options = parse_args()
    
    # Set default values for fine-tuning
    if not hasattr(options, 'inmodal') or not options.inmodal:
        options.inmodal = True  # in-modality consistency training
    
    if not hasattr(options, 'complete_finetune') or not options.complete_finetune:
        options.complete_finetune = True
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if options.log_dir_path is None:
        options.log_dir_path = os.path.join(
            options.logs,
            f"{timestamp}_{options.name}"
        )
    os.makedirs(options.log_dir_path, exist_ok=True)
    options.log_file_path = os.path.join(options.log_dir_path, "output.log")
    
    # Resolve training token file when train_data is a directory
    if options.train_data and os.path.isdir(options.train_data):
        if not getattr(options, "train_token_file", None):
            # Try to find token file or JSON file
            parent_dir = os.path.dirname(options.train_data.rstrip('/'))
            # Check for MSCOCO JSON file first
            for candidate in ("captions_train2017.json", "captions_val2017.json", "captions_train.json", "captions_val.json"):
                candidate_path = os.path.join(parent_dir, candidate)
                if os.path.exists(candidate_path):
                    options.train_token_file = candidate_path
                    break
            # If not found, try Flickr8k token files
            if not options.train_token_file:
                for candidate in ("Flickr8k.token.txt", "Flickr8k.lemma.token.txt"):
                    candidate_path = os.path.join(parent_dir, candidate)
                    if os.path.exists(candidate_path):
                        options.train_token_file = candidate_path
                        break
        # If annotations_file is provided and is JSON, use it as train_token_file
        if not options.train_token_file and hasattr(options, 'annotations_file') and options.annotations_file:
            if options.annotations_file.lower().endswith('.json') and os.path.exists(options.annotations_file):
                options.train_token_file = options.annotations_file
        if not options.train_token_file or not os.path.exists(options.train_token_file):
            raise ValueError(
                "train_token_file is required when train_data is a directory. "
                "Provide --train_token_file or place token file (Flickr8k.token.txt) or JSON file (captions_train2017.json) in the parent directory."
            )
    
    # Set output results file
    if not hasattr(options, 'output_results_file') or options.output_results_file is None:
        results_dir = "./outputs/defense"
        os.makedirs(results_dir, exist_ok=True)
        options.output_results_file = os.path.join(results_dir, "finetune_val.txt")
    
    # Set final model name
    if not hasattr(options, 'final_model_name') or options.final_model_name is None:
        options.final_model_name = "defense_finetune.pt"
    if not getattr(options, "checkpoints_dir_path", None):
        options.checkpoints_dir_path = "./outputs/checkpoints/defense"

    logger, listener = get_logger(options.log_file_path)
    listener.start()
    try:
        ngpus = torch.cuda.device_count()
        if ngpus == 0 or options.device == "cpu":
            options.device = "cpu"
            options.num_devices = 1
            options.distributed = False
            worker(0, options, logger)
        else:
            if ngpus == 1 or not options.distributed:
                options.device = "cuda"
                options.num_devices = 1
                options.distributed = False
                worker(0, options, logger)
            else:
                options.device = "cuda"
                if options.device_ids is None:
                    options.device_ids = list(range(ngpus))
                    options.num_devices = ngpus
                else:
                    options.device_ids = list(map(int, options.device_ids[0].split()))
                    options.num_devices = len(options.device_ids)
                options.distributed = True
                os.environ["NCCL_P2P_DISABLE"] = "1"
                mp.spawn(worker, nprocs=options.num_devices, args=(options, logger), join=True)
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
