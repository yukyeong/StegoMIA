#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""Data loader for mixed clean/stego image-text training."""
"""

import os
import re
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data._utils.collate import default_collate
from PIL import Image
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_captions(caption_file: str) -> Dict[str, List[str]]:
    """
    Load captions from token file.
    
    Args:
        caption_file: Path to caption file (e.g., Flickr8k.token.txt)
        
    Returns:
        Dictionary mapping image name to list of captions
    """
    captions_dict = {}
    
    with open(caption_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Format: image_name#caption_id\tcaption_text
            if '\t' in line:
                parts = line.split('\t', 1)
            elif ' ' in line and '#' in line:
                match = re.match(r'^([^#]+#\d+)\s+(.+)$', line)
                if match:
                    parts = [match.group(1), match.group(2)]
                else:
                    continue
            else:
                continue
            
            if len(parts) != 2:
                continue
            
            image_caption_id = parts[0].strip()
            caption_text = parts[1].strip()
            
            # Extract image name (remove #caption_id)
            if '#' in image_caption_id:
                image_name = image_caption_id.split('#')[0]
            else:
                image_name = image_caption_id
            
            if image_name not in captions_dict:
                captions_dict[image_name] = []
            captions_dict[image_name].append(caption_text)
    
    return captions_dict


class StegoImageTextDataset(Dataset):
    """
    Dataset for mixed clean/stego image-text training.
    """
    
    def __init__(
        self,
        image_dir: str,
        caption_dict: Dict[str, List[str]],
        processor,
        is_stego: bool = False,
        max_samples: Optional[int] = None,
        clean_image_dir: Optional[str] = None
    ):
        """
        Initialize dataset.
        
        Args:
            image_dir: Directory containing images
            caption_dict: Dictionary mapping image names to captions
            processor: Image/text processor
            is_stego: Whether this is stego dataset
            max_samples: Maximum number of samples (None for all)
            clean_image_dir: Directory containing clean images (for stego dataset)
        """
        self.image_dir = image_dir
        self.caption_dict = caption_dict
        self.processor = processor
        self.is_stego = is_stego
        self.clean_image_dir = clean_image_dir
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG']
        self.image_files = []
        
        if os.path.exists(image_dir):
            for ext in image_extensions:
                for img_file in os.listdir(image_dir):
                    if img_file.lower().endswith(ext.lower()):
                        # Get base name without extension
                        base_name = os.path.splitext(img_file)[0]
                        # Try to match with caption dict
                        if base_name in caption_dict or img_file in caption_dict:
                            self.image_files.append(img_file)
        
        # Limit samples if specified
        if max_samples is not None and max_samples < len(self.image_files):
            self.image_files = random.sample(self.image_files, max_samples)
        
        logger.info(f"Loaded {len(self.image_files)} images from {image_dir} (stego={is_stego})")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get caption (use first caption if multiple available)
        base_name = os.path.splitext(img_file)[0]
        if base_name in self.caption_dict:
            captions = self.caption_dict[base_name]
        elif img_file in self.caption_dict:
            captions = self.caption_dict[img_file]
        else:
            # Try without extension variations
            captions = None
            for key in self.caption_dict:
                if os.path.splitext(key)[0] == base_name:
                    captions = self.caption_dict[key]
                    break
            
            if captions is None:
                captions = [""]  # Empty caption as fallback
        
        caption_text = captions[0] if captions else ""
        
        # Process image and text
        pixel_values = self.processor.process_image(image)
        text_processed = self.processor.process_text([caption_text])
        
        result = {
            'pixel_values': pixel_values,
            'input_ids': text_processed['input_ids'][0],
            'attention_mask': text_processed['attention_mask'][0],
            'caption': caption_text,
            'image_path': img_path,
            'is_stego': self.is_stego,
            'clean_pixel_values': None,
            'clean_image_path': None
        }
        
        # Load corresponding clean image if this is stego dataset
        if self.is_stego and self.clean_image_dir is not None:
            clean_img_path = os.path.join(self.clean_image_dir, img_file)
            if os.path.exists(clean_img_path):
                try:
                    clean_image = Image.open(clean_img_path).convert('RGB')
                    clean_pixel_values = self.processor.process_image(clean_image)
                    result['clean_pixel_values'] = clean_pixel_values
                    result['clean_image_path'] = clean_img_path
                except Exception as e:
                    logger.debug(f"Failed to load clean image {clean_img_path}: {e}")
        
        return result


def custom_collate_fn(batch):
    """
    Custom collate function to handle None values in batch.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Collated batch with None values handled properly
    """
    # First, collate all standard fields
    collated = {}
    
    # Get all keys from first sample
    sample_keys = batch[0].keys()
    
    for key in sample_keys:
        values = [sample[key] for sample in batch]
        
        # Handle None values for optional fields
        if key in ['clean_pixel_values', 'clean_image_path']:
            # Keep as list, filtering None values will be done in training loop
            collated[key] = values
        else:
            # Use default collate for other fields
            try:
                collated[key] = default_collate(values)
            except TypeError as e:
                # If collate fails (e.g., due to None), keep as list
                logger.debug(f"Failed to collate {key}: {e}, keeping as list")
                collated[key] = values
    
    return collated


def create_mixed_dataloader(
    clean_image_dir: str,
    stego_image_dir: str,
    caption_file: str,
    processor,
    stego_ratio: float,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    max_clean_samples: Optional[int] = None,
    max_stego_samples: Optional[int] = None
) -> Tuple[DataLoader, Dict]:
    """
    Create mixed dataloader with clean and stego samples.
    
    Args:
        clean_image_dir: Directory containing clean images
        stego_image_dir: Directory containing stego images
        caption_file: Path to caption file
        processor: Image/text processor
        stego_ratio: Ratio of stego samples (0.01, 0.02, 0.05, 0.10, 0.20)
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        max_clean_samples: Maximum clean samples (None for all)
        max_stego_samples: Maximum stego samples (None for all)
        
    Returns:
        DataLoader and statistics dictionary
    """
    # Load captions
    logger.info(f"Loading captions from {caption_file}")
    caption_dict = load_captions(caption_file)
    logger.info(f"Loaded {len(caption_dict)} image captions")
    
    # Create clean dataset
    clean_dataset = StegoImageTextDataset(
        clean_image_dir,
        caption_dict,
        processor,
        is_stego=False,
        max_samples=max_clean_samples
    )
    
    # Create stego dataset (with reference to clean images)
    stego_dataset = StegoImageTextDataset(
        stego_image_dir,
        caption_dict,
        processor,
        is_stego=True,
        max_samples=max_stego_samples,
        clean_image_dir=clean_image_dir
    )
    
    # Calculate sample counts
    total_clean = len(clean_dataset)
    total_stego = len(stego_dataset)
    
    # Calculate how many stego samples to use
    if stego_ratio > 0:
        # If stego_ratio is 0.01 (1%), we want 1 stego per 99 clean
        # So: num_stego / (num_clean + num_stego) = stego_ratio
        # Solving: num_stego = stego_ratio * num_clean / (1 - stego_ratio)
        num_stego = int(total_clean * stego_ratio / (1 - stego_ratio))
        num_stego = min(num_stego, total_stego)
        
        # Sample stego dataset
        if num_stego < total_stego:
            stego_indices = random.sample(range(total_stego), num_stego)
            stego_dataset = torch.utils.data.Subset(stego_dataset, stego_indices)
    else:
        num_stego = 0
        stego_dataset = None
    
    # Combine datasets
    if stego_dataset is not None and num_stego > 0:
        mixed_dataset = ConcatDataset([clean_dataset, stego_dataset])
    else:
        mixed_dataset = clean_dataset
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        mixed_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    # Statistics
    stats = {
        'total_clean': total_clean,
        'total_stego': total_stego,
        'used_stego': num_stego,
        'stego_ratio': stego_ratio,
        'total_samples': len(mixed_dataset)
    }
    
    logger.info(f"Mixed dataset statistics:")
    logger.info(f"  Clean samples: {total_clean}")
    logger.info(f"  Stego samples: {num_stego} ({stego_ratio*100:.1f}%)")
    logger.info(f"  Total samples: {len(mixed_dataset)}")
    
    return dataloader, stats

