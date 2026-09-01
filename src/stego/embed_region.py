#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Steganography embedding for foreground images with full captions.

This script embeds the full caption (first caption for each image) into foreground images
using frequency-domain steganography, then composites them with original images.
"""

import os
import re
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np
from PIL import Image

# Import frequency steganography functions
import sys
from src.stego.frequency import embed_frequency, embed_frequency_dct, extract_frequency

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def index_images(directory: str) -> Dict[str, Path]:
    """
    Index images in a directory by stem (filename without extension).
    """
    images = {}
    root = Path(directory)
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            stem = path.stem
            if stem not in images:
                images[stem] = path
    return images


def extract_subject_from_caption(caption: str) -> str:
    """
    Extract subject from caption.
    
    The subject is typically the first part of the sentence before the main verb phrase.
    For example: "Two racer drive a white bike down a road" -> "Two racer drive"
    
    Args:
        caption: Input caption text
        
    Returns:
        Extracted subject string
    """
    caption = caption.strip()
    if not caption:
        return ""
    
    # Remove trailing period
    caption = caption.rstrip('.')
    
    # Common patterns for subject extraction:
    # 1. Subject + verb (e.g., "Two racer drive", "A child climbs")
    # 2. Subject + verb + object (e.g., "A man lays on a bench")
    
    # Common prepositions that often separate subject from object/location
    prepositions = [' down ', ' up ', ' into ', ' onto ', ' toward ', ' away ', 
                    ' on ', ' in ', ' at ', ' with ', ' to ', ' for ', ' of ', 
                    ' by ', ' from ', ' near ', ' behind ', ' in front of ']
    
    # Try to find the subject by splitting at prepositions
    caption_lower = caption.lower()
    for prep in prepositions:
        if prep in caption_lower:
            # Find the position (case-insensitive)
            idx = caption_lower.find(prep)
            if idx > 0:
                subject_part = caption[:idx].strip()
                words = subject_part.split()
                # Ensure we have a reasonable subject (2-6 words)
                if 2 <= len(words) <= 6:
                    return subject_part
    
    # If no preposition found, try to extract first part before common verbs
    # Common verbs that often mark the end of subject
    common_verbs = [' is ', ' are ', ' was ', ' were ', ' has ', ' have ', 
                    ' had ', ' do ', ' does ', ' did ', ' can ', ' could ',
                    ' will ', ' would ', ' should ', ' may ', ' might ']
    
    caption_lower = caption.lower()
    for verb in common_verbs:
        if verb in caption_lower:
            idx = caption_lower.find(verb)
            if idx > 0:
                subject_part = caption[:idx].strip()
                words = subject_part.split()
                if 2 <= len(words) <= 6:
                    return subject_part
    
    # If no preposition or verb found, try to extract first 2-4 words as subject
    words = caption.split()
    if len(words) >= 2:
        # Take first 2-4 words as subject
        subject_length = min(4, len(words))
        return ' '.join(words[:subject_length])
    
    return caption


def load_captions(caption_file: str) -> Dict[str, List[str]]:
    """
    Load captions from token file or COCO JSON file.

    Only the first caption encountered for each image is retained.

    Args:
        caption_file: Path to caption file (.token.txt or .json)

    Returns:
        Dictionary mapping image name to list of captions (length 1)
    """
    captions_dict = {}
    
    # Check if it's a JSON file (COCO format)
    if caption_file.endswith('.json'):
        return load_coco_captions(caption_file)
    
    # Otherwise, treat as token file format
    with open(caption_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Format: image_name#caption_id\tcaption_text
            if '\t' in line:
                parts = line.split('\t', 1)
            elif ' ' in line and '#' in line:
                # Alternative format: image_name#caption_id caption_text
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
                captions_dict[image_name] = [caption_text]
    
    return captions_dict


def load_coco_captions(caption_json_file: str) -> Dict[str, List[str]]:
    """
    Load COCO captions from JSON file.

    Supports two formats:
    1. COCO format: {"images": [...], "annotations": [...]}
    2. ImageNet-1k format: {"image_basename": ["caption1", "caption2", ...]}

    Only the first caption encountered for each image is retained.

    Args:
        caption_json_file: Path to COCO caption JSON file

    Returns:
        Dictionary mapping image filename (or basename) to list of captions (length 1)
    """
    with open(caption_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if it's ImageNet-1k format (direct dict mapping basename to captions)
    if isinstance(data, dict):
        # Check if it has COCO structure
        if 'images' in data and 'annotations' in data:
            # COCO format: create mapping from image_id to filename
            image_id_to_filename = {}
            for img in data.get('images', []):
                image_id_to_filename[img['id']] = img['file_name']
            
            # Create mapping from filename to list of captions
            filename_to_captions = {}
            for ann in data.get('annotations', []):
                image_id = ann['image_id']
                if image_id in image_id_to_filename:
                    filename = image_id_to_filename[image_id]
                    if filename not in filename_to_captions:
                        filename_to_captions[filename] = [ann['caption']]
            
            return filename_to_captions
        else:
            # ImageNet-1k format: direct mapping from basename to captions
            # Keys are image basenames (without extension), values are caption lists
            filtered: Dict[str, List[str]] = {}
            for image_name, captions in data.items():
                if isinstance(captions, list):
                    if captions:
                        filtered[image_name] = [captions[0]]
                elif isinstance(captions, str):
                    filtered[image_name] = [captions]
            return filtered
    
    return {}


def get_caption_for_image(image_name: str, captions_dict: Dict[str, List[str]]) -> str:
    """
    Get full caption for a specific image from captions.
    
    Uses the first caption for the image.
    
    Args:
        image_name: Image filename
        captions_dict: Dictionary of captions
        
    Returns:
        Full caption string (first caption for the image)
    """
    # Try exact match first
    if image_name in captions_dict:
        caption = captions_dict[image_name][0]
        return caption.strip()
    
    # Try without extension
    image_base = os.path.splitext(image_name)[0]
    if image_base in captions_dict:
        caption = captions_dict[image_base][0]
        return caption.strip()
    
    # Try with different extensions
    for ext in ['.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG']:
        test_name = image_base + ext
        if test_name in captions_dict:
            caption = captions_dict[test_name][0]
            return caption.strip()
    
    return ""


def get_subject_for_image(image_name: str, captions_dict: Dict[str, List[str]]) -> str:
    """
    Get subject for a specific image from captions (deprecated, use get_caption_for_image).
    
    This function is kept for backward compatibility but now returns the full caption.
    
    Args:
        image_name: Image filename
        captions_dict: Dictionary of captions
        
    Returns:
        Full caption string (first caption for the image)
    """
    return get_caption_for_image(image_name, captions_dict)


def composite_images(foreground: Image.Image, original: Image.Image, 
                    mask: Image.Image) -> Image.Image:
    """
    Composite stego foreground image with original image using mask.
    
    Args:
        foreground: Stego foreground image (RGBA or RGB)
        original: Original image (RGB)
        mask: Mask image (grayscale or RGB)
        
    Returns:
        Composited image (RGB)
    """
    # Ensure images are same size
    if foreground.size != original.size:
        foreground = foreground.resize(original.size, Image.LANCZOS)
    
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.LANCZOS)
    
    # Convert mask to grayscale if needed
    if mask.mode != 'L':
        mask = mask.convert('L')
    
    # Normalize mask to 0-1 range
    mask_array = np.array(mask, dtype=np.float32) / 255.0
    
    # Convert images to numpy arrays
    if foreground.mode == 'RGBA':
        fg_array = np.array(foreground)[:, :, :3].astype(np.float32)
    else:
        fg_array = np.array(foreground.convert('RGB')).astype(np.float32)
    
    orig_array = np.array(original.convert('RGB')).astype(np.float32)
    
    # Composite: use foreground where mask is high, original where mask is low
    mask_3d = mask_array[:, :, np.newaxis]
    composited = (fg_array * mask_3d + orig_array * (1 - mask_3d)).astype(np.uint8)
    
    return Image.fromarray(composited, 'RGB')


def process_single_image(
    image_name: str,
    fg_path: str,
    mask_path: str,
    orig_path: str,
    stego_foreground_dir: str,
    stego_output_dir: str,
    captions_dict: Dict[str, List[str]],
    domain: str = "fft",
    key: int = 0,
    block_size: int = 8,
    freq: Tuple[int, int] = (3, 3),
    min_magnitude: float = 2.0,
) -> bool:
    """
    Process a single image: embed subject and composite.
    
    Args:
        image_name: Name of the image file
        fg_path: Path to the foreground image
        mask_path: Path to the mask image
        orig_path: Path to the original image
        stego_foreground_dir: Output directory for stego foreground images
        stego_output_dir: Output directory for final composited images
        captions_dict: Dictionary of captions
        key: Steganography key
        block_size: Block size for FFT embedding
        freq: Frequency coordinate for embedding
        min_magnitude: Minimum magnitude for frequency coefficients
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get image base name (without extension)
        image_base = os.path.splitext(image_name)[0]
        
        # Get full caption from captions (use first caption)
        caption = get_caption_for_image(image_name, captions_dict)
        if not caption:
            print(f"Warning: No caption found for {image_name}")
            return False
        
        # Load foreground image
        foreground_img = Image.open(fg_path).convert('RGB')
        
        # Embed full caption into foreground image
        caption_bytes = caption.encode('utf-8')
        if domain == "dct":
            stego_foreground = embed_frequency_dct(
                foreground_img,
                caption_bytes,
                key=key,
                block_size=block_size,
                freq=freq,
                min_magnitude=min_magnitude,
            )
        else:
            stego_foreground = embed_frequency(
                foreground_img,
                caption_bytes,
                key=key,
                block_size=block_size,
                freq=freq,
                min_magnitude=min_magnitude,
            )
        
        # Save stego foreground image
        stego_fg_path = os.path.join(stego_foreground_dir, image_base + '.png')
        os.makedirs(os.path.dirname(stego_fg_path), exist_ok=True)
        stego_foreground.save(stego_fg_path)
        
        # Load original image and mask
        original_img = Image.open(orig_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L')
        
        # Composite stego foreground with original using mask
        composited_img = composite_images(stego_foreground, original_img, mask_img)
        
        # Save composited image (preserve original format)
        stego_output_path = os.path.join(stego_output_dir, image_name)
        os.makedirs(os.path.dirname(stego_output_path), exist_ok=True)
        
        # Preserve original image format
        orig_ext = os.path.splitext(image_name)[1].lower()
        if orig_ext in ['.jpg', '.jpeg']:
            composited_img.save(stego_output_path, format='JPEG', quality=95)
        elif orig_ext == '.png':
            composited_img.save(stego_output_path, format='PNG')
        else:
            # Default to JPEG
            if not stego_output_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                stego_output_path = os.path.splitext(stego_output_path)[0] + '.jpg'
            composited_img.save(stego_output_path, format='JPEG', quality=95)
        
        return True
        
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Embed full captions into foreground images and composite with originals'
    )
    parser.add_argument('--foreground_dir', type=str, required=True,
                       help='Directory containing foreground images')
    parser.add_argument('--mask_dir', type=str, required=True,
                       help='Directory containing mask images')
    parser.add_argument('--original_dir', type=str, required=True,
                       help='Directory containing original images')
    parser.add_argument('--caption_file', type=str, required=True,
                       help='Path to caption file (.token.txt or .json for COCO format)')
    parser.add_argument('--stego_foreground_dir', type=str, required=True,
                       help='Output directory for stego foreground images')
    parser.add_argument('--stego_output_dir', type=str, required=True,
                       help='Output directory for final composited images')
    parser.add_argument('--domain', type=str, default='fft',
                       choices=['fft', 'dct'],
                       help='Frequency transform domain for embedding')
    parser.add_argument('--key', type=int, default=0,
                       help='Steganography key')
    parser.add_argument('--block_size', type=int, default=8,
                       help='Block size for FFT embedding')
    parser.add_argument('--freq', type=int, nargs=2, default=[3, 3],
                       help='Frequency coordinate for embedding')
    parser.add_argument('--min_magnitude', type=float, default=2.0,
                       help='Minimum magnitude for frequency coefficients')
    parser.add_argument('--sample_fraction', type=float, default=None,
                       help='Process this fraction of matched images (0-1)')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Process at most this many matched images')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed for sampling matched images')
    
    args = parser.parse_args()
    
    # Load captions
    print("Loading captions...")
    captions_dict = load_captions(args.caption_file)
    print(f"Loaded {len(captions_dict)} image captions")
    
    # Index images from each directory
    print("Indexing images...")
    originals = index_images(args.original_dir)
    foregrounds = index_images(args.foreground_dir)
    masks = index_images(args.mask_dir)
    print(f"Found {len(originals)} originals, {len(foregrounds)} foregrounds, {len(masks)} masks")
    
    common_bases = sorted(set(originals) & set(foregrounds) & set(masks))
    if not common_bases:
        raise SystemExit("No images with matching foreground, mask, and original were found.")
    
    if len(common_bases) < len(originals):
        print(f"Note: only {len(common_bases)} of {len(originals)} originals have matching foreground + mask.")
    
    total_available = len(common_bases)
    target_count = total_available
    if args.sample_fraction is not None:
        if not (0 < args.sample_fraction <= 1):
            raise ValueError("--sample_fraction must be in the range (0, 1].")
        target_count = max(1, int(round(total_available * args.sample_fraction)))
    
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("--max_images must be a positive integer.")
        target_count = min(target_count, args.max_images)
    
    if target_count < total_available:
        rng = random.Random(args.seed)
        rng.shuffle(common_bases)
        common_bases = common_bases[:target_count]
        print(f"Sampling {len(common_bases)} images (seed={args.seed})")
    
    print(f"Processing {len(common_bases)} images")
    image_tasks = [
        (originals[base].name, foregrounds[base], masks[base], originals[base])
        for base in common_bases
    ]
    
    # Process each image
    success_count = 0
    for image_name, fg_path, mask_path, orig_path in tqdm(image_tasks, desc="Processing images"):
        if process_single_image(
            image_name,
            fg_path,
            mask_path,
            orig_path,
            args.stego_foreground_dir,
            args.stego_output_dir,
            captions_dict,
            domain=args.domain,
            key=args.key,
            block_size=args.block_size,
            freq=tuple(args.freq),
            min_magnitude=args.min_magnitude,
        ):
            success_count += 1
    
    print(f"\nProcessing completed: {success_count}/{len(image_tasks)} images processed successfully")


if __name__ == "__main__":
    main()
