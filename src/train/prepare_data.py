#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare training data with different steganography ratios.

This script:
1. Randomly selects 1%, 2%, 5% of stego images from perfect match set
2. Creates mixed datasets with clean and stego images
3. Generates CSV files for training
"""

import os
import random
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import argparse


def load_captions(caption_file: str) -> Dict[str, str]:
    """Load captions from token file."""
    captions = {}
    with open(caption_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '#0' in line:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    img_name = parts[0].split('#')[0].replace('.jpg', '').replace('.png', '')
                    caption = parts[1].strip()
                    captions[img_name] = caption
    return captions


def prepare_training_data(
    stego_dir: Path,
    clean_dir: Path,
    caption_file: str,
    output_dir: Path,
    stego_ratios: List[float] = [0.01, 0.02, 0.05],
    random_seed: int = 42,
    build_mixed_dirs: bool = False,
    mixed_output_dir: Optional[Path] = None,
    link_type: str = "symlink",
):
    """
    Prepare training data with different stego ratios.
    
    Args:
        stego_dir: Directory containing stego images
        clean_dir: Directory containing clean images
        caption_file: Path to caption file
        output_dir: Output directory for CSV files
        stego_ratios: List of stego ratios (e.g., [0.01, 0.02, 0.05])
        random_seed: Random seed for reproducibility
    """
    random.seed(random_seed)
    
    # Load captions
    print("Loading captions...")
    captions = load_captions(caption_file)
    print(f"Loaded {len(captions)} captions")
    
    # Get all stego images (perfect match set)
    stego_images = sorted([f.stem for f in stego_dir.glob('*.jpg')])
    print(f"\nFound {len(stego_images)} stego images")
    
    # Get all clean images
    clean_images = sorted([f.stem for f in clean_dir.glob('*.jpg')])
    print(f"Found {len(clean_images)} clean images")

    clean_files = {
        f.stem: f.name for f in clean_dir.glob('*.jpg')
    }
    stego_files = {
        f.stem: f.name for f in stego_dir.glob('*.jpg')
    }
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each stego ratio
    for ratio in stego_ratios:
        print(f"\n{'='*80}")
        print(f"Processing stego ratio: {ratio*100:.1f}%")
        print(f"{'='*80}")
        
        # Calculate number of stego images to select based on total clean images (8091)
        # Ratio is based on total dataset size, not stego images size
        num_stego_target = round(len(clean_images) * ratio)
        # But limit to available stego images
        num_stego = min(num_stego_target, len(stego_images))
        print(f"Target: {num_stego_target} stego images ({ratio*100:.1f}% of {len(clean_images)} total images)")
        print(f"Selecting {num_stego} stego images from {len(stego_images)} available")
        
        # Randomly select stego images
        selected_stego = random.sample(stego_images, num_stego)
        selected_stego_set = set(selected_stego)
        
        # Get corresponding clean images (same names as selected stego)
        selected_clean = [img for img in clean_images if img in selected_stego_set]
        
        # Get remaining clean images (not in selected stego set)
        remaining_clean = [img for img in clean_images if img not in selected_stego_set]
        
        print(f"Selected stego images: {len(selected_stego)}")
        print(f"Corresponding clean images: {len(selected_clean)}")
        print(f"Remaining clean images: {len(remaining_clean)}")
        
        # Create training data list
        training_data = []
        
        # Add stego images with their captions
        for img_name in selected_stego:
            if img_name in captions:
                training_data.append({
                    'filepath': f"{img_name}.jpg",
                    'caption': captions[img_name],
                    'is_stego': True,
                    'clean_filepath': f"{img_name}.jpg"  # Reference to clean image
                })
        
        # Add remaining clean images
        for img_name in remaining_clean:
            if img_name in captions:
                training_data.append({
                    'filepath': f"{img_name}.jpg",
                    'caption': captions[img_name],
                    'is_stego': False,
                    'clean_filepath': f"{img_name}.jpg"
                })
        
        print(f"Total training samples: {len(training_data)}")
        print(f"  Stego samples: {sum(1 for d in training_data if d['is_stego'])}")
        print(f"  Clean samples: {sum(1 for d in training_data if not d['is_stego'])}")
        
        # Create DataFrame
        df = pd.DataFrame(training_data)
        
        # Save CSV file
        csv_filename = output_dir / f"flickr8k_stego_{int(ratio*100)}pct_train.csv"
        df.to_csv(csv_filename, index=False, sep='\t')
        print(f"\nSaved training data to: {csv_filename}")
        
        # Save selected stego image list for reference
        stego_list_file = output_dir / f"flickr8k_stego_{int(ratio*100)}pct_selected.txt"
        with open(stego_list_file, 'w') as f:
            for img_name in sorted(selected_stego):
                f.write(f"{img_name}\n")
        print(f"Saved selected stego list to: {stego_list_file}")

        if build_mixed_dirs and mixed_output_dir is not None:
            ratio_tag = f"{int(ratio*100)}pct"
            out_dir = mixed_output_dir / f"Flicker8k_Dataset_stego_{ratio_tag}"
            out_dir.mkdir(parents=True, exist_ok=True)

            def link_or_copy(src: Path, dst: Path):
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                if link_type == "hardlink":
                    os.link(src, dst)
                elif link_type == "copy":
                    import shutil
                    shutil.copy2(src, dst)
                else:
                    os.symlink(src, dst)

            # Link all clean images first
            for stem, filename in clean_files.items():
                src = clean_dir / filename
                dst = out_dir / filename
                link_or_copy(src, dst)

            # Replace selected stego images
            replaced = 0
            for stem in selected_stego:
                stego_name = stego_files.get(stem)
                clean_name = clean_files.get(stem)
                if not stego_name or not clean_name:
                    continue
                src = stego_dir / stego_name
                dst = out_dir / clean_name
                link_or_copy(src, dst)
                replaced += 1

            print(f"Built mixed dataset: {out_dir}")
            print(f"  Replaced with stego images: {replaced}")
    
    print(f"\n{'='*80}")
    print("Data preparation completed!")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Prepare training data with different stego ratios')
    parser.add_argument('--stego_dir', type=str, required=True,
                       help='Directory containing stego images')
    parser.add_argument('--clean_dir', type=str, required=True,
                       help='Directory containing clean images')
    parser.add_argument('--caption_file', type=str, required=True,
                       help='Path to caption file')
    parser.add_argument('--output_dir', type=str, default='./outputs/data_configs',
                       help='Output directory for CSV files')
    parser.add_argument('--ratios', type=float, nargs='+', default=[0.01, 0.02, 0.05],
                       help='Stego ratios (e.g., 0.01 0.02 0.05)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--build_mixed_dirs', action='store_true',
                       help='Create mixed dataset directories with stego replacements')
    parser.add_argument('--mixed_output_dir', type=str, default=None,
                       help='Root directory for mixed datasets')
    parser.add_argument('--link_type', type=str, default='symlink',
                       choices=['symlink', 'hardlink', 'copy'],
                       help='How to materialize mixed datasets')
    
    args = parser.parse_args()
    
    mixed_output_dir = Path(args.mixed_output_dir) if args.mixed_output_dir else None

    prepare_training_data(
        Path(args.stego_dir),
        Path(args.clean_dir),
        args.caption_file,
        Path(args.output_dir),
        args.ratios,
        args.seed,
        build_mixed_dirs=args.build_mixed_dirs,
        mixed_output_dir=mixed_output_dir,
        link_type=args.link_type,
    )


if __name__ == "__main__":
    main()
