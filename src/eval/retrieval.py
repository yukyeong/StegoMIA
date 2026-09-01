#!/usr/bin/env python3
"""
Zero-shot image-text retrieval evaluation script.
Evaluates R@1, R@5, R@10 for both image-to-text and text-to-image retrieval
on both clean and steganography datasets.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import open_clip
    from open_clip import tokenize
except ImportError:
    print("Warning: open_clip not found. Please install it.")
    sys.exit(1)


class ImageTextDataset(Dataset):
    """Dataset for image-text pairs."""

    def __init__(
        self,
        image_dir: str,
        caption_file: str,
        transform=None,
        image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        image_list: Optional[Set[str]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            image_dir: Directory containing images
            caption_file: Path to caption file (format: image_filename\tcaption)
            transform: Image transformation function
            image_extensions: Allowed image extensions
            image_list: Optional set of image filenames to include
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_extensions = image_extensions

        # Load captions
        self.image_captions = {}
        if os.path.exists(caption_file):
            # Check if it's a JSON file (MSCOCO format)
            if caption_file.endswith('.json'):
                try:
                    with open(caption_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Handle MSCOCO2017 format
                    if 'images' in data and 'annotations' in data:
                        # Create mapping from image_id to image info
                        image_id_to_info = {img['id']: img for img in data['images']}
                        
                        # Create mapping from image_id to captions
                        image_id_to_captions = {}
                        for ann in data['annotations']:
                            img_id = ann['image_id']
                            if img_id not in image_id_to_captions:
                                image_id_to_captions[img_id] = []
                            if 'caption' in ann:
                                image_id_to_captions[img_id].append(ann['caption'])
                        
                        # Build image_name to caption mapping (use first caption)
                        available_files = set()
                        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                            available_files.update([f.name for f in self.image_dir.glob(f'*{ext}')])
                        
                        base_to_full = {}
                        for full_name in available_files:
                            base_name = Path(full_name).stem
                            if base_name not in base_to_full:
                                base_to_full[base_name] = []
                            base_to_full[base_name].append(full_name)
                        
                        # Match JSON file names with actual files
                        for img_id, img_info in image_id_to_info.items():
                            if img_id in image_id_to_captions and len(image_id_to_captions[img_id]) > 0:
                                json_file_name = img_info['file_name']
                                json_base_name = Path(json_file_name).stem
                                
                                if json_base_name in base_to_full:
                                    actual_file_name = base_to_full[json_base_name][0]
                                    # Use first caption for each image
                                    self.image_captions[actual_file_name] = [image_id_to_captions[img_id][0]]
                    # Handle ImageNet-1k format
                    elif isinstance(data, dict) and len(data) > 0:
                        first_value = list(data.values())[0]
                        if isinstance(first_value, list) and len(first_value) > 0 and isinstance(first_value[0], str):
                            available_files = set()
                            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                                available_files.update([f.name for f in self.image_dir.glob(f'*{ext}')])
                            
                            base_to_full = {}
                            for full_name in available_files:
                                base_name = Path(full_name).stem
                                if base_name not in base_to_full:
                                    base_to_full[base_name] = []
                                base_to_full[base_name].append(full_name)
                            
                            for img_base_name, captions in data.items():
                                if isinstance(captions, list) and len(captions) > 0:
                                    if img_base_name in base_to_full:
                                        actual_file_name = base_to_full[img_base_name][0]
                                        self.image_captions[actual_file_name] = [captions[0]]
                except Exception as e:
                    print(f"Error loading JSON file {caption_file}: {e}")
                    print("Falling back to text format parsing")
            
            # Parse text format (Flickr8k format)
            if not self.image_captions:
                with open(caption_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if "\t" in line:
                            img_name_with_suffix, caption = line.split("\t", 1)
                            # Handle format: image.jpg#0 or image.jpg
                            if "#" in img_name_with_suffix:
                                img_name = img_name_with_suffix.split("#")[0]
                            else:
                                img_name = img_name_with_suffix
                            if img_name not in self.image_captions:
                                self.image_captions[img_name] = []
                            self.image_captions[img_name].append(caption)

        # Get all images
        self.images = []
        for ext in self.image_extensions:
            self.images.extend(list(self.image_dir.glob(f"*{ext}")))
            self.images.extend(list(self.image_dir.glob(f"*{ext.upper()}")))

        self.images = sorted(self.images)
        if image_list:
            allowed_names = set(image_list)
            allowed_stems = {Path(name).stem for name in allowed_names}
            self.images = [
                img
                for img in self.images
                if img.name in allowed_names or img.stem in allowed_stems
            ]

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        """
        Get an item from the dataset.

        Args:
            idx: Index

        Returns:
            Tuple of (image, caption, image_path)
        """
        img_path = self.images[idx]
        img_name = img_path.name

        # Load image
        from PIL import Image

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image
            image = Image.new("RGB", (224, 224))
            if self.transform:
                image = self.transform(image)

        # Get caption
        caption = ""
        if img_name in self.image_captions:
            caption = self.image_captions[img_name][0]  # Use first caption
        elif img_name.replace(".jpg", "") in self.image_captions:
            caption = self.image_captions[img_name.replace(".jpg", "")][0]

        return image, caption, str(img_path)


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


def compute_recall_at_k(
    scores: np.ndarray, k_values: List[int] = [1, 5, 10]
) -> Dict[int, float]:
    """
    Compute Recall@K metrics.

    Args:
        scores: Similarity scores matrix [num_queries, num_candidates]
        k_values: List of K values to compute

    Returns:
        Dictionary mapping K to recall value
    """
    num_queries, num_candidates = scores.shape
    recalls = {}

    for k in k_values:
        k = min(k, num_candidates)
        recall_sum = 0.0

        for i in range(num_queries):
            # Get top-k indices for query i
            top_k_indices = np.argsort(scores[i])[::-1][:k]
            # Check if the correct match (diagonal) is in top-k
            if i in top_k_indices:
                recall_sum += 1.0

        recalls[k] = recall_sum / num_queries

    return recalls


def evaluate_zero_shot_retrieval(
    model, dataloader: DataLoader, device: torch.device, tokenizer
) -> Dict:
    """
    Evaluate zero-shot image-text retrieval.

    Args:
        model: CLIP model
        dataloader: DataLoader for evaluation data
        device: Device to run on
        tokenizer: Text tokenizer

    Returns:
        Dictionary of metrics (R@1, R@5, R@10 for both directions)
    """
    model.eval()

    all_image_features = []
    all_text_features = []

    with torch.no_grad():
        for images, texts, _ in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            texts = list(texts)

            # Get image features
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            # Get text features
            text_tokens = tokenizer(texts).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)

            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())

    # Concatenate all features
    image_features = torch.cat(all_image_features, dim=0).numpy()
    text_features = torch.cat(all_text_features, dim=0).numpy()

    # Compute similarity matrix
    similarity_matrix = image_features @ text_features.T  # [num_images, num_texts]

    # Image-to-Text retrieval: for each image, find matching text
    i2t_recalls = compute_recall_at_k(similarity_matrix, k_values=[1, 5, 10])

    # Text-to-Image retrieval: for each text, find matching image
    t2i_recalls = compute_recall_at_k(
        similarity_matrix.T, k_values=[1, 5, 10]
    )  # Transpose for text queries

    metrics = {
        "image_to_text_R@1": float(i2t_recalls[1]),
        "image_to_text_R@5": float(i2t_recalls[5]),
        "image_to_text_R@10": float(i2t_recalls[10]),
        "text_to_image_R@1": float(t2i_recalls[1]),
        "text_to_image_R@5": float(t2i_recalls[5]),
        "text_to_image_R@10": float(t2i_recalls[10]),
        "num_samples": len(image_features),
    }

    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Zero-shot Retrieval Evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B-16",
        help="CLIP model name",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="cc3m_12m",
        help="Pretrained dataset name",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint (e.g., ViT_B_16_cc3m_12m_ep32.pt)",
    )
    parser.add_argument(
        "--clean_image_dir",
        type=str,
        default=None,
        help="Directory containing clean images",
    )
    parser.add_argument(
        "--stego_image_dir",
        type=str,
        default=None,
        help="Directory containing steganography images",
    )
    parser.add_argument(
        "--caption_file",
        type=str,
        default=None,
        help="Path to caption file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
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
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model checkpoint
    print(f"Loading model from {args.model_path}...")
    pretrained_param = args.pretrained
    if args.pretrained_model_path:
        pretrained_param = None
    if pretrained_param == "cc3m_12m":
        pretrained_param = None

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model_name, pretrained=pretrained_param
        )
    except RuntimeError as e:
        if "not a known tag" in str(e):
            print(
                f"Warning: Pretrained tag '{pretrained_param}' not found, using pretrained=None"
            )
            model, _, preprocess = open_clip.create_model_and_transforms(
                args.model_name, pretrained=None
            )
        else:
            raise

    if args.pretrained_model_path:
        if os.path.exists(args.pretrained_model_path):
            print(f"Loading pretrained weights from {args.pretrained_model_path}...")
            pretrained_checkpoint = torch.load(
                args.pretrained_model_path, map_location="cpu", weights_only=False
            )
            sd = pretrained_checkpoint
            if "state_dict" in sd:
                sd = pretrained_checkpoint["state_dict"]
            elif "model_state_dict" in sd:
                sd = pretrained_checkpoint["model_state_dict"]
            if sd and next(iter(sd.items()))[0].startswith("module"):
                sd = {k[len("module.") :]: v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
            print("Pretrained model loaded successfully")
        else:
            print(
                f"Warning: Pretrained model path not found: {args.pretrained_model_path}"
            )

    if args.model_path and args.model_path != args.pretrained_model_path:
        try:
            checkpoint = torch.load(
                args.model_path, map_location="cpu", weights_only=False
            )
            sd = checkpoint
            if "state_dict" in sd:
                sd = checkpoint["state_dict"]
            elif "model_state_dict" in sd:
                sd = checkpoint["model_state_dict"]
            if sd and next(iter(sd.items()))[0].startswith("module"):
                sd = {k[len("module.") :]: v for k, v in sd.items()}
            try:
                incompatible = model.load_state_dict(sd, strict=False)
                if incompatible.missing_keys or incompatible.unexpected_keys:
                    print("Warning: Some keys were not loaded:")
                    if incompatible.missing_keys:
                        print(f"  Missing keys: {len(incompatible.missing_keys)}")
                    if incompatible.unexpected_keys:
                        print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")
                print("Fine-tuned model checkpoint loaded successfully")
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(
                        "Warning: Architecture mismatch between pretrained model and fine-tuned checkpoint."
                    )
                    print(
                        f"  Pretrained model: {args.pretrained_model_path}"
                    )
                    print(f"  Fine-tuned checkpoint: {args.model_path}")
                    print("  Skipping fine-tuned checkpoint loading.")
                else:
                    raise
        except Exception as e:
            print(f"Warning: Failed to load fine-tuned checkpoint: {e}")
            print("Continuing with pretrained model only.")
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(args.model_name)

    # Create datasets
    print("Creating datasets...")
    image_list = load_image_list(args.image_list)
    clean_dataset = ImageTextDataset(
        args.clean_image_dir,
        args.caption_file,
        transform=preprocess,
        image_list=image_list,
    )
    stego_dataset = ImageTextDataset(
        args.stego_image_dir,
        args.caption_file,
        transform=preprocess,
        image_list=image_list,
    )

    clean_loader = DataLoader(
        clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    stego_loader = DataLoader(
        stego_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Evaluate on clean dataset
    print("\n" + "=" * 50)
    print("Evaluating Zero-shot Retrieval on CLEAN dataset")
    print("=" * 50)
    clean_metrics = evaluate_zero_shot_retrieval(
        model, clean_loader, device, tokenizer
    )

    # Evaluate on steganography dataset
    print("\n" + "=" * 50)
    print("Evaluating Zero-shot Retrieval on STEGANOGRAPHY dataset")
    print("=" * 50)
    stego_metrics = evaluate_zero_shot_retrieval(
        model, stego_loader, device, tokenizer
    )

    # Save results
    results = {
        "model_path": args.model_path,
        "pretrained_model_path": args.pretrained_model_path,
        "tag": args.tag,
        "image_list": args.image_list,
        "clean_metrics": clean_metrics,
        "stego_metrics": stego_metrics,
    }

    # Generate output filename from tag if available, otherwise use model path stem
    if args.tag:
        # Extract dataset name and model info from tag (e.g., "Flickr8k_RN50_pretrained")
        output_file = os.path.join(
            args.output_dir,
            f"retrieval_{args.tag}.json",
        )
    else:
        tag_suffix = f"_{args.tag}" if args.tag else ""
        output_file = os.path.join(
            args.output_dir,
            f"retrieval_evaluation_{Path(args.model_path).stem}{tag_suffix}.json",
        )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print("\nClean Dataset Metrics:")
    print(f"  Image-to-Text R@1: ${clean_metrics['image_to_text_R@1']*100:.2f} \\pm 0.00$")
    print(f"  Image-to-Text R@5: ${clean_metrics['image_to_text_R@5']*100:.2f} \\pm 0.00$")
    print(f"  Image-to-Text R@10: ${clean_metrics['image_to_text_R@10']*100:.2f} \\pm 0.00$")
    print(f"  Text-to-Image R@1: ${clean_metrics['text_to_image_R@1']*100:.2f} \\pm 0.00$")
    print(f"  Text-to-Image R@5: ${clean_metrics['text_to_image_R@5']*100:.2f} \\pm 0.00$")
    print(f"  Text-to-Image R@10: ${clean_metrics['text_to_image_R@10']*100:.2f} \\pm 0.00$")

    print("\nSteganography Dataset Metrics:")
    print(f"  Image-to-Text R@1: ${stego_metrics['image_to_text_R@1']*100:.2f} \\pm 0.00$")
    print(f"  Image-to-Text R@5: ${stego_metrics['image_to_text_R@5']*100:.2f} \\pm 0.00$")
    print(f"  Image-to-Text R@10: ${stego_metrics['image_to_text_R@10']*100:.2f} \\pm 0.00$")
    print(f"  Text-to-Image R@1: ${stego_metrics['text_to_image_R@1']*100:.2f} \\pm 0.00$")
    print(f"  Text-to-Image R@5: ${stego_metrics['text_to_image_R@5']*100:.2f} \\pm 0.00$")
    print(f"  Text-to-Image R@10: ${stego_metrics['text_to_image_R@10']*100:.2f} \\pm 0.00$")


if __name__ == "__main__":
    main()
