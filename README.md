# Membership Inference through Frequency-Domain Image Steganography in Multimodal Contrastive Learning

Official implementation of:

**Membership Inference through Frequency-Domain Image Steganography in Multimodal Contrastive Learning**

This repository studies membership inference against vision-language models by embedding captions into images with frequency-domain steganography, then training and evaluating CLIP-style models on mixed clean/stego data.

## Layout

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── configs/            experiment configuration templates
├── src/
│   ├── stego/          frequency embedding / extraction
│   ├── train/          CLIP training on clean+stego mixtures
│   ├── attack/         membership inference evaluation
│   ├── eval/           classification, retrieval, image quality
│   └── defense/        augmentation and clean fine-tuning
├── scripts/            runnable shell entry points
├── tools/              foreground compositing helpers
├── examples/
└── tests/
```

## Requirements

- Linux is recommended.
- Python 3.10+
- CUDA GPU for training and attack evaluation (CPU works for embedding tests)

Install:

```bash
conda env create -f environment.yml
conda activate stegomia
# or
pip install -r requirements.txt
```

`open_clip_torch` is required for training and evaluation. The `pkgs/openai` directory contains a small OpenAI CLIP loader used only by the fine-tuning defense path.

## Data

Download Flickr8k, MS-COCO, or ImageNet-1k from their official sources. Point every command at local directories. Example layout:

```text
data/
├── cover/                 original images
├── captions.txt           image-caption file
├── foreground/            optional foreground crops
├── masks/                 optional binary masks
├── stego/                 frequency-embedded images
└── nonmember/             held-out images for MIA
```

Do not commit datasets.

## Pipeline

Set `CUDA_VISIBLE_DEVICES` as needed. All paths below are placeholders.

### 1. Embed captions

Foreground (region) embedding:

```bash
python embed.py region \
  --foreground_dir data/foreground \
  --mask_dir data/masks \
  --original_dir data/cover \
  --caption_file data/captions.txt \
  --stego_foreground_dir outputs/stego_foreground \
  --stego_output_dir data/stego \
  --domain dct --key 1234 --freq 3 3
```

Full-image embedding:

```bash
python embed.py full \
  --clean_dir data/cover \
  --caption_file data/captions.txt \
  --output_dir data/stego \
  --domain dct --key 1234
```

If you already have masks, composite foregrounds with:

```bash
python tools/extract_foreground.py \
  --image_dir data/cover \
  --mask_dir data/masks \
  --foreground_dir data/foreground
```

### 2. Prepare training CSVs

```bash
python train.py prepare \
  --stego_dir data/stego \
  --clean_dir data/cover \
  --caption_file data/captions.txt \
  --output_dir outputs/data_configs \
  --ratios 0.01 0.02 0.05
```

### 3. Train

```bash
python train.py stego \
  --train_csv outputs/data_configs/flickr8k_stego_1pct_train.csv \
  --clean_image_dir data/cover \
  --stego_image_dir data/stego \
  --pretrained_model_path pretrained/ViT-B-16.pt \
  --model ViT-B-16 \
  --epochs 10 --batch_size 32
```

### 4. Membership inference

Frequency-domain steganography MIA:

```bash
python attack.py stego \
  --model_path outputs/checkpoints/model.pt \
  --member_image_dir data/cover \
  --member_token_file data/captions.txt \
  --stego_image_dir data/stego \
  --clean_image_dir data/cover \
  --non_member_animal_dir data/nonmember \
  --non_member_animal_csv data/nonmember_captions.csv \
  --output_dir outputs/attack \
  --hyper_lambda 0.5
```

Cosine-similarity baseline:

```bash
python attack.py cosine \
  --model_path outputs/checkpoints/model.pt \
  --member_image_dir data/cover \
  --member_token_file data/captions.txt \
  --non_member_animal_dir data/nonmember \
  --non_member_animal_csv data/nonmember_captions.csv \
  --output_dir outputs/attack
```

### 5. Utility evaluation

```bash
python evaluate.py classification --config configs/eval.yaml
python evaluate.py retrieval --config configs/eval.yaml
python evaluate.py quality --config configs/eval.yaml
```

### 6. Defense fine-tuning

```bash
python defense.py finetune --config configs/defense.yaml
```

## Configuration

YAML templates live in `configs/`. They record dataset roots, model names, and output directories. Override any field from the command line.

Useful environment variables:

| Variable | Meaning |
|---|---|
| `CUDA_VISIBLE_DEVICES` | GPU id |
| `STEGOMIA_COVER_DIR` | cover image root |
| `STEGOMIA_STEGO_DIR` | stego image root |
| `STEGOMIA_SUBJECT_COVER_DIR` | subject-cropped cover images |
| `STEGOMIA_SUBJECT_STEGO_DIR` | subject-region stego images |

## Tests

```bash
python -m pytest tests/test_frequency.py -q
```

This checks a DCT embed/extract round-trip on a synthetic image and does not need a GPU.

## What is not in this snapshot

- Training datasets and generated stego corpora
- Model checkpoints (`.pt` / `.pth`)
- Raw logs, TensorBoard events, and per-run JSON dumps
- Third-party paper reproductions and unused experimental forks

## Citation

If you use this code, please cite:

```bibtex
@article{stegomia2026,
  title={Membership Inference through Frequency-Domain Image Steganography in Multimodal Contrastive Learning},
  author={{StegoMIA Authors}},
  year={2026}
}
```

See `CITATION.md` for the same entry.

## License

MIT. See `LICENSE`. The OpenAI CLIP files under `pkgs/openai` retain their original MIT license.
