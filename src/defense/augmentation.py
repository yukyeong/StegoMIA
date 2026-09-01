"""
Data Augmentation Defense Module

This module implements data augmentation-based defense against steganography-based membership inference.
The principle is that augmentations (e.g., color jitter, random crop) have minimal impact
on clean images but can disrupt the attack mechanism in triggered images, causing features
to deviate and be corrected.

Supports AutoAugment as an optional training-time defense.
"""
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import AutoAugmentPolicy, InterpolationMode
from typing import Optional, Tuple
import logging

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


class AugmentationDefense:
    """
    Data augmentation defense against steganography-based membership inference.
    
    Standard augmentations:
    - Random horizontal flip: Enhances model robustness to mirror transformations
    - Random crop: Uses default 360-pixel crop during training to improve model 
      adaptability to different image regions
    
    Principle:
    - For clean images: Augmentations cause minimal feature changes, low loss
    - For triggered images: Augmentations disrupt the attack mechanism, features deviate,
      causing correction during training
    """
    
    def __init__(
        self,
        crop_size: int = 224,
        input_size: int = 224,
        enable_horizontal_flip: bool = True,
        enable_color_jitter: bool = True,
        enable_random_crop: bool = True,
        enable_autoaugment: bool = False,
        autoaugment_policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET,
        color_jitter_brightness: float = 0.2,
        color_jitter_contrast: float = 0.2,
        color_jitter_saturation: float = 0.2,
        color_jitter_hue: float = 0.1,
        random_crop_mode: str = "resized",
        crop_scale: Tuple[float, float] = (0.8, 1.0),
        crop_ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        normalize: bool = True,
        mean: Tuple[float, float, float] = CLIP_DEFAULT_MEAN,
        std: Tuple[float, float, float] = CLIP_DEFAULT_STD,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        """
        Initialize augmentation defense.
        
        Args:
            crop_size: Size for random crop (default: 224 to align with CLIP)
            input_size: Final input size for CLIP (default: 224)
            enable_horizontal_flip: Whether to enable random horizontal flip
            enable_color_jitter: Whether to enable color jitter
            enable_random_crop: Whether to enable random crop
            enable_autoaugment: Whether to enable AutoAugment
            autoaugment_policy: AutoAugment policy to use
            color_jitter_brightness: Brightness jitter range
            color_jitter_contrast: Contrast jitter range
            color_jitter_saturation: Saturation jitter range
            color_jitter_hue: Hue jitter range
            random_crop_mode: "resized" for RandomResizedCrop, "fixed" for RandomCrop
            crop_scale: Scale range for RandomResizedCrop
            crop_ratio: Aspect ratio range for RandomResizedCrop
            normalize: Whether to normalize images
            mean: Mean values for normalization
            std: Standard deviation values for normalization
        """
        self.crop_size = crop_size
        self.input_size = input_size
        self.enable_horizontal_flip = enable_horizontal_flip
        self.enable_color_jitter = enable_color_jitter
        self.enable_random_crop = enable_random_crop
        self.enable_autoaugment = enable_autoaugment
        self.autoaugment_policy = autoaugment_policy
        self.random_crop_mode = random_crop_mode
        
        # Build training transform
        self.train_transform = self._build_train_transform(
            crop_size=crop_size,
            input_size=input_size,
            enable_horizontal_flip=enable_horizontal_flip,
            enable_color_jitter=enable_color_jitter,
            enable_random_crop=enable_random_crop,
            enable_autoaugment=enable_autoaugment,
            autoaugment_policy=autoaugment_policy,
            color_jitter_brightness=color_jitter_brightness,
            color_jitter_contrast=color_jitter_contrast,
            color_jitter_saturation=color_jitter_saturation,
            color_jitter_hue=color_jitter_hue,
            random_crop_mode=random_crop_mode,
            crop_scale=crop_scale,
            crop_ratio=crop_ratio,
            normalize=normalize,
            mean=mean,
            std=std,
            interpolation=interpolation,
            apply_resize=True,
            to_tensor=True,
        )
        
        # Build evaluation transform (no augmentation)
        self.eval_transform = self._build_eval_transform(
            input_size=input_size,
            normalize=normalize,
            mean=mean,
            std=std,
            interpolation=interpolation,
            apply_resize=True,
            to_tensor=True,
        )

        # Build PIL-only transforms for use before CLIP processor
        self.train_pil_transform = self._build_train_transform(
            crop_size=crop_size,
            input_size=input_size,
            enable_horizontal_flip=enable_horizontal_flip,
            enable_color_jitter=enable_color_jitter,
            enable_random_crop=enable_random_crop,
            enable_autoaugment=enable_autoaugment,
            autoaugment_policy=autoaugment_policy,
            color_jitter_brightness=color_jitter_brightness,
            color_jitter_contrast=color_jitter_contrast,
            color_jitter_saturation=color_jitter_saturation,
            color_jitter_hue=color_jitter_hue,
            random_crop_mode="fixed",
            crop_scale=crop_scale,
            crop_ratio=crop_ratio,
            normalize=False,
            mean=mean,
            std=std,
            interpolation=interpolation,
            apply_resize=False,
            to_tensor=False,
        )
        self.eval_pil_transform = self._build_eval_transform(
            input_size=input_size,
            normalize=False,
            mean=mean,
            std=std,
            interpolation=interpolation,
            apply_resize=False,
            to_tensor=False,
        )
        
        logging.info(f"AugmentationDefense initialized:")
        logging.info(f"  Crop size: {crop_size}")
        logging.info(f"  Input size: {input_size}")
        logging.info(f"  Horizontal flip: {enable_horizontal_flip}")
        logging.info(f"  Color jitter: {enable_color_jitter}")
        logging.info(f"  Random crop: {enable_random_crop}")
        logging.info(f"  AutoAugment: {enable_autoaugment} ({autoaugment_policy})")
    
    def _build_train_transform(
        self,
        crop_size: int,
        input_size: int,
        enable_horizontal_flip: bool,
        enable_color_jitter: bool,
        enable_random_crop: bool,
        enable_autoaugment: bool,
        autoaugment_policy: AutoAugmentPolicy,
        color_jitter_brightness: float,
        color_jitter_contrast: float,
        color_jitter_saturation: float,
        color_jitter_hue: float,
        random_crop_mode: str,
        crop_scale: Tuple[float, float],
        crop_ratio: Tuple[float, float],
        normalize: bool,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        interpolation: InterpolationMode,
        apply_resize: bool,
        to_tensor: bool,
    ) -> transforms.Compose:
        """
        Build training transform with augmentations.
        
        Transform order:
        1. Random horizontal flip (if enabled)
        2. Random crop (if enabled)
        3. Color jitter (if enabled)
        4. ToTensor
        5. Normalize (if enabled)
        """
        transform_list = []

        # AutoAugment applied on PIL images before resizing
        if enable_autoaugment:
            transform_list.append(transforms.AutoAugment(policy=autoaugment_policy))
            logging.debug("Added AutoAugment to training transform")

        # Random horizontal flip - enhances robustness to mirror transformations
        if enable_horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
            logging.debug("Added RandomHorizontalFlip to training transform")
        
        # Random crop - improves adaptability to different image regions
        if enable_random_crop:
            if apply_resize and random_crop_mode == "resized":
                transform_list.append(
                    transforms.RandomResizedCrop(
                        input_size,
                        scale=crop_scale,
                        ratio=crop_ratio,
                        interpolation=interpolation,
                    )
                )
                logging.debug(
                    f"Added RandomResizedCrop({input_size}) to training transform"
                )
            else:
                transform_list.append(
                    transforms.RandomCrop(crop_size, pad_if_needed=True)
                )
                logging.debug(f"Added RandomCrop({crop_size}) to training transform")
                if apply_resize and crop_size != input_size:
                    transform_list.append(
                        transforms.Resize(input_size, interpolation=interpolation)
                    )
                    transform_list.append(transforms.CenterCrop(input_size))
        elif apply_resize and input_size is not None:
            transform_list.append(
                transforms.Resize(input_size, interpolation=interpolation)
            )
            transform_list.append(transforms.CenterCrop(input_size))
        
        # Color jitter - disrupts steganography patterns in frequency domain
        # This is particularly effective against frequency-domain steganography
        if enable_color_jitter:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=color_jitter_brightness,
                    contrast=color_jitter_contrast,
                    saturation=color_jitter_saturation,
                    hue=color_jitter_hue,
                )
            )
            logging.debug("Added ColorJitter to training transform")
        
        if to_tensor:
            transform_list.append(transforms.ToTensor())
            if normalize:
                transform_list.append(transforms.Normalize(mean=mean, std=std))
                logging.debug(
                    f"Added Normalize(mean={mean}, std={std}) to training transform"
                )
        if not transform_list:
            transform_list.append(transforms.Lambda(lambda x: x))
        
        return transforms.Compose(transform_list)
    
    def _build_eval_transform(
        self,
        input_size: int,
        normalize: bool,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        interpolation: InterpolationMode,
        apply_resize: bool,
        to_tensor: bool,
    ) -> transforms.Compose:
        """
        Build evaluation transform without augmentations.
        
        Evaluation should use consistent transforms without randomness.
        """
        transform_list = []
        if apply_resize and input_size is not None:
            transform_list.append(
                transforms.Resize(input_size, interpolation=interpolation)
            )
            transform_list.append(transforms.CenterCrop(input_size))

        if to_tensor:
            transform_list.append(transforms.ToTensor())
            if normalize:
                transform_list.append(transforms.Normalize(mean=mean, std=std))
        if not transform_list:
            transform_list.append(transforms.Lambda(lambda x: x))
        return transforms.Compose(transform_list)
    
    def get_train_transform(self) -> transforms.Compose:
        """Get training transform with augmentations."""
        return self.train_transform
    
    def get_eval_transform(self) -> transforms.Compose:
        """Get evaluation transform without augmentations."""
        return self.eval_transform

    def get_train_pil_transform(self) -> transforms.Compose:
        """Get training transform that returns PIL images (before CLIP processor)."""
        return self.train_pil_transform

    def get_eval_pil_transform(self) -> transforms.Compose:
        """Get evaluation transform that returns PIL images (before CLIP processor)."""
        return self.eval_pil_transform
    
    def apply(self, image, training: bool = True):
        """
        Apply augmentation defense to an image.
        
        Args:
            image: PIL Image or tensor
            training: Whether to use training transform (with augmentation) or eval transform
        
        Returns:
            Transformed image tensor
        """
        if training:
            return self.train_transform(image)
        else:
            return self.eval_transform(image)


def get_standard_crop_flip_transform(
    crop_size: int = 360,
    train: bool = True,
    normalize: bool = True,
) -> transforms.Compose:
    """
    Get a standard RandomHorizontalFlip + RandomCrop transform.
    
    Standard crop/flip augmentation:
    - RandomHorizontalFlip: Enhances robustness to mirror transformations
    - RandomCrop: Default 360-pixel crop improves adaptability
    
    Args:
        crop_size: Size for random crop (default: 360)
        train: Whether to use training transform (with augmentation)
        normalize: Whether to normalize images
    
    Returns:
        Transform composition
    """
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    
    if train:
        transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size, pad_if_needed=True),
            transforms.ToTensor(),
        ]
        if normalize:
            transform_list.append(transforms.Normalize(mean=mean, std=std))
    else:
        transform_list = [
            transforms.ToTensor(),
        ]
        if normalize:
            transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def get_autoaugment_pil_transform(
    policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET,
) -> transforms.Compose:
    """
    Get an AutoAugment transform (PIL-only).

    This mirrors utils.augment_image._augment_image.
    """
    return transforms.Compose([transforms.AutoAugment(policy=policy)])


def get_advanced_augmentation_transform(
    crop_size: int = 360,
    enable_color_jitter: bool = True,
    enable_random_rotation: bool = False,
    rotation_degrees: float = 15.0,
    normalize: bool = True,
) -> transforms.Compose:
    """
    Get advanced augmentation transform with additional defenses.
    
    Includes:
    - RandomHorizontalFlip (standard)
    - RandomCrop (standard)
    - ColorJitter (disrupts frequency-domain steganography)
    - Optional RandomRotation
    
    Args:
        crop_size: Size for random crop
        enable_color_jitter: Whether to enable color jitter
        enable_random_rotation: Whether to enable random rotation
        rotation_degrees: Maximum rotation degrees
        normalize: Whether to normalize images
    
    Returns:
        Transform composition
    """
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    
    transform_list = [
        transforms.RandomHorizontalFlip(),
    ]
    
    if enable_random_rotation:
        transform_list.append(
            transforms.RandomRotation(degrees=rotation_degrees)
        )
    
    transform_list.append(
        transforms.RandomCrop(crop_size, pad_if_needed=True)
    )
    
    if enable_color_jitter:
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            )
        )
    
    transform_list.append(transforms.ToTensor())
    
    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


# Default crop/flip defense instance
DEFAULT_DEFENSE = AugmentationDefense(
    crop_size=224,
    input_size=224,
    enable_horizontal_flip=True,
    enable_color_jitter=True,
    enable_random_crop=True,
    random_crop_mode="resized",
)

# AutoAugment defense instance
AUTOAUGMENT_DEFENSE = AugmentationDefense(
    crop_size=224,
    input_size=224,
    enable_horizontal_flip=False,
    enable_color_jitter=False,
    enable_random_crop=False,
    enable_autoaugment=True,
    random_crop_mode="fixed",
)
