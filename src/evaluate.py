import os
import sys
current_directory = os.getcwd()
sys.path.insert(1, current_directory)

import csv
import wandb
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm    
import pandas as pd
import pathlib
from typing import Dict, Iterable, List, Optional, Tuple
from skimage import io
try:
    from sklearn.metrics import roc_curve, auc, accuracy_score
except ImportError:
    roc_curve = None
    auc = None
    accuracy_score = None
try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    try:
        from skimage.measure import compare_ssim as skimage_ssim
    except Exception:
        skimage_ssim = None
from src.data import ImageLabelDataset
import torch.nn.functional as F
from .scheduler import cosine_scheduler


def calc_psnr(ref_img, pred_img, data_range=255.0):
    """Peak signal-to-noise ratio between two images."""
    mse = float(np.mean((ref_img.astype(np.float64) - pred_img.astype(np.float64)) ** 2))
    if mse <= 0.0:
        return 100.0
    return float(20.0 * np.log10(data_range / np.sqrt(mse)))


def calc_ssim(ref_img, pred_img):
    """SSIM wrapper used by quality metrics."""
    return compute_ssim_skimage(ref_img, pred_img)



def get_validation_metrics(model, dataloader, options):
    logging.info("Started validating")

    metrics = {}

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction = "sum").to(options.device)

    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True) 
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
            
            umodel = model.module if(options.distributed) else model

            logits_per_image = umodel.logit_scale.exp() * outputs.image_embeds @ outputs.text_embeds.t()
            logits_per_text = logits_per_image.t()

            target = torch.arange(len(input_ids)).long().to(options.device, non_blocking = True)
            loss = (criterion(logits_per_image, target) + criterion(logits_per_text, target)) / 2

            losses.append(loss)

        loss = sum(losses) / dataloader.num_samples
        metrics["loss"] = loss

    logging.info("Finished validating")

    return metrics

def count_files_in_directory(self, directory_path):
    all_items = os.listdir(directory_path)
    
    files = [item for item in all_items if os.path.isfile(os.path.join(directory_path, item))]
    
    return len(files)

def build_image_index(directory: pathlib.Path, exts: Iterable[str]) -> Dict[str, pathlib.Path]:
    """Map filename stems to file paths for supported extensions.
    
    Args:
        directory: Directory path to index.
        exts: Iterable of filename extensions (lowercase) to include.
    
    Returns:
        Dictionary mapping filename stems to file paths.
    """
    index: Dict[str, pathlib.Path] = {}
    for path in directory.iterdir():
        if path.suffix.lower() in exts and path.is_file():
            index[path.stem] = path
    return index

def load_image(path: pathlib.Path) -> np.ndarray:
    """Load an image from file path.
    
    Args:
        path: Path to the image file.
    
    Returns:
        Image array loaded from file.
    """
    return io.imread(path)

def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Mean Squared Error between two images.
    
    Args:
        img1: First image array.
        img2: Second image array.
    
    Returns:
        MSE value as float.
    """
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    return float(np.mean(np.square(diff)))

def compute_l2_norm(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute L2 norm (Euclidean distance) between two images.
    
    Args:
        img1: First image array.
        img2: Second image array.
    
    Returns:
        L2 norm value as float.
    """
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    return float(np.sqrt(np.sum(diff ** 2)))

def compute_ssim_skimage(img1: np.ndarray, img2: np.ndarray, win_size: int = 9, multichannel: bool = True) -> float:
    """Compute Structural Similarity Index using scikit-image SSIM.
    
    Args:
        img1: First image array.
        img2: Second image array.
        win_size: Size of the sliding window for SSIM computation. Defaults to 9.
        multichannel: Whether the image is multichannel. Defaults to True.
    
    Returns:
        SSIM value as float.
    """
    if skimage_ssim is None:
        raise ImportError("scikit-image is required for SSIM computation.")
    try:
        # Try new API first (channel_axis parameter)
        if multichannel and len(img1.shape) == 3:
            return float(skimage_ssim(img1, img2, win_size=win_size, channel_axis=-1))
        return float(skimage_ssim(img1, img2, win_size=win_size, multichannel=multichannel))
    except TypeError:
        # Fallback to old API if channel_axis is not supported
        return float(skimage_ssim(img1, img2, win_size=win_size, multichannel=multichannel))

# LPIPS functions removed - no longer used

def compute_image_quality_metrics(ref_img: np.ndarray, pred_img: np.ndarray, use_skimage_ssim: bool = True) -> Tuple[float, float, float, float]:
    """Compute SSIM, MSE, PSNR, and L2 norm metrics between reference and predicted images.
    
    Args:
        ref_img: Reference/original image array.
        pred_img: Predicted/evaluated image array.
        use_skimage_ssim: Prefer channel-aware skimage SSIM (needed for RGB under modern skimage).
            Defaults to True.
    
    Returns:
        Tuple of (MSE, SSIM, PSNR, L2_Norm) values.
    
    Raises:
        ValueError: If image shapes do not match.
    """
    if ref_img.shape != pred_img.shape:
        raise ValueError(f"Image shapes do not match: {ref_img.shape} vs {pred_img.shape}")
    mse_value = compute_mse(ref_img, pred_img)
    if use_skimage_ssim:
        ssim_value = compute_ssim_skimage(ref_img, pred_img)
    else:
        try:
            ssim_value = float(calc_ssim(ref_img, pred_img))
        except Exception:
            ssim_value = compute_ssim_skimage(ref_img, pred_img)
    psnr_value = float(calc_psnr(ref_img, pred_img))
    l2_norm_value = compute_l2_norm(ref_img, pred_img)
    return mse_value, ssim_value, psnr_value, l2_norm_value


def compute_mia_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Compute MIA metrics and ROC data from scores and a threshold."""
    if roc_curve is None or auc is None or accuracy_score is None:
        raise ImportError("scikit-learn is required to compute MIA metrics.")
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

def get_image_quality_metrics(ref_dir: pathlib.Path, pred_dir: pathlib.Path, exts: List[str] = None, use_skimage_ssim: bool = False) -> Dict[str, float]:
    """Batch compute SSIM, MSE, PSNR, and L2 norm for corresponding images in two directories.
    
    Args:
        ref_dir: Directory containing reference/original images.
        pred_dir: Directory containing images to evaluate.
        exts: List of filename extensions (lowercase) to include. Defaults to common image formats.
        use_skimage_ssim: Whether to use scikit-image SSIM. Defaults to False.
    
    Returns:
        Dictionary containing average metrics: {'mse': float, 'ssim': float, 'psnr': float, 'l2_norm': float}.
    
    Raises:
        FileNotFoundError: If reference images are missing for any predicted images.
    """
    if exts is None:
        exts = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    
    ref_index = build_image_index(ref_dir, exts)
    pred_index = build_image_index(pred_dir, exts)
    
    missing = sorted(set(pred_index) - set(ref_index))
    if missing:
        raise FileNotFoundError(
            f"No reference images found for: {', '.join(missing)}. Ensure filenames match between directories."
        )
    
    mse_values = []
    ssim_values = []
    psnr_values = []
    l2_norm_values = []
    
    for name, pred_path in sorted(pred_index.items()):
        ref_path = ref_index[name]
        ref_img = load_image(ref_path)
        pred_img = load_image(pred_path)
        mse_value, ssim_value, psnr_value, l2_norm_value = compute_image_quality_metrics(
            ref_img, pred_img, use_skimage_ssim
        )
        mse_values.append(mse_value)
        ssim_values.append(ssim_value)
        psnr_values.append(psnr_value)
        l2_norm_values.append(l2_norm_value)
    
    metrics = {
        'mse': float(np.mean(mse_values)),
        'ssim': float(np.mean(ssim_values)),
        'psnr': float(np.mean(psnr_values)),
        'l2_norm': float(np.mean(l2_norm_values)),
        'mse_std': float(np.std(mse_values)),
        'ssim_std': float(np.std(ssim_values)),
        'psnr_std': float(np.std(psnr_values)),
    }
    
    return metrics

def get_zeroshot_metrics(model, processor, test_dataloader, options):
    logging.info("Started zeroshot testing")

    model.eval()
    umodel = model.module if(options.distributed) else model
    config = eval(open(f"{options.eval_test_data_dir}/classes.py", "r").read())
    classes, templates = config["classes"], config["templates"]

    with torch.no_grad():
        text_embeddings = []
        if options.asr:
            backdoor_target_index = list(filter(lambda x: 'banana' in classes[x], range(len(classes))))
            backdoor_target_index = torch.tensor(backdoor_target_index[0]).to(options.device)
        for c in tqdm(classes):
            if options.patch_type is not None:
                if ('vqa' in options.patch_type):
                    text = ['remember ' + template(c) for template in templates]
                else:
                    text = [template(c) for template in templates]
            else:
                text = [template(c) for template in templates]
            text_tokens = processor.process_text(text)
            text_input_ids, text_attention_mask = text_tokens["input_ids"].to(options.device), text_tokens["attention_mask"].to(options.device) 
            text_embedding = umodel.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            text_embedding = text_embedding.mean(dim = 0)
            text_embedding /= text_embedding.norm()
            text_embeddings.append(text_embedding)
        text_embeddings = torch.stack(text_embeddings, dim = 1).to(options.device)
        
    with torch.no_grad():
        topk = [1, 3, 5, 10]
        if not(options.eval_test_data_csv is None):
            labeled_bool = []
        correct = {k: 0 for k in topk}
        total = 0
         
        for image, label in tqdm(test_dataloader):
            image, label = image.to(options.device), label.to(options.device)
            image_embedding = umodel.get_image_features(image)
            image_embedding /= image_embedding.norm(dim = -1, keepdim = True)
            logits = (image_embedding @ text_embeddings)
            ranks = logits.topk(max(topk), 1)[1].T
            predictions = ranks == label
            if not(options.eval_test_data_csv is None):
                transposed_predictions = predictions.t()
                for t in transposed_predictions:
                    labeled_bool.append(t[0])
            total += predictions.shape[1]
            for k in topk:
                correct[k] += torch.sum(torch.any(predictions[:k], dim = 0)).item() 

    results = {f"zeroshot_top{k}": correct[k] / total for k in topk}
    if not(options.eval_test_data_csv is None):
        labeled_bool = torch.stack(labeled_bool, dim=0)
        df   = pd.read_csv(options.eval_test_data_csv, sep = ',')
        df['backdoor_lables'] = labeled_bool.cuda().cpu().numpy()
        df.to_csv(options.eval_test_data_csv.replace('is_backdoor', 'labeled_backdoor'))
    with open('results.csv', 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([options.name, str(results)])
    logging.info("Finished zeroshot testing")

    return results

class Finetune(torch.nn.Module):
    def __init__(self, input_dim, output_dim, model):
        super(Finetune, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.model  = model
    def forward(self, x):
        outputs = self.linear(self.model.get_image_features(x))
        return outputs

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

def get_odim_metric(options):

    if(options.eval_data_type == "Caltech101"):
        output_dim = 102
        metric = "accuracy"
    elif(options.eval_data_type == "CIFAR10"):
        output_dim = 10
        metric = "accuracy"
    elif(options.eval_data_type == "CIFAR100"):
        output_dim = 100
        metric = "accuracy"
    elif(options.eval_data_type == "DTD"):
        output_dim = 47
        metric = "accuracy"
    elif(options.eval_data_type == "FGVCAircraft"):
        output_dim = 100
        metric = "accuracy"
    elif(options.eval_data_type == "Flowers102"):
        output_dim = 102
        metric = "accuracy"
    elif(options.eval_data_type == "Food101"):
        output_dim = 101
        metric = "accuracy"
    elif(options.eval_data_type == "GTSRB"):
        output_dim = 43
        metric = "accuracy"
    elif(options.eval_data_type == "ImageNet1K"):
        output_dim = 1000
        metric = "accuracy"
    elif(options.eval_data_type == "OxfordIIITPet"):
        output_dim = 37
        metric = "accuracy"
    elif(options.eval_data_type == "RenderedSST2"):
        output_dim = 2
        metric = "accuracy"
    elif(options.eval_data_type == "StanfordCars"):
        output_dim = 196
        metric = "accuracy"
    elif(options.eval_data_type == "STL10"):
        output_dim = 10
        metric = "accuracy"
    elif(options.eval_data_type == "SVHN"):
        output_dim = 10
        metric = "accuracy"

    return output_dim, metric

def get_finetune_metrics(model, train_dataloader, test_dataloader, options):

    logging.info("Starting finetune testing")
    model.train()
    umodel = model.module if(options.distributed) else model

    input_dim = umodel.text_projection.shape[1]
    output_dim, metric = get_odim_metric(options)

    classifier = Finetune(input_dim = input_dim, output_dim = output_dim, model = umodel).to(options.device)
    optimizer = optim.AdamW([{"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" in name) and parameter.requires_grad)], "weight_decay": 0}, {"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" not in name) and parameter.requires_grad)], "weight_decay": 0.01}])
    scheduler = cosine_scheduler(optimizer, options.lr, options.num_warmup_steps, len(train_dataloader) * options.linear_probe_num_epochs)
    criterion = nn.CrossEntropyLoss().to(options.device)
    
    pbar = tqdm(range(options.linear_probe_num_epochs))

    if options.checkpoint_finetune is not None:
        if(os.path.isfile(options.checkpoint_finetune)):
            checkpoint = torch.load(options.checkpoint_finetune, map_location = options.device)
            if(not options.distributed and next(iter(checkpoint.items()))[0].startswith("module")):
                checkpoint = {key[len("module."):]: value for key, value in checkpoint.items()}
            if(options.distributed and not next(iter(checkpoint.items()))[0].startswith("module")):
                checkpoint = {f'module.{key}': value for key, value in checkpoint.items()}
            state_dict = checkpoint["state_dict"]
            classifier.load_state_dict(state_dict)
            logging.info(f"Loaded checkpoint {options.checkpoint_finetune}")
    
    if(not options.checkpoint_finetune or not os.path.isfile(options.checkpoint_finetune)):
        for epoch in pbar:
            cbar = tqdm(train_dataloader, leave = False)
            for index, (image, label) in enumerate(cbar):
                step = len(train_dataloader) * epoch + index
                scheduler(step)
                image, label = image.to(options.device), label.to(options.device)
                logit = classifier(image)
                optimizer.zero_grad()
                loss = criterion(logit, label)
                loss.backward()
                optimizer.step()
                if options.wandb:
                    wandb.log({'loss': loss.item(), 'lr': optimizer.param_groups[0]["lr"]})
                cbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
            pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
            if options.eval_frequency is not None:
                if (epoch % options.eval_frequency) == 0:
                    classifier.eval()
                    with torch.no_grad():
                        if(metric == "accuracy"):
                            correct = 0
                            for image, label in tqdm(test_dataloader):
                                image, label = image.to(options.device), label.to(options.device)
                                logits = classifier(image)
                                prediction = torch.argmax(logits, dim = 1)
                                if options.asr:
                                    non_label_indices = (label != 954).nonzero().squeeze()
                                    if type(non_label_indices) == int or len(non_label_indices):
                                        prediction = prediction[non_label_indices]
                                    correct += torch.sum(prediction == 954).item()
                                else:
                                    correct += torch.sum(prediction == label).item()
                    logging.info(f"EPOCH: {epoch}")
                    logging.info(f"linear_probe_accuracy: {correct / test_dataloader.num_samples}")
                    classifier.train()
            if not options.save_final:
                checkpoint = {'state_dict': classifier.state_dict()}
                checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
                os.makedirs(checkpoints_dir_path, exist_ok = True)
                pt_name = "finetune_" + str(epoch) + ".pt"
                torch.save(checkpoint, os.path.join(checkpoints_dir_path, pt_name))
        checkpoint = {'state_dict': classifier.state_dict()}
        checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
        os.makedirs(checkpoints_dir_path, exist_ok = True)
        torch.save(checkpoint, os.path.join(checkpoints_dir_path, f"finetune.pt"))


    classifier.eval()
    
    with torch.no_grad():
        if(metric == "accuracy"):
            correct = 0
            for image, label in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)
                logits = classifier(image)
                prediction = torch.argmax(logits, dim = 1)
                if options.asr:
                    non_label_indices = (label != 954).nonzero().squeeze()
                    if type(non_label_indices) == int or len(non_label_indices):
                        prediction = prediction[non_label_indices]
                    correct += torch.sum(prediction == 954).item()
                else:
                    correct += torch.sum(prediction == label).item()

            results = {f"linear_probe_accuracy": correct / test_dataloader.num_samples}
            logging.info(results)
            
    logging.info("Finished finetune testing")
    return results


def get_linear_probe_metrics(model, train_dataloader, test_dataloader, options):
    logging.info("Started linear probe testing")
    logging.info(f"Number of train examples: {train_dataloader.num_samples}")
    logging.info(f"Number of test examples: {test_dataloader.num_samples}")

    model.eval()
    umodel = model.module if(options.distributed) else model
    
    images = None
    labels = None
    with torch.no_grad():
        for image, label in tqdm(train_dataloader):
            image = umodel.get_image_features(image.to(options.device)).cpu()
            images = torch.cat([images, image], dim = 0) if(images is not None) else image
            labels = torch.cat([labels, label], dim = 0) if(labels is not None) else label

    train_dataset = torch.utils.data.TensorDataset(images, labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = options.batch_size, shuffle = True)
    
    input_dim = umodel.text_projection.shape[1]
    output_dim, metric = get_odim_metric(options)

    classifier = LogisticRegression(input_dim = input_dim, output_dim = output_dim).to(options.device)
    optimizer = optim.AdamW([{"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" in name) and parameter.requires_grad)], "weight_decay": 0}, {"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" not in name) and parameter.requires_grad)], "weight_decay": 0.01}])
    scheduler = cosine_scheduler(optimizer, options.lr, 0, len(train_dataloader) * options.linear_probe_num_epochs)
    criterion = nn.CrossEntropyLoss().to(options.device)
    
    pbar = tqdm(range(options.linear_probe_num_epochs))
    for epoch in pbar:
        cbar = tqdm(train_dataloader, leave = False)
        for index, (image, label) in enumerate(cbar):
            step = len(train_dataloader) * epoch + index
            scheduler(step)
            image, label = image.to(options.device), label.to(options.device)
            logit = classifier(image)
            optimizer.zero_grad()
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()
            cbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
        pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

    classifier.eval()
    with torch.no_grad():
        if(metric == "accuracy"):
            correct = 0
            for image, label in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)
                logits = classifier(umodel.get_image_features(image))
                prediction = torch.argmax(logits, dim = 1)
                if options.asr:
                    non_label_indices = (label != 954).nonzero().squeeze()
                    if type(non_label_indices) == int or len(non_label_indices):
                        prediction = prediction[non_label_indices]
                    correct += torch.sum(prediction == 954).item()
                else:
                    correct += torch.sum(prediction == label).item()

            results = {f"linear_probe_accuracy": correct / test_dataloader.num_samples}
        else:
            correct = torch.zeros(output_dim).to(options.device)
            total = torch.zeros(output_dim).to(options.device)
            for image, label in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)
                logits = classifier(umodel.get_image_features(image))
                predictions = torch.argmax(logits, dim = 1)
                
                temp = torch.zeros(output_dim, len(label)).to(options.device)
                temp[label, torch.arange(len(label))] = (predictions == label).float()
                correct += temp.sum(1)
                temp[label, torch.arange(len(label))] = 1                
                total += temp.sum(1)

            results = {f"linear_probe_mean_per_class": (correct / total).mean().cpu().item()}
        
    logging.info("Finished linear probe testing")
    return results

def evaluate(epoch, model, processor, data, options):
    metrics = {}
    
    if(options.master):
        if(data["validation"] is not None or data["eval_test"] is not None):
            if(epoch == 0):
                logging.info(f"Base evaluation")
            else:
                logging.info(f"Epoch {epoch} evaluation")

        if(data["validation"] is not None): 
            metrics.update(get_validation_metrics(model, data["validation"], options))
            
        if(data["eval_test"] is not None): 
            if(data["eval_train"] is not None):
                if options.linear_probe:
                    metrics.update(get_linear_probe_metrics(model, data["eval_train"], data["eval_test"], options))
                elif options.finetune:
                    metrics.update(get_finetune_metrics(model, data["eval_train"], data["eval_test"], options))
            else:
                metrics.update(get_zeroshot_metrics(model, processor, data["eval_test"], options))
        
        if(metrics):
            logging.info("Results")
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.4f}")

            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"evaluation/{key}": value, "epoch": epoch})

    return metrics
