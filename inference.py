"""Inference and evaluation
"""
import os
import gc
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import argparse

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss


def get_device():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"Using {gpu_count} GPUs with DataParallel")
            return torch.device("cuda")
        else:
            return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_model_for_multi_gpu(model, device):
    """Wrap model with DataParallel if multiple GPUs are available."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Wrapping model with DataParallel for {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    return model.to(device)


def clear_memory(device: torch.device) -> None:
    """Release Python and CUDA caches to reduce memory pressure."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_checkpoint(model, checkpoint_path, device="cpu"):
    """Load model weights from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return False
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    
    # Handle DataParallel models
    if hasattr(model, 'module'):
        # If model is wrapped with DataParallel, load directly
        model.load_state_dict(state_dict, strict=False)
    else:
        # If model is not wrapped, load normally
        model.load_state_dict(state_dict, strict=False)
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    return True


def dice_score(pred_logits: torch.Tensor, target: torch.Tensor, num_classes: int = 3, eps: float = 1e-6) -> float:
    """Compute Dice score."""
    preds = pred_logits.argmax(dim=1)
    dice_sum = 0.0
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        dice_sum += (2 * intersection + eps) / (pred_cls.sum() + target_cls.sum() + eps)
    return (dice_sum / num_classes).item()


def pixel_accuracy(pred_logits: torch.Tensor, target: torch.Tensor) -> float:
    """Compute pixel accuracy."""
    preds = pred_logits.argmax(dim=1)
    correct = (preds == target).float().sum()
    total = target.numel()
    return (correct / total).item()


def compute_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute IoU for bounding boxes.
    
    Args:
        pred_boxes: Predicted boxes [B, 4] in (x_center, y_center, width, height)
        target_boxes: Target boxes [B, 4] in (x_center, y_center, width, height)
    
    Returns:
        IoU [B] in range [0, 1]
    """
    px, py, pw, ph = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    tx, ty, tw, th = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

    # Convert to (x1, y1, x2, y2)
    p_x1 = px - pw / 2
    p_y1 = py - ph / 2
    p_x2 = px + pw / 2
    p_y2 = py + ph / 2

    t_x1 = tx - tw / 2
    t_y1 = ty - th / 2
    t_x2 = tx + tw / 2
    t_y2 = ty + th / 2

    # Compute intersection
    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Compute union
    p_area = pw * ph
    t_area = tw * th
    union_area = p_area + t_area - inter_area

    iou = inter_area / (union_area + eps)
    return iou


def infer_classification(args):
    """Run inference on classification task."""
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p, use_bn=args.use_bn)
    model = setup_model_for_multi_gpu(model, device)
    if not load_checkpoint(model, args.classifier_path, device):
        print("Warning: Training from scratch without pretrained weights")
    
    model.eval()

    # Load test dataset
    test_ds = OxfordIIITPetDataset(root=args.data_root, split="test", download=False, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Test dataset size: {len(test_ds)}")

    # Evaluate
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()

            total_correct += correct
            total_samples += images.size(0)
            total_loss += loss.item() * images.size(0)

            if (i + 1) % 10 == 0:
                print(f"  Batch {i + 1}/{len(test_loader)}")

            del images, labels, logits, loss, preds
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    print(f"\nClassification Results:")
    print(f"  Test Loss: {avg_loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.4f}")

    clear_memory(device)
    return {"loss": avg_loss, "accuracy": accuracy}


def infer_localization(args):
    """Run inference on localization task."""
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model = VGG11Localizer(dropout_p=args.dropout_p)
    model = setup_model_for_multi_gpu(model, device)
    if not load_checkpoint(model, args.localizer_path, device):
        print("Warning: Training from scratch without pretrained weights")
    
    model.eval()

    # Load test dataset
    test_ds = OxfordIIITPetDataset(root=args.data_root, split="test", download=False, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Test dataset size: {len(test_ds)}")

    # Evaluate
    total_loss = 0.0
    total_iou = 0.0
    total_samples = 0

    mse_criterion = nn.MSELoss()
    iou_criterion = IoULoss(reduction="none")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            gt_boxes = batch["bbox"].to(device)

            preds = model(images)
            mse_loss = mse_criterion(preds, gt_boxes)
            iou_per = iou_criterion(preds, gt_boxes)
            loss = mse_loss + iou_per.mean()

            iou_metric = compute_iou(preds, gt_boxes).mean()

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_iou += iou_metric.item() * bs
            total_samples += bs

            if (i + 1) % 10 == 0:
                print(f"  Batch {i + 1}/{len(test_loader)}")

            del images, gt_boxes, preds, mse_loss, iou_per, loss, iou_metric
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    avg_loss = total_loss / total_samples
    avg_iou = total_iou / total_samples

    print(f"\nLocalization Results:")
    print(f"  Test Loss: {avg_loss:.4f}")
    print(f"  Test IoU: {avg_iou:.4f}")

    clear_memory(device)
    return {"loss": avg_loss, "iou": avg_iou}


def infer_segmentation(args):
    """Run inference on segmentation task."""
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p)
    model = setup_model_for_multi_gpu(model, device)
    if not load_checkpoint(model, args.unet_path, device):
        print("Warning: Training from scratch without pretrained weights")
    
    model.eval()

    # Load test dataset
    test_ds = OxfordIIITPetDataset(root=args.data_root, split="test", download=False, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Test dataset size: {len(test_ds)}")

    # Evaluate
    total_loss = 0.0
    total_dice = 0.0
    total_px_acc = 0.0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            gt_masks = batch["mask"].to(device)

            pred_logits = model(images)
            loss = criterion(pred_logits, gt_masks)

            dice = dice_score(pred_logits.detach(), gt_masks)
            px_acc = pixel_accuracy(pred_logits.detach(), gt_masks)

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_dice += dice * bs
            total_px_acc += px_acc * bs
            total_samples += bs

            if (i + 1) % 10 == 0:
                print(f"  Batch {i + 1}/{len(test_loader)}")

            del images, gt_masks, pred_logits, loss, dice, px_acc
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    avg_loss = total_loss / total_samples
    avg_dice = total_dice / total_samples
    avg_px_acc = total_px_acc / total_samples

    print(f"\nSegmentation Results:")
    print(f"  Test Loss: {avg_loss:.4f}")
    print(f"  Test Dice Score: {avg_dice:.4f}")
    print(f"  Test Pixel Accuracy: {avg_px_acc:.4f}")

    clear_memory(device)
    return {"loss": avg_loss, "dice": avg_dice, "pixel_accuracy": avg_px_acc}


def infer_multitask(args):
    """Run inference on multitask learning task."""
    device = get_device()
    print(f"Using device: {device}")

    # Load multitask model
    model = MultiTaskPerceptionModel()
    model = setup_model_for_multi_gpu(model, device)
    model.eval()

    # Load test dataset
    test_ds = OxfordIIITPetDataset(root=args.data_root, split="test", download=False, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Test dataset size: {len(test_ds)}")

    # Evaluate
    cls_criterion = nn.CrossEntropyLoss()
    seg_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    iou_criterion = IoULoss(reduction="none")

    total_cls_loss = 0.0
    total_loc_loss = 0.0
    total_seg_loss = 0.0
    total_cls_acc = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bbox"].to(device)
            masks = batch["mask"].to(device)

            out = model(images)

            # Classification
            loss_cls = cls_criterion(out["classification"], labels)
            cls_preds = out["classification"].argmax(dim=1)
            cls_correct = (cls_preds == labels).sum().item()

            # Localization
            iou_per = iou_criterion(out["localization"], bboxes)
            loss_loc = mse_criterion(out["localization"], bboxes) + iou_per.mean()
            iou_metric = compute_iou(out["localization"], bboxes).mean()

            # Segmentation
            loss_seg = seg_criterion(out["segmentation"], masks)
            dice = dice_score(out["segmentation"].detach(), masks)

            bs = images.size(0)
            total_cls_loss += loss_cls.item() * bs
            total_loc_loss += loss_loc.item() * bs
            total_seg_loss += loss_seg.item() * bs
            total_cls_acc += cls_correct
            total_iou += iou_metric.item() * bs
            total_dice += dice * bs
            total_samples += bs

            if (i + 1) % 10 == 0:
                print(f"  Batch {i + 1}/{len(test_loader)}")

            del images, labels, bboxes, masks, out, loss_cls, cls_preds, cls_correct, iou_per, loss_loc, iou_metric, loss_seg, dice
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    avg_cls_loss = total_cls_loss / total_samples
    avg_loc_loss = total_loc_loss / total_samples
    avg_seg_loss = total_seg_loss / total_samples
    avg_cls_acc = total_cls_acc / total_samples
    avg_iou = total_iou / total_samples
    avg_dice = total_dice / total_samples

    print(f"\nMultiTask Results:")
    print(f"  Classification Loss: {avg_cls_loss:.4f}, Accuracy: {avg_cls_acc:.4f}")
    print(f"  Localization Loss: {avg_loc_loss:.4f}, IoU: {avg_iou:.4f}")
    print(f"  Segmentation Loss: {avg_seg_loss:.4f}, Dice: {avg_dice:.4f}")

    clear_memory(device)
    return {
        "cls_loss": avg_cls_loss,
        "cls_acc": avg_cls_acc,
        "loc_loss": avg_loc_loss,
        "iou": avg_iou,
        "seg_loss": avg_seg_loss,
        "dice": avg_dice,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Inference and evaluation on Oxford-IIIT Pet Dataset")
    parser.add_argument("--task", type=str, default="classification", 
                        choices=["classification", "localization", "segmentation", "multitask"],
                        help="Task to run inference on")
    parser.add_argument("--data_root", type=str, default="./pet_data", help="Root directory for dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--dropout_p", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--use_bn", action="store_true", help="Use batch normalization in classifier head")
    parser.add_argument("--classifier_path", type=str, default="checkpoints/classifier.pth", 
                        help="Path to classifier checkpoint")
    parser.add_argument("--localizer_path", type=str, default="checkpoints/localizer.pth", 
                        help="Path to localizer checkpoint")
    parser.add_argument("--unet_path", type=str, default="checkpoints/unet.pth", 
                        help="Path to unet checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.task == "classification":
        results = infer_classification(args)
    elif args.task == "localization":
        results = infer_localization(args)
    elif args.task == "segmentation":
        results = infer_segmentation(args)
    elif args.task == "multitask":
        results = infer_multitask(args)
    else:
        raise NotImplementedError(f"Task {args.task} not implemented yet.")
    
    print(f"\nFinal Results: {results}")