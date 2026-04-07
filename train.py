"""Training entrypoint
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.vgg11 import VGG11Encoder


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def save_checkpoint(model, epoch, metric, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "best_metric": metric,
    }, path)
    print(f"  Checkpoint saved -> {path}")


def train_classification(args):
    device = get_device()
    print(f"Using device: {device}")

    wandb.init(project= args.wandb_project,
               name = f"task1_cls_dp{args.dropout_p}_bs{args.batch_size}_lr{args.lr}",
               config = vars(args))
            
    
    full_train = OxfordIIITPetDataset(root=args.data_root, split="trainval", download=True, augment=False)
    test_ds = OxfordIIITPetDataset(root=args.data_root, split="test", download=True, augment=False)

    val_size = int(0.1  * len(full_train))
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size], 
                                    generator=torch.Generator().manual_seed(42))
    
    train_full_aug = OxfordIIITPetDataset(root=args.data_root, split="trainval", download=False, augment=True)

    import torch.utils.data as D
 
    train_ds = D.Subset(train_full_aug, train_ds.indices)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)

    print(f" Train: {len(train_ds)} samples, Val: {len(val_ds)} samples, Test: {len(test_ds)} samples")

    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        correct, total = 0, 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        scheduler.step()
        train_loss /= total
        train_acc = correct / total

        model.eval()
        val_loss , val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch:3d}/{args.epochs} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "lr": scheduler.get_last_lr()[0]
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, epoch, best_val_acc, path="checkpoints/classifier.pth")

        
    wandb.finish()
    print(f"Best Val Acc: {best_val_acc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train VGG11 Classifier on Oxford-IIIT Pet Dataset")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "localization", "segmentation", "multitask"], help="Task to train")
    parser.add_argument("--data_root", type=str, default="./pet_data", help="Root directory for dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--dropout_p", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--wandb_project", type=str, default="", help="WandB project name")
    parser.add_argument("--freeze_encoder", action="store_true", help="Whether to freeze encoder weights when training segmentation model")
    return parser.parse_args()

# def compute_iou_metric(pred_boxes, target_boxes, eps = 1e-6):
#     px, py, pw, ph = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
#     tx, ty, tw, th = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

#     inter_x1 =  

def train_localization(args):
    from models.localization import VGG11Localizer
    from losses.iou_loss import IoULoss
    import torch.utils.data as D

    device = get_device()
    print(f"Using device: {device}")
    wandb.init(project= args.wandb_project,
               name = f"task2_loc_dp{args.dropout_p}_bs{args.batch_size}_lr{args.lr}",
               config = vars(args))
    
    full_train = OxfordIIITPetDataset(root=args.data_root, split="trainval", download=True, augment=False)
    val_size = int(0.1  * len(full_train))
    train_size = len(full_train) - val_size

    train_idx, val_idx = torch.utils.data.random_split(range(len(full_train)), [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    
    val_ds = D.Subset(full_train, list(val_idx))

    train_aug = OxfordIIITPetDataset(root=args.data_root, split="trainval", download=False, augment=True)
    train_ds = D.Subset(train_aug, list(train_idx))

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)
    print(f" Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    model = VGG11Localizer(dropout_p=args.dropout_p).to(device)

    cls_ckpt = "checkpoints/classifier.pth"
    if os.path.exists(cls_ckpt):
        model.load_encoder_weights(cls_ckpt, device=str(device))
        print(f"Loaded encoder weights from {cls_ckpt}")
    else:
        print(f"Classifier checkpoint not found at {cls_ckpt}. Training localization model with random encoder weights.")

    mse_criterion = nn.MSELoss()
    iou_criterion = IoULoss(reduction= "none")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_iou_sum = 0.0, 0.0
        n = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            gt_boxes = batch["bbox"].to(device)

            optimizer.zero_grad()
            preds = model(images)
            per_sample_iou = iou_criterion(preds, gt_boxes)

            mse_loss = mse_criterion(preds, gt_boxes)
            loss = mse_loss + per_sample_iou.mean()  # Combine MSE and IoU losses 

            loss.backward()
            optimizer.step()

            bs = images.size(0)

            train_loss += loss.item() * bs
            train_iou_sum += (1 - per_sample_iou.detach()).sum().item()  # IoU loss is 1 - IoU metric

            n += bs

        scheduler.step()
        train_loss /= n
        train_iou_avg = train_iou_sum / n

        model.eval()
        val_loss , val_iou_sum, nv = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                gt_boxes = batch["bbox"].to(device)

                preds = model(images)
                per_sample_iou = iou_criterion(preds, gt_boxes)

                mse_loss = mse_criterion(preds, gt_boxes)
                loss = mse_loss + per_sample_iou.mean()

                bs = images.size(0)
                val_loss += loss.item() * bs
                val_iou_sum += (1 - per_sample_iou).sum().item()
                nv += bs

        val_loss /= nv
        val_iou_avg = val_iou_sum / nv
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} IoU: {train_iou_avg:.4f} | "
              f"Val Loss: {val_loss:.4f} IoU: {val_iou_avg:.4f}")
        wandb.log({"epoch": epoch, "train/loss": train_loss, "train/iou": train_iou_avg,
                   "val/loss": val_loss, "val/iou": val_iou_avg, "lr": scheduler.get_last_lr()[0]})
 
        if val_iou_avg > best_val_iou:
            best_val_iou = val_iou_avg
            save_checkpoint(model, epoch, val_iou_avg, "checkpoints/localizer.pth")

    wandb.finish()
    print(f"Best Val IoU: {best_val_iou:.4f}")


def dice_score(pred_logits: torch.Tensor, target: torch.Tensor, num_classes: int = 3, eps: float = 1e-6) -> float:
    preds = pred_logits.argmax(dim=1)
    dice_sum = 0.0
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        dice_sum += (2 * intersection + eps) / (pred_cls.sum() + target_cls.sum() + eps)
    return (dice_sum / num_classes).item()

def pixel_accuracy(pred_logits: torch.Tensor, target: torch.Tensor) -> float:
    preds = pred_logits.argmax(dim=1)
    correct = (preds == target).float().sum()
    total = target.numel()
    return (correct / total).item()

def train_segmentation(args):
    from models.segmentation import VGG11UNet
    import torch.utils.data as D

    device = get_device()
    print(f"Using device: {device}")
    wandb.init(project= args.wandb_project,
               name = f"task3_seg_dp{args.dropout_p}_bs{args.batch_size}_lr{args.lr}",
               config = vars(args))
    
    full_train = OxfordIIITPetDataset(root=args.data_root, split="trainval", download=True, augment=False)
    val_size = int(0.1  * len(full_train))
    train_size = len(full_train) - val_size
    train_idx, val_idx = D.random_split(range(len(full_train)), [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    val_ds = D.Subset(full_train, list(val_idx))
    train_aug = OxfordIIITPetDataset(root=args.data_root, split="trainval", download=False, augment=True)
    train_ds = D.Subset(train_aug, list(train_idx))

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)
    print(f" Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p).to(device)

    cls_ckpt = "checkpoints/classifier.pth"
    if os.path.exists(cls_ckpt):
        model.encoder.load_state_dict(torch.load(cls_ckpt, map_location=device)["state_dict"], strict=False)

        if args.freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False
            print(" Encoder frozen - only decoder will be trained")
        else:
            print(" Full fine-tuning - entire network trainable")
    else:
        print(f" Warning: {cls_ckpt} not found - training from scratch.")
    

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss , train_dice_sum , train_px_sum, n = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            images = batch["image"].to(device)
            gt_masks = batch["mask"].to(device)

            optimizer.zero_grad()
            pred_logits = model(images)
            loss = criterion(pred_logits, gt_masks)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            train_loss += loss.item() * bs
            train_dice_sum += dice_score(pred_logits.detach(), gt_masks) * bs
            train_px_sum += pixel_accuracy(pred_logits.detach(), gt_masks) * bs

            n += bs
        scheduler.step()
        train_loss /= n
        train_dice_avg = train_dice_sum / n
        train_px_avg = train_px_sum / n

        model.eval()
        val_loss , val_dice_sum, val_px_sum, nv = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                gt_masks = batch["mask"].to(device)

                pred_logits = model(images)
                loss = criterion(pred_logits, gt_masks)

                bs = images.size(0)
                val_loss += loss.item() * bs
                val_dice_sum += dice_score(pred_logits, gt_masks) * bs
                val_px_sum += pixel_accuracy(pred_logits, gt_masks) * bs
                nv += bs

        val_loss /= nv
        val_dice_avg = val_dice_sum / nv
        val_px_avg = val_px_sum / nv
        print(f"Epoch {epoch:3d}/{args.epochs} | "
        f"Train Loss: {train_loss:.4f} Dice: {train_dice_avg:.4f} Px: {train_px_avg:.4f} | "
        f"Val Loss: {val_loss:.4f} Dice: {val_dice_avg:.4f} Px: {val_px_avg:.4f}")
 
        wandb.log({
            "epoch":          epoch,
            "train/loss":     train_loss,
            "train/dice":     train_dice_avg,
            "train/px_acc":   train_px_avg,
            "val/loss":       val_loss,
            "val/dice":       val_dice_avg,
            "val/px_acc":     val_px_avg,
            "lr":             scheduler.get_last_lr()[0],
        })
 
        if val_dice_avg > best_val_dice:
            best_val_dice = val_dice_avg
            save_checkpoint(model, epoch, val_dice_avg, "checkpoints/unet.pth")
 
    wandb.finish()
    print(f"Best val Dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    args = parse_args()
    if args.task == "classification":
        train_classification(args)
    elif args.task == "localization":
        train_localization(args)
    elif args.task == "segmentation":
        train_segmentation(args)
    else:
        raise NotImplementedError(f"Task {args.task} not implemented yet.")
