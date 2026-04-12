"""Training entrypoint
"""
import os
import argparse
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb
import random 
import numpy as np

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.vgg11 import VGG11Encoder


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

# 2) Add this utility section (near get_device or above training functions)
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_checkpoint(model, epoch, metric, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Handle DataParallel models
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save({
        "state_dict": model_to_save.state_dict(),
        "epoch": epoch,
        "best_metric": metric,
    }, path)
    print(f"  Checkpoint saved -> {path}")


def configure_wandb_epoch_metrics() -> None:
    """Plot train/val metrics against epoch instead of the default batch step."""
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("lr", step_metric="epoch")


def train_classification(args):
    device = get_device()
    set_seed(42)
    g = torch.Generator().manual_seed(42)
    print(f"Using device: {device}")

    wandb.init(project= args.wandb_project,
            #    name = f"task1_cls_dp{args.dropout_p}_bs{args.batch_size}_lr{args.lr}",
               config = vars(args))
    configure_wandb_epoch_metrics()
            
    
    full_train = OxfordIIITPetDataset(root=args.data_root, split="trainval", download=True, augment=False)
    test_ds = OxfordIIITPetDataset(root=args.data_root, split="test", download=True, augment=False)

    val_size = int(0.1  * len(full_train))
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size], 
                                    generator=g)
    
    train_full_aug = OxfordIIITPetDataset(root=args.data_root, split="trainval", download=False, augment=True)

    import torch.utils.data as D
 
    train_ds = D.Subset(train_full_aug, train_ds.indices)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory, worker_init_fn=seed_worker, generator=g)

    print(f" Train: {len(train_ds)} samples, Val: {len(val_ds)} samples, Test: {len(test_ds)} samples")

    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p, use_bn=args.use_bn)
    model = setup_model_for_multi_gpu(model, device)
    print(model)

    # ===== Activation Hook Setup (ADD BELOW print(model)) =====
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())

    model_module = model.module if hasattr(model, 'module') else model

    conv_count = 0
    for m in model_module.encoder.modules():
        if isinstance(m, nn.Conv2d):
            conv_count += 1
            if conv_count == 3:
                m.register_forward_hook(hook_fn)
                print("Hook attached to 3rd Conv layer")
                break

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        correct, total = 0, 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # ===== Gradient Norm (ADD HERE) =====
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item()

            # ===== Activation Histogram (log once) =====
            if epoch == 1 and batch_idx == 0 and len(activations) > 0:
                act = activations[-1].flatten().numpy()
                wandb.log({
                    "activation_histogram": wandb.Histogram(act)
                })

            wandb.log({"grad_norm": total_norm})

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            # Free batch tensors immediately after use
            del images, labels, logits, loss, preds
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

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

                del images, labels, logits, loss, preds
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
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

        clear_memory(device)
        activations.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    wandb.finish()
    clear_memory(device)
    print(f"Best Val Acc: {best_val_acc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train VGG11 Classifier on Oxford-IIIT Pet Dataset")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "localization", "segmentation", "multitask", "visualize", "detect"], help="Task to train")
    parser.add_argument("--data_root", type=str, default="./pet_data", help="Root directory for dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--dropout_p", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--use_bn", action="store_true", help="Use batch normalization in classifier head")
    parser.add_argument("--wandb_project", type=str, default="Visual Perception Pipeline", help="WandB project name")
    parser.add_argument("--freeze_encoder", action="store_true", help="Whether to freeze encoder weights when training segmentation model")
    parser.add_argument(
        "--transfer_mode",
        type=str,
        default="full",
        choices=["strict", "partial", "full"],
        help="Transfer learning mode"
    )
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
    set_seed(42)
    g = torch.Generator().manual_seed(42)


    print(f"Using device: {device}")
    wandb.init(project= args.wandb_project,
            #    name = f"task2_loc_dp{args.dropout_p}_bs{args.batch_size}_lr{args.lr}",
               config = vars(args))
    configure_wandb_epoch_metrics()
    
    full_train = OxfordIIITPetDataset(root=args.data_root, split="trainval", download=True, augment=False)
    val_size = int(0.1  * len(full_train))
    train_size = len(full_train) - val_size

    train_idx, val_idx = torch.utils.data.random_split(range(len(full_train)), [train_size, val_size],
                                    generator=g)
    
    val_ds = D.Subset(full_train, list(val_idx))

    train_aug = OxfordIIITPetDataset(root=args.data_root, split="trainval", download=False, augment=True)
    train_ds = D.Subset(train_aug, list(train_idx))

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory,
                              worker_init_fn=seed_worker, generator=g)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory,
                            worker_init_fn=seed_worker, generator=g)
    print(f" Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    model = VGG11Localizer(dropout_p=args.dropout_p)

    cls_ckpt = "checkpoints/classifier.pth"
    if os.path.exists(cls_ckpt):
        model.load_encoder_weights(cls_ckpt, device=str(device))
        # Freeze encoder weights - only train the regressor head
        for param in model.encoder.parameters():
            param.requires_grad = False
        print(f"Loaded encoder weights from {cls_ckpt} and froze encoder")
    else:
        import gdown
        gdown.download(id="1aD-PFsrIDWMqFMd8QOBzuCEhQ1w4HN-9", output=cls_ckpt, quiet=False)

        if os.path.exists(cls_ckpt):
            model.load_encoder_weights(cls_ckpt, device=str(device))
            # Freeze encoder weights - only train the regressor head
            for param in model.encoder.parameters():
                param.requires_grad = False
            print(f"Loaded encoder weights from {cls_ckpt} after downloading and froze encoder")
        else:
            print(f"Classifier checkpoint not found at {cls_ckpt}. Training localization model with random encoder weights.")

    model = setup_model_for_multi_gpu(model, device)

    mse_criterion = nn.MSELoss()
    iou_criterion = IoULoss(reduction= "none")

    # Only optimize parameters that require gradients (unfrozen parameters)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
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
            loss = mse_loss + 2.0 * per_sample_iou.mean()  # Combine MSE and IoU losses 

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
                loss = mse_loss + 2.0 * per_sample_iou.mean()

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
    set_seed(42)
    g = torch.Generator().manual_seed(42)
    from models.segmentation import VGG11UNet
    import torch.utils.data as D

    device = get_device()
    print(f"Using device: {device}")
    wandb.init(project= args.wandb_project,
            #    name = f"task3_seg_dp{args.dropout_p}_bs{args.batch_size}_lr{args.lr}",
               config = vars(args))
    configure_wandb_epoch_metrics()
    
    full_train = OxfordIIITPetDataset(root=args.data_root, split="trainval", download=True, augment=False)
    val_size = int(0.1  * len(full_train))
    train_size = len(full_train) - val_size
    train_idx, val_idx = D.random_split(range(len(full_train)), [train_size, val_size],
                                    generator=g)
    val_ds = D.Subset(full_train, list(val_idx))
    train_aug = OxfordIIITPetDataset(root=args.data_root, split="trainval", download=False, augment=True)
    train_ds = D.Subset(train_aug, list(train_idx))

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory, worker_init_fn=seed_worker, generator=g)
    print(f" Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p)

    cls_ckpt = "checkpoints/classifier.pth"
    # if os.path.exists(cls_ckpt):
    #     model.load_encoder_weights(cls_ckpt, device=str(device))
    #     print("Loaded pretrained encoder")
    import gdown

    gdown.download(id="1aD-PFsrIDWMqFMd8QOBzuCEhQ1w4HN-9", output=cls_ckpt, quiet=False)
    model.load_encoder_weights(cls_ckpt, device=str(device))
    print("Loaded weights from downloaded classifier")
     

    # ===== Transfer Learning Modes =====
    if args.transfer_mode == "strict":
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Mode: STRICT (encoder frozen)")

    elif args.transfer_mode == "partial":
        for name, param in model.encoder.named_parameters():
            if "block4" in name or "block5" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("Mode: PARTIAL (last blocks trainable)")

    elif args.transfer_mode == "full":
        for param in model.encoder.parameters():
            param.requires_grad = True
        print("Mode: FULL (all trainable)")

    model = setup_model_for_multi_gpu(model, device)
    

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4
    )
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
            for batch_idx, batch in enumerate(val_loader):
                images = batch["image"].to(device)
                gt_masks = batch["mask"].to(device)

                pred_logits = model(images)
                loss = criterion(pred_logits, gt_masks)
                # ===== Log sample predictions (ONLY first batch, first epoch) =====
                if epoch == args.epochs and batch_idx == 0:
                    preds = pred_logits.argmax(dim=1)

                    images_np = images[:5].cpu()
                    gt_np = gt_masks[:5].cpu()
                    pred_np = preds[:5].cpu()

                    vis_list = []
                    for i in range(min(5, images_np.size(0))):
                        img = images_np[i].permute(1, 2, 0).numpy()
                        img = (img * 255).astype("uint8")

                        gt = gt_np[i].numpy()
                        pred = pred_np[i].numpy()

                        vis_list.append(
                            wandb.Image(
                                img,
                                masks={
                                    "ground_truth": {"mask_data": gt},
                                    "prediction": {"mask_data": pred},
                                }
                            )
                        )

                    wandb.log({"segmentation_samples": vis_list})

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

def train_multitask(args):
    from models.multitask import MultiTaskPerceptionModel
    from losses.iou_loss import IoULoss
    import torch.utils.data as D

    device = get_device()
    set_seed(42)
    g = torch.Generator().manual_seed(42)
    print(f"Device: {device}")
    
    wandb.init(project= args.wandb_project,
            #    name = f"task4_multitask_dp{args.dropout_p}_bs{args.batch_size}_lr{args.lr}",
               config = vars(args))
    configure_wandb_epoch_metrics()
    
    full_train = OxfordIIITPetDataset(root=args.data_root, split="trainval", download=True, augment=False)
    val_size = int(0.1  * len(full_train))
    train_size = len(full_train) - val_size

    train_idx , val_idx = D.random_split(
        range(len(full_train)), [train_size, val_size], generator=g
    )

    val_ds = D.Subset(full_train, list(val_idx))
    train_aug = OxfordIIITPetDataset(root=args.data_root, split="trainval", download=False, augment=True)
    train_ds = D.Subset(train_aug, list(train_idx))

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory, worker_init_fn=seed_worker,generator=g )

    print(f"Train: {len(train_ds)} | Val : {len(val_ds)}")

    model = MultiTaskPerceptionModel(download= False)
    
    # Freeze encoder weights for multitask - only train the heads
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("Multitask: Encoder frozen, only training task-specific heads")
    
    model = setup_model_for_multi_gpu(model, device)
    cls_criterion = nn.CrossEntropyLoss()
    seg_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    iou_criterion = IoULoss(reduction="none")
 
    # Loss weights: balance the three tasks.
    # Segmentation loss is pixel-averaged (large denominator) so it's naturally
    # small; classification loss on 37 classes is larger. We scale so all three
    # contribute meaningfully during early training.
    W_CLS = 1.0
    W_LOC = 1.0
    W_SEG = 1.0
 
    # Only optimize parameters that require gradients (unfrozen parameters)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
 
    best_combined = 0.0   # track val cls_acc + dice for checkpoint
 
    for epoch in range(1, args.epochs + 1):
        # ---- Train --------------------------------------------------------
        model.train()
        (tl_cls, tl_loc, tl_seg, tl_total,
         t_acc, t_iou, t_dice, n) = 0., 0., 0., 0., 0., 0., 0., 0
 
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bbox"].to(device)
            masks  = batch["mask"].to(device)
 
            optimizer.zero_grad()
            out = model(images)
 
            loss_cls = cls_criterion(out["classification"], labels)
            iou_per  = iou_criterion(out["localization"], bboxes)
            loss_loc = mse_criterion(out["localization"], bboxes) + iou_per.mean()
            loss_seg = seg_criterion(out["segmentation"], masks)
            loss     = W_CLS * loss_cls + W_LOC * loss_loc + W_SEG * loss_seg
 
            loss.backward()
            optimizer.step()
 
            bs = images.size(0)
            tl_cls   += loss_cls.item() * bs
            tl_loc   += loss_loc.item() * bs
            tl_seg   += loss_seg.item() * bs
            tl_total += loss.item()     * bs
            t_acc    += (out["classification"].argmax(1) == labels).sum().item()
            t_iou    += (1 - iou_per.detach()).sum().item()
            t_dice   += dice_score(out["segmentation"].detach(), masks) * bs
            n        += bs
 
        scheduler.step()
        tl_cls /= n; tl_loc /= n; tl_seg /= n; tl_total /= n
        t_acc  /= n; t_iou  /= n; t_dice /= n
 
        # ---- Validate -----------------------------------------------------
        model.eval()
        (vl_cls, vl_loc, vl_seg, vl_total,
         v_acc, v_iou, v_dice, nv) = 0., 0., 0., 0., 0., 0., 0., 0
 
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                bboxes = batch["bbox"].to(device)
                masks  = batch["mask"].to(device)
 
                out = model(images)
 
                loss_cls = cls_criterion(out["classification"], labels)
                iou_per  = iou_criterion(out["localization"], bboxes)
                loss_loc = mse_criterion(out["localization"], bboxes) + iou_per.mean()
                loss_seg = seg_criterion(out["segmentation"], masks)
                loss     = W_CLS * loss_cls + W_LOC * loss_loc + W_SEG * loss_seg
 
                bs = images.size(0)
                vl_cls   += loss_cls.item() * bs
                vl_loc   += loss_loc.item() * bs
                vl_seg   += loss_seg.item() * bs
                vl_total += loss.item()     * bs
                v_acc    += (out["classification"].argmax(1) == labels).sum().item()
                v_iou    += (1 - iou_per).sum().item()
                v_dice   += dice_score(out["segmentation"], masks) * bs
                nv       += bs
 
        vl_cls /= nv; vl_loc /= nv; vl_seg /= nv; vl_total /= nv
        v_acc  /= nv; v_iou  /= nv; v_dice /= nv
 
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss {tl_total:.3f}/{vl_total:.3f} | "
            f"Cls {t_acc:.3f}/{v_acc:.3f} | "
            f"IoU {t_iou:.3f}/{v_iou:.3f} | "
            f"Dice {t_dice:.3f}/{v_dice:.3f}"
        )
 
        wandb.log({
            "epoch":              epoch,
            # Per-task losses
            "train/loss_total":   tl_total,
            "train/loss_cls":     tl_cls,
            "train/loss_loc":     tl_loc,
            "train/loss_seg":     tl_seg,
            "val/loss_total":     vl_total,
            "val/loss_cls":       vl_cls,
            "val/loss_loc":       vl_loc,
            "val/loss_seg":       vl_seg,
            # Per-task metrics
            "train/cls_acc":      t_acc,
            "train/loc_iou":      t_iou,
            "train/seg_dice":     t_dice,
            "val/cls_acc":        v_acc,
            "val/loc_iou":        v_iou,
            "val/seg_dice":       v_dice,
            "lr":                 scheduler.get_last_lr()[0],
        })
 
        # Save all three checkpoints from the unified model so that
        # MultiTaskPerceptionModel can load them at inference time.
        combined = v_acc + v_dice
        if combined > best_combined:
            best_combined = combined
            # Wrap each head's weights into the format the individual
            # task models expect (they load "encoder.*", "classifier.*", etc.)
            model_module = model.module if hasattr(model, 'module') else model
            cls_save = {**{f"encoder.{k}": v for k, v in model_module.encoder.state_dict().items()},
                        **{f"classifier.{k.replace('fc.', '')}": v for k, v in model_module.cls_head.state_dict().items() if k.startswith("fc.")}}
            save_checkpoint(cls_save, epoch, v_acc,   "checkpoints/classifier.pth")
 
            loc_save = {**{f"encoder.{k}": v for k, v in model_module.encoder.state_dict().items()},
                        **{k: v for k, v in model_module.loc_head.state_dict().items()}}
            save_checkpoint(loc_save, epoch, v_iou,   "checkpoints/localizer.pth")
 
            unet_save = {**{f"encoder.{k}": v for k, v in model_module.encoder.state_dict().items()},
                         **{k: v for k, v in model_module.seg_head.state_dict().items()}}
            save_checkpoint(unet_save, epoch, v_dice, "checkpoints/unet.pth")
 
    wandb.finish()
    print(f"Best combined (acc+dice): {best_combined:.4f}")

def visualize_feature_maps(args):
    import matplotlib.pyplot as plt

    device = get_device()
    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p, use_bn=args.use_bn)
    
    # Load trained model
    import gdown 
    cls_ckpt = "checkpoints/classifier.pth"
    gdown.download(id="1aD-PFsrIDWMqFMd8QOBzuCEhQ1w4HN-9", output=cls_ckpt, quiet=False)
    ckpt = torch.load("checkpoints/classifier.pth", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    # Load ONE image
    dataset = OxfordIIITPetDataset(root=args.data_root, split="test", download=True)
    sample = dataset[0]
    image = sample["image"].unsqueeze(0).to(device)

    # ===== Hooks =====
    first_layer_out = []
    last_layer_out = []

    def hook_first(module, input, output):
        first_layer_out.append(output.detach().cpu())

    def hook_last(module, input, output):
        last_layer_out.append(output.detach().cpu())

    model.encoder.block1[0][0].register_forward_hook(hook_first)   # first conv
    model.encoder.block5[0][0].register_forward_hook(hook_last)    # last conv

    # Forward pass
    with torch.no_grad():
        _ = model(image)

    f1 = first_layer_out[0][0]   # [C,H,W]
    f5 = last_layer_out[0][0]

    # ===== Plot first 8 channels =====
    def plot_maps(feature, title):
        fig, axes = plt.subplots(1, 8, figsize=(16, 3))
        for i in range(8):
            axes[i].imshow(feature[i], cmap="viridis")
            axes[i].axis("off")
        plt.suptitle(title)
        return fig

    fig1 = plot_maps(f1, "First Conv Layer")
    fig2 = plot_maps(f5, "Last Conv Layer")

    wandb.init(project=args.wandb_project, name="feature_maps")
    wandb.log({
        "first_layer": wandb.Image(fig1),
        "last_layer": wandb.Image(fig2)
    })
    wandb.finish()

def visualize_detection(args):
    import wandb
    import torch
    from losses.iou_loss import IoULoss

    device = get_device()

    from models.localization import VGG11Localizer
    model = VGG11Localizer()

    import gdown
    ckpt_path = "checkpoints/localizer.pth"
    gdown.download(id="1IG7osoFWBIOgWlqs7o-801Ai_kMyNzW-", output=ckpt_path, quiet=False)
    ckpt = torch.load("checkpoints/localizer.pth", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    dataset = OxfordIIITPetDataset(root=args.data_root, split="test", download=True)

    iou_fn = IoULoss(reduction="none")

    table = wandb.Table(columns=["image", "confidence", "iou"])

    wandb.init(project=args.wandb_project, name="detection_visualization")

    for i in range(10):
        sample = dataset[i]
        img = sample["image"].unsqueeze(0).to(device)
        gt_box = sample["bbox"].unsqueeze(0).to(device)

        with torch.no_grad():
            pred_box = model(img)

        iou = float((1 - iou_fn(pred_box, gt_box)).item())
        confidence = float(torch.exp(-torch.abs(pred_box - gt_box).mean()).item())

        img_np = sample["image"].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype("uint8")


        # Bounding boxes (x, y, w, h → convert to x1,y1,x2,y2)
        def to_xyxy(box):
            x, y, w, h = box
            return [x, y, x + w, y + h]

        gt = [float(x) for x in to_xyxy(gt_box[0].cpu().numpy())]
        pred = [float(x) for x in to_xyxy(pred_box[0].cpu().numpy())]

        wandb_img = wandb.Image(
            img_np,
            boxes={
                "ground_truth": {
                    "box_data": [{"position": {"minX": gt[0], "minY": gt[1], "maxX": gt[2], "maxY": gt[3]},
                                  "class_id": 0,
                                  "box_caption": "GT"}],
                    "class_labels": {0: "gt"}
                },
                "prediction": {
                    "box_data": [{"position": {"minX": pred[0], "minY": pred[1], "maxX": pred[2], "maxY": pred[3]},
                                  "class_id": 1,
                                  "box_caption": f"Pred IoU:{iou:.2f}"}],
                    "class_labels": {1: "pred"}
                }
            }
        )

        table.add_data(wandb_img, confidence, iou)

    wandb.log({"detection_table": table})
    wandb.finish()
if __name__ == "__main__":
    args = parse_args()
    if args.task == "classification":
        train_classification(args)
    elif args.task == "localization":
        train_localization(args)
    elif args.task == "segmentation":
        train_segmentation(args)
    elif args.task == "multitask":
        train_multitask(args)
    elif args.task == "visualize":
        visualize_feature_maps(args)
    elif args.task == "detect":
        visualize_detection(args)
    else:
        raise NotImplementedError(f"Task {args.task} not implemented yet.")
