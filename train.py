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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.task == "classification":
        train_classification(args)
    else:
        raise NotImplementedError(f"Task {args.task} not implemented yet.")