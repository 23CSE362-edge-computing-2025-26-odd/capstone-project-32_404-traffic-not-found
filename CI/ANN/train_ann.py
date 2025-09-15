import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import TrafficANN, TrafficANNConfig
from dataset import build_datasets


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model: TrafficANN, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_b: nn.Module, loss_r: nn.Module, device: torch.device, lambda_risk: float = 1.0, acc_threshold: float = 0.5) -> Tuple[float, float, float, float]:
    model.train()
    total_b, total_r, total = 0.0, 0.0, 0
    total_correct, total_elems = 0, 0
    for images, behavior_t, risk_t in loader:
        images = images.to(device, non_blocking=True)
        behavior_t = behavior_t.to(device)
        risk_t = risk_t.to(device)

        out = model(images, apply_activation=False)
        b_loss = loss_b(out["behavior_logits"], behavior_t)
        r_loss = loss_r(out["risk"], risk_t)
        loss = b_loss + lambda_risk * r_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_b += b_loss.item() * bs
        total_r += r_loss.item() * bs
        total += bs

        # Micro accuracy over all labels
        preds = (torch.sigmoid(out["behavior_logits"]) > acc_threshold)
        target = (behavior_t > 0.5)
        total_correct += (preds == target).sum().item()
        total_elems += preds.numel()
    acc = total_correct / max(1, total_elems)
    return total_b / total, total_r / total, (total_b + total_r) / total, acc


@torch.no_grad()
def validate(model: TrafficANN, loader: DataLoader, loss_b: nn.Module, loss_r: nn.Module, device: torch.device, lambda_risk: float = 1.0, acc_threshold: float = 0.5) -> Tuple[float, float, float, float]:
    model.eval()
    total_b, total_r, total = 0.0, 0.0, 0
    total_correct, total_elems = 0, 0
    for images, behavior_t, risk_t in loader:
        images = images.to(device, non_blocking=True)
        behavior_t = behavior_t.to(device)
        risk_t = risk_t.to(device)

        out = model(images, apply_activation=False)
        b_loss = loss_b(out["behavior_logits"], behavior_t)
        r_loss = loss_r(out["risk"], risk_t)
        bs = images.size(0)
        total_b += b_loss.item() * bs
        total_r += r_loss.item() * bs
        total += bs
        preds = (torch.sigmoid(out["behavior_logits"]) > acc_threshold)
        target = (behavior_t > 0.5)
        total_correct += (preds == target).sum().item()
        total_elems += preds.numel()
    acc = total_correct / max(1, total_elems)
    return total_b / total, total_r / total, (total_b + total_r) / total, acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=str, default="archive", help="Path to dataset archive directory containing data.yaml")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "mobilenet_v3_large", "resnet18"]) 
    p.add_argument("--num_behaviors", type=int, default=None, help="Override number of behavior classes; default uses data.yaml names length.")
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--lambda_risk", type=float, default=1.0)
    p.add_argument("--acc_threshold", type=float, default=0.5)
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--save", type=str, default="ANN/checkpoints")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--freeze_backbone", action="store_true")
    args = p.parse_args()

    device = get_device(prefer_gpu=True)
    train_ds, val_ds, paths = build_datasets(
        Path(args.archive),
        image_size=(args.height, args.width),
        in_channels=args.in_channels,
        augment=True,
    )

    num_behaviors = args.num_behaviors or len(paths.class_names)
    cfg = TrafficANNConfig(
        num_behaviors=num_behaviors,
        backbone=args.backbone,
        pretrained_backbone=not args.no_pretrained,
        behavior_mode="multilabel",
        in_channels=args.in_channels,
    )
    model = TrafficANN(cfg)
    if args.freeze_backbone:
        model.freeze_backbone()
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    loss_behavior = nn.BCEWithLogitsLoss()
    loss_risk = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr_b, tr_r, tr_total, tr_acc = train_one_epoch(
            model, train_loader, optimizer, loss_behavior, loss_risk, device,
            lambda_risk=args.lambda_risk, acc_threshold=args.acc_threshold,
        )
        va_b, va_r, va_total, va_acc = validate(
            model, val_loader, loss_behavior, loss_risk, device,
            lambda_risk=args.lambda_risk, acc_threshold=args.acc_threshold,
        )
        print(
            f"Epoch {epoch:03d} | train: b={tr_b:.4f} r={tr_r:.4f} tot={tr_total:.4f} acc={tr_acc:.4f} "
            f"| val: b={va_b:.4f} r={va_r:.4f} tot={va_total:.4f} acc={va_acc:.4f}"
        )

        ckpt_path = save_dir / f"traffic_ann_epoch{epoch:03d}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "epoch": epoch,
            "val_loss": va_total,
        }, ckpt_path)

        if va_total < best_val:
            best_val = va_total
            best_path = save_dir / "traffic_ann_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"Saved best to {best_path}")


if __name__ == "__main__":
    main()
