import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms as T

from model import TrafficANN, TrafficANNConfig
from dataset import load_data_yaml


def build_transform(in_channels: int, size_hw: tuple[int, int]) -> T.Compose:
    h, w = size_hw
    if in_channels == 3:
        to_tensor = T.ToTensor()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif in_channels == 1:
        to_tensor = T.Compose([T.Grayscale(num_output_channels=1), T.ToTensor()])
        mean = [0.485]
        std = [0.229]
    else:
        raise ValueError("in_channels must be 1 or 3")
    return T.Compose([
        T.Resize((h, w), interpolation=T.InterpolationMode.BILINEAR),
        to_tensor,
        T.Normalize(mean=mean, std=std),
    ])


def build_model_from_checkpoint(ckpt_path: Path, num_behaviors: int, backbone: str, in_channels: int, pretrained_backbone: bool) -> TrafficANN:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt and "config" in ckpt:
        cfg_dict = ckpt["config"]
        # Ensure critical fields align with current environment
        cfg_dict["in_channels"] = in_channels
        cfg = TrafficANNConfig(**cfg_dict)
        model = TrafficANN(cfg)
        model.load_state_dict(ckpt["model_state"], strict=True)
        return model
    # Fallback: best checkpoint saved as raw state_dict
    cfg = TrafficANNConfig(
        num_behaviors=num_behaviors,
        backbone=backbone,
        pretrained_backbone=pretrained_backbone,
        behavior_mode="multilabel",
        in_channels=in_channels,
    )
    model = TrafficANN(cfg)
    state = ckpt if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    return model


def pretty_print(classes: List[str], probs: torch.Tensor, threshold: float, risk: float) -> None:
    probs = probs.tolist()
    picked = [(cls, p) for cls, p in zip(classes, probs) if p >= threshold]
    picked.sort(key=lambda x: x[1], reverse=True)
    print("Predicted classes (>= {:.2f}):".format(threshold))
    if not picked:
        print("- none (no class exceeded threshold)")
    else:
        for cls, p in picked:
            print(f"- {cls}: {p:.3f}")
    print(f"Risk score (0..1, density-proxy): {risk:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Run ANN multi-label classification + risk on one image")
    ap.add_argument("--image", required=True, type=str, help="Path to input image")
    ap.add_argument("--archive", type=str, default="archive", help="Dataset archive folder with data.yaml (for class names)")
    ap.add_argument("--checkpoint", type=str, default="ANN/checkpoints/traffic_ann_best.pt", help="Path to checkpoint (.pt)")
    ap.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "mobilenet_v3_large", "resnet18"], help="Backbone to instantiate if checkpoint lacks config")
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--in_channels", type=int, default=3, choices=[1, 3])
    ap.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for multi-label selection")
    ap.add_argument("--no_pretrained", action="store_true", help="Disable pretrained backbone when rebuilding from state_dict")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    data_yaml = Path(args.archive) / "data.yaml"
    paths = load_data_yaml(data_yaml)
    class_names = paths.class_names
    num_behaviors = len(class_names)

    model = build_model_from_checkpoint(
        Path(args.checkpoint),
        num_behaviors=num_behaviors,
        backbone=args.backbone,
        in_channels=args.in_channels,
        pretrained_backbone=not args.no_pretrained,
    )
    model.eval()

    transform = build_transform(args.in_channels, (args.height, args.width))
    img = Image.open(img_path)
    if args.in_channels == 3:
        img = img.convert("RGB")
    else:
        img = img.convert("L")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x, apply_activation=True)
        probs = out["behavior"][0].cpu()
        risk = float(out["risk"][0].cpu())

    pretty_print(class_names, probs, args.threshold, risk)


if __name__ == "__main__":
    main()
