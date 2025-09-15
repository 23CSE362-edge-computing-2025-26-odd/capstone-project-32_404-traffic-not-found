import argparse
import torch

from model import TrafficANN, TrafficANNConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small")
    parser.add_argument("--num_behaviors", type=int, default=4)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--export", type=str, default="none", choices=["none", "onnx"]) 
    parser.add_argument("--out", type=str, default="model.onnx")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--behavior_mode", type=str, default="multilabel", choices=["multilabel", "multiclass"]) 
    args = parser.parse_args()

    cfg = TrafficANNConfig(
        num_behaviors=args.num_behaviors,
        backbone=args.backbone,
        pretrained_backbone=not args.no_pretrained,
        in_channels=args.in_channels,
        behavior_mode=args.behavior_mode,
    )
    model = TrafficANN(cfg)
    model.eval()

    x = torch.randn(2, cfg.in_channels, args.height, args.width)
    out = model(x, apply_activation=True)
    print("behavior:", out["behavior"].shape, out["behavior"].min().item(), out["behavior"].max().item())
    print("risk:", out["risk"].shape, out["risk"].min().item(), out["risk"].max().item())

    if args.export == "onnx":
        model.export_onnx(args.out, input_shape=(1, cfg.in_channels, args.height, args.width))
        print("Exported ONNX to", args.out)


if __name__ == "__main__":
    main()
