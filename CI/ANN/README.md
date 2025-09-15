# Stage 1 ANN: Traffic Behavior + Risk Model

This folder contains a lightweight, edge-friendly PyTorch ANN for analyzing traffic camera images to detect risky behaviors and estimate a scalar risk signal. The model uses a configurable CNN backbone (MobileNetV3 or ResNet18) and two heads:

- Behavior head: predicts traffic behavior probabilities/logits (multi-label or multi-class)
- Risk head: predicts a scalar risk (0-1 with sigmoid or unbounded)

## Quick Start

### 1) Install

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install torch torchvision
```

### 2) Run a smoke test

```powershell
python ANN\test_model.py
```

Expected output: tensor shapes for behavior and risk.

### 3) Use in code

```python
from ANN.model import TrafficANN, TrafficANNConfig
import torch

cfg = TrafficANNConfig(num_behaviors=5, backbone="mobilenet_v3_small", behavior_mode="multilabel")
model = TrafficANN(cfg)
model.eval()

x = torch.randn(2, 3, 224, 224)
out = model(x, apply_activation=True)
print(out["behavior"].shape, out["risk"].shape)
```

### 4) Train on `archive/` dataset

The training script reads YOLO-format labels defined in `archive/data.yaml` and converts them into:
- Multi-label behavior targets (presence per class)
- A heuristic density-based risk target in [0,1]

Run training:

```powershell
python ANN\train_ann.py --archive archive --epochs 5 --batch 16 --height 224 --width 224 --no_pretrained
```

Checkpoints are saved to `ANN\checkpoints`. The best model is `traffic_ann_best.pt`.

### 4) Export to ONNX (optional)

```powershell
python ANN\test_model.py --export onnx --out model.onnx --height 224 --width 224
```

The exported ONNX contains two outputs: `behavior`, `risk`.

### 5) To run a small test on the CLI:
```powershell
python ANN\infer_ann.py --image "path\to\your.jpg" --archive archive --checkpoint ANN\checkpoints\traffic_ann_best.pt --threshold 0.5
```

## Notes

- For non-RGB inputs (e.g., 1-channel), set `in_channels` in `TrafficANNConfig`.
- If using `behavior_mode="multiclass"`, use cross-entropy loss and class labels. For `multilabel`, use BCE-with-logits loss and multi-hot labels.
- For edge deployment, prefer `mobilenet_v3_small`.

### Dataset Link:
https://www.kaggle.com/datasets/yusufberksardoan/traffic-detection-project