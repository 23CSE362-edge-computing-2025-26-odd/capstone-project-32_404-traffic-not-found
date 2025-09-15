from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class TrafficANNConfig:
    """Configuration for the TrafficANN model.

    Args:
        num_behaviors: Number of behavior classes to predict.
        backbone: Backbone name. Supported: 'mobilenet_v3_small', 'mobilenet_v3_large', 'resnet18'.
        pretrained_backbone: Whether to initialize the backbone with pretrained weights.
        hidden_dim: Hidden dimension used in heads.
        dropout: Dropout probability in heads.
        behavior_mode: 'multiclass' or 'multilabel' for interpretation in inference.
        risk_activation: Activation for risk output during inference. Supported: 'sigmoid' or 'none'.
        in_channels: Number of input image channels.
    """

    num_behaviors: int = 4
    backbone: str = "mobilenet_v3_small"
    pretrained_backbone: bool = True
    hidden_dim: int = 512
    dropout: float = 0.2
    behavior_mode: str = "multilabel"
    risk_activation: str = "sigmoid"
    in_channels: int = 3


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _get_torchvision_backbone(name: str, pretrained: bool, in_channels: int) -> Tuple[nn.Module, int]:
    """Builds a TorchVision backbone and returns a feature extractor and its feature dimension.

    The returned module must output a flattened feature vector of shape [N, feat_dim].
    """

    try:
        import torchvision
        from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, resnet18
    except Exception as e:
        raise RuntimeError("torchvision is required for the selected backbone") from e

    weights_kw: Dict[str, Optional[object]] = {}
    if pretrained:
        try:
            if name == "mobilenet_v3_small":
                weights_kw["weights"] = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
            elif name == "mobilenet_v3_large":
                weights_kw["weights"] = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
            elif name == "resnet18":
                weights_kw["weights"] = torchvision.models.ResNet18_Weights.DEFAULT
        except Exception:
            weights_kw["pretrained"] = True

    if name == "mobilenet_v3_small":
        m = mobilenet_v3_small(**weights_kw)
        if in_channels != 3:
            first = m.features[0][0]
            conv = nn.Conv2d(in_channels, first.out_channels, kernel_size=first.kernel_size, stride=first.stride, padding=first.padding, bias=False)
            m.features[0][0] = conv
        features = m.features
        feat_dim = m.classifier[0].in_features
        extractor = nn.Sequential(features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        return extractor, feat_dim
    if name == "mobilenet_v3_large":
        m = mobilenet_v3_large(**weights_kw)
        if in_channels != 3:
            first = m.features[0][0]
            conv = nn.Conv2d(in_channels, first.out_channels, kernel_size=first.kernel_size, stride=first.stride, padding=first.padding, bias=False)
            m.features[0][0] = conv
        features = m.features
        feat_dim = m.classifier[0].in_features
        extractor = nn.Sequential(features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        return extractor, feat_dim
    if name == "resnet18":
        m = resnet18(**weights_kw)
        if in_channels != 3:
            m.conv1 = nn.Conv2d(in_channels, m.conv1.out_channels, kernel_size=m.conv1.kernel_size, stride=m.conv1.stride, padding=m.conv1.padding, bias=False)
        m.fc = Identity()
        feat_dim = 512
        return m, feat_dim
    raise ValueError(f"Unsupported backbone: {name}")


class Head(nn.Module):
    """Two-layer MLP head with optional activation on output."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrafficANN(nn.Module):
    """TrafficANN for visual behavior understanding and risk regression.

    Forward inputs are images shaped [N, C, H, W].
    Forward outputs a dict with keys:
      - 'behavior_logits': Tensor [N, num_behaviors]
      - 'risk': Tensor [N, 1] (activation optionally applied via forward flags)
    """

    def __init__(self, config: TrafficANNConfig = TrafficANNConfig()):
        super().__init__()
        self.config = config
        self.backbone, feat_dim = _get_torchvision_backbone(
            name=config.backbone,
            pretrained=config.pretrained_backbone,
            in_channels=config.in_channels,
        )
        self.behavior_head = Head(feat_dim, config.num_behaviors, config.hidden_dim, config.dropout)
        self.risk_head = Head(feat_dim, 1, config.hidden_dim, config.dropout)

    @torch.no_grad()
    def _apply_inference_activation(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mode = self.config.behavior_mode
        if mode == "multilabel":
            behavior = torch.sigmoid(outputs["behavior_logits"])
        elif mode == "multiclass":
            behavior = torch.softmax(outputs["behavior_logits"], dim=-1)
        else:
            behavior = outputs["behavior_logits"]
        if self.config.risk_activation == "sigmoid":
            risk = torch.sigmoid(outputs["risk"]) 
        else:
            risk = outputs["risk"]
        return {"behavior": behavior, "risk": risk}

    def forward(self, images: torch.Tensor, apply_activation: bool = False) -> Dict[str, torch.Tensor]:
        feats = self.backbone(images)
        behavior_logits = self.behavior_head(feats)
        risk = self.risk_head(feats)
        out = {"behavior_logits": behavior_logits, "risk": risk}
        if apply_activation:
            return self._apply_inference_activation(out)
        return out

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def export_onnx(
        self,
        file_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
        opset: int = 13,
        apply_activation: bool = True,
        dynamic_batch: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        """Exports the model to ONNX for edge deployment.

        Args:
            file_path: Destination .onnx path.
            input_shape: Input tensor shape (N, C, H, W).
            opset: ONNX opset version.
            apply_activation: Whether to include inference activations in the exported graph.
            dynamic_batch: If True, export with dynamic batch axis.
            device: Device to place the model on for export.
        """

        was_training = self.training
        self.eval()
        dev = device or torch.device("cpu")
        self.to(dev)
        dummy = torch.randn(*input_shape, device=dev)

        class Wrapper(nn.Module):
            def __init__(self, m: "TrafficANN", act: bool):
                super().__init__()
                self.m = m
                self.act = act

            def forward(self, x: torch.Tensor):
                out = self.m(x, apply_activation=self.act)
                if "behavior" in out:
                    return out["behavior"], out["risk"]
                return out["behavior_logits"], out["risk"]

        wrapped = Wrapper(self, apply_activation)

        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {"input": {0: "batch"}, "behavior": {0: "batch"}, "risk": {0: "batch"}}

        torch.onnx.export(
            wrapped,
            dummy,
            file_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["behavior", "risk"],
            dynamic_axes=dynamic_axes,
        )

        if was_training:
            self.train()


__all__ = ["TrafficANNConfig", "TrafficANN"]
