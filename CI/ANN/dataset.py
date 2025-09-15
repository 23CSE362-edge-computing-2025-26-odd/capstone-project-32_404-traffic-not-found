from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


@dataclass
class YOLOPaths:
    root: Path
    train_images: Path
    val_images: Path
    train_labels: Path
    val_labels: Path
    class_names: List[str]


def load_data_yaml(yaml_path: Path) -> YOLOPaths:
    # Minimal YAML parser without dependency
    # Expects keys: train, val, names
    text = yaml_path.read_text(encoding="utf-8")
    train_dir: Optional[str] = None
    val_dir: Optional[str] = None
    names: List[str] = []

    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("train:"):
            train_dir = s.split(":", 1)[1].strip().strip("'\"")
        elif s.startswith("val:") or s.startswith("valid:"):
            val_dir = s.split(":", 1)[1].strip().strip("'\"")
        elif s.startswith("names:"):
            # Could be names: ['a','b'] or followed by list lines
            rhs = s.split(":", 1)[1].strip()
            if rhs.startswith("["):
                # Inline list
                inside = rhs.strip().strip("[]")
                names = [x.strip().strip("'\"") for x in inside.split(",") if x.strip()]
            else:
                names = []
        elif s.startswith("-") and names == [] and "names:" not in s:
            names.append(s.lstrip("- ").strip().strip("'\""))

    if train_dir is None or val_dir is None or not names:
        raise ValueError("data.yaml missing required keys: train, val, names")

    def images_and_labels(dir_str: str) -> Tuple[Path, Path]:
        base = yaml_path.parent
        candidates = []
        d = Path(dir_str)
        candidates.append(base / d)
        candidates.append(d)
        candidates.append((base / d) / "images")
        candidates.append(d / "images")

        for cand in candidates:
            if not cand.exists():
                continue
            # If cand is the parent folder containing images/labels
            if (cand / "images").exists() and (cand / "labels").exists():
                return cand / "images", cand / "labels"
            # If cand is already the images folder, derive labels sibling
            if cand.name.lower() == "images" and (cand.parent / "labels").exists():
                return cand, cand.parent / "labels"
        # Fallback by split keyword present in dir_str
        s = dir_str.replace("\\", "/").lower()
        for split in ("train", "valid", "val", "test"):
            if split in s:
                split_norm = "valid" if split == "val" else split
                base_split = base / split_norm
                if (base_split / "images").exists() and (base_split / "labels").exists():
                    return base_split / "images", base_split / "labels"
        raise FileNotFoundError(f"Could not resolve images/labels for {dir_str}")

    train_images, train_labels = images_and_labels(train_dir)
    val_images, val_labels = images_and_labels(val_dir)

    return YOLOPaths(
        root=yaml_path.parent,
        train_images=train_images,
        val_images=val_images,
        train_labels=train_labels,
        val_labels=val_labels,
        class_names=names,
    )


class YoloMultiTargetDataset(Dataset):
    """Converts YOLO detection labels into multi-label behavior targets and a heuristic risk.

    - behavior target: multi-hot presence vector over classes
    - risk target: heuristic density score in [0,1] based on number of boxes per image
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        class_names: List[str],
        image_size: Tuple[int, int] = (224, 224),
        risk_scale: float = 50.0,
        augment: bool = False,
        in_channels: int = 3,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.risk_scale = risk_scale
        self.in_channels = in_channels

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.images: List[Path] = [
            p for p in sorted(self.images_dir.rglob("*")) if p.suffix.lower() in exts
        ]

        if not self.images:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        if in_channels == 3:
            to_tensor = T.ToTensor()
        elif in_channels == 1:
            to_tensor = T.Compose([T.Grayscale(num_output_channels=1), T.ToTensor()])
        else:
            raise ValueError("in_channels must be 1 or 3")

        tf_train = [
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1) if augment else T.Lambda(lambda x: x),
            to_tensor,
            T.Normalize(mean=[0.485] * in_channels, std=[0.229] * in_channels),
        ]
        self.transform = T.Compose([t for t in tf_train if not isinstance(t, T.Lambda) or augment])

    def __len__(self) -> int:
        return len(self.images)

    def _label_path(self, img_path: Path) -> Path:
        return self.labels_dir / (img_path.stem + ".txt")

    def _read_labels(self, lbl_path: Path) -> List[int]:
        if not lbl_path.exists():
            return []
        try:
            lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
        except Exception:
            return []
        cls_ids: List[int] = []
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            parts = s.split()
            try:
                cid = int(float(parts[0]))
                cls_ids.append(cid)
            except Exception:
                continue
        return cls_ids

    def _targets_from_cls_ids(self, cls_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        behavior = torch.zeros(self.num_classes, dtype=torch.float32)
        for cid in cls_ids:
            if 0 <= cid < self.num_classes:
                behavior[cid] = 1.0
        # Heuristic density risk: scale by count
        risk = min(1.0, len(cls_ids) / self.risk_scale)
        risk_t = torch.tensor([risk], dtype=torch.float32)
        return behavior, risk_t

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        if self.in_channels == 1 and x.shape[0] == 3:
            x = x[:1]
        lbl_path = self._label_path(img_path)
        cls_ids = self._read_labels(lbl_path)
        behavior, risk = self._targets_from_cls_ids(cls_ids)
        return x, behavior, risk


def build_datasets(archive_dir: Path, image_size: Tuple[int, int] = (224, 224), risk_scale: float = 50.0, in_channels: int = 3, augment: bool = True) -> Tuple[YoloMultiTargetDataset, YoloMultiTargetDataset, YOLOPaths]:
    paths = load_data_yaml(Path(archive_dir) / "data.yaml")
    train_ds = YoloMultiTargetDataset(paths.train_images, paths.train_labels, paths.class_names, image_size=image_size, risk_scale=risk_scale, augment=augment, in_channels=in_channels)
    val_ds = YoloMultiTargetDataset(paths.val_images, paths.val_labels, paths.class_names, image_size=image_size, risk_scale=risk_scale, augment=False, in_channels=in_channels)
    return train_ds, val_ds, paths


__all__ = [
    "YOLOPaths",
    "load_data_yaml",
    "YoloMultiTargetDataset",
    "build_datasets",
]
