"""DarkDriving dataset loader.

Loads aligned day-night image pairs from the DarkDriving dataset.
Structure expected:
  root/
    train/
      night/  -- low-light images
      day/    -- ground-truth day images
      annotations/  -- 2D bbox annotations (COCO JSON or YOLO txt)
    test/
      night/
      day/
      annotations/

Paper specs:
  - 9,538 total pairs (5,906 train, 3,632 test)
  - Resolution: 2448x2048 (raw), resized to 512x512 for enhancement
  - 13,184 bounding box annotations (Car class only)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DarkDrivingDataset(Dataset):
    """DarkDriving day-night aligned pair dataset.

    Each sample returns a dict with:
        - night: (3, H, W) normalized low-light image tensor
        - day: (3, H, W) normalized ground-truth day image tensor
        - filename: str, image filename
        - bboxes: (N, 4) tensor of [x1, y1, x2, y2] bboxes (if available)
        - labels: (N,) tensor of class labels (if available)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        input_size: tuple[int, int] = (512, 512),
        augment: bool = False,
        crop_size: tuple[int, int] | None = None,
        horizontal_flip: bool = False,
        flip_prob: float = 0.5,
        random_rotation: bool = False,
        rotation_degrees: int = 90,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.input_size = input_size
        self.augment = augment
        self.crop_size = crop_size or input_size
        self.horizontal_flip = horizontal_flip
        self.flip_prob = flip_prob
        self.random_rotation = random_rotation
        self.rotation_degrees = rotation_degrees

        # Build file lists
        self.night_dir = self.root / split / "night"
        self.day_dir = self.root / split / "day"
        self.anno_dir = self.root / split / "annotations"

        self.filenames: list[str] = []
        self._annotations: dict[str, Any] = {}

        if self.night_dir.exists():
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
            self.filenames = sorted(
                f.name
                for f in self.night_dir.iterdir()
                if f.suffix.lower() in exts
            )

        # Load COCO-format annotations if available
        coco_path = self.anno_dir / "annotations.json"
        if coco_path.exists():
            self._load_coco_annotations(coco_path)

    def _load_coco_annotations(self, path: Path) -> None:
        """Load COCO-format annotations."""
        with open(path) as f:
            coco = json.load(f)

        # Build filename -> image_id mapping
        fname_to_id: dict[str, int] = {}
        for img_info in coco.get("images", []):
            fname_to_id[img_info["file_name"]] = img_info["id"]

        # Build image_id -> annotations mapping
        id_to_annos: dict[int, list[dict]] = {}
        for anno in coco.get("annotations", []):
            img_id = anno["image_id"]
            if img_id not in id_to_annos:
                id_to_annos[img_id] = []
            id_to_annos[img_id].append(anno)

        # Store as filename -> list of (bbox, label)
        for fname, img_id in fname_to_id.items():
            annos = id_to_annos.get(img_id, [])
            self._annotations[fname] = [
                {
                    "bbox": a["bbox"],  # [x, y, w, h] COCO format
                    "category_id": a.get("category_id", 0),
                }
                for a in annos
            ]

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        fname = self.filenames[idx]

        # Load images
        night_img = Image.open(self.night_dir / fname).convert("RGB")
        day_img = Image.open(self.day_dir / fname).convert("RGB")

        # Resize to target input size
        h, w = self.input_size
        night_img = night_img.resize((w, h), Image.BILINEAR)
        day_img = day_img.resize((w, h), Image.BILINEAR)

        # Convert to numpy
        night_np = np.array(night_img, dtype=np.float32) / 255.0
        day_np = np.array(day_img, dtype=np.float32) / 255.0

        # Augmentations (applied identically to both images)
        if self.augment:
            night_np, day_np = self._augment(night_np, day_np)

        # Convert to tensors (C, H, W)
        night_t = torch.from_numpy(night_np).permute(2, 0, 1).contiguous()
        day_t = torch.from_numpy(day_np).permute(2, 0, 1).contiguous()

        result: dict[str, Any] = {
            "night": night_t,
            "day": day_t,
            "filename": fname,
        }

        # Load bboxes if available
        if fname in self._annotations:
            annos = self._annotations[fname]
            if annos:
                bboxes = []
                labels = []
                for a in annos:
                    x, y, bw, bh = a["bbox"]
                    # Convert COCO [x, y, w, h] -> [x1, y1, x2, y2]
                    bboxes.append([x, y, x + bw, y + bh])
                    labels.append(a["category_id"])
                result["bboxes"] = torch.tensor(bboxes, dtype=torch.float32)
                result["labels"] = torch.tensor(labels, dtype=torch.long)
            else:
                result["bboxes"] = torch.zeros((0, 4), dtype=torch.float32)
                result["labels"] = torch.zeros((0,), dtype=torch.long)
        else:
            result["bboxes"] = torch.zeros((0, 4), dtype=torch.float32)
            result["labels"] = torch.zeros((0,), dtype=torch.long)

        return result

    def _augment(
        self, night: np.ndarray, day: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply synchronized augmentations to both images."""
        # Random horizontal flip
        if self.horizontal_flip and np.random.random() < self.flip_prob:
            night = np.fliplr(night).copy()
            day = np.fliplr(day).copy()

        # Random rotation (0, 90, 180, 270)
        if self.random_rotation and self.rotation_degrees > 0:
            k = np.random.randint(0, 4)  # number of 90-degree rotations
            if k > 0:
                night = np.rot90(night, k).copy()
                day = np.rot90(day, k).copy()

        # Random crop
        h, w = night.shape[:2]
        ch, cw = self.crop_size
        if ch < h or cw < w:
            top = np.random.randint(0, max(1, h - ch))
            left = np.random.randint(0, max(1, w - cw))
            night = night[top : top + ch, left : left + cw]
            day = day[top : top + ch, left : left + cw]

        return night, day


class SyntheticLowLightDataset(Dataset):
    """Synthetic low-light dataset for pretraining/augmentation.

    Generates low-light images from normal-light images using digital
    image processing (gamma correction, noise injection), following the
    paper's approach for nuScenes nighttime pair synthesis.
    """

    def __init__(
        self,
        root: str,
        input_size: tuple[int, int] = (512, 512),
        gamma_range: tuple[float, float] = (2.0, 5.0),
        noise_std: float = 0.02,
    ):
        super().__init__()
        self.root = Path(root)
        self.input_size = input_size
        self.gamma_range = gamma_range
        self.noise_std = noise_std

        exts = {".png", ".jpg", ".jpeg"}
        self.filenames = sorted(
            str(f)
            for f in self.root.rglob("*")
            if f.suffix.lower() in exts
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        fpath = self.filenames[idx]
        img = Image.open(fpath).convert("RGB")
        h, w = self.input_size
        img = img.resize((w, h), Image.BILINEAR)
        day_np = np.array(img, dtype=np.float32) / 255.0

        # Synthetic darkening: gamma correction + noise
        gamma = np.random.uniform(*self.gamma_range)
        night_np = np.power(day_np.clip(1e-8, 1.0), gamma)
        noise = np.random.normal(0, self.noise_std, night_np.shape).astype(np.float32)
        night_np = np.clip(night_np + noise, 0.0, 1.0)

        night_t = torch.from_numpy(night_np).permute(2, 0, 1).contiguous()
        day_t = torch.from_numpy(day_np).permute(2, 0, 1).contiguous()

        return {
            "night": night_t,
            "day": day_t,
            "filename": Path(fpath).name,
        }


def build_dataloaders(
    root: str,
    input_size: tuple[int, int] = (512, 512),
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    val_ratio: float = 0.1,
    seed: int = 42,
    augment_train: bool = True,
    crop_size: tuple[int, int] | None = None,
    horizontal_flip: bool = True,
    flip_prob: float = 0.5,
    random_rotation: bool = True,
    rotation_degrees: int = 90,
) -> dict[str, torch.utils.data.DataLoader]:
    """Build train, val, and test dataloaders.

    Carves a validation set from the training split (default 10%).

    Returns:
        Dict with keys 'train', 'val', 'test' mapping to DataLoaders.
    """
    from torch.utils.data import DataLoader, Subset

    # Full training dataset (augmented)
    train_full = DarkDrivingDataset(
        root=root,
        split="train",
        input_size=input_size,
        augment=augment_train,
        crop_size=crop_size,
        horizontal_flip=horizontal_flip,
        flip_prob=flip_prob,
        random_rotation=random_rotation,
        rotation_degrees=rotation_degrees,
    )

    # Validation dataset (no augmentation)
    val_full = DarkDrivingDataset(
        root=root,
        split="train",
        input_size=input_size,
        augment=False,
    )

    # Split train into train/val
    n = len(train_full)
    if n == 0:
        # Dataset not available -- return empty loaders
        empty_loader = DataLoader(train_full, batch_size=1)
        return {"train": empty_loader, "val": empty_loader, "test": empty_loader}

    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val

    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=gen).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_ds = Subset(train_full, train_indices)
    val_ds = Subset(val_full, val_indices)

    # Test dataset
    test_ds = DarkDrivingDataset(
        root=root,
        split="test",
        input_size=input_size,
        augment=False,
    )

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    return loaders
