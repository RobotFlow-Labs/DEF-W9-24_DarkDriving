"""Multi-source dataset for DarkDriving enhancement training.

Combines nuScenes day images + KITTI daytime images with synthetic
low-light generation to create training pairs. Surpasses paper's
9.5K DarkDriving pairs with 37K+ synthetic pairs.

Sources:
- nuScenes CAM_FRONT day images (30K+) → synthetic pairs
- KITTI training images (7.5K) → synthetic pairs
- nuScenes CAM_FRONT night images (4K) → evaluation only (no GT)
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


class NuScenesLowLightDataset(Dataset):
    """nuScenes day images with synthetic low-light generation.

    Reads nuScenes metadata to separate day/night scenes.
    Day images are synthetically darkened for training pairs.
    Night images are kept for evaluation (no ground truth).
    """

    def __init__(
        self,
        nuscenes_root: str = "/mnt/forge-data/datasets/nuscenes",
        mode: str = "day",  # "day" for training pairs, "night" for eval
        input_size: tuple[int, int] = (512, 512),
        gamma_range: tuple[float, float] = (2.0, 5.0),
        noise_std: float = 0.02,
        augment: bool = False,
        max_samples: int | None = None,
    ):
        super().__init__()
        self.root = Path(nuscenes_root)
        self.mode = mode
        self.input_size = input_size
        self.gamma_range = gamma_range
        self.noise_std = noise_std
        self.augment = augment

        self.image_paths: list[str] = []
        self._load_file_list(max_samples)

    def _load_file_list(self, max_samples: int | None) -> None:
        """Parse nuScenes metadata to get day/night camera images."""
        meta_dir = self.root / "v1.0-trainval"
        if not meta_dir.exists():
            return

        with open(meta_dir / "scene.json") as f:
            scenes = json.load(f)
        with open(meta_dir / "sample.json") as f:
            samples = json.load(f)
        with open(meta_dir / "sample_data.json") as f:
            sample_data = json.load(f)
        with open(meta_dir / "sensor.json") as f:
            sensors = json.load(f)
        with open(meta_dir / "calibrated_sensor.json") as f:
            cal_sensors = json.load(f)

        # Map calibrated_sensor_token -> channel name
        sensor_map = {s["token"]: s["channel"] for s in sensors}
        cal_to_channel = {}
        for cs in cal_sensors:
            cal_to_channel[cs["token"]] = sensor_map.get(
                cs["sensor_token"], ""
            )

        # Identify night scene tokens
        night_tokens = set()
        for s in scenes:
            desc = s.get("description", "").lower()
            if "night" in desc:
                night_tokens.add(s["token"])

        # Get sample tokens for day/night
        if self.mode == "night":
            target_tokens = {
                s["token"]
                for s in samples
                if s["scene_token"] in night_tokens
            }
        else:
            target_tokens = {
                s["token"]
                for s in samples
                if s["scene_token"] not in night_tokens
            }

        # Filter CAM_FRONT keyframes
        for sd in sample_data:
            channel = cal_to_channel.get(
                sd.get("calibrated_sensor_token", ""), ""
            )
            if (
                channel == "CAM_FRONT"
                and sd.get("is_key_frame", False)
                and sd["sample_token"] in target_tokens
            ):
                img_path = str(self.root / sd["filename"])
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)

        if max_samples and len(self.image_paths) > max_samples:
            random.shuffle(self.image_paths)
            self.image_paths = self.image_paths[:max_samples]

        self.image_paths.sort()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Robust loading — skip corrupt images
        for offset in range(min(10, len(self.image_paths))):
            try:
                real_idx = (idx + offset) % len(self.image_paths)
                return self._load_sample(real_idx)
            except Exception:
                continue
        # Last resort: return black image
        h, w = self.input_size
        blank = torch.zeros(3, h, w)
        return {
            "night": blank, "day": blank,
            "filename": "fallback", "source": "fallback",
        }

    def _load_sample(self, idx: int) -> dict[str, Any]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        h, w = self.input_size
        img = img.resize((w, h), Image.BILINEAR)
        day_np = np.array(img, dtype=np.float32) / 255.0

        if self.mode == "night":
            night_t = torch.from_numpy(day_np).permute(2, 0, 1).contiguous()
            return {
                "night": night_t,
                "day": night_t,
                "filename": Path(self.image_paths[idx]).name,
                "source": "nuscenes_night",
            }

        gamma = np.random.uniform(*self.gamma_range)
        night_np = np.power(np.clip(day_np, 1e-8, 1.0), gamma)
        noise = np.random.normal(0, self.noise_std, night_np.shape)
        night_np = np.clip(night_np + noise, 0.0, 1.0).astype(np.float32)

        if self.augment:
            night_np, day_np = self._augment(night_np, day_np)

        night_t = torch.from_numpy(night_np).permute(2, 0, 1).contiguous()
        day_t = torch.from_numpy(day_np).permute(2, 0, 1).contiguous()

        return {
            "night": night_t,
            "day": day_t,
            "filename": Path(self.image_paths[idx]).name,
            "source": "nuscenes_day",
        }

    def _augment(
        self, night: np.ndarray, day: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.random() < 0.5:
            night = np.fliplr(night).copy()
            day = np.fliplr(day).copy()
        k = np.random.randint(0, 4)
        if k > 0:
            night = np.rot90(night, k).copy()
            day = np.rot90(day, k).copy()
        return night, day


class KITTILowLightDataset(Dataset):
    """KITTI daytime images with synthetic low-light generation."""

    def __init__(
        self,
        kitti_root: str = "/mnt/forge-data/datasets/kitti",
        input_size: tuple[int, int] = (512, 512),
        gamma_range: tuple[float, float] = (2.0, 5.0),
        noise_std: float = 0.02,
        augment: bool = False,
        max_samples: int | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.gamma_range = gamma_range
        self.noise_std = noise_std
        self.augment = augment

        img_dir = Path(kitti_root) / "training" / "image_2"
        self.image_paths: list[str] = []
        if img_dir.exists():
            exts = {".png", ".jpg", ".jpeg"}
            self.image_paths = sorted(
                str(f) for f in img_dir.iterdir()
                if f.suffix.lower() in exts
            )
            if max_samples and len(self.image_paths) > max_samples:
                self.image_paths = self.image_paths[:max_samples]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        for offset in range(min(10, len(self.image_paths))):
            try:
                real_idx = (idx + offset) % len(self.image_paths)
                return self._load_sample(real_idx)
            except Exception:
                continue
        h, w = self.input_size
        blank = torch.zeros(3, h, w)
        return {
            "night": blank, "day": blank,
            "filename": "fallback", "source": "fallback",
        }

    def _load_sample(self, idx: int) -> dict[str, Any]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        h, w = self.input_size
        img = img.resize((w, h), Image.BILINEAR)
        day_np = np.array(img, dtype=np.float32) / 255.0

        gamma = np.random.uniform(*self.gamma_range)
        night_np = np.power(np.clip(day_np, 1e-8, 1.0), gamma)
        noise = np.random.normal(0, self.noise_std, night_np.shape)
        night_np = np.clip(night_np + noise, 0.0, 1.0).astype(np.float32)

        if self.augment:
            if np.random.random() < 0.5:
                night_np = np.fliplr(night_np).copy()
                day_np = np.fliplr(day_np).copy()

        night_t = torch.from_numpy(night_np).permute(2, 0, 1).contiguous()
        day_t = torch.from_numpy(day_np).permute(2, 0, 1).contiguous()

        return {
            "night": night_t,
            "day": day_t,
            "filename": Path(self.image_paths[idx]).name,
            "source": "kitti",
        }


def build_multi_dataloaders(
    nuscenes_root: str = "/mnt/forge-data/datasets/nuscenes",
    kitti_root: str = "/mnt/forge-data/datasets/kitti",
    input_size: tuple[int, int] = (512, 512),
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    val_ratio: float = 0.1,
    seed: int = 42,
    gamma_range: tuple[float, float] = (2.0, 5.0),
    noise_std: float = 0.02,
    max_nuscenes: int | None = None,
    max_kitti: int | None = None,
) -> dict[str, DataLoader]:
    """Build train/val/test dataloaders from multiple sources.

    Train: nuScenes day (synthetic pairs) + KITTI (synthetic pairs)
    Val: 10% held out from train
    Test: nuScenes real night images (no-ref evaluation)
    """
    # Training datasets
    nuscenes_train = NuScenesLowLightDataset(
        nuscenes_root=nuscenes_root,
        mode="day",
        input_size=input_size,
        gamma_range=gamma_range,
        noise_std=noise_std,
        augment=True,
        max_samples=max_nuscenes,
    )

    kitti_train = KITTILowLightDataset(
        kitti_root=kitti_root,
        input_size=input_size,
        gamma_range=gamma_range,
        noise_std=noise_std,
        augment=True,
        max_samples=max_kitti,
    )

    # Combine
    combined = ConcatDataset([nuscenes_train, kitti_train])
    total = len(combined)

    if total == 0:
        empty_loader: DataLoader = DataLoader(
            combined, batch_size=1
        )
        return {
            "train": empty_loader,
            "val": empty_loader,
            "test": empty_loader,
        }

    # Train/val split
    n_val = max(1, int(total * val_ratio))
    n_train = total - n_val

    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=gen).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_ds = Subset(combined, train_indices)

    # Val without augmentation — rebuild datasets
    nuscenes_val = NuScenesLowLightDataset(
        nuscenes_root=nuscenes_root,
        mode="day",
        input_size=input_size,
        gamma_range=gamma_range,
        noise_std=noise_std,
        augment=False,
        max_samples=max_nuscenes,
    )
    kitti_val = KITTILowLightDataset(
        kitti_root=kitti_root,
        input_size=input_size,
        gamma_range=gamma_range,
        noise_std=noise_std,
        augment=False,
        max_samples=max_kitti,
    )
    combined_val = ConcatDataset([nuscenes_val, kitti_val])
    val_ds = Subset(combined_val, val_indices)

    # Test: real nuScenes night images
    test_ds = NuScenesLowLightDataset(
        nuscenes_root=nuscenes_root,
        mode="night",
        input_size=input_size,
        augment=False,
    )

    print(f"[DATA] nuScenes day: {len(nuscenes_train)}")
    print(f"[DATA] KITTI: {len(kitti_train)}")
    print(f"[DATA] Combined train: {n_train}, val: {n_val}")
    print(f"[DATA] nuScenes night (test): {len(test_ds)}")

    return {
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
