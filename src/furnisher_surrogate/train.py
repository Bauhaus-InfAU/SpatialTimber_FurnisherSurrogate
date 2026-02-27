"""Dataset and training utilities for CNN training.

Building blocks for the training notebook — not a standalone script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch.utils.data import Dataset

from .data import ROOM_TYPES, Apartment, Split, assign_splits
from .features import area as compute_area, aspect_ratio as compute_aspect_ratio, n_vertices as compute_n_vertices

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_NPZ = _PROJECT_ROOT / "data" / "rooms_rasterized.npz"


# ── Dataset ──────────────────────────────────────────────────


class RoomDataset(Dataset):
    """PyTorch dataset for rasterized rooms.

    Loads from pre-computed NPZ, filters by split using apartment-level
    assignment. Augmentation (hflip, vflip, rot90) applied on train only.
    """

    def __init__(
        self,
        apartments: list[Apartment],
        split: Split,
        npz_path: Path | str = _DEFAULT_NPZ,
        area_mean: float | None = None,
        area_std: float | None = None,
        augment: bool = False,
        include_geometry: bool = False,
        n_verts_mean: float | None = None,
        n_verts_std: float | None = None,
        aspect_mean: float | None = None,
        aspect_std: float | None = None,
    ):
        npz = np.load(npz_path)
        split_map = assign_splits(apartments)

        # Build set of apartment seeds in this split
        seeds_in_split = {seed for seed, s in split_map.items() if s == split}

        # Filter indices
        all_seeds = npz["apartment_seeds"]
        mask = np.array([s in seeds_in_split for s in all_seeds])
        idx = np.where(mask)[0]

        self.images = npz["images"][idx]          # (N, 3, 64, 64) uint8
        self.scores = npz["scores"][idx]           # (N,) float32
        self.room_type_idx = npz["room_type_idx"][idx]  # (N,) int8
        self.area = npz["area"][idx]               # (N,) float32
        self.door_rel_x = npz["door_rel_x"][idx]  # (N,) float32
        self.door_rel_y = npz["door_rel_y"][idx]  # (N,) float32

        # apartment_type_idx: present in new NPZ files, fallback to 0 for old ones
        if "apartment_type_idx" in npz:
            self.apartment_type_idx = npz["apartment_type_idx"][idx]  # (N,) int8
        else:
            self.apartment_type_idx = np.zeros(len(idx), dtype=np.int8)

        # Standardize area
        if area_mean is None or area_std is None:
            self.area_mean = float(self.area.mean())
            self.area_std = float(self.area.std())
        else:
            self.area_mean = area_mean
            self.area_std = area_std

        # Optional: n_vertices and aspect_ratio (not in NPZ, compute from rooms)
        self.include_geometry = include_geometry
        if include_geometry:
            all_rooms = [room for apt in apartments for room in apt.rooms]
            all_n_verts = np.array([compute_n_vertices(r) for r in all_rooms], dtype=np.float32)
            all_aspect = np.array([compute_aspect_ratio(r) for r in all_rooms], dtype=np.float32)
            self.n_verts = all_n_verts[idx]
            self.aspect_ratio = all_aspect[idx]

            if n_verts_mean is None or n_verts_std is None:
                self.n_verts_mean = float(self.n_verts.mean())
                self.n_verts_std = float(self.n_verts.std())
            else:
                self.n_verts_mean = n_verts_mean
                self.n_verts_std = n_verts_std

            if aspect_mean is None or aspect_std is None:
                self.aspect_mean = float(self.aspect_ratio.mean())
                self.aspect_std = float(self.aspect_ratio.std())
            else:
                self.aspect_mean = aspect_mean
                self.aspect_std = aspect_std

        self.augment = augment

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = torch.from_numpy(self.images[idx].astype(np.float32) / 255.0)
        room_type = torch.tensor(int(self.room_type_idx[idx]), dtype=torch.long)
        apt_type = torch.tensor(int(self.apartment_type_idx[idx]), dtype=torch.long)

        area_std = (self.area[idx] - self.area_mean) / (self.area_std + 1e-8)
        tab_values = [area_std, self.door_rel_x[idx], self.door_rel_y[idx]]

        if self.include_geometry:
            nv_std = (self.n_verts[idx] - self.n_verts_mean) / (self.n_verts_std + 1e-8)
            ar_std = (self.aspect_ratio[idx] - self.aspect_mean) / (self.aspect_std + 1e-8)
            tab_values.extend([nv_std, ar_std])

        tabular = torch.tensor(tab_values, dtype=torch.float32)
        score = torch.tensor(self.scores[idx], dtype=torch.float32)

        # Augmentation: hflip, vflip, rot90 (train only)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[-1])
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[-2])
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                image = torch.rot90(image, k, dims=[-2, -1])

        return {
            "image": image,
            "room_type_idx": room_type,
            "apt_type_idx": apt_type,
            "tabular": tabular,
            "score": score,
        }


# ── Training utilities ───────────────────────────────────────


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Single training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        room_types = batch["room_type_idx"].to(device)
        apt_types = batch.get("apt_type_idx")
        if apt_types is not None:
            apt_types = apt_types.to(device)
        tabular = batch["tabular"].to(device)
        scores = batch["score"].to(device)

        optimizer.zero_grad()
        preds = model(images, room_types, tabular, apt_types).squeeze(-1)
        loss = criterion(preds, scores)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inference on a DataLoader. Returns (y_true, y_pred, room_type_idx) as numpy."""
    model.eval()
    all_true, all_pred, all_rt = [], [], []

    for batch in loader:
        images = batch["image"].to(device)
        room_types = batch["room_type_idx"].to(device)
        apt_types = batch.get("apt_type_idx")
        if apt_types is not None:
            apt_types = apt_types.to(device)
        tabular = batch["tabular"].to(device)

        preds = model(images, room_types, tabular, apt_types).squeeze(-1)
        all_true.append(batch["score"].numpy())
        all_pred.append(preds.cpu().numpy())
        all_rt.append(batch["room_type_idx"].numpy())

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_rt),
    )


# ── Metrics ──────────────────────────────────────────────────


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    """Compute regression + binary fail/pass metrics. Mirrors baseline exactly."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # Binary fail/pass: score=0 vs score>0, threshold at 5
    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 5).astype(int)
    acc = float(accuracy_score(y_true_bin, y_pred_bin))
    f1 = float(f1_score(y_true_bin, y_pred_bin, average="binary"))

    # Conditional MAE
    mask_zero = y_true == 0
    mask_nonzero = y_true > 0
    mae_zeros = float(mean_absolute_error(y_true[mask_zero], y_pred[mask_zero])) if mask_zero.any() else 0.0
    mae_nonzero = float(mean_absolute_error(y_true[mask_nonzero], y_pred[mask_nonzero])) if mask_nonzero.any() else 0.0

    p = f"{prefix}/" if prefix else ""
    return {
        f"{p}mae": mae,
        f"{p}rmse": rmse,
        f"{p}r2": r2,
        f"{p}binary_accuracy": acc,
        f"{p}binary_f1": f1,
        f"{p}mae_on_zeros": mae_zeros,
        f"{p}mae_on_nonzero": mae_nonzero,
    }


def per_type_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    room_type_idx: np.ndarray,
) -> dict[str, dict]:
    """Per-room-type breakdown with full metrics."""
    results = {}
    for i, rt in enumerate(ROOM_TYPES):
        mask = room_type_idx == i
        if not mask.any():
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        n = int(mask.sum())
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2 = float(r2_score(yt, yp)) if n > 1 else 0.0
        naive_mae = float(np.mean(np.abs(yt - yt.mean())))

        y_true_bin = (yt > 0).astype(int)
        y_pred_bin = (yp > 5).astype(int)
        n_zeros = int((yt == 0).sum())
        acc = float(accuracy_score(y_true_bin, y_pred_bin)) if n > 0 else 0.0
        f1 = float(f1_score(y_true_bin, y_pred_bin, average="binary", zero_division=0))

        results[rt] = {
            "n": n,
            "n_zeros": n_zeros,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "naive_mae": naive_mae,
            "binary_accuracy": acc,
            "binary_f1": f1,
        }
    return results
