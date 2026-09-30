"""Seeding, devices, preprocessing, logging and checkpoint helpers."""
from __future__ import annotations

import csv
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess(frame: np.ndarray, size: int = 64) -> np.ndarray:
    """96x96x3 uint8 env frame -> size x size x 3 uint8 (antialiased)."""
    return np.array(Image.fromarray(frame).resize((size, size), Image.BILINEAR))


def to_tensor(frames_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    """(N,H,W,3) uint8 -> (N,3,H,W) float in [0,1] on device."""
    x = torch.from_numpy(np.require(frames_u8, requirements=["C", "W"])).to(device)
    return x.permute(0, 3, 1, 2).float().div_(255.0)


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """(N,3,H,W) float [0,1] -> (N,H,W,3) uint8 on CPU."""
    return x.detach().clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()


def upscale(img: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((size, size), Image.NEAREST))


class CSVLogger:
    """Append-only CSV logger; creates the header on first write."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()
        if self.path.exists():
            data = self.path.read_bytes()
            if data and not data.endswith(b"\n"):   # torn last line from a crash: drop it
                data = data[: data.rfind(b"\n") + 1]
                self.path.write_bytes(data)
            rows = data.decode().splitlines()
            if len(rows) > 1:                        # keep the clock running across restarts
                try:
                    self._t0 -= float(rows[-1].split(",")[0])
                except ValueError:
                    pass

    def log(self, **row):
        """The first row fixes the columns; later rows may leave some blank."""
        row = {"wall_s": round(time.time() - self._t0, 2), **row}
        if self.path.exists():
            with self.path.open() as f:
                fields = next(csv.reader(f))
        else:
            fields = list(row)
        extra = set(row) - set(fields)
        assert not extra, f"columns {extra} missing from {self.path.name}; include them in row 1"
        new = not self.path.exists()
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            if new:
                w.writeheader()
            w.writerow(row)


def save_ckpt(path: Path, **payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic: never leaves a half-written checkpoint


def save_pickle(path: Path, obj) -> None:
    """Atomic pickle: a crash mid-write leaves the previous file intact."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(obj))
    os.replace(tmp, path)


def load_ckpt(path: Path, device: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=False)
