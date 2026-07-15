from __future__ import annotations

import numpy as np

from .metrics import outlier_channel_metrics


def next_power_of_two(value: int) -> int:
    out = 1
    while out < int(value):
        out *= 2
    return out


def previous_power_of_two(value: int) -> int:
    out = 1
    while out * 2 <= int(value):
        out *= 2
    return out


def fwht(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    n = out.shape[-1]
    step = 1
    while step < n:
        reshaped = out.reshape(*out.shape[:-1], -1, 2 * step)
        a = reshaped[..., :step].copy()
        b = reshaped[..., step : 2 * step].copy()
        reshaped[..., :step] = a + b
        reshaped[..., step : 2 * step] = a - b
        step *= 2
    return out / np.sqrt(float(n))


def hadamard_rotate_columns(weight: np.ndarray) -> np.ndarray:
    w = np.asarray(weight, dtype=np.float32)
    _rows, cols = w.shape
    out = np.empty_like(w)
    start = 0
    remaining = cols
    while remaining > 0:
        width = previous_power_of_two(remaining)
        signs = np.where(((np.arange(width) + start) % 2) == 0, 1.0, -1.0).astype(np.float32)
        out[:, start : start + width] = fwht(w[:, start : start + width] * signs[None, :])
        start += width
        remaining -= width
    return out


def butterfly_rotate_columns(weight: np.ndarray, *, steps: int = 3, angle: float = np.pi / 4) -> np.ndarray:
    out = np.asarray(weight, dtype=np.float32).copy()
    cols = out.shape[1]
    for level in range(max(int(steps), 1)):
        stride = 2**level
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        for start in range(0, cols - stride, 2 * stride):
            left = out[:, start : start + stride].copy()
            right = out[:, start + stride : min(start + 2 * stride, cols)].copy()
            width = min(left.shape[1], right.shape[1])
            if width <= 0:
                continue
            out[:, start : start + width] = c * left[:, :width] + s * right[:, :width]
            out[:, start + stride : start + stride + width] = -s * left[:, :width] + c * right[:, :width]
    return out


def learned_butterfly_rotate_columns(weight: np.ndarray, *, steps: int = 16, lr: float = 0.05) -> np.ndarray:
    try:
        import torch
    except ImportError:
        return butterfly_rotate_columns(weight, steps=3)
    w = torch.as_tensor(np.asarray(weight, dtype=np.float32))
    angle = torch.nn.Parameter(torch.tensor(0.7853982))
    optimizer = torch.optim.Adam([angle], lr=float(lr))
    for _ in range(max(int(steps), 1)):
        optimizer.zero_grad()
        rotated = torch.as_tensor(butterfly_rotate_columns(w.detach().cpu().numpy(), steps=3, angle=float(angle.detach().cpu())))
        col_norm = rotated.norm(dim=0)
        loss = col_norm.max() / torch.clamp(col_norm.median(), min=1e-6)
        # Straight-through scalar search: finite-difference signal around current angle.
        eps = 1e-3
        plus = torch.as_tensor(butterfly_rotate_columns(w.detach().cpu().numpy(), steps=3, angle=float(angle.detach().cpu() + eps)))
        minus = torch.as_tensor(butterfly_rotate_columns(w.detach().cpu().numpy(), steps=3, angle=float(angle.detach().cpu() - eps)))
        plus_loss = plus.norm(dim=0).max() / torch.clamp(plus.norm(dim=0).median(), min=1e-6)
        minus_loss = minus.norm(dim=0).max() / torch.clamp(minus.norm(dim=0).median(), min=1e-6)
        angle.grad = ((plus_loss - minus_loss) / (2 * eps)).reshape_as(angle)
        optimizer.step()
        if not np.isfinite(float(loss)):
            break
    return butterfly_rotate_columns(weight, steps=3, angle=float(angle.detach().cpu()))


def rotation_metrics(weight: np.ndarray, rotated: np.ndarray, *, rotation_type: str) -> dict[str, object]:
    before = outlier_channel_metrics(weight)
    after = outlier_channel_metrics(rotated)
    before_norm = float(np.linalg.norm(np.asarray(weight, dtype=np.float32)))
    after_norm = float(np.linalg.norm(np.asarray(rotated, dtype=np.float32)))
    return {
        "rotation_type": rotation_type,
        "relative_norm_change": abs(after_norm - before_norm) / max(before_norm, 1e-12),
        "in_channel_max_over_median_before": before["in_channel_max_over_median"],
        "in_channel_max_over_median_after": after["in_channel_max_over_median"],
        "out_channel_max_over_median_before": before["out_channel_max_over_median"],
        "out_channel_max_over_median_after": after["out_channel_max_over_median"],
        "in_channel_outlier_count_before": before["in_channel_outlier_count"],
        "in_channel_outlier_count_after": after["in_channel_outlier_count"],
        "out_channel_outlier_count_before": before["out_channel_outlier_count"],
        "out_channel_outlier_count_after": after["out_channel_outlier_count"],
    }
