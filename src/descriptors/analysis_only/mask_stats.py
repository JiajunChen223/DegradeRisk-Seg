from __future__ import annotations

from collections import deque

import numpy as np


def _count_connected_components(mask: np.ndarray) -> tuple[int, int]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    num_components = 0
    small_components = 0
    for y in range(h):
        for x in range(w):
            if visited[y, x] or mask[y, x] == 0:
                continue
            num_components += 1
            q: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            size = 0
            while q:
                cy, cx = q.popleft()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and mask[ny, nx] == 1:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            if size <= 4:
                small_components += 1
    return num_components, small_components


def extract_mask_descriptors(y: np.ndarray) -> dict[str, float]:
    mask = (y > 0.5).astype(np.uint8)
    area = float(mask.sum())
    coverage = float(mask.mean())
    if area <= 0:
        return {
            "coverage": coverage,
            "boundary_complexity": 0.0,
            "num_components": 0.0,
            "small_component_ratio": 0.0,
            "largest_component_ratio": 0.0,
        }
    horizontal = np.abs(mask[:, 1:] - mask[:, :-1]).sum()
    vertical = np.abs(mask[1:, :] - mask[:-1, :]).sum()
    perimeter = float(horizontal + vertical)
    components, small = _count_connected_components(mask)
    largest = 0
    if components > 0:
        labels = np.zeros_like(mask, dtype=np.int32)
        label = 0
        h, w = mask.shape
        for yy in range(h):
            for xx in range(w):
                if labels[yy, xx] != 0 or mask[yy, xx] == 0:
                    continue
                label += 1
                q: deque[tuple[int, int]] = deque([(yy, xx)])
                labels[yy, xx] = label
                size = 0
                while q:
                    cy, cx = q.popleft()
                    size += 1
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] == 0 and mask[ny, nx] == 1:
                            labels[ny, nx] = label
                            q.append((ny, nx))
                largest = max(largest, size)
    return {
        "coverage": coverage,
        "boundary_complexity": float(perimeter / max(area, 1.0)),
        "num_components": float(components),
        "small_component_ratio": float(small / max(components, 1)),
        "largest_component_ratio": float(largest / max(area, 1.0)),
    }

