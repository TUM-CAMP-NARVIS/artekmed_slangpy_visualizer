"""Pack per-point RGB colors into a 2D RGBA texture + UV coordinates."""

from __future__ import annotations

import math

import numpy as np


def pack_colors_to_texture(colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-point RGB colors to a 2D RGBA texture and UV coordinates.

    Args:
        colors: (N, 3) uint8 RGB array.

    Returns:
        texture: (H, W, 4) uint8 RGBA array — roughly square, padded with zeros.
        uvs: (N, 2) float32 array — texel-center UVs mapping each point to
            its color in the texture.
    """
    n = colors.shape[0]
    w = math.ceil(math.sqrt(n))
    h = math.ceil(n / w)

    texture = np.zeros((h, w, 4), dtype=np.uint8)
    indices = np.arange(n)
    rows = indices // w
    cols = indices % w
    texture[rows, cols, :3] = colors
    texture[rows, cols, 3] = 255

    uvs = np.empty((n, 2), dtype=np.float32)
    uvs[:, 0] = (cols + 0.5) / w
    uvs[:, 1] = (rows + 0.5) / h
    return texture, uvs
