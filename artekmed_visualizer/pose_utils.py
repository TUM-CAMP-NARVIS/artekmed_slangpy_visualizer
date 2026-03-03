"""Convert artekmed RigidTransform poses to 4x4 matrices."""

from __future__ import annotations

from typing import Optional

import numpy as np

from artekmed_dataset_reader import RigidTransform


def rigid_transform_to_matrix(pose: Optional[RigidTransform]) -> np.ndarray:
    """Convert a RigidTransform (quaternion + translation) to a 4x4 row-major matrix.

    Args:
        pose: A RigidTransform with translation (Vec3) and rotation (Quaternion
            in x, y, z, w order), or None for identity.

    Returns:
        4x4 float32 matrix (row-major, suitable for Slang ``mul(M, v)``).
    """
    if pose is None:
        return np.eye(4, dtype=np.float32)

    x, y, z, w = pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w
    tx, ty, tz = pose.translation.x, pose.translation.y, pose.translation.z

    # Rotation matrix from quaternion (row-major)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    M = np.eye(4, dtype=np.float32)
    M[0, 0] = 1.0 - 2.0 * (yy + zz)
    M[0, 1] = 2.0 * (xy - wz)
    M[0, 2] = 2.0 * (xz + wy)
    M[0, 3] = tx
    M[1, 0] = 2.0 * (xy + wz)
    M[1, 1] = 1.0 - 2.0 * (xx + zz)
    M[1, 2] = 2.0 * (yz - wx)
    M[1, 3] = ty
    M[2, 0] = 2.0 * (xz - wy)
    M[2, 1] = 2.0 * (yz + wx)
    M[2, 2] = 1.0 - 2.0 * (xx + yy)
    M[2, 3] = tz
    return M
