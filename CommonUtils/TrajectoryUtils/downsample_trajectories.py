"""
Created on May 14 2025 11:24

@author: ISAC - pettirsch
"""

import numpy as np

def downsample_by_distance(traj, num_points=5):
    """
    Downsample trajectory based on % of total distance.
    traj: (N, 2) array of x, y
    Returns: (num_points * 2,) flattened array
    """
    traj_2d = traj[:, :2]  # Ensure we only use x, y coordinates
    traj_3d = traj[:, :3]  # Keep the original 3D coordinates if needed

    # Compute cumulative distance along the trajectory
    diffs = np.diff(traj_2d, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    cumdist = np.insert(np.cumsum(dists), 0, 0)

    # Normalize to 0-1
    cumdist /= cumdist[-1]

    # Target distances: evenly spaced
    target_cumdist = np.linspace(0, 1, num_points)

    # Interpolate
    new_points = []
    for t in target_cumdist:
        idx = np.searchsorted(cumdist, t)
        if idx >= len(traj):
            idx = len(traj) - 1
        new_points.append(traj_3d[idx])

    return np.array(new_points)