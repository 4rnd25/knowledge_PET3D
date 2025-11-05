"""
Created on Mar 05 2025 14:40

@author: ISAC - pettirsch
"""
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

class MultiFileTrajectoryDataset(Dataset):
    def __init__(self, files: list[str], class_map, mask_ratio=0.2, noise_std=0.3):
        self.files = files
        self.class_map = class_map
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std
        self.index = []

        for file in files:
            df = pd.read_csv(file, usecols=['idVehicle', 'ObjectClass', 'FrameTimeStamp']).drop_duplicates(subset=['idVehicle'])
            for vehicle_id in df['idVehicle'].unique():
                self.index.append((file, vehicle_id))

    def __len__(self):
        return len(self.index)

    def load_trajectory(self, file, vehicle_id):
        df = pd.read_csv(file)
        traj = df[df['idVehicle'] == vehicle_id].copy()

        traj = to_ego_coordinates(traj)
        traj = normalize_time(traj)
        traj = compute_displacements(traj)

        return traj

    def __getitem__(self, idx):
        file, vehicle_id = self.index[idx]
        traj = self.load_trajectory(file, vehicle_id)
        masked_traj = mask_and_corrupt(traj, self.mask_ratio, self.noise_std)

        cls = traj['ObjectClass'].iloc[0]
        cls_id = self.class_map[cls]

        return masked_traj, traj, cls_id


def to_ego_coordinates(traj):
    """
    1. Rotate all coordinates so that the initial yaw aligns with the +Y axis (forward).
    2. After rotation, shift the coordinates so the first point is (0, 0).

    Parameters:
        traj (DataFrame): A DataFrame containing posX, posY, Yaw (in radians).

    Returns:
        traj (DataFrame): Transformed DataFrame with ego-aligned coordinates.
    """

    # Get initial yaw (assumed to be in radians)
    initial_yaw = traj['YawFit'].iloc[0]

    # Build rotation matrix to align initial yaw with +Y axis (vehicle forward)
    cos_yaw = np.cos(-initial_yaw)  # Negative to rotate the world into vehicle frame
    sin_yaw = np.sin(-initial_yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])

    # Rotate all positions
    xy = traj[['posXFit', 'posYFit']].values
    rotated_xy = xy @ rotation_matrix.T

    # Overwrite with rotated coordinates
    traj['posXFit'], traj['posYFit'] = rotated_xy[:, 0], rotated_xy[:, 1]

    # Subtract first point after rotation to make first point (0,0)
    origin = traj[['posXFit', 'posYFit']].iloc[0].values
    traj[['posXFit', 'posYFit']] -= origin

    return traj

def normalize_time(traj):
    """
    Convert mixed-format timestamps into consistent datetime objects,
    then compute relative time (deltaT) in seconds.
    """

    # Step 1: Convert FrameTimeStamp to datetime (allowing mixed formats)
    traj['FrameTimeStamp'] = pd.to_datetime(traj['FrameTimeStamp'], format='mixed', errors='coerce')

    # Step 2: Convert microseconds column (ensure it's numeric)
    traj['FrameTimeStamp_MicroSec'] = pd.to_numeric(traj['FrameTimeStamp_MicroSec'], errors='coerce').fillna(0)

    # Step 3: Add microseconds to full timestamp
    traj['FullTimeStamp'] = traj['FrameTimeStamp'] + pd.to_timedelta(traj['FrameTimeStamp_MicroSec'], unit='us')

    # Step 4: Convert to seconds relative to first timestamp
    start_time = traj['FullTimeStamp'].iloc[0]
    traj['FrameTimeStamp'] = (traj['FullTimeStamp'] - start_time).dt.total_seconds()

    # Drop temporary column
    traj.drop(columns=['FullTimeStamp'], inplace=True)

    return traj

def compute_displacements(traj):
    """ Compute Δx, Δy. """
    traj['deltaX'] = traj['posXFit'].diff().fillna(0)
    traj['deltaY'] = traj['posYFit'].diff().fillna(0)
    traj['deltaT'] = traj['FrameTimeStamp'].diff().fillna(0)
    return traj

def mask_and_corrupt(
    traj,
    mask_ratio: float = 0.4,
    noise_std: float = 0.1,
    max_mask_length: int = 5,
    debug: bool = False
):
    """
    1. Mask contiguous runs (set Δx,Δy to NaN).
    2. Add Gaussian noise only to the *un*-masked Δx,Δy.
    """

    masked_traj = traj.copy()
    N = len(traj)
    num_to_mask = int(np.ceil(mask_ratio * N))

    # 1) Build a boolean mask of length N
    to_mask = np.zeros(N, dtype=bool)
    masked_count = 0

    while masked_count < num_to_mask:
        start = np.random.randint(0, N)
        length = np.random.randint(1, max_mask_length + 1)
        end = min(start + length, N)

        # how many *new* points would this run cover?
        new_pts = np.sum(~to_mask[start:end])
        if new_pts == 0:
            continue

        pts_needed = num_to_mask - masked_count
        if new_pts > pts_needed:
            # mask exactly the number we still need
            avail = np.where(~to_mask[start:end])[0]
            choose = np.random.choice(avail, pts_needed, replace=False)
            to_mask[start + choose] = True
            masked_count += pts_needed
        else:
            to_mask[start:end] = True
            masked_count += new_pts

    # 2) Apply the mask
    masked_traj.loc[to_mask, ['deltaX', 'deltaY']] = np.nan

    if debug:
        print(f"Masking {to_mask.sum()} / {N} points "
              f"({100 * to_mask.sum()/N:.1f}%)")

    # 3) Add noise *only* to the unmasked points
    visible_idx = np.where(~to_mask)[0]
    noise = np.random.normal(0.0, noise_std, size=(len(visible_idx), 2))

    clean_vals = masked_traj.iloc[visible_idx][['deltaX','deltaY']].values
    masked_traj.iloc[visible_idx, masked_traj.columns.get_indexer(['deltaX','deltaY'])] = clean_vals + noise

    return masked_traj