"""
Created on Nov 19 2024 12:37

@author: ISAC - pettirsch
"""


import torch
import numpy as np


def get_3d_corners(bc_3d = None, dims = None, yaws = None, local=False):
    """

    :param bc_3d: n,3
    :param dims: n,3
    :param yaws: n,
    :return:
    """

    # # Check if dims is numpy array and convert to torch tensor if necessary
    # if isinstance(dims, torch.Tensor):
    #     dims = dims.float()
    # elif isinstance(dims, (list, tuple)):
    #     dims = torch.tensor(dims, dtype=torch.float64)
    # elif isinstance(dims, np.ndarray):
    #     dims = torch.tensor(dims, dtype=torch.float64)
    # else:
    #     raise TypeError("dims must be a torch tensor, list, or tuple")


    if bc_3d is None:
        bc_3d = torch.tensor([[0, 0, 0]], dtype=torch.float64).repeat(dims.shape[0], 1)
        bc_3d = bc_3d.to(dims.device)

    half_dim = dims / 2
    corners_Loc_bc = torch.tensor([
        [1, -1, 0],  # front-right-bottom
        [1, 1, 0],  # front-left-bottom
        [-1, 1, 0],  # back-left-bottom
        [-1, -1, 0],  # back-right-bottom
        [1, -1, 2],  # front-right-top
        [1, 1, 2],  # front-left-top
        [-1, 1, 2],  # back-left-top
        [-1, -1, 2]  # back-right-top
    ], dtype=torch.float64).unsqueeze(0)  # Shape: (1, 8, 3)

    # Repeat corners loc bc to have first dimension n with n = first dimension of dims
    corners_Loc_bc = corners_Loc_bc.repeat(bc_3d.shape[0], 1, 1)
    corners_Loc_bc = corners_Loc_bc.to(dims.device)

    # Scale the local corners by dimensions
    scaled_corners = half_dim.unsqueeze(1) * corners_Loc_bc  # Shape: (n, 8, 3)
    # scaled_corners = half_dim.unsqueeze(1) * corners_Loc_bc.unsqueeze(0)

    # Compute rotation matrices
    cos_yaw = torch.cos(yaws)
    sin_yaw = torch.sin(yaws)
    rotation_matrices = torch.zeros((bc_3d.shape[0], 3, 3), dtype=torch.float64, device=dims.device)
    rotation_matrices[:, 0, 0] = cos_yaw
    rotation_matrices[:, 0, 1] = -sin_yaw
    rotation_matrices[:, 1, 0] = sin_yaw
    rotation_matrices[:, 1, 1] = cos_yaw
    rotation_matrices[:, 2, 2] = 1.0
    rotation_matrices = rotation_matrices.to(dims.device)
    if len(rotation_matrices.shape) == 4:
        rotation_matrices = rotation_matrices.squeeze(-1)

    # Rotate local corners
    scaled_corners = scaled_corners.permute(0, 2, 1)  # Shape: (n, 3, 8)
    rotated_corners = torch.bmm(rotation_matrices, scaled_corners)  # Shape: (n, 3, 8)
    rotated_corners = rotated_corners.permute(0, 2, 1)  # Back to (n, 8, 3)

    # Translate to world coordinates
    bc_3d_all_corn = bc_3d.unsqueeze(1).repeat(1, 8, 1)
    corners_World_bc = bc_3d_all_corn+ rotated_corners  # Shape: (n, 8, 3)

    return corners_World_bc


import numpy as np


def get_3d_corners_numpy(bc_3d=None, dims=None, yaws=None, local=False):
    """
    Compute the 3D corners of bounding boxes given the center, dimensions, and rotation (yaw).

    :param bc_3d: ndarray of shape (n, 3), bounding box centers in 3D space
    :param dims: ndarray of shape (n, 3), dimensions of bounding boxes (length, width, height)
    :param yaws: ndarray of shape (n,), yaw angles in radians
    :param local: bool, whether to compute in local coordinates
    :return: ndarray of shape (n, 8, 3), 3D coordinates of the 8 bounding box corners
    """
    if bc_3d is None:
        bc_3d = np.zeros((dims.shape[0], 3))

    # If bc_3d, dims is shape 3, convert to 1,3
    if bc_3d.ndim == 1:
        bc_3d = np.expand_dims(bc_3d, axis=0)
    if dims.ndim == 1:
        dims = np.expand_dims(dims, axis=0)
    # if yaws is float convert to array
    if isinstance(yaws, float):
        yaws = np.array([yaws])
    elif isinstance(yaws, (list, tuple)):
        yaws = np.array(yaws)

    half_dim = dims / 2

    # Define the corners in local bounding box coordinates
    corners_Loc_bc = np.array([
        [1, -1, 0],  # front-right-bottom
        [1, 1, 0],  # front-left-bottom
        [-1, 1, 0],  # back-left-bottom
        [-1, -1, 0],  # back-right-bottom
        [1, -1, 2],  # front-right-top
        [1, 1, 2],  # front-left-top
        [-1, 1, 2],  # back-left-top
        [-1, -1, 2]  # back-right-top
    ], dtype=np.float64)

    # Repeat corners for each bounding box
    n = bc_3d.shape[0]
    corners_Loc_bc = np.tile(corners_Loc_bc, (n, 1, 1))  # Shape: (n, 8, 3)

    # Scale corners by half dimensions
    scaled_corners = half_dim[:, np.newaxis, :] * corners_Loc_bc  # Shape: (n, 8, 3)

    # Compute rotation matrices
    cos_yaw = np.cos(yaws)
    sin_yaw = np.sin(yaws)

    rotation_matrices = np.zeros((n, 3, 3), dtype=np.float64)
    rotation_matrices[:, 0, 0] = cos_yaw
    rotation_matrices[:, 0, 1] = -sin_yaw
    rotation_matrices[:, 1, 0] = sin_yaw
    rotation_matrices[:, 1, 1] = cos_yaw
    rotation_matrices[:, 2, 2] = 1.0  # Z-axis remains unchanged

    # Rotate the corners
    rotated_corners = np.einsum('nij,nkj->nki', rotation_matrices, scaled_corners)  # Shape: (n, 8, 3)

    # Translate to world coordinates
    bc_3d_all_corn = bc_3d[:, np.newaxis, :]  # Shape: (n, 1, 3)
    corners_World_bc = bc_3d_all_corn + rotated_corners  # Shape: (n, 8, 3)

    return corners_World_bc
