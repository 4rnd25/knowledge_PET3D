"""
Created on Nov 19 2024 12:37

@author: ISAC - pettirsch
"""

import torch
import numpy as np

from utils.general import scale_coords


def calc_3d_output(pred, bottom_map, cam_pos, cls_mean_lookup, img_shape = (640,480)):
    """
    :param pred: n,13 Tensor [xyxy, conf, cls, kpt, d_length, d_width, d_height, s_angle, c_angle]
    :param bottom_map: 640,480,3 Tensor
    :param cam_pos: 3,1 Tensor

    :return: n,15 Tensor [xyxy, conf, cls, kpt_img_x, kpt_img_y, kpt_world_x, kpt_world_y, kpt_world_z, length, width, height, yaw]
    """

    # Stack predictions from all imagees, pred is a list of n,13 Tensors and add image index
    pred_all = torch.cat(
        [torch.cat([p, torch.ones((p.shape[0], 1), device=p.device) * i], dim=1) for i, p in enumerate(pred)], dim=0)

    if pred_all.shape[0] == 0:
        return []

    # Scale 2D boxes
    scale_coords((bottom_map.shape[1], bottom_map.shape[0]), pred_all[:, :4], img_shape)  # native-space pred

    # Extract bottom, center image, dimensions, and angles
    pred_bc_img = pred_all[:, 6:8]
    pred_dim = pred_all[:, 8:11]
    pred_obs_angle_sin = pred_all[:, 11]
    pred_obs_angle_cos = pred_all[:, 12]

    # Handle padding
    gain = min(img_shape[0] / bottom_map.shape[1], img_shape[1] / bottom_map.shape[0])  # gain  = old / new
    pad = (img_shape[1] - bottom_map.shape[0] * gain) / 2, (img_shape[0] - bottom_map.shape[1] * gain) / 2  # wh padding
    pred_bc_img[:, 0] -= pad[0]
    pred_bc_img[:, 1] -= pad[1]

    # Filter bottom center image coordinates
    pred_bc_img_x = torch.round(pred_bc_img[:, 0])
    pred_bc_img_y = torch.round(pred_bc_img[:, 1])
    out_of_bounds = (pred_bc_img_x < 0) | (pred_bc_img_x >= bottom_map.shape[0]) | (pred_bc_img_y < 0) | (
            pred_bc_img_y >= bottom_map.shape[1])
    pred_bc_img_x_valid = pred_bc_img_x[~out_of_bounds].long()
    pred_bc_img_y_valid = pred_bc_img_y[~out_of_bounds].long()

    # Get BC world
    bottom_map = bottom_map.to(pred_bc_img.device)
    cam_pos = cam_pos.to(pred_bc_img.device)
    pred_bc_world = bottom_map[pred_bc_img_x_valid, pred_bc_img_y_valid, :] + cam_pos

    pred_bc_world_x = pred_bc_world[:, 0]
    pred_bc_world_y = pred_bc_world[:, 1]
    pred_bc_world_z = pred_bc_world[:, 2]
    in_bounds_world = (pred_bc_world_x >= 0) & (pred_bc_world_y >= 0) & (pred_bc_world_z >= 0)
    pred_bc_world_x_valid = pred_bc_world_x[in_bounds_world]
    pred_bc_world_y_valid = pred_bc_world_y[in_bounds_world]
    pred_bc_world_z_valid = pred_bc_world_z[in_bounds_world]

    # Expand camera position
    cam_pos_expanded = cam_pos.unsqueeze(0).repeat(pred_bc_world_x_valid.shape[0], 1).to(pred_bc_img.device)
    osin_valid = pred_obs_angle_sin[~out_of_bounds][in_bounds_world]
    ocos_valid = pred_obs_angle_cos[~out_of_bounds][in_bounds_world]
    magnitude = torch.sqrt(osin_valid ** 2 + ocos_valid ** 2 + 1e-8)  # Add small epsilon for numerical stability
    sin_alpha = osin_valid / magnitude
    cos_alpha = ocos_valid / magnitude
    obs_angle = torch.atan2(sin_alpha, cos_alpha)
    #obs_angle = torch.atan2(osin_valid, ocos_valid)

    atan_xy = torch.atan2(pred_bc_world_x_valid - cam_pos_expanded[:, 0],
                          pred_bc_world_y_valid - cam_pos_expanded[:, 1])
    yaw_angle = atan_xy + obs_angle

    # dim 3d
    delta_lenght = pred_dim[:, 0][~out_of_bounds][in_bounds_world]
    delta_width = pred_dim[:, 1][~out_of_bounds][in_bounds_world]
    delta_height = pred_dim[:, 2][~out_of_bounds][in_bounds_world]

    # Get class
    cls = pred_all[:, 5].long()
    cls = cls[~out_of_bounds][in_bounds_world]

    # Get mean values
    cls_mean_lookup = cls_mean_lookup.to(pred_bc_img.device)
    mean_values = cls_mean_lookup[cls.long()]
    mean_lenght = mean_values[:,0]
    mean_width = mean_values[:,1]
    mean_height = mean_values[:,2]
    length = mean_lenght * torch.exp(delta_lenght)
    width = mean_width * torch.exp(delta_width)
    height = mean_height * torch.exp(delta_height)

    if pred_all.shape[0] == 0:
        return []

    xmin_img = pred_all[:, 0][~out_of_bounds][in_bounds_world]
    ymin_img = pred_all[:, 1][~out_of_bounds][in_bounds_world]
    xmax_img = pred_all[:, 2][~out_of_bounds][in_bounds_world]
    ymax_img = pred_all[:, 3][~out_of_bounds][in_bounds_world]
    conf = pred_all[:, 4][~out_of_bounds][in_bounds_world]

    # Create output tensor [xyxy, conf, cls, kpt_img_x, kpt_img_y, kpt_world_x, kpt_world_y, kpt_world_z, length, width, height, yaw]
    output = torch.cat([xmin_img.unsqueeze(1), ymin_img.unsqueeze(1), xmax_img.unsqueeze(1), ymax_img.unsqueeze(1),
                       conf.unsqueeze(1), cls.unsqueeze(1), pred_bc_img_x_valid.unsqueeze(1),
                       pred_bc_img_y_valid.unsqueeze(1),
                       pred_bc_world_x_valid.unsqueeze(1), pred_bc_world_y_valid.unsqueeze(1),
                       pred_bc_world_z_valid.unsqueeze(1), length.unsqueeze(1), width.unsqueeze(1), height.unsqueeze(1),
                       yaw_angle.unsqueeze(1)], dim = 1)

    # Create list of output tensors for each image
    image_indices = pred_all[:, 13][~out_of_bounds][in_bounds_world]
    output_list = []
    for i in range(len(pred)):
        indices_image = torch.nonzero(image_indices == i, as_tuple=False).squeeze()
        image_output = output[indices_image]
        if len(image_output.shape) == 1:
            image_output = image_output.unsqueeze(0)
        output_list.append(image_output)

    return output_list


def calc_3d_output_numpy(pred, bottom_map, cam_pos, cls_mean_lookup, img_shape=(640, 480)):
    """
    :param pred: List of n,13 NumPy arrays [xyxy, conf, cls, kpt, d_length, d_width, d_height, s_angle, c_angle]
    :param bottom_map: 640,480,3 NumPy array
    :param cam_pos: 3,1 NumPy array

    :return: List of n,15 NumPy arrays [xyxy, conf, cls, kpt_img_x, kpt_img_y, kpt_world_x, kpt_world_y, kpt_world_z, length, width, height, yaw]
    """
    # Stack predictions from all images and add image index
    pred_all = np.vstack([np.hstack([p, np.ones((p.shape[0], 1)) * i]) for i, p in enumerate(pred)])

    if pred_all.shape[0] == 0:
        return []

    # Scale 2D boxes
    scale_coords((bottom_map.shape[1], bottom_map.shape[0]), pred_all[:, :4], img_shape)  # Native-space pred

    # Extract bottom center image, dimensions, and angles
    pred_bc_img = pred_all[:, 6:8]
    pred_dim = pred_all[:, 8:11]
    pred_obs_angle_sin = pred_all[:, 11]
    pred_obs_angle_cos = pred_all[:, 12]

    # Handle padding
    gain = min(img_shape[0] / bottom_map.shape[1], img_shape[1] / bottom_map.shape[0])  # Gain = old / new
    pad = ((img_shape[1] - bottom_map.shape[0] * gain) / 2,
           (img_shape[0] - bottom_map.shape[1] * gain) / 2)  # Wh padding
    pred_bc_img[:, 0] -= pad[0]
    pred_bc_img[:, 1] -= pad[1]

    # Filter bottom center image coordinates
    pred_bc_img_x = np.round(pred_bc_img[:, 0]).astype(int)
    pred_bc_img_y = np.round(pred_bc_img[:, 1]).astype(int)
    out_of_bounds = (
        (pred_bc_img_x < 0) | (pred_bc_img_x >= bottom_map.shape[0]) |
        (pred_bc_img_y < 0) | (pred_bc_img_y >= bottom_map.shape[1])
    )
    pred_bc_img_x_valid = pred_bc_img_x[~out_of_bounds]
    pred_bc_img_y_valid = pred_bc_img_y[~out_of_bounds]

    # Get BC world
    pred_bc_world = bottom_map[pred_bc_img_x_valid, pred_bc_img_y_valid, :] + cam_pos.squeeze()
    pred_bc_world_x = pred_bc_world[:, 0]
    pred_bc_world_y = pred_bc_world[:, 1]
    pred_bc_world_z = pred_bc_world[:, 2]
    in_bounds_world = (pred_bc_world_x >= 0) & (pred_bc_world_y >= 0) & (pred_bc_world_z >= 0)

    pred_bc_world_x_valid = pred_bc_world_x[in_bounds_world]
    pred_bc_world_y_valid = pred_bc_world_y[in_bounds_world]
    pred_bc_world_z_valid = pred_bc_world_z[in_bounds_world]

    # Expand camera position
    cam_pos_expanded = np.repeat(cam_pos.T, pred_bc_world_x_valid.shape[0], axis=0)
    osin_valid = pred_obs_angle_sin[~out_of_bounds][in_bounds_world]
    ocos_valid = pred_obs_angle_cos[~out_of_bounds][in_bounds_world]
    magnitude = np.sqrt(osin_valid ** 2 + ocos_valid ** 2 + 1e-8)  # Add small epsilon for numerical stability
    sin_alpha = osin_valid / magnitude
    cos_alpha = ocos_valid / magnitude
    obs_angle = np.arctan2(sin_alpha, cos_alpha)

    atan_xy = np.arctan2(pred_bc_world_x_valid - cam_pos_expanded[:, 0],
                         pred_bc_world_y_valid - cam_pos_expanded[:, 1])
    yaw_angle = atan_xy + obs_angle

    # Dimensions 3D
    delta_length = pred_dim[:, 0][~out_of_bounds][in_bounds_world]
    delta_width = pred_dim[:, 1][~out_of_bounds][in_bounds_world]
    delta_height = pred_dim[:, 2][~out_of_bounds][in_bounds_world]

    # Get class
    cls = pred_all[:, 5].astype(int)
    cls = cls[~out_of_bounds][in_bounds_world]

    # Get mean values
    mean_values = cls_mean_lookup[cls]
    mean_length = mean_values[:, 0]
    mean_width = mean_values[:, 1]
    mean_height = mean_values[:, 2]
    length = mean_length * np.exp(delta_length)
    width = mean_width * np.exp(delta_width)
    height = mean_height * np.exp(delta_height)

    if pred_all.shape[0] == 0:
        return []

    xmin_img = pred_all[:, 0][~out_of_bounds][in_bounds_world]
    ymin_img = pred_all[:, 1][~out_of_bounds][in_bounds_world]
    xmax_img = pred_all[:, 2][~out_of_bounds][in_bounds_world]
    ymax_img = pred_all[:, 3][~out_of_bounds][in_bounds_world]
    conf = pred_all[:, 4][~out_of_bounds][in_bounds_world]

    # Create output array [xyxy, conf, cls, kpt_img_x, kpt_img_y, kpt_world_x, kpt_world_y, kpt_world_z, length, width, height, yaw]
    output = np.column_stack([
        xmin_img, ymin_img, xmax_img, ymax_img, conf, cls,
        pred_bc_img_x_valid, pred_bc_img_y_valid,
        pred_bc_world_x_valid, pred_bc_world_y_valid, pred_bc_world_z_valid,
        length, width, height, yaw_angle
    ])

    # Create list of output arrays for each image
    image_indices = pred_all[:, 13][~out_of_bounds][in_bounds_world]
    output_list = []
    for i in range(len(pred)):
        indices_image = np.where(image_indices == i)[0]
        image_output = output[indices_image]
        output_list.append(image_output)

    return output_list
