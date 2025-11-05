

import numpy as np

def get_bev_corners(centers, dimensions, yaws):
    """
    Compute the 4 corners of oriented bounding boxes in BEV.

    centers: (N, 2) array of x, y positions.
    dimensions: (N, 2) array of length, width.
    yaws: (N,) array of yaw angles in radians.

    Returns an (N, 4, 2) array of box corners.
    """
    num_boxes = centers.shape[0]
    corners = np.zeros((num_boxes, 4, 2))

    # Half dimensions
    l = dimensions[:, 0] / 2.0
    w = dimensions[:, 1] / 2.0

    # Box corners in local coordinates (before rotation)
    local_corners = np.array([
        [1, -1],
        [1, 1],
        [-1, 1],
        [-1, -1]
    ])  # Shape: (4,2)

    # scaled_corners = np.zeros((num_boxes, 4, 2))
    # scaled_corners[:, :, 0] = local_corners[:, 0] * l[:, None]
    # scaled_corners[:, :, 1] = local_corners[:, 1] * w[:, None]

    # # Add center to corners
    # corners[:, :, 0] = centers[:, 0][:, None] + scaled_corners[:, :, 0]
    # corners[:, :, 1] = centers[:, 1][:, None] + scaled_corners[:, :, 1]
    #
    # # Compute rotated corners
    # cos_yaw = np.cos(yaws)
    # sin_yaw = np.sin(yaws)
    #
    # rotation_matrices = np.zeros((num_boxes, 2, 2))
    # rotation_matrices[:, 0, 0] = cos_yaw
    # rotation_matrices[:, 0, 1] = -sin_yaw
    # rotation_matrices[:, 1, 0] = sin_yaw
    # rotation_matrices[:, 1, 1] = cos_yaw
    #
    # # Rotate corners
    # for i in range(num_boxes):
    #     corners[i] = np.dot(rotation_matrices[i], corners[i].T).T
    #
    # return corners

    # Compute rotated corners
    cos_yaw = np.cos(yaws)
    sin_yaw = np.sin(yaws)

    for i in range(4):
        x_offset = local_corners[i, 0] * l
        y_offset = local_corners[i, 1] * w

        corners[:, i, 0] = centers[:, 0] + x_offset * cos_yaw - y_offset * sin_yaw
        corners[:, i, 1] = centers[:, 1] + x_offset * sin_yaw + y_offset * cos_yaw

    return corners




    # # # rotation_matrices[:, 0, 0] = cos_yaw
    # # # rotation_matrices[:, 0, 1] = -sin_yaw
    # # # rotation_matrices[:, 1, 0] = sin_yaw
    # # # rotation_matrices[:, 1, 1] = cos_yaw
    # #
    # # for i in range(4):
    # #     x_offset = local_corners[i, 0] * l
    # #     y_offset = local_corners[i, 1] * w
    # #
    # #     corners[:, i, 0] = centers[:, 0] + x_offset * cos_yaw - y_offset * sin_yaw
    # #     corners[:, i, 1] = centers[:, 1] + x_offset * sin_yaw + y_offset * cos_yaw
    #
    # return corners