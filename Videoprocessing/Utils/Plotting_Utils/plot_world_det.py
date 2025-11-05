"""
Created on Dec 17 2024 09:26

@author: ISAC - pettirsch
"""
import numpy as np
import open3d as o3d

def plot_world_det(corners_3d, ax, color, line_thickness=1):

    if max(color) > 1:
        color = [c / 255 for c in color]
        # RGB to BGR
        color = color[::-1]


    # Draw bottom face
    ax.plot([corners_3d[0, 0], corners_3d[1, 0]], [corners_3d[0, 1], corners_3d[1, 1]],
            [corners_3d[0, 2], corners_3d[1, 2]], color=color, linewidth=line_thickness)
    ax.plot([corners_3d[1, 0], corners_3d[2, 0]], [corners_3d[1, 1], corners_3d[2, 1]],
            [corners_3d[1, 2], corners_3d[2, 2]], color=color, linewidth=line_thickness)
    ax.plot([corners_3d[2, 0], corners_3d[3, 0]], [corners_3d[2, 1], corners_3d[3, 1]],
            [corners_3d[2, 2], corners_3d[3, 2]], color=color, linewidth=line_thickness)
    ax.plot([corners_3d[3, 0], corners_3d[0, 0]], [corners_3d[3, 1], corners_3d[0, 1]],
            [corners_3d[3, 2], corners_3d[0, 2]], color=color, linewidth=line_thickness)

    # Draw top face
    ax.plot([corners_3d[4, 0], corners_3d[5, 0]], [corners_3d[4, 1], corners_3d[5, 1]],
            [corners_3d[4, 2], corners_3d[5, 2]], color=color, linewidth=line_thickness)
    ax.plot([corners_3d[5, 0], corners_3d[6, 0]], [corners_3d[5, 1], corners_3d[6, 1]],
            [corners_3d[5, 2], corners_3d[6, 2]], color=color, linewidth=line_thickness)
    ax.plot([corners_3d[6, 0], corners_3d[7, 0]], [corners_3d[6, 1], corners_3d[7, 1]],
            [corners_3d[6, 2], corners_3d[7, 2]], color=color, linewidth=line_thickness)
    ax.plot([corners_3d[7, 0], corners_3d[4, 0]], [corners_3d[7, 1], corners_3d[4, 1]],
            [corners_3d[7, 2], corners_3d[4, 2]], color=color, linewidth=line_thickness)

    # Draw lines connecting top and bottom face
    ax.plot([corners_3d[0, 0], corners_3d[4, 0]], [corners_3d[0, 1], corners_3d[4, 1]],
            [corners_3d[0, 2], corners_3d[4, 2]], color=color, linewidth=line_thickness)
    ax.plot([corners_3d[1, 0], corners_3d[5, 0]], [corners_3d[1, 1], corners_3d[5, 1]],
            [corners_3d[1, 2], corners_3d[5, 2]], color=color, linewidth=line_thickness)
    ax.plot([corners_3d[2, 0], corners_3d[6, 0]], [corners_3d[2, 1], corners_3d[6, 1]],
            [corners_3d[2, 2], corners_3d[6, 2]], color=color, linewidth=line_thickness)
    ax.plot([corners_3d[3, 0], corners_3d[7, 0]], [corners_3d[3, 1], corners_3d[7, 1]],
            [corners_3d[3, 2], corners_3d[7, 2]], color=color, linewidth=line_thickness)

    return ax


def set_axes_equal(ax):
    '''
    Set equal scaling for 3D axes to maintain aspect ratio.
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


def get_axes_data(ax):
    """
    Retrieve x, y, and z data currently plotted on a 3D axis.
    Works for Line and Scatter plots.
    """
    x_data, y_data, z_data = [], [], []

    # Loop through all plotted lines (for line plots)
    for line in ax.lines:
        x, y, z = line._verts3d  # Extract vertex data
        x_data.extend(x)
        y_data.extend(y)
        z_data.extend(z)

    # # Loop through all scatter or surface collections (if applicable)
    # for collection in ax.collections:
    #     segments = collection._segments3d  # For surface plots or other 3D plots
    #     for seg in segments:
    #         x_data.extend(seg[0])
    #         y_data.extend(seg[1])
    #         z_data.extend(seg[2])

    return np.array(x_data), np.array(y_data), np.array(z_data)


def set_narrow_axes_limits_from_data(ax, margin=0.1):
    """
    Dynamically set axes limits based on the data currently plotted on ax.
    """
    x_data, y_data, z_data = get_axes_data(ax)

    if len(x_data) == 0 or len(y_data) == 0 or len(z_data) == 0:
        print("No data found on the axes.")
        return

    # Calculate limits with margin
    x_margin = (x_data.max() - x_data.min()) * margin
    y_margin = (y_data.max() - y_data.min()) * margin
    z_margin = (z_data.max() - z_data.min()) * margin

    ax.set_xlim(x_data.min() - x_margin, x_data.max() + x_margin)
    ax.set_ylim(y_data.min() - y_margin, y_data.max() + y_margin)
    ax.set_zlim(z_data.min() - z_margin, z_data.max() + z_margin)

def set_axes_limits_from_thresholds(ax, x_min, x_max, y_min, y_max, z_min, z_max, margin=0.1):
    """
    Set axes limits based on the given thresholds.
    """
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin
    z_margin = (z_max - z_min) * margin

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_zlim(z_min - z_margin, z_max + z_margin)


def add_street_markings(ax, street_marking_path = None, marking=None, x_min=None, x_max=None, y_min=None, y_max=None,
                        BEV=False):

    if x_min is None:
        # Get data
        x_data, y_data, z_data = get_axes_data(ax)

        if len(x_data) == 0 or len(y_data) == 0 or len(z_data) == 0:
            print("No data found on the axes.")
            return

        x_min = x_data.min()-15
        x_max = x_data.max()+15
        y_min = y_data.min()-15
        y_max = y_data.max()+15

    if marking is None:
        # Load point cloud data
        pcd = o3d.io.read_point_cloud(street_marking_path)  # Load point cloud data
        # Convert point cloud to NumPy array
        marking = np.asarray(pcd.points)  # Extract XYZ coordinates as NumPy array

    # Get all markings in the range of the current plot
    markings_in_range = marking[(marking[:, 0] > x_min) & (marking[:, 0] < x_max) & (marking[:, 1] > y_min) & (marking[:, 1] < y_max)]

    # Plot markings
    if not BEV:
        ax.scatter(markings_in_range[:, 0], markings_in_range[:, 1], markings_in_range[:, 2], c='grey', s=1)
    else:
        ax.scatter(markings_in_range[:, 0], markings_in_range[:, 1], c='grey', s=1)


