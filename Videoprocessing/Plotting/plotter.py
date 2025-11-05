"""
Created on Jan 17 2025 13:11

@author: ISAC - pettirsch
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import open3d as o3d
import time

from Videoprocessing.Utils.Cuboid_calc.calc_3d_corners import get_3d_corners
from Videoprocessing.Utils.Plotting_Utils.plot_3d_box import plot_3d_corners
from Videoprocessing.Utils.Plotting_Utils.plot_world_det import add_street_markings, plot_world_det, set_axes_equal, \
    set_narrow_axes_limits_from_data


class Plotter:
    def __init__(self, plotting_config, detectionZoneBoardersImage, detectionZoneBoardersWorld, verbose=False):

        self.plotWorldCoord = plotting_config["plotWorldOutput"]
        self.detectionZoneBoadersImage = detectionZoneBoardersImage

        if self.plotWorldCoord:
            self.detectionZoneBoadersWorld = detectionZoneBoardersWorld
            self.streetMarkingsPath = plotting_config["streetMarkingPath"]
        else:
            self.detectionZoneBoadersWorld = None
            self.streetMarkingsPath = None
            self.cached_markings = None

        self.verbose = verbose
        self.colors = plotting_config["colors"]
        self.colors_thermal = {}
        for key in self.colors.keys():
            self.colors_thermal[key] = [self.colors[key][2], self.colors[key][1], self.colors[key][0]]


        if self.plotWorldCoord:
            self.fig, self.ax = plt.subplots(subplot_kw={'projection': '3d'})
            self.fig.set_size_inches(640 / 100, 480 / 100)
            self.fig.set_dpi(100)
            set_axes_equal(self.ax)

        if self.streetMarkingsPath is not None and self.plotWorldCoord:
            pcd = o3d.io.read_point_cloud(self.streetMarkingsPath)  # Load point cloud data
            # Convert point cloud to NumPy array
            self.cached_markings = np.asarray(pcd.points)  # Extract XYZ coordinates as NumPy array
            if self.detectionZoneBoadersWorld:
                all_borders = np.vstack(self.detectionZoneBoadersWorld)
                self.xmin, self.ymin, self.zmin = all_borders.min(axis=0)
                self.xmax, self.ymax, self.zmax = all_borders.max(axis=0)
                self.ax.set_xlim([self.xmin, self.xmax])
                self.ax.set_ylim([self.ymin, self.ymax])
                self.ax.set_zlim([self.zmin, self.zmax])

    def plot_frame(self, frame, objects, perspective_transform):

        start_time = time.time()

        if self.plotWorldCoord:
            self.ax.clear()  # Clear the plot
            if self.cached_markings is not None:
                add_street_markings(self.ax, street_marking_path=None, marking=self.cached_markings, x_min=self.xmin,
                                    x_max=self.xmax, y_min=self.ymin, y_max=self.ymax)


        for key in objects.keys():

            if objects[key].is_banned():
                continue

            if not objects[key].get_valid_position():
                continue
            else:
                curr_track = objects[key].to_dict()


            label = f'{curr_track["class_name"]} {curr_track["track_id"]}'

            bc_3d = torch.tensor(
                [curr_track["keypoints_world"][-1][0], curr_track["keypoints_world"][-1][1],
                 curr_track["keypoints_world"][-1][2]],
                dtype=torch.float64).unsqueeze(0)
            # corners_3d = get_3d_corners(bc_3d, torch.tensor(

            corners_3d = get_3d_corners(bc_3d, torch.tensor(
                [curr_track["mean_dimensions"][0], curr_track["mean_dimensions"][1],
                 curr_track["mean_dimensions"][2]],
                dtype=torch.float64).unsqueeze(0),
                                        torch.tensor(curr_track["yaw"], dtype=torch.float64))[0, :, :]

            if self.plotWorldCoord:
                plot_world_det(corners_3d, self.ax, color=self.colors[curr_track["class_name"]], line_thickness=1)

            corners_3d_img = perspective_transform.worldToPixel(corners_3d.cpu().numpy())

            # Check if the object is in the field of view
            # if np.any(corners_3d_img[:, 0] < 0) or np.any(corners_3d_img[:, 0] >= frame.shape[1]) or \
            #         np.any(corners_3d_img[:, 1] < 0) or np.any(corners_3d_img[:, 1] >= frame.shape[0]):
            #     continue

            try:
                plot_3d_corners(corners_3d_img, frame, label=label, color=self.colors_thermal[curr_track["class_name"]],
                                line_thickness=2)
            except:
                continue

            keypoint_image = curr_track["keypoints_image"][-1]
            if keypoint_image is None:
                keypoint_image = perspective_transform.worldToPixel(bc_3d.cpu().numpy())
            else:
                keypoint_image = keypoint_image
            keypoint_image = keypoint_image.astype(int)
            frame = cv2.circle(frame, (keypoint_image[0], keypoint_image[1]), 5,
                               self.colors_thermal[curr_track["class_name"]],
                               -1)

        if self.plotWorldCoord:
            # self.ax.set_xlabel('East')
            # self.ax.set_ylabel('North')
            # self.ax.set_zlabel('Z')

            if self.streetMarkingsPath is not None and self.detectionZoneBoadersWorld is None:
                add_street_markings(self.ax, self.streetMarkingsPath)
                # Set narrow axes limits
                set_narrow_axes_limits_from_data(ax)

            # Apply aspect ratio correction
            set_axes_equal(self.ax)

            # Render Matplotlib plot to NumPy array
            canvas = FigureCanvas(self.fig)
            canvas.draw()
            plot_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(
                canvas.get_width_height()[::-1] + (3,))

            stacked_image = np.hstack((frame, plot_image))
            frame = stacked_image

        return frame
