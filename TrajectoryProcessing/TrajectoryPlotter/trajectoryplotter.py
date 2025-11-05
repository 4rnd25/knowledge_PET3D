"""
Created on Feb 04 2025 15:33

@author: ISAC - pettirsch
"""
import os
import matplotlib.pyplot as plt
import matplotlib
from sympy.printing.pretty.pretty_symbology import line_width
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# matplotlib.use('TkAgg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import open3d as o3d
import cv2
import numpy as np
import random
import torch

from Videoprocessing.Utils.Cuboid_calc.calc_3d_corners import get_3d_corners
from CommonUtils.BoxCalcUtils.calc_3d_box_corners import get_bev_corners
from Videoprocessing.Utils.Plotting_Utils.plot_world_det import add_street_markings, set_axes_equal, \
    set_narrow_axes_limits_from_data
from Videoprocessing.Utils.Plotting_Utils.plot_world_det import add_street_markings, plot_world_det


class TrajectoryPlotter:
    def __init__(self, output_folder, video_filename, plotting_config, verbose=False):
        self.mother_output_folder = output_folder
        self.recordName = video_filename.split('.')[0]
        self.recordFolder = os.path.join(self.mother_output_folder, self.recordName)
        self.outputfolder_trajectories = os.path.join(self.recordFolder, 'Trajectories')

        self.plotting_config = plotting_config
        self.streetMarkingsPath = None #plotting_config["streetMarkingPath"]

        self.colors = plotting_config["colors"]
        self.colors_thermal = {}
        for key in self.colors.keys():
            self.colors_thermal[key] = [self.colors[key][2], self.colors[key][1], self.colors[key][0]]

        self.fig, self.ax = plt.subplots()
        self.cluster_color_dict = {}
        self.cluster_color_dict_thermal = {}
        # self.fig.set_size_inches(640 / 100, 480 / 100)
        # self.fig.set_dpi(100)
        # set_axes_equal(self.ax)

        if self.streetMarkingsPath is not None:
            pcd = o3d.io.read_point_cloud(self.streetMarkingsPath)  # Load point cloud data
            # Convert point cloud to NumPy array
            self.cached_markings = np.asarray(pcd.points)  # Extract XYZ coordinates as NumPy array
        else:
            self.cached_markings = None

        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None

        self.verbose = verbose

    def update_record(self, filename):
        self.recordName = filename.split('.')[0]
        self.recordFolder = os.path.join(self.mother_output_folder, self.recordName)
        self.outputfolder_trajectories = os.path.join(self.recordFolder, 'Trajectories')

    def plot_all_trajectories(self, trajectories, frame, plot_type, mean_trajectories=None, cluster=False, num_cluster=100,
                              name=None):

        overlay = frame.copy()

        if mean_trajectories == {}:
            mean_trajectories = None

        # Iterate through all trajectories
        self.xmin = 10000000000000000000000000
        self.ymin = 10000000000000000000000000
        self.xmax = -10000000000000000000000000
        self.ymax = -10000000000000000000000000
        for i, trajectory in enumerate(trajectories):

            # Check color
            if cluster:
                if trajectory.get_cluster() == -1:
                    continue
                if trajectory.get_cluster() == str(-1):
                    continue

                self.check_color(trajectory.get_cluster(), num_cluster)

            overlay = self.plot_image_trajectory(trajectory, overlay, cluster)
            if mean_trajectories is not None:
                self.plot_world_trajectory(trajectory, cluster, transparency=0.5)
            else:
                self.plot_world_trajectory(trajectory, cluster)

            all_world_coords = trajectory.get_world_positions()
            if i == 0:
                all_world_coords = all_world_coords[:359,:]
            if i == 1:
                all_world_coords = all_world_coords[:339,:]
            loc_x_min = min(all_world_coords, key=lambda x: x[0])[0]
            loc_x_max = max(all_world_coords, key=lambda x: x[0])[0]
            loc_y_min = min(all_world_coords, key=lambda x: x[1])[1]
            loc_y_max = max(all_world_coords, key=lambda x: x[1])[1]

            if loc_x_min < self.xmin:
                self.xmin = loc_x_min
            if loc_x_max > self.xmax:
                self.xmax = loc_x_max
            if loc_y_min < self.ymin:
                self.ymin = loc_y_min
            if loc_y_max > self.ymax:
                self.ymax = loc_y_max

        if self.cached_markings is not None:
            add_street_markings(self.ax, street_marking_path=None, marking=self.cached_markings, x_min=self.xmin,
                                y_min=self.ymin, x_max=self.xmax, y_max=self.ymax, BEV=True)

        if mean_trajectories is not None:
            alpha = 0.5
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            for key in mean_trajectories.keys():
                frame = self.plot_image_trajectory(mean_trajectories[key], frame, cluster, thickness=6)
        else:
            frame = overlay

        if mean_trajectories is not None:
            for key in mean_trajectories.keys():
                self.plot_world_trajectory(mean_trajectories[key], cluster, label=key, linewidth=6)

        # Ensure axes are updated
        self.ax.relim()  # Recompute limits based on data
        self.ax.autoscale_view()  # Adjust view to include new points

        if mean_trajectories is not None:
            self.ax.legend(loc='upper left', bbox_to_anchor=(0.5, -0.2), borderaxespad=0.)

        # Convert Matplotlib figure to an OpenCV image
        fig_img = self.fig_to_image()

        # Show OpenCV image with trajectory overlay
        # cv2.imshow('Trajectories', frame)
        # cv2.imshow('World Trajectories', fig_img)  # Display Matplotlib figure as an image
        # cv2.waitKey(0)

        # Save the image
        if name is None:
            cv2.imwrite(f"{self.outputfolder_trajectories}/Trajectories_{plot_type}.png", frame)
            cv2.imwrite(f"{self.outputfolder_trajectories}/World_Trajectories_{plot_type}.png", fig_img)
        else:
            cv2.imwrite(f"{self.outputfolder_trajectories}/{name}.png", frame)
            cv2.imwrite(f"{self.outputfolder_trajectories}/World_{name}.png", fig_img)

        print("Trajectories saved at: ", f"{self.outputfolder_trajectories}/Trajectories_{plot_type}.png")

        self.plot_cluster_legend()

        return frame

    def plot_world_trajectories_with_time(self, trajectories, frame, plot_type, mean_trajectories=None, cluster=False, num_cluster=50):

        overlay = frame.copy()

        if mean_trajectories == {}:
            mean_trajectories = None

        # Iterate through all trajectories
        for i, trajectory in enumerate(trajectories):

            # Check color
            if cluster:
                if trajectory.get_cluster() == -1:
                    continue

                self.check_color(trajectory.get_cluster(), num_cluster=num_cluster)

            overlay = self.plot_image_trajectory(trajectory, overlay, cluster)
            if mean_trajectories is not None:
                self.plot_world_trajectory(trajectory, cluster, transparency=0.5)
            else:
                self.plot_world_trajectory(trajectory, cluster)

        if mean_trajectories is not None:
            alpha = 0.5
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            for key in mean_trajectories.keys():
                frame = self.plot_image_trajectory(mean_trajectories[key], frame, cluster, thickness=3)
        else:
            frame = overlay

        if mean_trajectories is not None:
            for key in mean_trajectories.keys():
                self.plot_world_trajectory(mean_trajectories[key], cluster, label=key, linewidth=3)

        # Ensure axes are updated
        self.ax.relim()  # Recompute limits based on data
        self.ax.autoscale_view()  # Adjust view to include new points

        if mean_trajectories is not None:
            self.ax.legend(loc='upper left', bbox_to_anchor=(0.5, -0.2), borderaxespad=0.)

        # Convert Matplotlib figure to an OpenCV image
        fig_img = self.fig_to_image()

        # Show OpenCV image with trajectory overlay
        # cv2.imshow('Trajectories', frame)
        # cv2.imshow('World Trajectories', fig_img)  # Display Matplotlib figure as an image
        # cv2.waitKey(0)

        # Save the image
        cv2.imwrite(f"{self.outputfolder_trajectories}/Trajectories_{plot_type}.png", frame)
        cv2.imwrite(f"{self.outputfolder_trajectories}/World_Trajectories_{plot_type}.png", fig_img)

        self.plot_cluster_legend()

        return frame

    def plot_image_trajectory(self, trajectory, frame, cluster=False, thickness=1):

        # Get all image positions
        image_positions = trajectory.get_image_positions()
        image_positions = np.array(image_positions, dtype=np.int32)
        cls = trajectory.get_class()

        if cluster:
            if cluster == str(-1):
                color = (0, 0, 0)
            else:
                try:
                    color = self.cluster_color_dict_thermal[trajectory.get_cluster()]
                except:
                    color = (0,0,0)
        else:
            color = self.colors_thermal[cls]
        color = tuple(map(int, color))  # Ensure it's a tuple of integers

        # Draw lines connecting the points
        for i in range(len(image_positions) - 1):
            cv2.line(frame, tuple(image_positions[i]), tuple(image_positions[i + 1]), color, 4)

        # White circle at the start
        cv2.circle(frame, tuple(image_positions[0]), 5, (0, 0, 0), -1)

        # Black circle at the end
        cv2.circle(frame, tuple(image_positions[-1]), 5, (255, 255, 255), -1)

        return frame

    def plot_world_trajectory(self, trajectory, cluster=False, transparency=1.0, label=None, linewidth=1):

        # Get all world positions
        world_positions = trajectory.get_world_positions()
        world_positions = np.array(world_positions)
        cls = trajectory.get_class()

        if cluster:
            color = self.cluster_color_dict[trajectory.get_cluster()]
        else:
            color = self.colors[cls]

        if transparency < 1.0:
            color = color + [transparency]

        if max(color) > 1:
            color = [c / 255 for c in color]
            # RGB to BGR
            # color = color[::-1]

        # Plot the trajectory
        if label is not None:
            self.ax.plot(world_positions[:, 0], world_positions[:, 1], color=color, linewidth=4, label=label)
        else:
            self.ax.plot(world_positions[:, 0], world_positions[:, 1], color=color, linewidth=4)

        # Plot the start and end points green start red end
        self.ax.plot(world_positions[0, 0], world_positions[0, 1], 'go', markersize=5)
        self.ax.plot(world_positions[-1, 0], world_positions[-1, 1], 'ro', markersize=5)

    def add_street_markings(self):
        if self.streetMarkingsPath is not None:
            pcd = o3d.io.read_point_cloud(self.streetMarkingsPath)  # Load point cloud data
            # Convert point cloud to NumPy array
            self.cached_markings = np.asarray(pcd.points)  # Extract XYZ coordinates as NumPy array
            # if self.detectionZoneBoadersWorld:
            #     all_borders = np.vstack(self.detectionZoneBoadersWorld)
            #     self.xmin, self.ymin, self.zmin = all_borders.min(axis=0)
            #     self.xmax, self.ymax, self.zmax = all_borders.max(axis=0)
            #     self.ax.set_xlim([self.xmin, self.xmax])
            #     self.ax.set_ylim([self.ymin, self.ymax])
            #     self.ax.set_zlim([self.zmin, self.zmax])

    def fig_to_image(self):
        """
        Converts the Matplotlib figure (self.fig) into a NumPy image array for OpenCV.
        """
        self.fig.canvas.draw()  # Render the figure
        width, height = self.fig.canvas.get_width_height()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape((height, width, 3))  # Convert to NumPy array (H, W, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        return img

    def plot_cluster_legend(self):
        """
        Plots small lines representing clusters, creates a legend, and saves it as an OpenCV image.
        """
        if not self.cluster_color_dict:
            print("No clusters available for legend.")
            return

        num_clusters = len(self.cluster_color_dict)

        fig_width = 4  # Fixed width
        fig_height = max(1, num_clusters * 0.3)  # Dynamic height for visibility

        # Create a separate figure for the legend
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # fig, ax = plt.subplots(figsize=(3, 1.5))  # Small figure size
        legend_patches = []  # Store legend handles

        for i, (cluster_id, color) in enumerate(self.cluster_color_dict.items()):
            color_rgb = [c / 255 for c in color]  # Normalize color to 0-1 for Matplotlib
            line, = ax.plot([i, 10], [i, 620], color=color_rgb, linewidth=2, label=f"Cluster {cluster_id}")
            legend_patches.append(line)

        # Remove axes and only keep the legend
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

        # Create a legend
        ax.legend(fontsize=8, loc='best', ncol=2)

        # Ensure proper layout
        fig.tight_layout()

        # Save to OpenCV
        legend_path = os.path.join(self.outputfolder_trajectories, "Cluster_Legend.png")
        fig.canvas.draw()

        # Convert figure to OpenCV format
        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape((height, width, 3))  # Convert to NumPy array (H, W, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        cv2.imwrite(legend_path, img)  # Save image
        plt.close(fig)  # Close to free memory

        print(f"Cluster legend saved at: {legend_path}")

    def check_color(self, key, num_cluster=50):
        if key not in self.cluster_color_dict:
            # Use current number of unique keys to space colors
            cluster_count = len(self.cluster_color_dict)
            max_clusters = num_cluster  # max expected number of clusters
            color = plt.cm.hsv(cluster_count / max_clusters)[:3]  # RGB tuple in [0,1]
            rgb_255 = (np.array(color) * 255).astype(int)
            self.cluster_color_dict[key] = rgb_255
            self.cluster_color_dict_thermal[key] = [rgb_255[2], rgb_255[1], rgb_255[0]]  # BGR for OpenCV/thermal

    def plot_single_cluster(self, trajectories, subplot, frame=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for trajectory in trajectories:
            timestampes = trajectory.get_normalized_timeStamps()
            positions = trajectory.get_world_positions()
            positions_x = [pos[0] for pos in positions]
            positions_y = [pos[1] for pos in positions]

            # Generate random RGB color (each channel 0-255)
            color = [random.randint(0, 255) for _ in range(3)]
            if max(color) > 1:
                color = [c / 255 for c in color]

            # Create plot with x = positions_x, y = positions_y and z = timestampes
            ax.plot(positions_x, positions_y, timestampes, color=color, linewidth=1)

            if frame is not None:
                frame = self.plot_image_trajectory(trajectory, frame, cluster=subplot, thickness=1)

        # Save the frame
        cv2.imwrite(f"{self.outputfolder_trajectories}/Trajectories_{subplot}.png", frame)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Time')

        ax.relim()  # Recompute limits based on data
        ax.autoscale_view()  # Adjust view to include new points

        # plt.show()
        # plt.pause(10)
        # plt.close()

        fig.canvas.draw()  # Render the figure
        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape((height, width, 3))  # Convert to NumPy array (H, W, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        # Show the plot
        # cv2.imshow(f'Cluster_{subplot}', img)
        # cv2.waitKey(0)

        # Save the plot
        cv2.imwrite(f"{self.outputfolder_trajectories}/cluster_{subplot}.png", img)

    def plot_conflict(self, trajectories, label, conflict_time=None, label_1=None, label_2=None,
                      outputfolder=None):

        # Plot trajectorie classic
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        labels = [label_1, label_2]

        for i, trajectory in enumerate(trajectories):
            timestampes = trajectory.get_normalized_timeStamps(conflict_time)
            positions = trajectory.get_world_positions()
            positions_x = [pos[0] for pos in positions]
            positions_y = [pos[1] for pos in positions]

            # Generate random RGB color (each channel 0-255)
            color = [random.randint(0, 255) for _ in range(3)]
            if max(color) > 1:
                color = [c / 255 for c in color]

            # Create plot with x = positions_x, y = positions_y and z = timestampes
            ax.plot(positions_x, positions_y, timestampes, color=color, linewidth=1, label=labels[i])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Time')

        ax.relim()  # Recompute limits based on data
        ax.autoscale_view()  # Adjust view to include new points
        ax.legend()

        # plt.show()
        # plt.pause(10)
        # plt.close()

        fig.canvas.draw()  # Render the figure
        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape((height, width, 3))  # Convert to NumPy array (H, W, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        # Show the plot
        # cv2.imshow(f'Cluster_{subplot}', img)
        # cv2.waitKey(0)

        # Save the plot
        cv2.imwrite(f"{outputfolder}/{label}.png", img)

        # Plot trajectorie BEV
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        labels = [label_1, label_2]

        for i, trajectory in enumerate(trajectories):
            timestampes = trajectory.get_normalized_timeStamps(conflict_time)
            positions = trajectory.get_world_positions()
            positions_x = [pos[0] for pos in positions]
            positions_y = [pos[1] for pos in positions]

            # Generate random RGB color (each channel 0-255)
            color = [random.randint(0, 255) for _ in range(3)]
            if max(color) > 1:
                color = [c / 255 for c in color]

            # Create plot with x = positions_x, y = positions_y and z = timestampes
            ax.plot(positions_x, positions_y, timestampes, color=color, linewidth=1, label=labels[i])

            dimensions = [[trajectory.get_length(), trajectory.get_width(), trajectory.get_height()] for _ in
                          range(len(positions))]
            dimensions = np.asarray(dimensions)
            yaws = trajectory.get_yaws()

            box_corners_3d = get_bev_corners(positions, dimensions, yaws)

            if label == "Interaction_263_264_PET_0.0s":
                for q, timestamp in enumerate(timestampes):

                    if q % 10 != 0:
                        continue

                    # Define the faces of the rectangle
                    faces_2d = [[box_corners_3d[q, 0, :], box_corners_3d[q, 1, :], box_corners_3d[q, 2, :],
                                 box_corners_3d[q, 3, :]]]
                    faces = [np.array([[face[0], face[1], timestamp]]) for face in faces_2d[0]]

                    # Create 3D polygon collection
                    # poly3d = Poly3DCollection(faces, alpha=0.5, edgecolor='k')
                    # ax.add_collection3d(poly3d)

                    # Define edges of the cuboid
                    edges = [
                        [faces[0], faces[1]], [faces[1], faces[2]], [faces[2], faces[3]],
                        [faces[3], faces[0]]  # Sides
                    ]

                    # Plot edges
                    for edge in edges:
                        xs = (edge[0][0][0], edge[1][0][0])
                        ys = (edge[0][0][1], edge[1][0][1])
                        zs = (edge[0][0][2], edge[1][0][2])
                        ax.plot(xs, ys, zs, color=color)

                    # Scatter plot the corner points
                    # ax.scatter(faces[:, 0], faces[:, 1], faces[:, 2], faces[:,3], color='r', s=50)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Time')

        ax.relim()  # Recompute limits based on data
        ax.autoscale_view()  # Adjust view to include new points
        ax.legend()

        # plt.show()
        # plt.pause(10)
        # plt.close()

        # if label == "Interaction_263_264_PET_0.0s":
        #     fig.show()
        #     cv2.waitKey()

        fig.canvas.draw()  # Render the figure
        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape((height, width, 3))  # Convert to NumPy array (H, W, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        # Show the plot
        # cv2.imshow(f'Cluster_{subplot}', img)
        # cv2.waitKey(0)

        # Save the plot
        cv2.imwrite(f"{outputfolder}/BEV_{label}.png", img)

    def plot_trajectories_image_wise(self, frames, trajectories, perspective_transform, label, colors,
                                     video_output_path):

        image_output_path = os.path.join(video_output_path, label + "_3d_images", label)
        os.makedirs(image_output_path, exist_ok=True)
        prev_positions = [[], []]

        counter = 0
        for timestamp, _ in frames:

            self.fig, self.ax = plt.subplots(subplot_kw={'projection': '3d'})
            self.fig.set_size_inches(1280 / 100, 530 / 100) #1060
            self.fig.set_dpi(500)

            if counter > 29:
                continue

            if self.cached_markings is not None:
                add_street_markings(self.ax, street_marking_path=None, marking=self.cached_markings, x_min=self.xmin,
                                    y_min=self.ymin, x_max=self.xmax, y_max=self.ymax, BEV=False)

            for i, trajectory in enumerate(trajectories):
                position, yaw = trajectory.get_position_and_yaw(timestamp)

                length = trajectory.get_length()
                width = trajectory.get_width()
                height = trajectory.get_height()
                cls = trajectory.get_class()

                if position is None:
                    continue

                bc_3d = torch.tensor(
                    [position[0], position[1], position[2]], dtype=torch.float64).unsqueeze(0)

                # if counter != 29:
                #     prev_positions[i].append(bc_3d)
                #     counter += 1
                #     continue

                corners_3d = get_3d_corners(bc_3d, torch.tensor(
                    [length, width, height],
                    dtype=torch.float64).unsqueeze(0),
                                            torch.tensor(yaw, dtype=torch.float64))[0, :, :]

                plot_world_det(corners_3d, self.ax, color=self.colors[cls], line_thickness=1)
                color_curr = colors[cls]
                if max(color_curr) > 1:
                    color_curr = [c / 255 for c in color_curr]
                    # RGB to BGR
                    color_curr = color_curr[::-1]

                if len(prev_positions[i]) > 1:
                    for q, prev_pos in enumerate(prev_positions[i]):
                        if q > 0:
                            self.ax.plot([prev_pos[0,0], prev_positions[i][q - 1][0,0]], [prev_pos[0,1], prev_positions[i][q - 1][0,1]],
                                    [prev_pos[0,2], prev_positions[i][q - 1][0,2]],
                                    color=color_curr, linewidth=2)

                prev_positions[i].append(bc_3d)

            # if counter == 25:
            #     print("25")
            #     self.fig.show()

            self.ax.relim()  # Recompute limits based on data
            self.ax.autoscale_view()

            # Ax set zlim 165 and 185
            self.ax.set_zlim(160, 165)
            self.ax.view_init(elev=30, azim=-80)  # Side view #120 ok

            def set_axes_equal(ax):
                """Set 3D plot axes to equal scale."""
                x_limits = ax.get_xlim3d()
                y_limits = ax.get_ylim3d()
                z_limits = ax.get_zlim3d()

                x_range = abs(x_limits[1] - x_limits[0])
                x_middle = np.mean(x_limits)
                y_range = abs(y_limits[1] - y_limits[0])
                y_middle = np.mean(y_limits)
                z_range = abs(z_limits[1] - z_limits[0])
                z_middle = np.mean(z_limits)

                # The plot bounding box is a cube in data space
                plot_radius = 0.5 * max([x_range, y_range, z_range])

                ax.set_xlim3d([x_middle - plot_radius-2, x_middle + plot_radius-2])
                ax.set_ylim3d([y_middle - plot_radius + 10, y_middle + plot_radius + 10])
                ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

            # Apply it
            set_axes_equal(self.ax)


            # Save the image
            # self.ax.set_zlim(160, 170)
            plt.tight_layout()


            # Save the image
            plt.savefig(f"{image_output_path}/{label}_{counter}.png")
            counter += 1

    def release(self):
        plt.close(self.fig)
        self.fig.clf()
        plt.cla()
        plt.clf()
        plt.close('all')
        cv2.destroyAllWindows()
        self.ax.clear()
        self.fig.clear()
        plt.close()
        self.ax.remove()
        self.fig = None
        self.ax = None
        self.cluster_color_dict = {}
        self.cluster_color_dict_thermal = {}
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None
        self.cached_markings = None