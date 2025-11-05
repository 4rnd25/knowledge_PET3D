"""
Created on Jun 11 2025 13:43

@author: ISAC - pettirsch
"""

from datetime import timedelta
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from CommonTools.VideoReader.videoreader import VideoReader
from TrajectoryProcessing.TrajectoryReader.trajectorymanager import TrajectoryManager
from CommonTools.PerspectiveTransform.perspectiveTransform import PerspectiveTransform

from TrajectoryProcessing.InteractionDetector.pet3D_interactiondetector import compute_bev_iou_matrix
from Videoprocessing.Utils.Cuboid_calc.calc_3d_corners import get_3d_corners_numpy

class ConflictPlotter:
    def __init__(self, config, height_time_factor = 0.2, verbose=False):
        """
        Initialize the ConflictPlotter.

        Parameters:
            config (dict): Configuration dictionary.
            verbose (bool): If True, print additional information.
        """
        self.config = config
        self.height_time_factor = height_time_factor  # Factor to scale height with time
        self.verbose = verbose

    def svgplot_conflict(self, outputpath, recordID, idVehicle1, idVehicle2):
        # Set startRecordID and endRecordID in config
        self.config['database_config']['startrecordID'] = recordID
        self.config['database_config']['endrecordID'] = recordID

        # Load video reader
        video_reader = VideoReader(self.config['input_config'], self.config['conflict_config']['buffer_duration'],
                                   self.config['database_config'])

        # Intialize perstrans
        persTrans = PerspectiveTransform(
            calibrationPath=self.config["calibration_config"]["calibration_matrix_file"],
            triangulationFacesPath=self.config["calibration_config"]["calibration_faces_file"],
            triangulationPointsPath=self.config["calibration_config"]["calibration_points_file"],
            calibration_type=self.config["calibration_config"]["calibration_type"],
            imageSize=video_reader.get_frame_size(), verbose=self.verbose)

        # Load trajectories
        trajectory_manager = TrajectoryManager(self.config['output_config']['output_folder'],
                                               video_reader.getFilename(),
                                               self.config['database_config'], recordID,
                                               videoprocessing_config=None,
                                               imgsize=video_reader.get_frame_size(),
                                               database=False, lean=False, verbose=self.verbose)

        # Get involved trajectores
        involved_trajectory_1 = trajectory_manager.get_trajectory_by_idVehicle(idVehicle1)
        involved_trajectory_2 = trajectory_manager.get_trajectory_by_idVehicle(idVehicle2)
        involved_trajectories = [involved_trajectory_1, involved_trajectory_2]

        start_time_traj1 = involved_trajectory_1.get_time_stamps()[0].replace(
            microsecond=involved_trajectory_1.get_time_stamps_microsec()[0])
        end_time_traj1 = involved_trajectory_1.get_time_stamps()[-1].replace(
            microsecond=involved_trajectory_1.get_time_stamps_microsec()[-1])
        start_time_traj2 = involved_trajectory_2.get_time_stamps()[0].replace(
            microsecond=involved_trajectory_2.get_time_stamps_microsec()[0])
        end_time_traj2 = involved_trajectory_2.get_time_stamps()[-1].replace(
            microsecond=involved_trajectory_2.get_time_stamps_microsec()[-1])

        # Get start and end time of video snippet
        start_time_snippet = min(start_time_traj1, start_time_traj2)
        end_time_snippet = max(end_time_traj1, end_time_traj2)

        # Check if time is above 20s
        # if (end_time_snippet - start_time_snippet) > timedelta(seconds=20):
        #     # Get interaction time
        #     timeStamp = datetime.strptime(timeStamp, '%Y-%m-%d %H:%M:%S')
        #     interaction_time = timeStamp.replace(microsecond=timeStamp_micro)
        #
        #     start_time = interaction_time - timedelta(seconds=10)
        #     start_time_new = max(start_time_snippet, start_time)
        #
        #     end_time = interaction_time + timedelta(seconds=10)
        #     end_time_new = min(end_time_snippet, end_time)
        #
        #     start_time_snippet = start_time_new
        #     end_time_snippet = end_time_new

        # Plot trajectorie
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = 11  # Base font size
        plt.rcParams["axes.labelsize"] = 11  # Axis label size
        plt.rcParams["axes.titlesize"] = 11  # Title size
        plt.rcParams["legend.fontsize"] = 11  # Legend font size
        plt.rcParams["xtick.labelsize"] = 11  # Tick label sizes
        plt.rcParams["ytick.labelsize"] = 11

        from matplotlib.ticker import MultipleLocator



        fig = plt.figure(figsize=(7, 4.5))
        ax = fig.add_subplot(111, projection='3d')
        ax.yaxis.set_major_locator(MultipleLocator(5))



        xmin = np.inf
        xmax = 0
        ymin = np.inf
        ymax = 0
        zmin = np.inf
        zmax = 0

        for i, trajectory in enumerate(involved_trajectories):

            if i == 0:
                # Face color blue and edge color dark blue
                face_color = (0.6, 0.8, 1.0)
                edge_color = (0, 0, 0.5)
            else:
                # Face color red and edge color dark red
                face_color = (1.0, 0.6, 0.6)
                edge_color = (0.5, 0, 0)

            timestamps = trajectory.get_time_stamps()

            positions = trajectory.get_world_positions()
            positions_old = trajectory.get_world_positions()
            positions_x = [pos[0] for pos in positions]
            positions_y = [pos[1] for pos in positions]
            positions_z = []
            for timestamp in timestamps:
                pos_z = (timestamp - start_time_snippet).total_seconds()
                positions_z.append(pos_z)
            positions = np.array([positions_x, positions_y, positions_z]).T

            dimensions = [[trajectory.get_length(), trajectory.get_width(), trajectory.get_height()] for _ in
                          range(len(positions))]
            dimensions = np.array(dimensions)
            yaws = trajectory.get_yaws()

            iou_mat = compute_bev_iou_matrix(positions_old, dimensions, yaws, positions_old, dimensions,
                                             yaws)

            current_plot_idx = 0
            plotted_current = False
            offset = 0
            for pos_idx, position in enumerate(positions):
                # if timestamps[pos_idx] < start_time_snippet or timestamps[pos_idx] > end_time_snippet:
                #     continue

                pos_z = (timestamps[pos_idx]- start_time_snippet).total_seconds()

                # Plot current box
                if not plotted_current:
                    bc_3d = np.array([positions_x[pos_idx], positions_y[pos_idx], pos_z]).reshape(1, 3)
                    if positions_x[pos_idx] < xmin:
                        xmin = positions_x[pos_idx]
                    if positions_y[pos_idx] < ymin:
                        ymin = positions_y[pos_idx]
                    if pos_z < zmin:
                        zmin = pos_z
                    if positions_x[pos_idx] > xmax:
                        xmax = positions_x[pos_idx]
                    if positions_y[pos_idx] > ymax:
                        ymax = positions_y[pos_idx]
                    if pos_z > zmax:
                        zmax = pos_z

                    length = dimensions[pos_idx][0]
                    width = dimensions[pos_idx][1]
                    height = dimensions[pos_idx][2] * self.height_time_factor  # Scale height with time
                    yaw = yaws[pos_idx]
                    box_corners_3d = get_3d_corners_numpy(bc_3d, dimensions[pos_idx], yaw, local=False)
                    box_corners_3d = box_corners_3d[0,:,:]

                    ax =plot_world_det(box_corners_3d, ax, face_color=face_color, edge_color=edge_color, line_thickness=1)
                    plotted_current = True

                else:
                    if iou_mat[current_plot_idx, current_plot_idx+offset] > 0:
                        # Check time distance
                        if positions_z[current_plot_idx] + dimensions[current_plot_idx][2] * self.height_time_factor  < positions_z[current_plot_idx + offset]-1:
                            plotted_current = False
                            current_plot_idx = current_plot_idx + offset
                            offset = 0
                        else:
                            plotted_current = True
                            offset += 1
                    else:
                        plotted_current = False
                        current_plot_idx = current_plot_idx+offset
                        offset = 0

            # Plot trajectory line with z=0 for all timestamps in face color
            positions_ground = [-5 for _ in range(len(positions_x))]
            ax.plot(positions_x, positions_y, positions_ground, color=face_color, linewidth=2)


        # Save plot
        #ax.set_title(f"Trajectory {i + 1} - ID: {trajectory.get_idVehicle()}")
        ax.set_xlabel('X [m]')
        ax.set_ylabel('\n\nY [m]')
        ax.set_zlabel('Time [s]')
        zmin = min(zmin, -5)
        ax.set_xlim3d([xmin - 5, xmax + 5])
        ax.set_ylim3d([ymin - 5, ymax + 5])
        ax.set_zlim3d([zmin - 2, zmax + 2])
        plt.tight_layout()

        for degree in range(0, 360, 10):
            output_file = f"conflict_plot_{idVehicle1}_{idVehicle2}_{degree}.png"
            output_file = os.path.join(outputpath, output_file)
            ax.view_init(elev=30, azim=degree)  # Example: 30° up from Z, 45° around Z
            plt.savefig(output_file)
            # Save as svg
            output_file_svg = output_file.replace('.png', '.svg')
            plt.savefig(output_file_svg, format='svg')

    def svgplot_conflict_per_timestamp(self, recordName, recordID, conflict_indicator, idVehicle1,
                              idVehicle2, class1, class2, cluster1, cluster2, conflictVideoPath, timeStamp, timeStamp_micro,
                              auto_maneuver, value, degree):

        # Set startRecordID and endRecordID in config
        self.config['database_config']['startrecordID'] = recordID
        self.config['database_config']['endrecordID'] = recordID

        # Load video reader
        video_reader = VideoReader(self.config['input_config'], self.config['conflict_config']['buffer_duration'],
                                   self.config['database_config'])

        # Intialize perstrans
        persTrans = PerspectiveTransform(
            calibrationPath=self.config["calibration_config"]["calibration_matrix_file"],
            triangulationFacesPath=self.config["calibration_config"]["calibration_faces_file"],
            triangulationPointsPath=self.config["calibration_config"]["calibration_points_file"],
            calibration_type=self.config["calibration_config"]["calibration_type"],
            imageSize=video_reader.get_frame_size(), verbose=self.verbose)

        # Load trajectories
        trajectory_manager = TrajectoryManager(self.config['output_config']['output_folder'],
                                               video_reader.getFilename(),
                                               self.config['database_config'], recordID,
                                               videoprocessing_config=None,
                                               imgsize=video_reader.get_frame_size(),
                                               database=False, lean=False, verbose=self.verbose)

        # Get involved trajectores
        involved_trajectory_1 = trajectory_manager.get_trajectory_by_idVehicle(idVehicle1)
        involved_trajectory_2 = trajectory_manager.get_trajectory_by_idVehicle(idVehicle2)
        involved_trajectories = [involved_trajectory_1, involved_trajectory_2]

        # start and end time of video snippet
        start_time_traj_1 = involved_trajectory_1.get_time_stamps()[0].replace(
            microsecond=involved_trajectory_1.get_time_stamps_microsec()[0])
        end_time_traj_1 = involved_trajectory_1.get_time_stamps()[-1].replace(
            microsecond=involved_trajectory_1.get_time_stamps_microsec()[-1])
        start_time_traj_2 = involved_trajectory_2.get_time_stamps()[0].replace(
            microsecond=involved_trajectory_2.get_time_stamps_microsec()[0])
        end_time_traj_2 = involved_trajectory_2.get_time_stamps()[-1].replace(
            microsecond=involved_trajectory_2.get_time_stamps_microsec()[-1])

        # Get start and end time of video snippet
        start_time_snippet = min(start_time_traj_1, start_time_traj_2)
        end_time_snippet = max(end_time_traj_1, end_time_traj_2)

        # Check if time is above 20s
        if (end_time_snippet - start_time_snippet) > timedelta(seconds=20):
            # Get interaction time
            timeStamp = datetime.strptime(timeStamp, '%Y-%m-%d %H:%M:%S')
            interaction_time = timeStamp.replace(microsecond=timeStamp_micro)

            start_time = interaction_time - timedelta(seconds=10)
            start_time_new = max(start_time_snippet, start_time)

            end_time = interaction_time + timedelta(seconds=10)
            end_time_new = min(end_time_snippet, end_time)

            start_time_snippet = start_time_new
            end_time_snippet = end_time_new
        video_snippet = video_reader.get_video_snippet(start_time_snippet, end_time_snippet)


        # Fonts
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = 11
        plt.rcParams["axes.labelsize"] = 11
        plt.rcParams["axes.titlesize"] = 11
        plt.rcParams["legend.fontsize"] = 11
        plt.rcParams["xtick.labelsize"] = 11
        plt.rcParams["ytick.labelsize"] = 11

        from matplotlib.ticker import MultipleLocator

        # Set label
        if isinstance(timeStamp, datetime):
            interaction_time = timeStamp
        else:
            # Convert timeStamp string to datetime object
            if timeStamp_micro is not None:
                interaction_time = datetime.strptime(timeStamp, '%Y-%m-%d %H:%M:%S')

        value_rounded = round(value, 2)
        label = f"{interaction_time.strftime('%Y%m%dT%H%M%S')}_{auto_maneuver}_{class1}_{idVehicle1}_{cluster1}_{class2}_{idVehicle2}_{cluster2}_PET3D_{value_rounded}"


        # Get plot limits
        xmin = np.inf
        xmax = 0
        ymin = np.inf
        ymax = 0
        zmin = np.inf
        zmax = 0
        for curr_timestamp, _ in video_snippet:
            for i, trajectory in enumerate(involved_trajectories):
                timestamps = trajectory.get_time_stamps()
                if curr_timestamp < timestamps[0] or curr_timestamp > timestamps[-1]:
                    continue

                def find_closest_timestamp_index(timestamps, curr_timestamp):
                    # Convert to numeric seconds for efficiency
                    diffs = [abs((t - curr_timestamp).total_seconds()) for t in timestamps]
                    return int(np.argmin(diffs))

                curr_idx = find_closest_timestamp_index(timestamps, curr_timestamp)
                positions = trajectory.get_world_positions()[0:curr_idx + 1]
                positions_x = [pos[0] for pos in positions]
                positions_y = [pos[1] for pos in positions]
                positions_z = [pos[2] for pos in positions]

                if positions_x[curr_idx] < xmin:
                    xmin = positions_x[curr_idx]
                if positions_y[curr_idx] < ymin:
                    ymin = positions_y[curr_idx]
                if positions_z[curr_idx] < zmin:
                    zmin = positions_z[curr_idx]
                if positions_x[curr_idx] > xmax:
                    xmax = positions_x[curr_idx]
                if positions_y[curr_idx] > ymax:
                    ymax = positions_y[curr_idx]
                if positions_z[curr_idx] > zmax:
                    zmax = positions_z[curr_idx]


        frame_counter = 0
        old_positions_0 = []
        old_positions_1 = []
        for curr_timestamp, _ in video_snippet:

            print(f"Plotting frame {frame_counter} for timestamp {curr_timestamp}...")

            # Create figure
            fig = plt.figure(figsize=(7, 4.5))
            ax = fig.add_subplot(111, projection='3d')
            ax.yaxis.set_major_locator(MultipleLocator(5))

            for i, trajectory in enumerate(involved_trajectories):
                if i == 0:
                    # Face color blue and edge color dark blue
                    face_color = (0.6, 0.8, 1.0)
                    edge_color = (0, 0, 0.5)
                else:
                    # Face color red and edge color dark red
                    face_color = (1.0, 0.6, 0.6)
                    edge_color = (0.5, 0, 0)

                timestamps = trajectory.get_time_stamps()
                if curr_timestamp >= timestamps[0] or curr_timestamp <= timestamps[-1]:

                    def find_closest_timestamp_index(timestamps, curr_timestamp):
                        # Convert to numeric seconds for efficiency
                        diffs = [abs((t - curr_timestamp).total_seconds()) for t in timestamps]
                        return int(np.argmin(diffs))

                    curr_idx = find_closest_timestamp_index(timestamps, curr_timestamp)

                    positions = trajectory.get_world_positions()[0:curr_idx+1]
                    dimensions = [[trajectory.get_length(), trajectory.get_width(), trajectory.get_height()] for _ in
                                  range(len(positions))]
                    dimensions = np.array(dimensions)
                    yaws = trajectory.get_yaws()[0:curr_idx+1]

                    # Plot dot at current position
                    position_curr = positions[-1]
                    ax.scatter(position_curr[0], position_curr[1], position_curr[2], color=edge_color, s=4)

                    # Plot current
                    curr_yaw = yaws[-1]
                    curr_position = np.array([positions[-1][0], positions[-1][1], positions[-1][2]]).reshape(1, 3)
                    box_corners_3d = get_3d_corners_numpy(curr_position, dimensions[curr_idx], curr_yaw, local=False)
                    box_corners_3d = box_corners_3d[0,:,:]

                    ax = plot_world_det(box_corners_3d, ax, face_color=face_color, edge_color=edge_color, line_thickness=1)

                    # Add current position to old positions
                    if i == 0:
                        old_positions_0.append(positions[-1])
                    else:
                        old_positions_1.append(positions[-1])

                # Plot previous positions
                if len(old_positions_0) > 1 and i == 0:
                    # Exclude last old position
                    positions = old_positions_0[:-1]
                    positions_x = [pos[0] for pos in positions]
                    positions_y = [pos[1] for pos in positions]
                    positions_z = [pos[2] for pos in positions]
                    ax.plot(positions_x, positions_y, positions_z, color=face_color, linewidth=2)
                elif len(old_positions_1) > 1 and i == 1:
                    positions = old_positions_1[:-1]
                    positions_x = [pos[0] for pos in positions]
                    positions_y = [pos[1] for pos in positions]
                    positions_z = [pos[2] for pos in positions]
                    ax.plot(positions_x, positions_y, positions_z, color=face_color, linewidth=2)

            # Save plot
            ax.set_xlabel('X [m]')
            ax.set_ylabel('\n\nY [m]')
            ax.set_zlabel('Z [m]')
            ax.set_xlim3d([xmin -1, xmax + 1])
            ax.set_ylim3d([ymin - 1, ymax + 1])
            ax.set_zlim3d([zmin - 1, zmax + 2])
            plt.tight_layout()

            output_file_svg = f"conflict_plot_{idVehicle1}_{idVehicle2}_{frame_counter}.svg"
            ax.view_init(elev=30, azim=degree)  # Example: 30° up from Z, 45° around Z
            output_file_svg = os.path.join(conflictVideoPath, output_file_svg)
            plt.savefig(output_file_svg, format='svg')
            frame_counter += 1




def plot_world_det(corners_3d, ax, face_color=(0.6, 0.8, 1.0), edge_color=(0, 0, 0.5), line_thickness=1):
    """
    Plot a filled cuboid with edge lines (no vertex dots).

    Parameters:
    - corners_3d: np.array of shape (8, 3), the 3D coordinates of the cuboid corners.
    - ax: matplotlib 3D axis to plot on.
    - face_color: RGB tuple for the cuboid face color.
    - edge_color: RGB tuple for the cuboid edge lines.
    - line_thickness: Thickness of edge lines.
    """

    # Normalize color if needed
    if max(face_color) > 1:
        face_color = [c / 255 for c in face_color]
    if max(edge_color) > 1:
        edge_color = [c / 255 for c in edge_color]

    # Define the 6 cuboid faces
    faces = [
        [corners_3d[i] for i in [0, 1, 2, 3]],  # bottom
        [corners_3d[i] for i in [4, 5, 6, 7]],  # top
        [corners_3d[i] for i in [0, 1, 5, 4]],  # front
        [corners_3d[i] for i in [1, 2, 6, 5]],  # right
        [corners_3d[i] for i in [2, 3, 7, 6]],  # back
        [corners_3d[i] for i in [3, 0, 4, 7]],  # left
    ]

    # Plot filled faces
    ax.add_collection3d(Poly3DCollection(faces, facecolors=face_color, edgecolors=edge_color, linewidths=line_thickness, alpha=0.5))

    # Draw edges explicitly (for sharp lines)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7)   # verticals
    ]
    for start, end in edges:
        ax.plot([corners_3d[start, 0], corners_3d[end, 0]],
                [corners_3d[start, 1], corners_3d[end, 1]],
                [corners_3d[start, 2], corners_3d[end, 2]],
                color=edge_color, linewidth=line_thickness)

    return ax