"""
Created on Feb 03 2025 17:08

@author: ISAC - pettirsch
"""
import numpy as np
from scipy.interpolate import CubicSpline
from datetime import timedelta


class Trajectory:
    def __init__(self, idVehicle, cls, length, width, height, positions, yaws, velocities, accelerations, yaw_rates,
                 timeStamps, timeStampsMicroSec, cluster=None, perspectiveTransform=None):
        self.idVehicle = idVehicle
        self.cls = cls
        self.length = length
        self.width = width
        self.height = height
        self.positions = positions
        self.yaws = yaws

        self.velocities = velocities
        self.accelerations = accelerations
        self.yaw_rates = yaw_rates
        self.timeStamps = timeStamps
        self.timeStampsMicroSec = timeStampsMicroSec
        self.cluster_label = cluster

        if perspectiveTransform is not None:
            self.positionsPixel = perspectiveTransform.worldToPixel(positions)

    def get_image_positions(self):
        return self.positionsPixel

    def get_class(self):
        return self.cls

    def get_world_positions(self):
        return self.positions

    def get_current_track(self, curr_time_stamp):

        curr_microsec = curr_time_stamp.microsecond
        curr_time_stamp = curr_time_stamp.replace(microsecond=0)

        # Get the index of the current timestamp
        timestamps_np = np.array(self.timeStamps)
        all_second_indices = np.where(timestamps_np == curr_time_stamp)[0]
        all_second_indices = all_second_indices.tolist()

        # Check if all_second_indices is single value
        if not isinstance(all_second_indices, list):
            all_second_indices = [all_second_indices]

        # Get the index of the current microsecond
        filtered_microsecs = [self.timeStampsMicroSec[i] for i in all_second_indices]
        curr_microsec_index = filtered_microsecs.index(curr_microsec)

        curr_index = all_second_indices[curr_microsec_index]

        track = {}
        track["cls"] = self.cls
        track["length"] = self.length
        track["width"] = self.width
        track["height"] = self.height
        track["position"] = self.positions[0:(curr_index + 1)]
        track["yaw"] = self.yaws[0:(curr_index + 1)]
        track["velocity"] = self.velocities[0:(curr_index + 1)]
        track["acceleration"] = self.accelerations[0:(curr_index + 1)]
        track["yaw_rate"] = self.yaw_rates[0:(curr_index + 1)]

        return track

    def get_start_position(self):
        return self.positions[0]

    def get_pixel_positions(self):
        return self.positionsPixel

    def get_end_position(self):
        return self.positions[-1]

    def get_world_positions(self):
        return self.positions

    def set_cluster_label(self, label):
        self.cluster_label = label

    def get_cluster(self):
        return self.cluster_label

    def get_class(self):
        return self.cls

    def get_time_stamps(self):
        return self.timeStamps

    def get_time_stamps_microsec(self):
        return self.timeStampsMicroSec

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width

    def get_length(self):
        return self.length

    def get_yaws(self):
        return self.yaws

    def get_dimensions(self):
        return self.length, self.width, self.height

    def get_velocities(self):
        return self.velocities

    def get_accelerations(self):
        return self.accelerations

    def get_yaw_rates(self):
        return self.yaw_rates

    def get_first_yaw(self):
        return self.yaws[0]

    def get_last_yaw(self):
        return self.yaws[-1]

    def get_idVehicle(self):
        return self.idVehicle

    def get_start_time(self):
        timeStamp = self.timeStamps[0]
        timeStamp.replace(microsecond=self.timeStampsMicroSec[0])
        return timeStamp.timestamp()

    def get_end_time(self):
        timeStamp = self.timeStamps[-1]
        timeStamp.replace(microsecond=self.timeStampsMicroSec[-1])
        return timeStamp.timestamp()

    def get_all_complete_timestamps(self):
        all_complete_timeStamps = []
        for i, timeStamp in enumerate(self.timeStamps):
            curr_timeStamp = timeStamp.replace(microsecond=self.timeStampsMicroSec[i])
            all_complete_timeStamps.append(curr_timeStamp)

        all_complete_timeStamps.sort()

        return all_complete_timeStamps

    def get_position_and_yaw(self, search_timestamp):
        all_complete_timeStamps = []
        for i, timeStamp in enumerate(self.timeStamps):
            curr_timeStamp = timeStamp.replace(microsecond=self.timeStampsMicroSec[i])
            all_complete_timeStamps.append(curr_timeStamp)

        all_complete_timeStamps.sort()

        closest_timeStamp = min(all_complete_timeStamps, key=lambda t: abs(t - search_timestamp))

        if abs((search_timestamp - closest_timeStamp).total_seconds()) > 1 / 30:
            return None, None

        else:
            idx = all_complete_timeStamps.index(closest_timeStamp)
            return self.positions[idx], self.yaws[idx]

    def get_normalized_timeStamps(self, baseTimestamp=None):
        all_complete_timeStamps = []
        for i, timeStamp in enumerate(self.timeStamps):
            curr_timeStamp = timeStamp.replace(microsecond=self.timeStampsMicroSec[i])
            all_complete_timeStamps.append(curr_timeStamp)

        # Allow negative values
        if baseTimestamp is None:
            # Subtract the first timestamp and give result in seconds with microsecond precision
            return [(timeStamp - all_complete_timeStamps[0]).total_seconds() for timeStamp in all_complete_timeStamps]
        else:
            # Subtract the baseTimestamp and give result in seconds with microsecond precision
            return [(timeStamp - baseTimestamp).total_seconds() for timeStamp in all_complete_timeStamps]

    def append_trajectory(self, trajectory, perspectiveTransform=None):

        # Check time distance between last timestamp of current trajectory and first timestamp of appended trajectory
        all_complete_timestamps_1 = self.get_all_complete_timestamps()
        all_complete_timestamps_2 = trajectory.get_all_complete_timestamps()

        worldPositions_1 = self.get_world_positions()
        worldPositions_2 = trajectory.get_world_positions()

        if len(all_complete_timestamps_2) != len(worldPositions_2):
            print("Error: Length of timestamps and world positions do not match.")

        intial_length_1 = len(all_complete_timestamps_1)
        intial_length_2 = len(all_complete_timestamps_2)

        if (all_complete_timestamps_2[0] - all_complete_timestamps_1[-1]).total_seconds() > 1 / 30:
            # Duration to interpolate
            time_gap = (all_complete_timestamps_2[0] - all_complete_timestamps_1[-1]).total_seconds()
            n_steps = int(time_gap / (1 / 30))

            # Create cubic splines
            x_spline = CubicSpline([0, 1, n_steps + 3, n_steps + 4],
                                   [worldPositions_1[-2][0], worldPositions_1[-1][0], worldPositions_2[0][0],
                                    worldPositions_2[1][0]])
            y_spline = CubicSpline([0, 1, n_steps + 3, n_steps + 4],
                                   [worldPositions_1[-2][1], worldPositions_1[-1][1], worldPositions_2[0][1],
                                    worldPositions_2[1][1]])
            z_spline = CubicSpline([0, 1, n_steps + 3, n_steps + 4],
                                   [worldPositions_1[-2][2], worldPositions_1[-1][2], worldPositions_2[0][2],
                                    worldPositions_2[1][2]])

            # Get values for missing timestamps
            interpolated_x_positions = x_spline(np.arange(2, n_steps + 2))
            interpolated_y_positions = y_spline(np.arange(2, n_steps + 2))
            interpolated_z_positions = z_spline(np.arange(2, n_steps + 2))
            interpolated_world_positions = np.stack(
                [interpolated_x_positions, interpolated_y_positions, interpolated_z_positions], axis=1)

            interpolated_timeStamps = [
                all_complete_timestamps_1[-1] + timedelta(seconds=i * (1 / 30))
                for i in range(1, n_steps + 1)
            ]

            # Fuse all complete timestamps and interpolated timestamps
            all_complete_timestamps_1.extend(interpolated_timeStamps)

            # Fuse all world positions and interpolated world positions
            worldPositions_1 = np.concatenate([worldPositions_1, interpolated_world_positions], axis=0)
            if len(all_complete_timestamps_1) != len(worldPositions_1):
                print("Error: Length of timestamps and world positions do not match.")

        elif (all_complete_timestamps_2[0] - all_complete_timestamps_1[-1]).total_seconds() <= 0:
            # Get common timestamps
            common_timestamps = [timeStamp for timeStamp in all_complete_timestamps_2 if
                                 (timeStamp in all_complete_timestamps_1)]
            try:
                first_common_idx_1 = all_complete_timestamps_1.index(common_timestamps[0])
                last_common_idx_2 = all_complete_timestamps_2.index(common_timestamps[-1])
            except:
                print("Error: Common timestamps not found.")

            if first_common_idx_1 == len(all_complete_timestamps_1) - 1:
                common_postions_1 = worldPositions_1[first_common_idx_1,:]
            else:
                common_postions_1 = worldPositions_1[first_common_idx_1:]
            common_postions_2 = worldPositions_2[:last_common_idx_2 + 1]

            # Calc mean positions
            try:
                mean_positions = common_postions_1 + common_postions_2
            except:
                print("Error: Could not calculate mean positions.")
            mean_positions = mean_positions/2

            # Update positions
            worldPositions_1 = worldPositions_1[:first_common_idx_1]
            all_complete_timestamps_1 = all_complete_timestamps_1[:first_common_idx_1]
            if last_common_idx_2 == 0:
                worldPositions_2[last_common_idx_2,:] = mean_positions
            else:
                worldPositions_2[:last_common_idx_2 + 1] = mean_positions

        # Fuse all complete timestamps and all complete timestamps of trajectory
        all_complete_timestamps_1.extend(all_complete_timestamps_2)
        worldPositions_1 = np.concatenate([worldPositions_1, worldPositions_2], axis=0)

        try:
            assert len(all_complete_timestamps_1) == len(worldPositions_1)
        except:
            print("Error: Length of timestamps and world positions do not match.")

        # Fuse dims
        length = intial_length_1 * self.length + intial_length_2 * trajectory.length
        width = intial_length_1 * self.width + intial_length_2 * trajectory.width
        height = intial_length_1 * self.height + intial_length_2 * trajectory.height
        length = length / (intial_length_1 + intial_length_2)
        width = width / (intial_length_1 + intial_length_2)
        height = height / (intial_length_1 + intial_length_2)

        # Update object attributes
        self.length = length
        self.width = width
        self.height = height
        self.positions = worldPositions_1
        self.timeStamps = []
        self.timeStampsMicroSec = []
        for timeStamp in all_complete_timestamps_1:
            microsecond = timeStamp.microsecond
            timeStamp = timeStamp.replace(microsecond=0)
            self.timeStamps.append(timeStamp)
            self.timeStampsMicroSec.append(microsecond)


        # Fit smoothed trajectory to update velocity, acceleration, yaw, etc.
        self.fit_finished_track()

        try:
            assert len(self.timeStamps) == len(self.positions)
        except:
            print("Error: Length of timestamps and world positions do not match.")

        if perspectiveTransform is not None:
            self.positionsPixel = perspectiveTransform.worldToPixel(self.positions)

    def fit_finished_track(self):
        worldPositions = self.get_world_positions()
        worldPositions_smooth = self.smooth_trajectory_moving_average(worldPositions)

        smoothed_px = worldPositions_smooth[:, 0]
        smoothed_py = worldPositions_smooth[:, 1]
        smoothed_pz = worldPositions_smooth[:, 2]
        self.positions = worldPositions_smooth

        smoothed_vx = np.gradient(smoothed_px) * 30
        smoothed_vy = np.gradient(smoothed_py) * 30
        smoothed_vz = np.gradient(smoothed_pz) * 30
        self.velocities = np.vstack((smoothed_vx, smoothed_vy, smoothed_vz)).T

        smoothed_ax = np.gradient(smoothed_vx)
        smoothed_ay = np.gradient(smoothed_vy)
        smoothed_az = np.gradient(smoothed_vz)
        self.accelerations = np.vstack((smoothed_ax, smoothed_ay, smoothed_az)).T

        smoothed_yaw = np.arctan2(smoothed_vy, smoothed_vx)
        distance = 0
        start_idx = 0
        for i in range(len(worldPositions_smooth)):
            if i > 1:
                distance = np.linalg.norm(worldPositions_smooth[i] - worldPositions_smooth[start_idx])
            if distance > 1:
                yaw = worldPositions_smooth[i, :] - worldPositions_smooth[start_idx, :]
                yaw = np.arctan2(yaw[1], yaw[0])
                for j in range(start_idx, i):
                    smoothed_yaw[j] = yaw
                start_idx = i
                distance = 0

        smoothed_yaw = self._fix_yaw_jumps(smoothed_yaw)
        self.yaws = smoothed_yaw
        self.yaw_rates = np.gradient(smoothed_yaw)

    def _fix_yaw_jumps(self, yaw_array):
        def detect_jumps(lst):
            return [i for i in range(len(lst) - 1) if abs(lst[i + 1] - lst[i]) > 3]

        def moving_average(data, window_size):
            if window_size < 2:
                return data
            if window_size % 2 == 0:
                window_size += 1
            pad_size = window_size // 2
            padded_data = np.pad(data, (pad_size, pad_size), mode='edge')
            return np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')

        smoothed_yaw2 = yaw_array.copy()
        jumps = detect_jumps(smoothed_yaw2)
        start_idx = 0
        if len(jumps) > 0:
            for q, jump in enumerate(jumps):
                window_size = min(jump - start_idx, 60)
                smoothed_yaw2[start_idx:jump + 1] = moving_average(smoothed_yaw2[start_idx:jump + 1], window_size)
                start_idx = jump
                if q == len(jumps) - 1:
                    window_size = min(len(smoothed_yaw2) - start_idx, 60)
                    smoothed_yaw2[start_idx:] = moving_average(smoothed_yaw2[start_idx:], window_size)
        else:
            window_size = min(len(smoothed_yaw2) - start_idx, 60)
            smoothed_yaw2[start_idx:] = moving_average(smoothed_yaw2[start_idx:], window_size)

        return smoothed_yaw2

    def smooth_trajectory_moving_average(self, positions, window_size=10):
        """
        Applies a moving average to smooth the trajectory.
        Args:
            positions (np.ndarray): Array of shape (N, 3) representing world positions.
            window_size (int): Size of the moving average window.
        Returns:
            np.ndarray: Smoothed positions array of same shape.
        """
        if window_size < 2:
            return positions

        if window_size % 2 == 0:
            window_size += 1  # Ensure symmetric window

        pad = window_size // 2
        smoothed = []

        for dim in range(positions.shape[1]):
            padded = np.pad(positions[:, dim], (pad, pad), mode='edge')
            smoothed_dim = np.convolve(padded, np.ones(window_size) / window_size, mode='valid')
            smoothed.append(smoothed_dim)

        return np.stack(smoothed, axis=1)

    def get_timestamps_video(self):
        return self.timeStamps

    def get_microseconds_video(self):
        return self.timeStampsMicroSec

