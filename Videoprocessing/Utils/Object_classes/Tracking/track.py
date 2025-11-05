"""
Created on Jan 14 2025 13:09

@author: ISAC - pettirsch
"""
import torch
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, KalmanFilter

import pdb

# from Videoprocessing.Utils.Tracking_Utils.kalman_filter_with_yaw import KalmanFilter, \
#     UKFVehicleTracker  # , KalmanFilter


class Track:
    def __init__(self, track_id, detection, curr_time_stamp_video, curr_time_stamp_system, dt=1 / 30,
                 process_noise=2e-2, measurement_noise=3e-2):
        # process noise = 1e-2, measurement noise = 1e-1

        self.track_id = track_id
        self.dt = dt
        initial_pos = detection["keypoint_world"]

        initial_speed_dict = {"motorcycle": 30, "car": 50, "truck": 25, "bus": 25, "bicycle": 15, "person": 7,
                              "e-scooter": 20}

        process_noise = 5e-1
        dim_X = 9  # [x, y, z, w, l, h, vx, vy, vz]
        dim_z = 6  # [x, y, z, h, w, l]

        self.kf = KalmanFilter(dim_x=dim_X, dim_z=dim_z)

        # State transition matrix (F)
        self.kf.F = np.eye(dim_X)
        for i in range(3):  # Position update from velocity (x, y, z)
            self.kf.F[i, i + 6] = dt

        # Measurement matrix (H) - maps state to measurement
        self.kf.H = np.zeros((dim_z, dim_X))
        self.kf.H[0, 0] = 1  # x
        self.kf.H[1, 1] = 1  # y
        self.kf.H[2, 2] = 1  # z
        self.kf.H[3, 3] = 1  # h
        self.kf.H[4, 4] = 1  # w
        self.kf.H[5, 5] = 1  # l

        self.kf.P *= 10.  # Large initial uncertainty

        # Process noise covariance (Q)
        q = 1.0  # Tune this
        self.kf.Q = q * np.eye(dim_X)
        self.kf.Q[0:3, 0:3] *= 1.  # position
        self.kf.Q[3:6, 3:6] *= 1.  # size (l, w, h)
        self.kf.Q[6:9, 6:9] *= 0.01  # velocity

        # Measurement noise covariance (R)
        r = 1.0  # Tune this
        self.kf.R = r * np.eye(6)
        self.kf.R[0:3, 0:3] *= 1.  # x, y, z
        self.kf.R[3:6, 3:6] *= 10.  # l, w, h

        vx_init = initial_speed_dict[detection["class_name"]] / 3.6 * np.cos(detection["yaw"])
        vy_init = initial_speed_dict[detection["class_name"]] / 3.6 * np.sin(detection["yaw"])
        vz_init = 0
        w_init = detection["dimensions"][0]
        l_init = detection["dimensions"][1]
        h_init = detection["dimensions"][2]

        self.kf.x = np.array(
            [initial_pos[0], initial_pos[1], initial_pos[2], w_init, l_init, h_init, vx_init, vy_init, vz_init])

        # History
        self.history = {
            "frame_ids": [],
            "keypoints_image": [],
            "keypoints_world": [],
            "keypoints_world_raw": [],
            "scores": [],
            "class_names": [],
            "dimensions": [],
            "yaws": [],
            "detected": [],
            "movement_flag": [],
            "valid_position": [],
            "time_stamps_video": [],
            "time_stamps_system": [],
            "vel_x": [],
            "vel_y": [],
            "vel_z": [],
            "acc_x": [],
            "acc_y": [],
            "acc_z": [],
            "yaw_rate": [],
            "banned": [],
            "keypoints_world_fitted": None,
            "vel_x_fitted": None,
            "vel_y_fitted": None,
            "vel_z_fitted": None,
            "acc_x_fitted": None,
            "acc_y_fitted": None,
            "acc_z_fitted": None,
            "yaw_rate_fitted": None,
            "yaws_fitted": None
        }

        # Fill history with initial detection
        kf_state = self.kf.x  # [x, y, z, w, l, h, vx, vy, vz]
        self.history["frame_ids"].append(detection["frame_id"])
        self.history["keypoints_image"].append(detection["keypoint_image"])
        self.history["keypoints_world"].append(np.array([kf_state[0], kf_state[1], kf_state[2]]))
        self.history["keypoints_world_raw"].append(detection["keypoint_world"])
        self.history["scores"].append(detection["score"])
        self.history["class_names"].append(detection['class_name'])
        self.history["dimensions"].append(detection["dimensions"])
        yaw = np.arctan2(kf_state[7], kf_state[6])  # Yaw angle
        self.history["yaws"].append(yaw)
        self.history["yaws"][-1] = self.history["yaws"][-1] % (2 * np.pi)
        self.history["detected"].append(True)
        self.history["movement_flag"].append(True)
        self.history["valid_position"].append(True)
        self.history["time_stamps_video"].append(curr_time_stamp_video)
        self.history["time_stamps_system"].append(curr_time_stamp_system)
        self.history["banned"].append(False)

        self.history["vel_x"].append(kf_state[6])
        self.history["vel_y"].append(kf_state[7])
        self.history["vel_z"].append(kf_state[8])

        self.history["acc_x"].append(0)
        self.history["acc_y"].append(0)
        self.history["acc_z"].append(0)
        self.history["yaw_rate"].append(0)

        self.not_detected_counter = 0
        self.track_length = 0
        self.track_length_detected = 0
        self.track_length_conti = []
        self.counter_outside_detection_zone = 0

        self.curr_predicted_position = None
        self.curr_predicted_yaw = None
        self.vote_class()
        self.calc_mean_dimensions()
        self.last_not_standing_index = 0
        self.last_seen = detection["frame_id"]

    @classmethod
    def from_history(cls, track_id, history, dt=1 / 30):

        obj = cls.__new__(cls)  # Create an instance without calling __init__

        obj.track_id = track_id
        obj.dt = dt

        obj.kf = None  # No Kalman filter after split

        obj.history = history

        last_detected_index = len(obj.history["detected"]) - 1 - obj.history["detected"][::-1].index(True)
        obj.not_detected_counter = len(obj.history["detected"]) - last_detected_index - 1

        if obj.history["vel_x_fitted"] is not None:
            vx = obj.history["vel_x_fitted"]
            vy = obj.history["vel_y_fitted"]
            vz = obj.history["vel_z_fitted"]
        else:
            vx = np.array(obj.history["vel_x"])
            vy = np.array(obj.history["vel_y"])
            vz = np.array(obj.history["vel_z"])
        obj.track_length_conti = list(np.sqrt(vx ** 2 + vy ** 2 + vz ** 2))
        obj.track_length = sum(obj.track_length_conti)
        obj.track_length_detected = None

        obj.curr_predicted_position = None
        obj.curr_predicted_yaw = None
        obj.vote_class()
        obj.calc_mean_dimensions()
        obj.last_not_standing_index = None
        obj.last_seen = None

        return obj

    def predict(self):

        self.kf.predict()
        prediction = self.kf.x  # x,y,z, w, l, h, vx, vy, vz
        self.curr_predicted_position = np.array([prediction[0], prediction[1], prediction[2]])
        yaw = np.arctan2(prediction[7], prediction[6])  # Yaw angle
        self.curr_predicted_yaw = yaw

    def update(self, detection, perspectiveTransform, curr_time_stamp_video, curr_time_stamp_system,
               detectionZoneFilter=None):

        if curr_time_stamp_video in self.history["time_stamps_video"]:
            self.vote_class()
            self.calc_mean_dimensions()
            return

        # Handle detection not detection
        if detection["keypoint_world"] is not None:
            measurement = np.array(
                [detection["keypoint_world"][0], detection["keypoint_world"][1], detection["keypoint_world"][2],
                 detection["dimensions"][0], detection["dimensions"][1], detection["dimensions"][2]])
            self.kf.update(measurement)
            self.history["detected"].append(True)
            self.not_detected_counter = 0
            self.last_seen = detection["frame_id"]
            self.history["class_names"].append(detection['class_name'])
            self.history["scores"].append(detection["score"])
        else:
            self.history["detected"].append(False)
            self.not_detected_counter += 1
            self.history["class_names"].append(self.history["class_names"][-1])
            self.history["scores"].append(None)
            if self.history["banned"][-1]:
                self.kf.x[6] = 0
                self.kf.x[7] = 0
                self.kf.x[8] = 0

        kf_state = self.kf.x  # [x, y, z, w, l, h, vx, vy, vz]
        self.history["keypoints_world"].append(np.array([kf_state[0], kf_state[1], kf_state[2]]))
        self.history["dimensions"].append(kf_state[3:6])
        self.history["frame_ids"].append(detection["frame_id"])
        keypoints_image = perspectiveTransform.worldToPixel(self.history["keypoints_world"][-1])
        keypoints_image = np.round(keypoints_image).astype(int)
        self.history["keypoints_image"].append(keypoints_image)
        self.history["keypoints_world_raw"].append(detection["keypoint_world"])
        self.history["movement_flag"].append(False)
        self.history["time_stamps_video"].append(curr_time_stamp_video)
        self.history["time_stamps_system"].append(curr_time_stamp_system)

        yaw = np.arctan2(kf_state[7], kf_state[6])  # Yaw angle
        self.history["yaws"].append(yaw)

        self.history["yaws"][-1] = self.history["yaws"][-1] % (2 * np.pi)
        self.history["vel_x"].append(kf_state[6])
        self.history["vel_y"].append(kf_state[7])
        self.history["vel_z"].append(kf_state[8])

        acc_x = (self.history["vel_x"][-1] - self.history["vel_x"][-2]) / self.dt
        acc_y = (self.history["vel_y"][-1] - self.history["vel_y"][-2]) / self.dt
        acc_z = (self.history["vel_z"][-1] - self.history["vel_z"][-2]) / self.dt
        self.history["acc_x"].append(acc_x)
        self.history["acc_y"].append(acc_y)
        self.history["acc_z"].append(acc_z)

        yaw_old = self.history["yaws"][-2]
        yaw = self.history["yaws"][-1]
        yaw_rate = (yaw - yaw_old) / self.dt  # Yaw rate
        self.history["yaw_rate"].append(yaw_rate)

        if detectionZoneFilter.validDetection(
                (self.history["keypoints_image"][-1][0], self.history["keypoints_image"][-1][1])):
            self.history["valid_position"].append(True)
            self.counter_outside_detection_zone = 0
        else:
            self.history["valid_position"].append(False)
            self.counter_outside_detection_zone += 1

        # Calculate movement flag
        movement_length = np.linalg.norm(
            np.array(self.history["keypoints_world"][-1]) - np.array(self.history["keypoints_world"][-2]))
        self.track_length += movement_length
        self.track_length_conti.append(self.track_length)
        if self.history["detected"][-1]:
            self.track_length_detected += movement_length

        # Calc banned flag
        if self.history["banned"][-1]:
            self.history["banned"].append(True)
        else:
            if len(self.history["keypoints_world"]) > 10 and len(self.history["keypoints_world"]) < 30:
                if self.history["valid_position"][-1] and self.history["valid_position"][-10]:
                    if self.track_length_conti[-1] < 0.5:
                    #if np.linalg.norm(self.history["keypoints_world"][-1] - self.history["keypoints_world"][-10]) < 0.5:
                        #movement= np.linalg.norm(self.history["keypoints_world"][-1] - self.history["keypoints_world"][-10])
                        self.history["banned"].append(True)
                        self.kf.x[6] = 0
                        self.kf.x[7] = 0
                        self.kf.x[8] = 0
                    else:
                        self.history["banned"].append(False)
                else:
                    self.history["banned"].append(True)
                    # Set kf velocity to 0
                    self.kf.x[6] = 0
                    self.kf.x[7] = 0
                    self.kf.x[8] = 0
            elif len(self.history["keypoints_world"]) == 100:
                movement = np.linalg.norm(
                    np.array(self.history["keypoints_world"][-1]) - np.array(self.history["keypoints_world"][-100]))
                if self.history["keypoints_image"][-1] is not None and self.history["keypoints_image"][-100] is not None:
                    movement_pixel = np.linalg.norm(
                        np.array(self.history["keypoints_image"][-1]) - np.array(self.history["keypoints_image"][-100]))
                else:
                    # Search for last not none keypoint image
                    last_not_none_index = len(self.history["keypoints_image"]) - 1 - self.history["keypoints_image"][
                        ::-1].index(None)
                    first_not_none_index = len(self.history["keypoints_image"]) - 1 - self.history["keypoints_image"].index(None)
                    movement_pixel = np.linalg.norm(
                        np.array(self.history["keypoints_image"][last_not_none_index]) - np.array(
                            self.history["keypoints_image"][first_not_none_index]))
                if movement_pixel < 5 or movement < 0.5:
                    self.history["banned"].append(True)
                    # Set kf velocity to 0
                    self.kf.x[6] = 0
                    self.kf.x[7] = 0
                    self.kf.x[8] = 0
            else:
                self.history["banned"].append(False)

        self.vote_class()
        self.calc_mean_dimensions()

    def is_banned(self):
        return self.history["banned"][-1]

    def get_banned_counter(self):
        # get last non banned index
        last_false_index = len(self.history["banned"]) - 1 - self.history["banned"][::-1].index(False) if False in self.history[
            "banned"] else None
        if last_false_index is not None:
            return len(self.history["banned"]) - last_false_index - 1
        else:
            return 0

    def fit_trajectory(self):
        k = 5  # half-window size (kernel size = 2k + 1 = 11)
        dt = self.dt

        positions = np.array(self.history["keypoints_world"])
        if len(positions) < 3:
            return  # not enough points

        # --- Smooth position with moving average ---
        padded = np.pad(positions, ((k, k), (0, 0)), mode='edge')
        smoothed = np.array([np.mean(padded[i:i + 2 * k + 1], axis=0) for i in range(len(positions))])
        self.set_world_positions_fitted(smoothed)

        # --- Compute velocity (central difference) ---
        vel = (np.roll(smoothed, -1, axis=0) - np.roll(smoothed, 1, axis=0)) / (2 * dt)
        vel[0] = vel[1]  # fix border effects
        vel[-1] = vel[-2]
        self.set_velocities_fitted(vel)

        # --- Compute acceleration (central difference) ---
        acc = (np.roll(smoothed, -1, axis=0) - 2 * smoothed + np.roll(smoothed, 1, axis=0)) / (dt ** 2)
        acc[0] = acc[1]
        acc[-1] = acc[-2]
        self.set_accelerations_fitted(acc)

        # --- Yaw from velocity vector ---
        yaw = np.arctan2(vel[:, 1], vel[:, 0])  # atan2(v_y, v_x)

        # --- Circular smoothing of yaw using unit complex averaging ---
        complex_yaw = np.exp(1j * yaw)
        padded_cyaw = np.pad(complex_yaw, (k, k), mode='edge')
        smoothed_cyaw = np.array([np.mean(padded_cyaw[i:i + 2 * k + 1]) for i in range(len(yaw))])
        yaw_smooth = np.angle(smoothed_cyaw)
        self.set_yaws_fitted(yaw_smooth)

        # Caclulate movement distance based on smoothed positions for each step
        movement_distances = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
        movement_distances = np.insert(movement_distances, 0, 0)  # Insert 0 for the first element

        # Loop through the movement distances and update yaw values for non-moving positions
        last_valid_yaw = yaw_smooth[0]  # Initialize with the first yaw value
        for i in range(1, len(movement_distances)):
            if movement_distances[i] < 0.1:
                yaw_smooth[i] = last_valid_yaw  # Set yaw to the last valid moving value
            else:
                last_valid_yaw = yaw_smooth[i]  # Update the last valid yaw value for moving positions

        assert len(yaw_smooth) == len(self.history["keypoints_world"])

        # --- Yaw rate ---
        yaw_rate = np.gradient(yaw_smooth, dt)
        self.set_yaw_rates_fitted(yaw_rate)

        # Remove datapoints where dt in timeStamp video is below 1/30 seconds
        # time_stamps = np.array(self.history["time_stamps_video"])
        # time_diffs = np.diff(time_stamps)
        # valid_indices = np.where(time_diffs >= 1 / 30)[0] + 1
        # self.history["keypoints_world_fitted"] = smoothed[valid_indices]
        # self.history["vel_x_fitted"] = vel[valid_indices, 0]
        # self.history["vel_y_fitted"] = vel[valid_indices, 1]
        # self.history["vel_z_fitted"] = vel[valid_indices, 2]
        # self.history["acc_x_fitted"] = acc[valid_indices, 0]
        # self.history["acc_y_fitted"] = acc[valid_indices, 1]
        # self.history["acc_z_fitted"] = acc[valid_indices, 2]
        # self.history["yaw_rate_fitted"] = yaw_rate[valid_indices]
        # self.history["yaws_fitted"] = yaw_smooth[valid_indices]
        # self.history["dimensions"] = self.history["dimensions"][valid_indices]
        # self.history["class_names"] = self.history["class_names"][valid_indices]
        # self.history["time_stamps_video"] = self.history["time_stamps_video"][valid_indices]
        # self.history["time_stamps_system"] = self.history["time_stamps_system"][valid_indices]
        # self.history["detected"] = self.history["detected"][valid_indices]
        # self.history["keypoints_world"] = self.history["keypoints_world_fitted"]

        return

    def moving_average(self, data, window_size):
        """Applies a moving average filter with proper edge handling."""
        if window_size < 2:
            return data  # No smoothing needed for window_size < 2

        # Ensure the window size is odd for symmetric smoothing
        if window_size % 2 == 0:
            window_size += 1

        pad_size = window_size // 2
        padded_data = np.pad(data, (pad_size, pad_size), mode='edge')  # Replicate edges
        return np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')

    def vote_class(self):
        # Get all not None class names
        try:
            class_names = [class_name for i, class_name in enumerate(self.history["class_names"]) if
                       self.history["detected"][i]]
        except:
            pdb.set_trace()

        # Get the most common class name
        self.voted_class = max(set(class_names), key=class_names.count)
        # self.voted_class = self.history["class_names"][-1]

    def get_uncertain_counter(self):
        return self.uncertain_counter

    def calc_mean_dimensions(self):
        # Get all not None dimensions
        dimensions = [dimension for i, dimension in enumerate(self.history["dimensions"]) if
                      self.history["detected"][i]]

        # dimensions_array = torch.tensor(dimensions).detach().cpu().numpy()
        dimensions_array = np.array(dimensions)

        # Calculate mean dimensions
        self.mean_dimensions = tuple(dimensions_array.mean(axis=0))
        # self.mean_dimensions = self.history["dimensions"][-1]

    def get_current_world_position(self):
        return self.curr_predicted_position

    def get_hist_world_position(self):
        if len(self.history["keypoints_world"]) < 3:
            return self.history["keypoints_world"][-1]
        else:
            return self.history["keypoints_world"][-3]

    def get_positions(self):
        return self.history["keypoints_world"]

    def detected_last_frame(self):
        return self.history["detected"][-1]

    def to_dict(self):
        return {
            "track_id": self.track_id,
            "class_name": self.voted_class,
            "mean_dimensions": self.mean_dimensions,
            "yaw": self.history["yaws"][-1],
            "keypoints_world": self.history["keypoints_world"],
            "keypoints_image": self.history["keypoints_image"]
        }

    # def increase_not_detected_counter(self):
    #     self.not_detected_counter += 1
    #
    # def reset_not_detected_counter(self):
    #     self.not_detected_counter = 0

    def get_not_detected_counter(self):
        return self.not_detected_counter

    # def calculate_orientation(self):
    #     # Ensure there are at least two positions to calculate orientations
    #     if len(self.history["keypoints_world"]) < 2:
    #         return self.history["yaws"]
    #
    #     orientations = []
    #
    #     for i in range(1, len(self.history["keypoints_world"])):
    #         # Extract consecutive positions
    #         x1 = self.history["keypoints_world"][i - 1][0].detach().cpu().numpy()
    #         y1 = self.history["keypoints_world"][i - 1][1].detach().cpu().numpy()
    #         z1 = self.history["keypoints_world"][i - 1][2].detach().cpu().numpy()
    #         x2 = self.history["keypoints_world"][i][0].detach().cpu().numpy()
    #         y2 = self.history["keypoints_world"][i][1].detach().cpu().numpy()
    #         z2 = self.history["keypoints_world"][i][2].detach().cpu().numpy()
    #
    #         # Calculate the angle in the x-y plane
    #         delta_x = x2 - x1
    #         delta_y = y2 - y1
    #
    #         # Compute the heading angle using atan2
    #         angle_rad = np.arctan2(delta_y, delta_x)
    #
    #         orientations.append(angle_rad)
    #
    #     # Apply smoothing to the calculated orientations
    #     smoothed_orientations = []
    #     smoothing_factor = 0.8  # Adjust this factor as needed (0 < smoothing_factor <= 1)
    #
    #     for i, angle in enumerate(orientations):
    #         if i == 0:
    #             smoothed_orientations.append(angle)  # First angle remains unchanged
    #         else:
    #             smoothed_angle = (
    #                     smoothing_factor * angle + (1 - smoothing_factor) * smoothed_orientations[i - 1]
    #             )
    #             smoothed_orientations.append(smoothed_angle)
    #
    #     return smoothed_orientations

    def get_share_detected(self):
        return sum(self.history["detected"]) / len(self.history["detected"])

    def get_start_position(self):
        if self.history["keypoints_world_fitted"] is not None:
            return self.history["keypoints_world_fitted"][0]
        else:
            return self.history["keypoints_world"][0]

    def get_end_position(self):
        if self.history["keypoints_world_fitted"] is not None:
            return self.history["keypoints_world_fitted"][-1]
        else:
            return self.history["keypoints_world"][-1]

    def get_num_detected(self):
        return sum(self.history["detected"])

    def get_max_bev_dimension(self):
        return np.sqrt(self.mean_dimensions[0] ** 2 + self.mean_dimensions[1] ** 2) / 2

    def get_mean_dimensions(self):
        return self.mean_dimensions

    def get_first_yaw(self):
        if self.history["yaws_fitted"] is not None:
            return self.history["yaws_fitted"][0]
        else:
            return self.history["yaws"][0]

    def get_last_yaw(self):
        if self.history["yaws_fitted"] is not None:
            return self.history["yaws_fitted"][-1]
        else:
            return self.history["yaws"][-1]

    def get_idVehicle(self):
        return self.track_id

    def get_start_time(self):
        timeStamp = self.history["time_stamps_video"][0]
        return timeStamp

    def get_end_time(self):
        timeStamp = self.history["time_stamps_video"][-1]
        return timeStamp

    def get_voted_class(self):
        return self.voted_class

    def get_orientation(self):
        return self.history["yaws"][-1]

    def get_valid_position(self):
        return self.history["valid_position"][-1]

    def get_last_valid_world_position(self):
        last_true_index = len(self.history["valid_position"]) - 1 - self.history["valid_position"][::-1].index(
            True) if True in self.history["valid_position"] else None
        return self.history["keypoints_world"][last_true_index]

    def get_last_valid_pixel_position(self):
        last_true_index = len(self.history["valid_position"]) - 1 - self.history["valid_position"][::-1].index(
            True) if True in self.history["valid_position"] else None
        return self.history["keypoints_image"][last_true_index]

    def get_track_duration(self):
        return len(self.history["keypoints_world"])

    def get_track_length(self):
        return self.track_length

    def get_timestamps_video(self):
        return self.history["time_stamps_video"]


    def get_distance_to_start(self):
        dist_to_start = np.linalg.norm(self.history["keypoints_world"][-1] - self.history["keypoints_world"][0])
        return dist_to_start

    def detected_last_frame(self):
        return self.history["detected"][-1]

    def get_detected_counter(self):
        return sum(self.history["detected"])

    def get_counter_outside_detection_zone(self):
        return self.counter_outside_detection_zone

    def get_world_positions(self):
        if self.history["keypoints_world_fitted"] is not None:
            return self.history["keypoints_world_fitted"]
        else:
            return self.history["keypoints_world"]

    def get_yaws(self):
        if self.history["yaws_fitted"] is not None:
            return self.history["yaws_fitted"]
        else:
            return self.history["yaws"]

    def get_timestamps_video(self):
        return self.history["time_stamps_video"]

    def get_timestamps_system(self):
        return self.history["time_stamps_system"]

    def get_velocities(self):
        if self.history["vel_x_fitted"] is not None:
            return self.history["vel_x_fitted"], self.history["vel_y_fitted"], self.history["vel_z_fitted"]
        else:
            return self.history["vel_x"], self.history["vel_y"], self.history["vel_z"]

    def get_accelerations(self):
        if self.history["acc_x_fitted"] is not None:
            return self.history["acc_x_fitted"], self.history["acc_y_fitted"], self.history["acc_z_fitted"]
        else:
            return self.history["acc_x"], self.history["acc_y"], self.history["acc_z"]

    def get_yaw_rates(self):
        if self.history["yaw_rate_fitted"] is not None:
            return self.history["yaw_rate_fitted"]
        else:
            return self.history["yaw_rate"]

    def remove_last_non_valid_elements(self):
        last_true_index = len(self.history["valid_position"]) - 1 - self.history["valid_position"][::-1].index(
            True) if True in self.history["valid_position"] else None

        changed = False
        if self.history["valid_position"][-1] == False:
            changed = True

        for key in self.history.keys():
            if self.history[key] is not None:
                self.history[key] = self.history[key][:last_true_index + 1]

        self.track_length_conti = self.track_length_conti[:last_true_index + 1]

        return changed

    def remove_last_non_detected_elements(self):
        last_true_index = len(self.history["detected"]) - 1 - self.history["detected"][::-1].index(
            True) if True in self.history["detected"] else None

        changed = False
        if self.history["detected"][-1] == False:
            changed = True

        for key in self.history.keys():
            if self.history[key] is not None:
                self.history[key] = self.history[key][:last_true_index + 1]

        self.track_length_conti = self.track_length_conti[:last_true_index + 1]

        return changed

    def get_tracklength_detected(self):
        return self.track_length_detected

    def set_world_positions_fitted(self, world_positions):
        self.history["keypoints_world_fitted"] = world_positions

    def get_world_positions_fitted(self):
        return self.history["keypoints_world_fitted"]

    def set_velocities_fitted(self, velocities):
        vel_x_fitted = velocities[:, 0]
        vel_y_fitted = velocities[:, 1]
        vel_z_fitted = velocities[:, 2]
        self.history["vel_x_fitted"] = vel_x_fitted
        self.history["vel_y_fitted"] = vel_y_fitted
        self.history["vel_z_fitted"] = vel_z_fitted

    def get_velocities_fitted(self):
        return self.history["vel_x_fitted"], self.history["vel_y_fitted"], self.history["vel_z_fitted"]

    def set_accelerations_fitted(self, accelerations):
        acc_x_fitted = accelerations[:, 0]
        acc_y_fitted = accelerations[:, 1]
        acc_z_fitted = accelerations[:, 2]
        self.history["acc_x_fitted"] = acc_x_fitted
        self.history["acc_y_fitted"] = acc_y_fitted
        self.history["acc_z_fitted"] = acc_z_fitted

    def get_accelerations_fitted(self):
        return self.history["acc_x_fitted"], self.history["acc_y_fitted"], self.history["acc_z_fitted"]

    def set_yaw_rates_fitted(self, yaw_rates):
        self.history["yaw_rate_fitted"] = yaw_rates

    def get_yaw_rates_fitted(self):
        return self.history["yaw_rate_fitted"]

    def set_yaws_fitted(self, yaws):
        self.history["yaws_fitted"] = yaws

    def get_yaws_fitted(self):
        if self.history["yaws_fitted"] is None:
            return self.history["yaws"]
        else:
            return self.history["yaws_fitted"]

    def get_distance_to_x(self, x=90):
        distance = np.linalg.norm(self.history["keypoints_world"][-1] - self.history["keypoints_world"][-x])
        return distance

    def remove_boarders(self):
        first_value_gt_1 = next((i for i, x in enumerate(self.track_length_conti) if x > 1), None)
        first_value_gt_1 = int(round(first_value_gt_1, 0))
        max_value = self.track_length
        last_index_lt_max_minus_1 = next(
            (i for i in range(len(self.track_length_conti) - 1, -1, -1) if self.track_length_conti[i] < max_value - 1),
            None)
        last_index_lt_max_minus_1 = int(round(last_index_lt_max_minus_1, 0))

        for key in self.history.keys():
            if self.history[key] is not None:
                self.history[key] = self.history[key][first_value_gt_1:last_index_lt_max_minus_1 + 1]

    def get_kf_covariance(self):
        return self.kf.P

    def get_predicted_state(self):
        return self.kf.x

    def get_last_position(self):
        return self.history["keypoints_world"][-1]

    def get_last_detected_position(self):
        last_true_index = len(self.history["detected"]) - 1 - self.history["detected"][::-1].index(True)
        return self.history["keypoints_world"][last_true_index]

    def get_last_detected_yaw(self):
        last_true_index = len(self.history["detected"]) - 1 - self.history["detected"][::-1].index(True)
        return self.history["yaws"][last_true_index]

    def get_fitted_start_position(self):
        return self.history["keypoints_world_fitted"][0]

    def get_fitted_end_position(self):
        return self.history["keypoints_world_fitted"][-1]

    def get_fitted_dimensions(self):
        return self.mean_dimensions

    def get_fitted_start_yaw(self):
        return self.history["yaws_fitted"][0]

    def get_fitted_end_yaw(self):
        return self.history["yaws_fitted"][-1]

    def get_fitted_start_time(self):
        time_stamp = self.history["time_stamps_video"][0]

        # return as seconds
        return time_stamp.timestamp()

    def get_fitted_end_time(self):
        time_stamp = self.history["time_stamps_video"][-1]

        # return as seconds
        return time_stamp.timestamp()

    def get_history(self, start_idx, end_idx):
        history = {}
        for key in self.history.keys():
            # Check if key is list
            if isinstance(self.history[key], list):
                # Check if history[key] is not None
                if self.history[key] is not None:
                    # Check if self.history[key] is empty
                    if len(self.history[key]) == 0:
                        history[key] = []
                    else:
                        # Slice the list
                        history[key] = self.history[key][start_idx:end_idx]
            elif isinstance(self.history[key], np.ndarray):
                # Check if history[key] is not None
                if self.history[key] is not None:
                    # Check if self.history[key] is empty
                    if len(self.history[key]) == 0:
                        history[key] = np.array([])
                    else:
                        # check if shape is 1 or 2
                        if len(self.history[key].shape) == 1:
                            # Slice the list
                            history[key] = self.history[key][start_idx:end_idx]
                        else:
                            # check if shape is 2
                            if len(self.history[key].shape) != 2:
                                raise ValueError(
                                    f"Unsupported shape for history[{key}]: {self.history[key].shape}. Expected 1D or 2D array.")
                            else:
                                # check if shape is 2
                                history[key] = self.history[key][start_idx:end_idx, :]
            else:
                raise TypeError(
                    f"Unsupported type for history[{key}]: {type(self.history[key])}. Expected list or numpy array.")

        return history

    def set_idVehicle(self, idVehicle):
        self.track_id = idVehicle

    def add_history(self, history):
        if len(history["keypoints_world"]) == 0:
            return

        # Iterate over history time_stamps_video and remove those index which are in self.history["time_stamps_video"]
        idx_to_remove = []
        for i, time_stamp in enumerate(history["time_stamps_video"]):
            if time_stamp in self.history["time_stamps_video"]:
                idx_to_remove.append(i)

        # Remove those index from history
        for key in history.keys():
            if isinstance(history[key], list):
                history[key] = [history[key][i] for i in range(len(history[key])) if i not in idx_to_remove]
            elif isinstance(history[key], np.ndarray):
                history[key] = np.delete(history[key], idx_to_remove, axis=0)
            else:
                raise TypeError(
                    f"Unsupported type for history[{key}]: {type(history[key])}. Expected list or numpy array.")

        for key in self.history.keys():
            if not key in history.keys():
                continue
            # Check if self.history[key] is list
            if isinstance(self.history[key], list):
                # Check if history[key] is not None
                if history[key] is not None:
                    # Check if self.history[key] is empty
                    if len(self.history[key]) == 0:
                        self.history[key] = history[key]
                    else:
                        # Extend the list
                        self.history[key].extend(history[key])
            # Check if self.history[key] is numpy array
            elif isinstance(self.history[key], np.ndarray):
                # Check if history[key] is not None
                if history[key] is not None:
                    # Check if self.history[key] is empty
                    if len(self.history[key]) == 0:
                        self.history[key] = history[key]
                    else:
                        # Concatenate the numpy arrays
                        self.history[key] = np.concatenate((self.history[key], history[key]), axis=0)
            else:
                raise TypeError(
                    f"Unsupported type for history[{key}]: {type(self.history[key])}. Expected list or numpy array.")

        self.vote_class()
        self.calc_mean_dimensions()
        self.fit_trajectory()



        # if self.history["frame_ids"] is not None:
        #     self.history["frame_ids"].extend(history["frame_ids"])
        # if self.history["keypoints_image"] is not None:
        #     self.history["keypoints_image"].extend(history["keypoints_image"])
        # if self.history["keypoints_world"] is not None:
        #     self.history["keypoints_world"] = np.concatenate(
        #             [self.history["keypoints_world"], history["keypoints_world"]], axis=0)
        # if self.history["keypoints_world_raw"] is not None:
        #     self.history["keypoints_world_raw"].extend(history["keypoints_world_raw"])
        # if self.history["scores"] is not None:
        #     self.history["scores"].extend(history["scores"])
        # if self.history["class_names"] is not None:
        #     self.history["class_names"].extend(history["class_names"])
        # if self.history["dimensions"] is not None:
        #     self.history["dimensions"].extend(history["dimensions"])
        # if self.history["yaws"] is not None:
        #     self.history["yaws"].extend(history["yaws"])
        # if self.history["detected"] is not None:
        #     self.history["detected"].extend(history["detected"])
        # if self.history["movement_flag"] is not None:
        #     self.history["movement_flag"].extend(history["movement_flag"])
        # if self.history["valid_position"] is not None:
        #     self.history["valid_position"].extend(history["valid_position"])
        # if self.history["time_stamps_video"] is not None:
        #     self.history["time_stamps_video"].extend(history["time_stamps_video"])
        # if self.history["time_stamps_system"] is not None:
        #     self.history["time_stamps_system"].extend(history["time_stamps_system"])
        #
        # if self.history["vel_x"] is not None:
        #     self.history["vel_x"].extend(history["vel_x"])
        # if self.history["vel_y"] is not None:
        #     self.history["vel_y"].extend(history["vel_y"])
        # if self.history["vel_z"] is not None:
        #     self.history["vel_z"].extend(history["vel_z"])
        #
        # if self.history["acc_x"] is not None:
        #     self.history["acc_x"].extend(history["acc_x"])
        # if self.history["acc_y"] is not None:
        #     self.history["acc_y"].extend(history["acc_y"])
        # if self.history["acc_z"] is not None:
        #     self.history["acc_z"].extend(history["acc_z"])
        #
        # if self.history["yaw_rate"] is not None:
        #     self.history["yaw_rate"].extend(history["yaw_rate"])
        # self.vote_class()
        # self.calc_mean_dimensions()
        # self.fit_trajectory()

    def get_yaw_fluctuation(self):
        if self.history["yaws_fitted"] is not None:
            yaws = np.array(self.history["yaws_fitted"])
        else:
            yaws = np.array(self.history["yaws"])

        if len(yaws) < 2:
            return 0.0  # Not enough data to compute fluctuation

        # Map to unit circle
        complex_yaws = np.exp(1j * yaws)

        # Compute mean direction
        mean_angle = np.angle(np.mean(complex_yaws))

        # Compute angular distance to mean (shortest arc)
        angular_diffs = np.angle(np.exp(1j * (yaws - mean_angle)))

        # deviation from mean
        fluctuation = np.max(angular_diffs) - np.min(angular_diffs)

        return fluctuation  # Total fluctuation from min to max

    def set_history(self, history):
        self.history = history

    def get_distance(self):
        if self.history["keypoints_world_fitted"] is not None:
            # Sum all distances between keypoints world fitted
            distances = np.linalg.norm(np.diff(self.history["keypoints_world_fitted"], axis=0), axis=1)
        else:
            # Sum all distances between keypoints world
            distances = np.linalg.norm(np.diff(self.history["keypoints_world"], axis=0), axis=1)
        return np.sum(distances)

    def get_non_valid_counter(self):
        # Get last valid index
        last_true_index = len(self.history["valid_position"]) - 1 - self.history["valid_position"][::-1].index(
            True) if True in self.history["valid_position"] else None

        valid_counter = len(self.history["valid_position"]) - last_true_index - 1

        return valid_counter