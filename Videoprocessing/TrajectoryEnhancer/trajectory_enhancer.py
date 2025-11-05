"""
Created on Apr 15 2025 10:38

@author: ISAC - pettirsch
"""
import pdb

import numpy as np
from scipy.interpolate import CubicSpline
from datetime import timedelta
import datetime

import copy

from Videoprocessing.Utils.Object_classes.Tracking.track import Track
from sympy.codegen.fnodes import dimension


class TrajectoryEnhancer:
    def __init__(self, config, cost_threshold_dict, persTrans, verbose=False):

        self.initial_min_share_detected = config["initial_min_share_detected"]
        self.initial_min_duration = config["initial_min_duration"] * 30
        self.track_split_yaw_threshold = np.deg2rad(config["track_split_yaw_threshold"])
        self.time_thresh_matching = config["time_thresh_matching"]

        self.final_min_share = config["final_min_share_detected"]
        self.final_min_duration = config["final_min_duration"] * 30
        self.final_min_distance = config["final_min_distance"]
        self.finral_yaw_fluctuation = config["finral_yaw_fluctuation"]

        self.cost_threshold_dict = cost_threshold_dict
        self.persTrans = persTrans

        self.verbose = verbose

        self.tracks = {}

    def add_tracks(self, finished_tracks):
        if finished_tracks == {}:
            return

        # Sharpen finished tracks
        for track_id, track in finished_tracks.items():
            changed_valid = track.remove_last_non_valid_elements()
            changed_detected = track.remove_last_non_detected_elements()

        # Fine filtering
        self.fine_filter_tracks(finished_tracks)

    def fine_filter_tracks(self, finished_tracks):
        """
        Fine filtering of tracks
        """
        for track_id, track in finished_tracks.items():
            if track.get_share_detected() >= self.initial_min_share_detected and \
                    track.get_track_duration() >= self.initial_min_duration:
                if track_id not in self.tracks:
                    self.tracks[track_id] = track

    def enhance_trajectories(self):

        # Fit all trajectories
        for track_id, track in self.tracks.items():
            track.fit_trajectory()

        # Split trajectories
        self.split_trajectories()

        # Match trajectories
        self.match_trajectories()

        # Final filter trajectories
        self.final_filter_tracks()


    def split_trajectories(self):

        max_trajectory_id = self.get_max_trajectory_id()

        tracks_to_remove = []
        new_tracks = {}
        for track_id, track in self.tracks.items():
            yaws = track.get_yaws()
            yaw_diff = np.diff(yaws)

            # Normalize yaw difference
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))

            # World positions
            world_positions = track.get_world_positions()
            movement_distances = np.linalg.norm(np.diff(world_positions, axis=0), axis=1)
            movement_distances = np.insert(movement_distances, 0, 0)  # Insert 0 for the first element

            # Get split positions
            split_indices = np.where(np.abs(yaw_diff) > self.track_split_yaw_threshold)[0]
            if len(split_indices) == 0:
                continue
            elif len(split_indices) == 1:
                if split_indices[0] == 0:
                    continue
            split_indices = np.append(split_indices, len(yaws))
            # Add 0 to the beginning of split_indices
            split_indices = np.insert(split_indices, 0, 0)

            # Split track
            for i, split_idx in enumerate(split_indices):
                if i == len(split_indices) - 1:
                    continue

                split_start = split_indices[i]
                split_end = split_indices[i + 1]

                history_new = track.get_history(split_start, split_end)
                dt_new = track.dt

                if True in history_new["detected"]:
                    track_id_new = max_trajectory_id + 1
                    max_trajectory_id = track_id_new
                    new_track = Track.from_history(track_id, history_new, dt_new)
                    new_tracks[track_id_new] = new_track

            tracks_to_remove.append(track_id)

        # Remove old tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        # Add new tracks
        for track_id_new, new_track in new_tracks.items():
            self.tracks[track_id_new] = new_track

    def get_max_trajectory_id(self):
        max_trajectory_id = 0
        for track_id, track in self.tracks.items():
            if track_id > max_trajectory_id:
                max_trajectory_id = track_id
        return max_trajectory_id

    def match_trajectories(self):

        start_positions = []
        end_positions = []
        track_dims = []
        track_classes = []
        track_last_orientation = []
        track_first_orientation = []
        track_idx_list = []
        start_times = []
        end_times = []

        for track_id, track in self.tracks.items():
            start_positions.append(track.get_start_position())
            end_positions.append(track.get_end_position())
            track_dims.append(track.get_mean_dimensions())
            track_classes.append(track.get_voted_class())
            track_first_orientation.append(track.get_first_yaw())
            track_last_orientation.append(track.get_last_yaw())
            track_idx_list.append(track_id)
            start_times.append(track.get_start_time())
            end_times.append(track.get_end_time())

        # Get matching costs
        n = len(self.tracks)
        if n == 0:
            self.cost_matrix = None
            return []

        # Get matching cost
        matching_cost = self.get_matching_cost(start_positions, end_positions, track_dims, track_classes,
                                               track_first_orientation, track_last_orientation, track_idx_list,
                                               start_times, end_times)

        # Apply greedy matching
        matches, unmatched_origins, unmatched_continuations = self.greedy_match_fast(matching_cost, track_idx_list,
                                                                                     threshold=1000)

        # Matches 2nd trajectory, first_trajectory
        track_ids_to_delete = []
        track_id_match = {}
        for match in matches:
            history_second_trajectory = self.tracks[match[0]].history
            found_match = False
            idx_first = match[1]
            while not found_match:
                if idx_first in track_id_match.keys():
                    idx_first = track_id_match[idx_first]
                else:
                    found_match = True

            # Check time distance between last timestamp of current trajectory and first timestamp of appended trajectory
            all_complete_timestamps_first_traj = self.tracks[idx_first].history["time_stamps_video"]
            all_complete_timestamps_second_traj = self.tracks[match[0]].history["time_stamps_video"]

            world_positions_first_traj = self.tracks[idx_first].get_world_positions()
            world_positions_second_traj = self.tracks[match[0]].get_world_positions()

            if len(world_positions_first_traj) <= 5 or len(world_positions_second_traj) <= 5:
                # If trajectory is too short, skip matching
                continue

            len_first_traj_old = len(world_positions_first_traj)

            if (all_complete_timestamps_second_traj[0] - all_complete_timestamps_first_traj[
                -1]).total_seconds() > 1 / 30:
                # Interpolation needed
                time_gap = (all_complete_timestamps_second_traj[0] - all_complete_timestamps_first_traj[
                    -1]).total_seconds()
                n_steps = int(time_gap / (1 / 30))

                # Create cubic splines
                x_spline = CubicSpline([0, 1, n_steps + 3, n_steps + 4],
                                       [world_positions_first_traj[-2][0], world_positions_first_traj[-1][0],
                                        world_positions_second_traj[0][0],
                                        world_positions_second_traj[1][0]])
                y_spline = CubicSpline([0, 1, n_steps + 3, n_steps + 4],
                                       [world_positions_first_traj[-2][1], world_positions_first_traj[-1][1],
                                        world_positions_second_traj[0][1],
                                        world_positions_second_traj[1][1]])
                z_spline = CubicSpline([0, 1, n_steps + 3, n_steps + 4],
                                       [world_positions_first_traj[-2][2], world_positions_first_traj[-1][2],
                                        world_positions_second_traj[0][2],
                                        world_positions_second_traj[1][2]])

                # Get values for missing timestamps
                interpolated_x_positions = x_spline(np.arange(2, n_steps + 2))
                interpolated_y_positions = y_spline(np.arange(2, n_steps + 2))
                interpolated_z_positions = z_spline(np.arange(2, n_steps + 2))
                interpolated_world_positions = np.stack(
                    [interpolated_x_positions, interpolated_y_positions, interpolated_z_positions], axis=1)

                interpolated_timeStamps = [
                    all_complete_timestamps_first_traj[-1] + timedelta(seconds=i * (1 / 30))
                    for i in range(1, n_steps + 1)
                ]
                for timeStamp in interpolated_timeStamps:
                    if timeStamp.microsecond == 66666:
                        # Set to 66667
                        timeStamp.replace(microsecond=66667)
                    elif timeStamp.microsecond == 166666:
                        # Set to 166667
                        timeStamp.replace(microsecond=166667)
                    elif timeStamp.microsecond == 266666:
                        # Set to 266667
                        timeStamp.replace(microsecond=266667)
                    elif timeStamp.microsecond == 366666:
                        # Set to 366667
                        timeStamp.replace(microsecond=366667)
                    elif timeStamp.microsecond == 466666:
                        # Set to 466667
                        timeStamp.replace(microsecond=466667)
                    elif timeStamp.microsecond == 566666:
                        # Set to 566667
                        timeStamp.replace(microsecond=566667)
                    elif timeStamp.microsecond == 666666:
                        # Set to 666667
                        timeStamp.replace(microsecond=666667)
                    elif timeStamp.microsecond == 766666:
                        # Set to 766667
                        timeStamp.replace(microsecond=766667)
                    elif timeStamp.microsecond == 866666:
                        # Set to 866667
                        timeStamp.replace(microsecond=866667)
                    elif timeStamp.microsecond == 966666:
                        # Set to 966667
                        timeStamp.replace(microsecond=966667)


                # Fuse all complete timestamps and interpolated timestamps
                all_complete_timestamps_first_traj.extend(interpolated_timeStamps)

                # Fuse all world positions and interpolated world positions
                world_positions_first_traj = np.concatenate([world_positions_first_traj, interpolated_world_positions],
                                                            axis=0)
                offset = len(world_positions_first_traj) - len_first_traj_old

                history_first_old = self.tracks[idx_first].history

                appended_frame_id = [history_first_old["frame_ids"][-1] + i for i in range(offset)]
                class_names_append = [self.tracks[idx_first].get_voted_class() for i in range(offset)]
                dimensions_append = [self.tracks[idx_first].get_mean_dimensions() for i in range(offset)]
                detected_append = [False for i in range(offset)]
                time_stamp_system = [datetime.datetime.now() for i in range(offset)]
                valid_position = [True for i in range(offset)]

                history_first_new = {}
                history_first_new["keypoints_world"] = world_positions_first_traj
                history_first_new["time_stamps_video"] = all_complete_timestamps_first_traj

                history_first_new["frame_ids"] = history_first_old["frame_ids"]
                history_first_new["frame_ids"].extend(appended_frame_id)
                history_first_new["class_names"] = history_first_old["class_names"]
                history_first_new["class_names"].extend(class_names_append)
                history_first_new["dimensions"] = history_first_old["dimensions"]
                history_first_new["dimensions"].extend(dimensions_append)
                history_first_new["detected"] = history_first_old["detected"]
                history_first_new["detected"].extend(detected_append)
                history_first_new["time_stamps_system"] = history_first_old["time_stamps_system"]
                history_first_new["time_stamps_system"].extend(time_stamp_system)
                history_first_new["valid_position"] = history_first_old["valid_position"]
                history_first_new["valid_position"].extend(valid_position)


                self.tracks[idx_first].set_history(history_first_new)
                assert (len(self.tracks[idx_first].history["detected"]) == len(
                    self.tracks[idx_first].history["class_names"]))

                self.tracks[idx_first].add_history(history_second_trajectory)

                assert (len(self.tracks[idx_first].history["detected"]) == len(
                    self.tracks[idx_first].history["class_names"]))

            elif (all_complete_timestamps_second_traj[0] - all_complete_timestamps_first_traj[-1]).total_seconds() <= 0:
                common_timestamps = [timeStamp for timeStamp in all_complete_timestamps_second_traj if
                    timeStamp in all_complete_timestamps_first_traj]
                common_idx_first = all_complete_timestamps_first_traj.index(common_timestamps[0])
                common_idx_second = all_complete_timestamps_second_traj.index(common_timestamps[-1])

                common_positions_first_traj = world_positions_first_traj[common_idx_first:]
                common_positions_second_traj = world_positions_second_traj[:common_idx_second+1]

                detected_state_1 = self.tracks[idx_first].history["detected"][common_idx_first:]
                detected_state_2 = self.tracks[match[0]].history["detected"][:common_idx_second+1]

                # If detected_state_1 == detected_state_2 calc mean pos else choose the one which is true
                resulting_positions = []
                resulting_detections = []
                for pos1, pos2, state1, state2 in zip(common_positions_first_traj, common_positions_second_traj,
                                                     detected_state_1, detected_state_2):
                    if state1 and state2:
                        resulting_positions.append((pos1 + pos2) / 2)
                        resulting_detections.append(True)
                    elif state1:
                        resulting_positions.append(pos1)
                        resulting_detections.append(True)
                    elif state2:
                        resulting_positions.append(pos2)
                        resulting_detections.append(True)
                    else:
                        resulting_positions.append((pos1 + pos2) / 2)
                        resulting_detections.append(False)
                resulting_positions = np.array(resulting_positions)


                # Update history
                history_first_old = self.tracks[idx_first].history

                # Make a deep copy so we can safely modify
                history_first_new = copy.deepcopy(history_first_old)

                history_first_new["keypoints_world"][common_idx_first:] = resulting_positions
                history_first_new["detected"][common_idx_first:] = resulting_detections

                # Ensure all history keys remain the same length
                self.tracks[idx_first].set_history(history_first_new)
                assert (len(self.tracks[idx_first].history["detected"]) == len(
                    self.tracks[idx_first].history["class_names"]))

                for key in history_second_trajectory.keys():
                    history_second_trajectory[key] = history_second_trajectory[key][common_idx_second+1:]

                self.tracks[idx_first].add_history(history_second_trajectory)
                assert (len(self.tracks[idx_first].history["detected"]) == len(
                    self.tracks[idx_first].history["class_names"]))

            else:
                self.tracks[idx_first].add_history(history_second_trajectory)
                assert (len(self.tracks[idx_first].history["detected"]) == len(
                    self.tracks[idx_first].history["class_names"]))

            track_id_match[match[0]] = idx_first
            track_ids_to_delete.append(match[0])

        # Remove old tracks
        for track_id in track_ids_to_delete:
            del self.tracks[track_id]

        # Fit all trajectories
        for track_id, track in self.tracks.items():
            track.set_idVehicle(track_id)
            track.fit_trajectory()




    def get_matching_cost(self, start_positions, end_positions, track_dims, track_classes,
                          track_first_orientation, track_last_orientation, track_idx_list,
                          start_times, end_times):

        """
        Get matching cost
        """
        # Stack trajectory info
        start_positions = np.array(start_positions)  # (n, 3)
        end_positions = np.array(end_positions)  # (n, 3)
        track_dims = np.array(track_dims)  # (n, 3)
        track_classes = np.array(track_classes)  # (n,)
        track_first_orientation = np.array(track_first_orientation)  # (n,)
        track_last_orientation = np.array(track_last_orientation)  # (n,)
        start_times = np.array(start_times)  # (n,)
        end_times = np.array(end_times)  # (n,)

        # Compute ρ = [x, y, z, h, w, l]
        rho_start = np.concatenate([start_positions, track_dims], axis=1)  # (n, 6)
        rho_end = np.concatenate([end_positions, track_dims], axis=1)  # (n, 6)

        # Pairwise differences (start_i vs end_j)
        diff = rho_start[:, None, :] - rho_end[None, :, :]  # (n, n, 6)
        # Multiply the center difference by 2 to give it more weight
        diff[:, :, :3] *= 2
        euclidean = np.linalg.norm(diff, axis=2)  # (n, n)

        # Compute pairwise orientation cosine distance
        vec_start = np.stack([np.cos(track_first_orientation), np.sin(track_first_orientation)], axis=1)
        vec_end = np.stack([np.cos(track_last_orientation), np.sin(track_last_orientation)], axis=1)

        # Normalize (in case angles aren't perfectly unit-length)
        vec_start /= np.linalg.norm(vec_start, axis=1, keepdims=True)
        vec_end /= np.linalg.norm(vec_end, axis=1, keepdims=True)

        # Cosine similarity (n x n)
        cos_sim = np.dot(vec_start, vec_end.T)
        alpha = 2.0 - cos_sim

        base_cost = euclidean * alpha  # (n, n)

        # Class penalty
        cls_start = np.array(track_classes)[:, None]
        cls_end = np.array(track_classes)[None, :]
        match_mask = (cls_start == cls_end)

        # Apply infinite cost for class mismatch
        base_cost[~match_mask] = 1001

        # === Time threshold check ===
        start_times_j = start_times[:, None]
        start_times_i = start_times[None, :]
        end_times_j = end_times[:, None]
        end_times_i = end_times[None, :]

        # Compute time differences (in seconds)
        time_diff_start = np.vectorize(lambda a, b: (a - b).total_seconds())(start_times[:, None], start_times[None, :])
        time_diff_end = np.vectorize(lambda a, b: (a - b).total_seconds())(end_times[:, None], end_times[None, :])
        time_diff_total = np.vectorize(lambda a, b: abs((a - b).total_seconds()))(start_times[:, None],
                                                                                  end_times[None, :])

        # Penalize negative deltas
        base_cost[(time_diff_start <= 0) & (time_diff_end <= 0)] = 1001
        base_cost[end_times_j < end_times_i] = 1001
        # base_cost[time_diff_end <= 0] = 1001

        # Penalize if outside acceptable threshold
        base_cost[time_diff_total > self.time_thresh_matching] = 10001

        # Set all values on diagonal to np.inf
        np.fill_diagonal(base_cost, 1001)

        # Apply cost threshold
        for id, cls in enumerate(cls_end[0]):
            cost_threshold = self.cost_threshold_dict[cls]
            mask = base_cost[:, id] > cost_threshold
            base_cost[mask, id] = 1001

        return base_cost

    def greedy_match_fast(self, cost_matrix, track_idx_list, threshold=np.inf):
        n, m = cost_matrix.shape
        matched_dets = set()
        matched_tracks = set()
        matches = []

        # Flatten cost matrix into triplets (i, j, cost)
        det_indices, track_indices = np.nonzero(cost_matrix <= threshold)
        costs = cost_matrix[det_indices, track_indices]

        # Create a structured array for fast sorting
        dtype = [('i', int), ('j', int), ('cost', float)]
        pairs = np.array(list(zip(det_indices, track_indices, costs)), dtype=dtype)
        pairs.sort(order='cost')

        for i, j, _ in pairs:
            if track_idx_list[i] not in matched_dets and track_idx_list[j] not in matched_tracks:
                matches.append((track_idx_list[i], track_idx_list[j]))
                matched_dets.add(track_idx_list[i])
                matched_tracks.add(track_idx_list[j])

        unmatched_dets = [track_idx_list[i] for i in range(n) if track_idx_list[i] not in matched_dets]
        unmatched_tracks = [track_idx_list[j] for j in range(m) if track_idx_list[j] not in matched_tracks]

        return matches, unmatched_dets, unmatched_tracks

    def final_filter_tracks(self):

        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if len(track.history["detected"]) == 0:
                tracks_to_remove.append(track_id)
                continue

            # Remove not detected timestamps at the end
            changed_valid = track.remove_last_non_valid_elements()
            changed_detected = track.remove_last_non_detected_elements()

            # Fit all trajectories
            if changed_valid or changed_detected:
                track.fit_trajectory()

            if track.get_share_detected() >= self.final_min_share and \
                    track.get_track_duration() >= self.final_min_duration and \
                    track.get_distance() >= self.final_min_distance and \
                    track.get_yaw_fluctuation() <= self.finral_yaw_fluctuation:
                continue
            else:
                tracks_to_remove.append(track_id)

        # Remove old tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]

    def get_enhanced_trajectories(self):
        """
        Get enhanced trajectories
        """
        return self.tracks

    def restart(self):
        """
        Restart the trajectory enhancer
        """
        self.tracks = {}
