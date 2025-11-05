"""
Created on Jan 14 2025 13:09

@author: ISAC - pettirsch
"""
import pdb

import numpy as np
import torch
import math
from shapely.geometry import Polygon

from Videoprocessing.Utils.Object_classes.Tracking.track import Track


class ObjectTracker:
    def __init__(self, verbose=False):

        self.tracks = {}
        self.next_track_id = 0

        self.verbose = verbose


    def update_predictions(self):
        for track_id, track in self.tracks.items():
            track.predict()

    def get_center_array_tracks(self):
        center_list = []
        track_idx_list = []
        for track_id, track in self.tracks.items():
            curr_world_position = track.get_current_world_position()
            if curr_world_position is not None:
                center_list.append(curr_world_position)
                track_idx_list.append(track_id)
        return torch.tensor(center_list).detach().cpu().numpy(), track_idx_list

    def get_tracks(self):
        return self.tracks

    def update(self, matched_objects, unmatched_dets, unmatched_tracks, detections,
        frame_id, curr_time_stamp_video, curr_time_stamp_system, perspectiveTransform, cost_matrix,
        detectionZoneFilter = None):

        track_idx_list = [track_id for track_id in self.tracks.keys()]

        matched_det_dicts = []
        unmatched_det_dicts = []
        unmatched_idx_id_list = []
        matched_idx_id_list = []


        for det_idx, track_id in matched_objects:
            if self.tracks[track_id].is_banned():
                self.tracks[track_id].update({
                    "frame_id": frame_id,
                    "bbox_2d": None,
                    "keypoint_image": None,
                    "keypoint_world": None,
                    "score": None,
                    "class_name": None,
                    "dimensions": None,
                    "yaw": None,
                }, perspectiveTransform,
                    curr_time_stamp_video,
                    curr_time_stamp_system,
                    detectionZoneFilter=detectionZoneFilter)
            else:
                self.tracks[track_id].update(detections[det_idx].to_dict(), perspectiveTransform,
                                             curr_time_stamp_video, curr_time_stamp_system,
                                             detectionZoneFilter=detectionZoneFilter)


        for track_id in unmatched_tracks:
            self.tracks[track_id].update({
                "frame_id": frame_id,
                "bbox_2d": None,
                "keypoint_image": None,
                "keypoint_world": None,
                "score": None,
                "class_name": None,
                "dimensions": None,
                "yaw": None,
            }, perspectiveTransform,
                curr_time_stamp_video,
                curr_time_stamp_system,
                detectionZoneFilter=detectionZoneFilter)

        for det_idx in unmatched_dets:
            unmatched_det_dicts.append(detections[det_idx].to_dict())
            unmatched_idx_id_list.append(det_idx)

        for track_idx in self.tracks.keys():
            track_dict= {}
            track_dict["keypoint_world"] = self.tracks[track_idx].get_current_world_position()
            track_dict["dimensions"] = self.tracks[track_idx].get_mean_dimensions()
            track_dict["yaw"] = self.tracks[track_idx].get_orientation()

            matched_det_dicts.append(track_dict)


        intersection_matrix = self.calc_intersection(matched_det_dicts, unmatched_det_dicts)

        # Handle new detections
        for det_idx in unmatched_dets:
            if intersection_matrix.shape[1] == 0:
                # Create Track
                self.tracks[self.next_track_id] = Track(self.next_track_id, detections[det_idx].to_dict(),
                                                        curr_time_stamp_video, curr_time_stamp_system)
                self.next_track_id += 1
            else:
                # check if the detection has intersection with any of the matched detections
                # if not, create a new track
                det_id = unmatched_idx_id_list.index(det_idx)
                if np.sum(intersection_matrix[det_id, :]) == 0:
                    # Create Track
                    self.tracks[self.next_track_id] = Track(self.next_track_id, detections[det_idx].to_dict(),
                                                            curr_time_stamp_video, curr_time_stamp_system)
                    self.next_track_id += 1

    def calc_intersection(self, matched_det_dicts, unmatched_det_dicts, iou_threshold=0):
        """
        Calculates intersection matrix between unmatched and matched detections
        using 2D rotated rectangles (x, y, yaw) in world coordinates.

        Args:
            matched_det_dicts: List of detection dicts with 'keypoint_world', 'dimensions', and 'yaw'
            unmatched_det_dicts: Same for unmatched detections
            iou_threshold: Minimum IoU to count as intersection

        Returns:
            np.ndarray: binary matrix (len(unmatched), len(matched)) with 0 or 1
        """

        def create_box_polygon(center, dimensions, yaw_rad):
            cx, cy = center[:2]
            w, l = dimensions[:2]  # width across x, length across y

            # Corner points relative to center
            dx = w / 2
            dy = l / 2
            corners = np.array([
                [-dx, -dy],
                [dx, -dy],
                [dx, dy],
                [-dx, dy]
            ])

            # Rotation matrix
            cos_yaw = math.cos(yaw_rad)
            sin_yaw = math.sin(yaw_rad)
            R = np.array([[cos_yaw, -sin_yaw],
                          [sin_yaw, cos_yaw]])

            # Rotate and shift to global coordinates
            rotated = (R @ corners.T).T + np.array([cx, cy])
            return Polygon(rotated)

        n_unmatched = len(unmatched_det_dicts)
        n_matched = len(matched_det_dicts)
        intersection_matrix = np.zeros((n_unmatched, n_matched), dtype=int)

        for i, det_u in enumerate(unmatched_det_dicts):
            try:
                poly_u = create_box_polygon(det_u["keypoint_world"], det_u["dimensions"], det_u["yaw"])
            except:
                poly_u = None
            for j, det_m in enumerate(matched_det_dicts):

                if poly_u is None:
                    intersection_matrix[i, j] = 1
                    continue

                try:
                    poly_m = create_box_polygon(det_m["keypoint_world"], det_m["dimensions"], det_m["yaw"])
                except:
                    poly_m = None
                if poly_m is None:
                    intersection_matrix[i, j] = 0
                    continue

                if not poly_u.is_valid or not poly_m.is_valid:
                    continue

                intersection_area = poly_u.intersection(poly_m).area
                union_area = poly_u.union(poly_m).area

                iou = intersection_area / union_area if union_area > 0 else 0

                if iou > iou_threshold:
                    intersection_matrix[i, j] = 1

        return intersection_matrix

    def validate_motion_consistency(self, track, detection, velocity_thresh=2.0,
                                    angle_thresh=np.pi / 2, speed_factor=3, min_speed = 0.1):
        """
        Validate motion consistency for a match.

        Args:
            track (object): Existing track object.
            detection (array): [x, y] detection center.
            kalman_filter (object): Kalman filter instance.
            velocity_thresh (float): Allowed distance deviation.
            angle_thresh (float): Allowed angle deviation.
            speed_factor (float): Allowed speed change factor.

        Returns:
            bool: True if the match is valid, False otherwise.
        """

        # Predict next state using Kalman filter
        predicted_state = track.get_predicted_state()
        predicted_position = predicted_state[:3]
        predicted_velocity = predicted_state[3:6]

        # Get observed velocity from last known position
        last_position = np.array(track.get_last_position())
        observed_velocity = detection.keypoint_world - last_position

        # Distance validation
        distance_error = np.linalg.norm(predicted_position - detection.keypoint_world)

        ## Adjust thresholds for early-stage tracks
        track_duration = track.get_track_duration()
        adaptive_velocity_thresh = max(velocity_thresh, (distance_error*1.5) * 30)

        if track_duration < 150:
            return distance_error < adaptive_velocity_thresh

        # Use Kalman velocity directly
        predicted_speed = np.linalg.norm(predicted_velocity)

        last_3_position = track.get_hist_world_position()
        observed_velocity = detection.keypoint_world - last_3_position
        observed_speed = np.linalg.norm(observed_velocity)/3

        predicted_velocity = predicted_position - last_3_position
        predicted_speed = np.linalg.norm(predicted_velocity)/3


        #observed_speed = np.linalg.norm(observed_velocity)

        # Prevent division errors & handle low-speed cases
        # predicted_speed = max(predicted_speed, min_speed)
        # observed_speed = max(observed_speed, min_speed)

        # Speed consistency check with clamping for extreme cases
        speed_consistent = (observed_speed <= speed_factor * predicted_speed) and (
            observed_speed >= predicted_speed / speed_factor)
        # speed_consistent = True

        predicted_velocity = predicted_position - last_position
        observed_velocity = detection.keypoint_world - last_position

        # Angle consistency check
        if predicted_speed > 0 and observed_speed > 0:
            predicted_angle = np.arctan2(predicted_velocity[1], predicted_velocity[0])
            observed_angle = np.arctan2(observed_velocity[1], observed_velocity[0])
            angle_diff = np.abs(predicted_angle - observed_angle) % (2 * np.pi)
            angle_consistent = angle_diff < angle_thresh
        else:
            angle_consistent = True

        return (distance_error < velocity_thresh) and speed_consistent and angle_consistent


    def remove_finished_tracks(self):
        tracks_to_remove = []
        for track_id in self.tracks.keys():
            # Check whether the last 30 frames the object was not detected
            if not self.tracks[track_id].detected_last_frame():
                self.tracks[track_id].increase_not_detected_counter()
            else:
                self.tracks[track_id].reset_not_detected_counter()

            if self.tracks[track_id].get_not_detected_counter() > 30:
                if self.tracks[track_id].get_num_detected() > 10 and len(
                        self.tracks[track_id].history["detected"]) > 50:
                    if self.tracks[track_id].get_not_detected_counter() > 90:
                        tracks_to_remove.append(track_id)
                else:
                    tracks_to_remove.append(track_id)

            if len(self.tracks[track_id].history["detected"]) < 10 and self.tracks[
                track_id].get_not_detected_counter() > 6:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

    def get_active_tracks_as_dict(self, candidate_track_ids):

        active_tracks = {}
        for track_id, track in self.tracks.items():
            if track_id not in candidate_track_ids:
                active_tracks[track_id] = track

        return active_tracks


    def get_number_of_tracks(self):
        return len(self.tracks)

    def setTracks(self, tracks):

        self.tracks = tracks