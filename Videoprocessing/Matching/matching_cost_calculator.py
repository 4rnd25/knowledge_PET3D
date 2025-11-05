"""
Created on Jan 22 2025 09:27

@author: ISAC - pettirsch
"""

import numpy as np
import pdb
from scipy.linalg import inv

from shapely.affinity import rotate
from shapely.geometry import Polygon

from CommonUtils.BoxCalcUtils.calc_3d_box_corners import get_bev_corners


class MatchingCostCalculator:
    def __init__(self, matching_config, verbose=False):

        self.cost_threshold_dict = matching_config["cost_thresholds"]
        self.weight_factor_class_consistency = matching_config["weight_factor_class_consistency"]
        self.verbose = verbose

    def calculate_matching_cost(self, frameObjectManager, tracker):
        n = frameObjectManager.get_number_of_frame_objects()
        m = tracker.get_number_of_tracks()

        if m == 0:
            if n != 0:
                return n, None
            else:
                return None, None
        if n == 0:
            track_idx_list = []
            for track_id, track in tracker.get_tracks().items():
                track_idx_list.append(track_id)
            return None, track_idx_list

        # === Extract data ===
        det_center, det_dims, det_class, det_orientation, det_conf = self.get_detections_infos(frameObjectManager)
        track_center, track_dims, track_class, track_orientation, track_idx_list = self.get_tracks_infos(
            tracker)

        # === Rho = position + dimensions ===
        det_rho = np.concatenate([det_center, det_dims], axis=1)  # (n, 6)
        track_rho = np.concatenate([track_center, track_dims], axis=1)  # (m, 6)

        # === Compute pairwise Euclidean distances (n x m) ===
        diff = det_rho[:, None, :] - track_rho[None, :, :]  # shape: (n, m, 6)
        # Multiply the center difference by 2 to give it more weight
        diff[:, :, :3] *= 2
        euclidean = np.linalg.norm(diff, axis=2)  # (n, m)

        # === Compute pairwise orientation cosine distance ===
        det_vec = np.stack([np.cos(det_orientation), np.sin(det_orientation)], axis=1)  # (n, 2)
        track_vec = np.stack([np.cos(track_orientation), np.sin(track_orientation)], axis=1)  # (m, 2)

        # Normalize (in case angles aren't perfectly unit-length)
        det_vec /= np.linalg.norm(det_vec, axis=1, keepdims=True)
        track_vec /= np.linalg.norm(track_vec, axis=1, keepdims=True)

        # Cosine similarity (n x m)
        cos_sim = np.dot(det_vec, track_vec.T)  # (n, m)
        alpha = 2.0 - cos_sim  # ∈ [1, 2]

        base_cost = euclidean * alpha  # (n, m)

        # === Class penalty ===
        det_cls = np.array(det_class)[:, None]  # (n, 1)
        track_cls = np.array(track_class)[None, :]  # (1, m)
        match_mask = (det_cls == track_cls).astype(np.float32)  # (n, m)

        det_conf = np.array(det_conf).reshape(-1, 1)  # (n, 1)
        class_penalty = (1 - match_mask) * det_conf + match_mask * (1 - det_conf)  # (n, m)

        # === Final cost ===
        w = 0.5
        final_cost = base_cost * (1 + w * class_penalty)

        # === Apply cost threshold ===
        for id, cls in enumerate(track_cls[0]):
            assert cls in self.cost_threshold_dict, f"Class {cls} not found in cost threshold dictionary"
            cost_threshold = self.cost_threshold_dict[cls]
            mask = final_cost[:, id] > cost_threshold
            final_cost[mask, id] = 1001

        return final_cost, track_idx_list

    def get_detections_infos(self, frameObjectManager):
        center_list = []
        dimensions_list = []
        class_list = []
        orientation_list = []
        det_conf_list = []
        for detection in frameObjectManager.get_frame_objects():
            # Center
            center_list.append(detection.get_world_center())

            # Dimensions
            dimensions_list.append(detection.get_dimensions())

            # Class
            class_list.append(detection.get_class_name())

            # Orientation
            yaw = detection.get_yaw() % (2 * np.pi)  # Normalize to [0, 2pi]
            orientation_list.append(yaw)

            det_conf_list.append(detection.get_confidence())

        return np.array(center_list), np.array(dimensions_list), np.array(class_list), np.array(
            orientation_list), np.array(det_conf_list)

    def get_tracks_infos(self, tracker):
        center_list = []
        dimensions_list = []
        class_list = []
        orientation_list = []
        track_idx_list = []
        for track_id, track in tracker.get_tracks().items():
            track_idx_list.append(track_id)

            # Center
            center_list.append(track.get_current_world_position())

            # Dimensions
            dimensions_list.append(track.get_mean_dimensions())

            # Class
            class_list.append(track.get_voted_class())

            # Orientation
            orientation_list.append(track.get_orientation())

        return np.array(center_list), np.array(dimensions_list), np.array(class_list), np.array(
            orientation_list), track_idx_list
