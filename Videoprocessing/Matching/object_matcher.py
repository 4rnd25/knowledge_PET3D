"""
Created on Jan 17 2025 09:51

@author: ISAC - pettirsch
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


class ObjectMatcher:
    def __init__(self, matching_config, verbose=False):

        self.verbose = verbose

    def match_objects(self, cost_matrix, track_idx_list):
        """
        """
        self.cost_threshold = 8.15

        # Check if cost_matrix is single value
        if isinstance(cost_matrix, (int, float)):
            matches = []
            unmatched_dets = []
            unmatched_tracks = []
            for i in range(cost_matrix):
                unmatched_dets.append(i)
                # matches.append((i, -1))
            return matches, unmatched_dets, unmatched_tracks

        if cost_matrix is None:
            matches = []
            unmatched_dets = []
            if track_idx_list is None:
                unmatched_tracks = []
            else:
                unmatched_tracks = track_idx_list
            return matches, unmatched_dets, unmatched_tracks

        # Apply greedy matching
        matches, unmatched_dets, unmatched_tracks = greedy_match_fast(cost_matrix, track_idx_list, threshold=1000)

        return matches, unmatched_dets, unmatched_tracks



def greedy_match_fast(cost_matrix, track_idx_list, threshold=np.inf):
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
        if i not in matched_dets and track_idx_list[j] not in matched_tracks:
            matches.append((i, track_idx_list[j]))
            matched_dets.add(i)
            matched_tracks.add(track_idx_list[j])

    unmatched_dets = [i for i in range(n) if i not in matched_dets]
    unmatched_tracks = [track_idx_list[j] for j in range(m) if track_idx_list[j] not in matched_tracks]

    return matches, unmatched_dets, unmatched_tracks
