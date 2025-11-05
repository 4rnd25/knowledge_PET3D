"""
Created on Mar 12 2025 10:47

@author: ISAC - pettirsch
"""

import numpy as np
from CommonUtils.BoxCalcUtils.calc_3d_box_corners import get_bev_corners
from shapely.geometry import Polygon
import datetime
from TrajectoryProcessing.Utils.Object_classes.conflict import Conflict


def compute_bev_iou_matrix(n_centers, n_dims, n_yaws, m_centers, m_dims, m_yaws):
    """
    Compute the BEV IoU matrix for two sets of oriented 2D bounding boxes.

    Returns an (N, M) IoU matrix.
    """
    # Compute 4-corner polygons for both sets
    n_corners = get_bev_corners(n_centers, n_dims, n_yaws)
    m_corners = get_bev_corners(m_centers, m_dims, m_yaws)

    # Convert to shapely polygons
    n_polygons = [Polygon(n_corners[i]) for i in range(n_corners.shape[0])]
    m_polygons = [Polygon(m_corners[i]) for i in range(m_corners.shape[0])]

    # Compute IoU matrix
    iou_matrix = np.zeros((len(n_polygons), len(m_polygons)))

    for i, n_poly in enumerate(n_polygons):
        for j, m_poly in enumerate(m_polygons):
            inter_area = n_poly.intersection(m_poly).area
            union_area = n_poly.union(m_poly).area
            iou_matrix[i, j] = inter_area / union_area if union_area > 0 else 0.0

    return iou_matrix


class Pet3DInteractionDetector:
    def __init__(self, threshold = 1, maneuver_df = None, verbose=False):
        self.threshold = threshold
        self.name = "PET3D"
        self.maneuver_df = maneuver_df
        self.verbose = verbose

    def detect_interactions(self, trajectories):

        # pre-checked tube intersection
        tube_collisions = self.pre_check_tube_intersection(trajectories)

        conflicts = []

        # Check for actual tube intersection
        for tube_collision in tube_collisions:
            if tube_collision[3] > self.threshold:
                continue

            traj1 = trajectories[tube_collision[0]]
            traj2 = trajectories[tube_collision[1]]

            traj1_centers = traj1.get_world_positions()
            traj1_dims = [[traj1.get_length(), traj1.get_width(), traj1.get_height()] for i in
                          range(len(traj1_centers))]
            traj1_dims = np.asarray(traj1_dims)
            traj1_yaws = traj1.get_yaws()

            traj2_centers = traj2.get_world_positions()
            traj2_dims = [[traj2.get_length(), traj2.get_width(), traj2.get_height()] for i in
                          range(len(traj2_centers))]
            traj2_dims = np.asarray(traj2_dims)
            traj2_yaws = traj2.get_yaws()

            iou_mat = compute_bev_iou_matrix(traj1_centers, traj1_dims, traj1_yaws, traj2_centers, traj2_dims,
                                             traj2_yaws)

            conlfict = np.where(iou_mat > 0)

            min_time_dif = np.inf
            all_complete_time_stamps_traj1 = traj1.get_all_complete_timestamps()
            all_complete_time_stamps_traj2 = traj2.get_all_complete_timestamps()
            smaller_thresh_counter = []
            smaller_thresh_i = []
            iou_list = []
            for i in range(len(conlfict[0])):
                time_diff = abs((all_complete_time_stamps_traj1[conlfict[0][i]] -
                                 all_complete_time_stamps_traj2[conlfict[1][i]]).total_seconds())
                if time_diff < min_time_dif:
                    min_time_dif = time_diff
                if time_diff < self.threshold:
                    if conlfict[0][i] not in smaller_thresh_counter:
                        smaller_thresh_counter.append(conlfict[0][i])
                        if all_complete_time_stamps_traj2[conlfict[1][i]] not in smaller_thresh_i:
                            smaller_thresh_i.append(all_complete_time_stamps_traj2[conlfict[1][i]])
                            iou_list.append(iou_mat[conlfict[0][i], conlfict[1][i]])

            smaller_thresh_list = smaller_thresh_counter
            smaller_thresh_counter = len(smaller_thresh_counter)

            # if smaller_thresh_counter < 10:
            #     continue
            # else:
            #     print("ok")

            # Check if the minimum time difference is within the threshold
            if min_time_dif > self.threshold:
                continue

            # Define conflict timestamp
            timestamp_1 = all_complete_time_stamps_traj1[conlfict[0][0]]
            timestamp_2 = all_complete_time_stamps_traj2[conlfict[1][0]]

            idx_traj_1 = conlfict[0][0]
            idx_traj_2 = conlfict[1][0]

            # Select the newer one
            if timestamp_1 > timestamp_2:
                timestamp_microsec = int(timestamp_1.microsecond)
                timestamp = timestamp_1.replace(microsecond=0)
                first_veh_idx = traj1.get_idVehicle()
            else:
                timestamp_microsec = int(timestamp_2.microsecond)
                timestamp = timestamp_2.replace(microsecond=0)
                first_veh_idx = traj2.get_idVehicle()

            timeStampSystem = datetime.datetime.now()
            timeStampSystem_microsec = int(timeStampSystem.microsecond)
            timeStampSystem = timeStampSystem.replace(microsecond=0)

            idVehicles = [traj1.get_idVehicle(), traj2.get_idVehicle()]
            classes = [traj1.get_class(), traj2.get_class()]
            clusters = [traj1.get_cluster(), traj2.get_cluster()]

            index_veh1 = idVehicles.index(first_veh_idx)
            if index_veh1 == 0:
                index_veh2 = 1
            else:
                index_veh2 = 0
            idVehicle1 = idVehicles[index_veh1]
            idVehicle2 = idVehicles[index_veh2]
            classVehicle1 = classes[index_veh1]
            classVehicle2 = classes[index_veh2]
            cluster1 = clusters[index_veh1]
            cluster2 = clusters[index_veh2]

            indicator = self.name
            posX = (traj1.get_world_positions()[conlfict[0][0]][0] + traj2.get_world_positions()[conlfict[1][0]][0]) / 2
            posY = (traj1.get_world_positions()[conlfict[0][0]][1] + traj2.get_world_positions()[conlfict[1][0]][1]) / 2
            value = min_time_dif

            # Check if cluster combination is in self.maneuver_df
            if self.maneuver_df is not None:
                try:
                    maneuver_type = self.maneuver_df.loc[
                        (self.maneuver_df['ClusterVehicle1'] == cluster1) &
                        (self.maneuver_df['ClusterVehicle2'] == cluster2)
                        ]['ManeuverClass'].values[0]
                except:
                    maneuver_type = None

            if maneuver_type is None or maneuver_type == "Check individual":
                yaws_1 = traj1.get_yaws()
                yaws_2 = traj2.get_yaws()

                yaws_1 = yaws_1[max(0, min(idx_traj_1, idx_traj_1 - 30)):idx_traj_1]
                yaws_2 = yaws_2[max(0, min(idx_traj_2, idx_traj_2 - 30)):idx_traj_2]
                yaw_1 = np.mean(yaws_1)
                yaw_2 = np.mean(yaws_2)
                delta_yaw_abs = abs(yaw_1 - yaw_2)
                delta_yaw = min(delta_yaw_abs, 2 * np.pi - delta_yaw_abs)
                # to degree
                delta_yaw = np.degrees(delta_yaw)
                if delta_yaw > 45:
                    maneuver_type = "Crossing"
                elif delta_yaw < 45 and delta_yaw > 15:
                    maneuver_type = "Merging"
                else:
                    maneuver_type = "Following"

            conflict = Conflict(timeStampVideo=timestamp,
                                timeStampVideoMicrosec=timestamp_microsec,
                                timeStamp=timeStampSystem,
                                timeStampMicrosec=timeStampSystem_microsec,
                                indicator=indicator,
                                idVehicle1=idVehicle1,
                                idVehicle2=idVehicle2,
                                vehicle_class_1=classVehicle1,
                                vehicle_class_2=classVehicle2,
                                vehicle_cluster_1=cluster1,
                                vehicle_cluster_2=cluster2,
                                posX=posX,
                                posY=posY,
                                value=value,
                                maneuverType=maneuver_type)
            conflicts.append(conflict)

            if self.verbose:
                print(f"[Pet3DTime] {maneuver_type} {idVehicle2} & {idVehicle1} hit at ({posX:.2f},{posY:.2f}) "
                      f"around {timestamp} Δt={min_time_dif:.3f}s")


        return conflicts

    def pre_check_tube_intersection(self, trajectories):
        """ Detect intersecting tubes between different clusters """
        tube_collisions = []

        # Compare only across different clusters
        for i, traj1 in enumerate(trajectories):
            for j, traj2 in enumerate(trajectories):

                # Avoid redundant checks
                if i >= j:
                    continue  # Avoid redundant checks

                # Avoid checks for only VRU trajectories
                if (
                        traj1.get_class() == "person" or traj1.get_class() == "bicycle" or traj1.get_class() == "e-scooter") and (
                        traj2.get_class() == "person" or traj2.get_class() == "bicycle" or traj2.get_class() == "e-scooter"):
                    continue

                # Check if trajectories have common time frames
                timeStamps_1 = traj1.get_time_stamps()
                timeStamps_2 = traj2.get_time_stamps()
                timeStamps_1_microsec = traj1.get_time_stamps_microsec()
                timeStamps_2_microsec = traj2.get_time_stamps_microsec()

                if not bool(set(timeStamps_1) & set(timeStamps_2)):
                    continue

                # Step 2: Check for actual tube intersection
                min_dist, time_stamp_idx_1, time_stamp_idx_2 = self.min_distance_between_trajectories(traj1, traj2)

                max_dim_1 = max(traj1.get_length(), traj1.get_width(), traj1.get_height())
                max_dim_2 = max(traj2.get_length(), traj2.get_width(), traj2.get_height())
                conflict_time_1 = timeStamps_1[time_stamp_idx_1].replace(
                    microsecond=timeStamps_1_microsec[time_stamp_idx_1])
                conflict_time_2 = timeStamps_2[time_stamp_idx_2].replace(
                    microsecond=timeStamps_2_microsec[time_stamp_idx_2])

                if conflict_time_2 > conflict_time_1:
                    pet = (conflict_time_2 - conflict_time_1).total_seconds()
                    conflict_time = conflict_time_1
                else:
                    pet = (conflict_time_1 - conflict_time_2).total_seconds()
                    conflict_time = conflict_time_2

                pet = round(pet, 2)

                if min_dist <= (max_dim_1 + max_dim_2) / 2:
                    tube_collisions.append([i, j, conflict_time, pet])

        return tube_collisions

    def min_distance_between_trajectories(self, traj1, traj2):
        """ Calculate the minimum distance between two trajectories """
        positions_1 = traj1.get_world_positions()
        positions_2 = traj2.get_world_positions()
        yaws_1 = traj1.get_yaws()
        yaws_2 = traj2.get_yaws()

        vel_1 = traj1.get_velocities()
        vel_2 = traj2.get_velocities()

        diff = positions_1[:, np.newaxis] - positions_2
        dist = np.linalg.norm(diff, axis=2)

        all_complete_time_stamps_traj1 = traj1.get_all_complete_timestamps()
        all_complete_time_stamps_traj2 = traj2.get_all_complete_timestamps()

        max_dim_1 = max(traj1.get_length(), traj1.get_width(), traj1.get_height())
        max_dim_2 = max(traj2.get_length(), traj2.get_width(), traj2.get_height())
        dim_thresh = (max_dim_1 + max_dim_2) / 2

        # Get all values smaller then dim_thresh
        dim_thresh_idx = np.where(dist < dim_thresh)
        if len(dim_thresh_idx[0]) == 0:
            min_dist = np.min(np.linalg.norm(diff, axis=2))
            time_stamp_idx_1 = np.where(dist == min_dist)[0][0]
            time_stamp_idx_2 = np.where(dist == min_dist)[1][0]

            return min_dist, time_stamp_idx_1, time_stamp_idx_2
        min_time_diff = 1000
        for i in range(dim_thresh_idx[0].shape[0]):
            if abs((all_complete_time_stamps_traj1[dim_thresh_idx[0][i]] - all_complete_time_stamps_traj2[
                dim_thresh_idx[1][i]]).total_seconds()) < min_time_diff:
                time_stamp_idx_1 = dim_thresh_idx[0][i]
                time_stamp_idx_2 = dim_thresh_idx[1][i]
                min_dist = dist[dim_thresh_idx[0][i], dim_thresh_idx[1][i]]
                min_time_diff = abs((all_complete_time_stamps_traj1[dim_thresh_idx[0][i]] -
                                     all_complete_time_stamps_traj2[dim_thresh_idx[1][i]]).total_seconds())

        return min_dist, time_stamp_idx_1, time_stamp_idx_2

    def get_name(self):
        return self.name

    def get_threshold(self):
        return self.threshold
