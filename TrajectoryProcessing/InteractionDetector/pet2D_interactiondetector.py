"""
Created on May 07 2025 07:48

@author: ISAC - pettirsch
"""

import numpy as np
from TrajectoryProcessing.Utils.Object_classes.conflict import Conflict
import datetime
from shapely.geometry import LineString, Point

from CommonUtils.TrajectoryUtils.downsample_trajectories import downsample_by_distance

class Pet2DInteractionDetector:
    def __init__(self, threshold = 1, maneuver_df = None, verbose=False):
        self.threshold = threshold
        self.name = "PET2D"
        self.maneuver_df = maneuver_df
        self.verbose = verbose

    def detect_interactions(self, trajectories):
        """
        Returns a list of (i, j) index pairs whose 2D center-lines intersect.
        """
        conflicts  = []

        for i, traj1 in enumerate(trajectories):
            pts1 = [(x, y) for x, y, *_ in traj1.get_world_positions()]
            line1 = LineString(pts1)
            times1 = traj1.get_all_complete_timestamps()

            for j, traj2 in enumerate(trajectories[i+1:], start=i+1):

                # Avoid redundant checks
                if i >= j:
                    continue

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

                pts2 = [(x, y) for x, y, *_ in traj2.get_world_positions()]
                times2 = traj2.get_all_complete_timestamps()
                line2 = LineString(pts2)
                events = []

                line1 = LineString(pts1)
                if not line1.intersects(line2):
                    continue

                # walk every segment in traj1 × every segment in traj2
                for a in range(len(pts1) - 1):
                    seg1 = LineString([tuple(pts1[a]), tuple(pts1[a + 1])])
                    t1a, t1b = times1[a], times1[a + 1]
                    if seg1.length == 0:
                        continue
                    for b in range(len(pts2) - 1):
                        seg2 = LineString([tuple(pts2[b]), tuple(pts2[b + 1])])
                        t2a, t2b = times2[b], times2[b + 1]
                        if seg2.length == 0:
                            continue

                        if not seg1.intersects(seg2):
                            continue

                        inter = seg1.intersection(seg2)
                        # if they overlap in a segment, pick midpoint
                        if inter.geom_type == 'LineString':
                            inter = Point(inter.interpolate(0.5, normalized=True))

                        # compute fractional travel along each segment
                        f1 = seg1.project(inter) / seg1.length
                        f2 = seg2.project(inter) / seg2.length

                        # interpolate timestamps
                        dt1 = (t1b - t1a).total_seconds()
                        inter_t1 = t1a + np.timedelta64(round(f1 * dt1 * 1e6), 'us')
                        dt2 = (t2b - t2a).total_seconds()
                        inter_t2 = t2a + np.timedelta64(round(f2 * dt2 * 1e6), 'us')

                        if inter_t1 <= inter_t2:
                            fist_veh_idx = traj1.get_idVehicle()
                            second_veh_idx = traj2.get_idVehicle()
                        else:
                            fist_veh_idx = traj2.get_idVehicle()
                            second_veh_idx = traj1.get_idVehicle()

                        # Δt and chosen conflict timestamp
                        dt_sec = abs((inter_t1 - inter_t2).total_seconds())
                        conflict_ts = max(inter_t1, inter_t2)

                        events.append((conflict_ts, dt_sec, inter.x, inter.y, fist_veh_idx))

                if not events:
                    continue

                # pick the event with smallest Δt
                best = min(events, key=lambda e: e[1])
                ts, dt_min, x, y, fist_veh_idx = best

                # Check if the event is within the threshold
                if dt_min > self.threshold:
                    continue



                def find_closest_index(timestamps, target):
                    """
                    Return the index of the element in `timestamps` whose absolute difference
                    to `target` is smallest.
                    """
                    return min(
                        range(len(timestamps)),
                        key=lambda i: abs(timestamps[i] - target)
                    )

                timestamp_microsec = int(ts.microsecond)
                timeStamp = ts.replace(microsecond=0)

                timeStampSystem = datetime.datetime.now()
                timeStampSystem_microsec = int(timeStampSystem.microsecond)
                timeStampSystem = timeStampSystem.replace(microsecond=0)

                idVehicles = [traj1.get_idVehicle(), traj2.get_idVehicle()]
                classes = [traj1.get_class(), traj2.get_class()]
                clusters = [traj1.get_cluster(), traj2.get_cluster()]

                index_veh1 = idVehicles.index(fist_veh_idx)
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
                posX = x
                posY = y
                value = dt_min

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
                    trajectories = [traj1, traj2]
                    traj1_downsampled = downsample_by_distance(trajectories[index_veh1].get_world_positions(), num_points=5)
                    traj2_downsampled = downsample_by_distance(trajectories[index_veh2].get_world_positions(), num_points=5)

                    traj1_conflict = trajectories[index_veh1]
                    traj2_conflict = trajectories[index_veh2]

                    yaws_1 = traj1_conflict.get_yaws()
                    yaws_2 = traj2_conflict.get_yaws()

                    all_complete_timestamps_1 = traj1_conflict.get_all_complete_timestamps()
                    all_complete_timestamps_2 = traj2_conflict.get_all_complete_timestamps()
                    idx_traj_1 = find_closest_index(all_complete_timestamps_1, ts)
                    idx_traj_2 = find_closest_index(all_complete_timestamps_2, ts)

                    yaws_1 = yaws_1[max(0,min(idx_traj_1, idx_traj_1-30)):idx_traj_1]
                    yaws_2 = yaws_2[max(0,min(idx_traj_2, idx_traj_2-30)):idx_traj_2]
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

                conflict = Conflict(timeStampVideo=timeStamp,
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
                    print(f"[Pet2DTime] {maneuver_type} {idVehicle2} & {idVehicle1} hit at ({x:.2f},{y:.2f}) "
                          f"around {ts} Δt={dt_min:.3f}s")

        return conflicts

    def get_name(self):
        return self.name

    def get_threshold(self):
        return self.threshold
