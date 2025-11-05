"""
Created on Feb 03 2025 14:03

@author: ISAC - pettirsch
"""
import os
import pandas as pd
import numpy as np

from TrajectoryProcessing.Utils.Object_classes.trajectory import Trajectory

from CommonTools.PerspectiveTransform.perspectiveTransform import PerspectiveTransform
from CommonTools.Filtering.detection_zone_filter import DetectionZoneFilter
from Videoprocessing.Plotting.plotter import Plotter

from tslearn.barycenters import dtw_barycenter_averaging


class TrajectoryManager:
    def __init__(self, output_folder, video_filename, database_config, record_id, fitted=True, database=False,
                 videoprocessing_config=None, imgsize=None, lean=False, verbose=False):

        self.mother_output_folder = output_folder
        self.recordName = video_filename.split('.')[0]
        self.recordFolder = os.path.join(self.mother_output_folder, self.recordName)

        self.database_config = database_config
        self.videoprocessing_config = videoprocessing_config
        self.record_id = record_id
        self.database = database
        self.verbose = verbose
        self.fitted = fitted
        self.lean = lean

        self.number_of_trajectories = 0

        self.measurementID = database_config["measurementID"]
        self.sensID = database_config["sensorID"]

        self.meanTrajectories = {}

        if videoprocessing_config is not None:
            self.perspectiveTransform = PerspectiveTransform(
                calibrationPath=videoprocessing_config["calibration_config"]["calibration_matrix_file"],
                triangulationFacesPath=videoprocessing_config["calibration_config"]["calibration_faces_file"],
                triangulationPointsPath=videoprocessing_config["calibration_config"]["calibration_points_file"],
                calibration_type=videoprocessing_config["calibration_config"]["calibration_type"],
                imageSize=imgsize, create_bottom_map=False, verbose=verbose)
            # Get Detectionzonefilter for the plotter
            self.detectionZoneFilter = DetectionZoneFilter(videoprocessing_config["detection_zone_config"])
            imageBoarders, worldBoarders = self.getDetectionZoneBoarders()
            self.framePlotter = Plotter(videoprocessing_config['plotting_config'], imageBoarders, worldBoarders,
                                        verbose=verbose)
        else:
            self.perspectiveTransform = None

        self.trajectories = {}

        cls_id_dict = {"motorcycle": 110, "car": 210, "truck": 410, "bus": 610, "bicycle": 910, "person": 810,
                       "e-scooter": 920}
        self.id_cls_dict = {v: k for k, v in cls_id_dict.items()}

        self.timeStamp_trajectory_df = None
        self.read_trajectories()

    def update_record(self, filename, recordID):
        self.recordName = filename.split('.')[0]
        self.recordFolder = os.path.join(self.mother_output_folder, self.recordName)
        self.record_id = recordID
        self.timeStamp_trajectory_df = None
        self.trajectories = {}
        self.read_trajectories()


    def read_trajectories(self):
        if self.database:
            pass
        else:
            self.read_Trajectories_FromCSV()

    def load_recordID(self, video_filename, record_id, fitted=True, database=False,
                      videoprocessing_config=None, imgsize=None, clear=False, verbose=False):
        if clear:
            self.timeStamp_trajectory_df = None
            self.trajectories = {}
            # self.read_trajectories()

        self.record_id = record_id
        self.recordName = video_filename.split('.')[0]
        self.recordFolder = os.path.join(self.mother_output_folder, self.recordName)

        self.read_Trajectories_FromCSV()

    def match_trajectories(self, origin_id, continuation_id, recordID, persTrans=None):
        origin_id = str(self.measurementID) + "_" + str(self.sensID) + "_" + str(recordID) + "_" + str(origin_id)
        continuation_id = str(self.measurementID) + "_" + str(self.sensID) + "_" + str(recordID) + "_" + str(
            continuation_id)

        self.trajectories[origin_id].append_trajectory(self.trajectories[continuation_id], persTrans)

    def remove_trajectory(self, traj_id, recordID):
        traj_id = str(self.measurementID) + "_" + str(self.sensID) + "_" + str(recordID) + "_" + str(traj_id)
        if traj_id in self.trajectories.keys():
            del self.trajectories[traj_id]

    def get_trajectory(self, traj_id, recordID):
        traj_id = str(self.measurementID) + "_" + str(self.sensID) + "_" + str(recordID) + "_" + str(traj_id)
        return self.trajectories[traj_id]

    def get_max_idVehicle(self, recordID):
        recIdVehicles = [int(veh.split("_")[-1]) for veh in self.trajectories.keys() if
                         int(veh.split("_")[-2]) == recordID]
        if len(recIdVehicles) == 0:
            return -1
        return max(recIdVehicles)

    def split_trajectory(self, trajectory_id, split_indices, recordID, maxIdVehicle, persTrans):
        trajectory_id = str(self.measurementID) + "_" + str(self.sensID) + "_" + str(recordID) + "_" + str(
            trajectory_id)
        trajectory_to_split = self.trajectories[trajectory_id]
        remove = False
        for i, split_idx in enumerate(split_indices):
            if i == len(split_indices) - 1:
                continue
            split_start = split_indices[i]
            split_end = split_indices[i + 1]
            if split_end - split_start < 2:  # 15
                continue

            veh_name = str(self.measurementID) + "_" + str(self.sensID) + "_" + str(recordID) + "_" + str(
                maxIdVehicle)
            positions = trajectory_to_split.positions[split_start:split_end + 1]
            yaws = trajectory_to_split.yaws[split_start:split_end + 1]
            velocities = trajectory_to_split.velocities[split_start:split_end + 1]
            accelerations = trajectory_to_split.accelerations[split_start:split_end + 1]
            yaw_rates = trajectory_to_split.yaw_rates[split_start:split_end + 1]
            timeStamps = trajectory_to_split.timeStamps[split_start:split_end + 1]
            timeStampsMicroSec = trajectory_to_split.timeStampsMicroSec[split_start:split_end + 1]

            if not self.lean:
                self.trajectories[veh_name] = Trajectory(maxIdVehicle,
                                                         trajectory_to_split.cls,
                                                         trajectory_to_split.length, trajectory_to_split.width,
                                                         trajectory_to_split.height, positions, yaws, velocities,
                                                         accelerations, yaw_rates,
                                                         timeStamps, timeStampsMicroSec,
                                                         trajectory_to_split.cluster_label,
                                                         persTrans)
            else:
                self.trajectories[veh_name] = Trajectory(maxIdVehicle, trajectory_to_split.cls,
                                                         trajectory_to_split.length, trajectory_to_split.width,
                                                         trajectory_to_split.height, positions, yaws, None,
                                                         None, None, timeStamps,
                                                         timeStampsMicroSec, trajectory_to_split.cluster_label,
                                                         persTrans)
            maxIdVehicle += 1
            remove = True

        return maxIdVehicle, remove

    def read_Trajectories_FromCSV(self):

        self.prefereEnhanced = True
        self.useEnhanced = False
        trajectoryFolder = os.path.join(self.recordFolder, 'Trajectories')
        if not self.prefereEnhanced:
            trajectoryFileName = self.recordName + '_trajectories.csv'
            trajectoryFile = os.path.join(trajectoryFolder, trajectoryFileName)
        else:
            trajectoryFileName = self.recordName + '_enhanced_trajectories.csv'
            trajectoryFile = os.path.join(trajectoryFolder, trajectoryFileName)
            # Check if file exists
            if not os.path.exists(trajectoryFile):
                trajectoryFileName = self.recordName + '_trajectories.csv'
                trajectoryFile = os.path.join(trajectoryFolder, trajectoryFileName)
            else:
                self.useEnhanced = True
                self.fitted = True

        # Check if file exists
        if not os.path.exists(trajectoryFile):
            raise FileNotFoundError(f"Trajectory file not found: {trajectoryFile}")

        trajectory_df = pd.read_csv(trajectoryFile)

        # Convert FrameTimeStamp to datetime and remove microseconds
        trajectory_df['FrameTimeStamp'] = pd.to_datetime(trajectory_df['FrameTimeStamp'], format="mixed").dt.floor('S')

        # Save the columns timeStampFrame and timeStampFrameMicroSec and idVehicle
        self.timeStamp_trajectory_df = trajectory_df[["FrameTimeStamp", "FrameTimeStamp_MicroSec", "idVehicle"]]

        # Get unique idVehicle values
        unique_vehicles = trajectory_df["idVehicle"].unique()

        for idVeh in unique_vehicles:
            vehicle_df = trajectory_df[trajectory_df["idVehicle"] == idVeh]
            cls = self.id_cls_dict[int(vehicle_df["ObjectClass"].iloc[0])]
            length = vehicle_df["Length"].iloc[0]
            width = vehicle_df["Width"].iloc[0]
            height = vehicle_df["Height"].iloc[0]

            if "Cluster" in vehicle_df.columns:
                cluster = vehicle_df["Cluster"].iloc[0]
            else:
                cluster = None

            timeStamps = vehicle_df["FrameTimeStamp"].tolist()
            timeStampsMicroSec = vehicle_df["FrameTimeStamp_MicroSec"].tolist()

            if self.fitted:
                positions_x = vehicle_df["posXFit"].tolist()
                positions_y = vehicle_df["posYFit"].tolist()
                positions_z = vehicle_df["posZFit"].tolist()
                positions = np.array([positions_x, positions_y, positions_z]).T

                yaws = vehicle_df["YawFit"].tolist()

                if not self.lean:
                    yaw_rates = vehicle_df["YawRateFit"].tolist()

                    vx = vehicle_df["VxFit"].tolist()
                    vy = vehicle_df["VyFit"].tolist()
                    vz = vehicle_df["VzFit"].tolist()
                    velocities = np.array([vx, vy, vz]).T

                    ax = vehicle_df["AxFit"].tolist()
                    ay = vehicle_df["AyFit"].tolist()
                    az = vehicle_df["AzFit"].tolist()
                    accelerations = np.array([ax, ay, az]).T
            else:
                positions_x = vehicle_df["posX"].tolist()
                positions_y = vehicle_df["posY"].tolist()
                positions_z = vehicle_df["posZ"].tolist()
                positions = np.array([positions_x, positions_y, positions_z]).T

                yaws = vehicle_df["Yaw"].tolist()

                if not self.lean:
                    yaw_rates = vehicle_df["YawRate"].tolist()

                    vx = vehicle_df["Vx"].tolist()
                    vy = vehicle_df["Vy"].tolist()
                    vz = vehicle_df["Vz"].tolist()
                    velocities = np.array([vx, vy, vz]).T

                    ax = vehicle_df["Ax"].tolist()
                    ay = vehicle_df["Ay"].tolist()
                    az = vehicle_df["Az"].tolist()
                    accelerations = np.array([ax, ay, az]).T

            veh_name = str(self.measurementID) + "_" + str(self.sensID) + "_" + str(self.record_id) + "_" + str(idVeh)

            if not self.lean:
                positions = positions[10:,:]
                positions = positions[:-10,:]
                yaws = yaws[10:]
                yaws = yaws[:-10]
                velocities = velocities[10:,:]
                velocities = velocities[:-10,:]
                accelerations = accelerations[10:,:]
                accelerations = accelerations[:-10,:]
                yaw_rates = yaw_rates[10:]
                yaw_rates = yaw_rates[:-10]
                timeStamps = timeStamps[10:]
                timeStamps = timeStamps[:-10]
                timeStampsMicroSec = timeStampsMicroSec[10:]
                timeStampsMicroSec = timeStampsMicroSec[:-10]
                self.trajectories[veh_name] = Trajectory(idVeh, cls, length, width, height, positions, yaws, velocities,
                                                         accelerations, yaw_rates,
                                                         timeStamps, timeStampsMicroSec, cluster,
                                                         self.perspectiveTransform)
            else:
                self.trajectories[veh_name] = Trajectory(idVeh, cls, length, width, height, positions, yaws, None,
                                                         None, None,
                                                         timeStamps, timeStampsMicroSec, cluster,
                                                         self.perspectiveTransform)

    def update_csv(self, filename, record_id):
        """
        Updates the CSV file by adding/updating the "Cluster" column based on trajectory data.
        If the column doesn't exist, it is created.
        """
        self.record_id = record_id
        self.recordName = filename.split('.')[0]
        self.recordFolder = os.path.join(self.mother_output_folder, self.recordName)

        trajectoryFolder = os.path.join(self.recordFolder, 'Trajectories')
        self.useEnhanced = False
        if not self.prefereEnhanced:
            trajectoryFileName = self.recordName + '_trajectories.csv'
            trajectoryFile = os.path.join(trajectoryFolder, trajectoryFileName)
        else:
            trajectoryFileName = self.recordName + '_enhanced_trajectories.csv'
            trajectoryFile = os.path.join(trajectoryFolder, trajectoryFileName)
            # Check if file exists
            if not os.path.exists(trajectoryFile):
                trajectoryFileName = self.recordName + '_trajectories.csv'
                trajectoryFile = os.path.join(trajectoryFolder, trajectoryFileName)
            else:
                self.useEnhanced = True
                self.fitted = True

        # Load CSV file
        trajectory_df = pd.read_csv(trajectoryFile)

        # Check if "Cluster" column exists, if not, create it
        if "Cluster" not in trajectory_df.columns:
            trajectory_df["Cluster"] = np.nan  # Initialize column with NaN

        # Update cluster names based on trajectories
        for veh_name, trajectory in self.trajectories.items():
            if veh_name.split("_")[-2] != str(record_id):
                continue
            if hasattr(trajectory, "cluster_label"):
                if trajectory.cluster_label is not None:
                    trajectory_df.loc[
                        trajectory_df["idVehicle"] == trajectory.idVehicle, "Cluster"] = trajectory.cluster_label

        # Save the updated DataFrame back to the CSV file
        trajectory_df.to_csv(trajectoryFile, index=False)
        print(f"CSV file updated with cluster names: {trajectoryFile}")

    def get_all_trajectories(self):
        return [trajectory for trajectory in self.trajectories.values()]

    def get_number_of_trajectories(self):
        # test = self.trajectories.keys()
        # test_len = len(test)
        return len(self.trajectories.keys())

    def get_vehicle_distribution(self):
        day_hours_vehicle_distribution = {}
        # Iterate over trajectories
        for trajectory in self.trajectories.values():
            # Check first timestamp to get day_hour key
            day = trajectory.timeStamps[0].strftime("%Y%m%d")
            hour = trajectory.timeStamps[0].hour
            day_hour_key = day + "_" + str(hour)
            if not day_hour_key in day_hours_vehicle_distribution.keys():
                day_hours_vehicle_distribution[day_hour_key] = {"Motorcycles": 0, "Cars": 0, "Trucks": 0, "Buses": 0,
                                                                "Pedestrians": 0,
                                                                "Bicycles": 0, "E-Scooters": 0, "all": 0}
            cls = trajectory.cls
            if cls == "motorcycle":
                day_hours_vehicle_distribution[day_hour_key]["Motorcycles"] += 1
            elif cls == "car":
                day_hours_vehicle_distribution[day_hour_key]["Cars"] += 1
            elif cls == "truck":
                day_hours_vehicle_distribution[day_hour_key]["Trucks"] += 1
            elif cls == "bus":
                day_hours_vehicle_distribution[day_hour_key]["Buses"] += 1
            elif cls == "person":
                day_hours_vehicle_distribution[day_hour_key]["Pedestrians"] += 1
            elif cls == "bicycle":
                day_hours_vehicle_distribution[day_hour_key]["Bicycles"] += 1
            elif cls == "e-scooter":
                day_hours_vehicle_distribution[day_hour_key]["E-Scooters"] += 1
            else:
                raise ValueError(f"Unknown class: {cls}")
            day_hours_vehicle_distribution[day_hour_key]["all"] += 1

        return day_hours_vehicle_distribution

    def getDetectionZoneBoarders(self):
        imageBoarders = self.detectionZoneFilter.getImageBorders()
        if imageBoarders is not None:
            worldBoarders = [self.perspectiveTransform.pixelToStreePlane(np.asarray(imageBoarders_curr)) for
                             imageBoarders_curr in
                             imageBoarders]
        else:
            worldBoarders = None
        return imageBoarders, worldBoarders

    def get_active_tracks(self, curr_time_stamp):

        active_tracks = {}

        curr_time_stamp_microsec = curr_time_stamp.microsecond
        curr_time_stamp_without_microsec = curr_time_stamp.replace(microsecond=0)

        # Search self.timeStamp_trajectory_df for the current time stamp
        # Target timestamp
        target_timestamp = pd.to_datetime(curr_time_stamp_without_microsec).floor('S')
        target_microsec = curr_time_stamp_microsec

        # Get all track IDs with timestamp == target_timestamp and timestamp_microsec == target_microsec
        track_ids = self.timeStamp_trajectory_df[
            (self.timeStamp_trajectory_df["FrameTimeStamp"] == target_timestamp) & (
                    self.timeStamp_trajectory_df["FrameTimeStamp_MicroSec"] == target_microsec)]["idVehicle"].tolist()
        track_ids = [str(self.measurementID) + "_" + str(self.sensID) + "_" + str(self.record_id) + "_" + str(track_id)
                    for track_id in track_ids]

        for track_id in track_ids:
            active_tracks[track_id] = self.trajectories[track_id].get_current_track(curr_time_stamp)

        return active_tracks

    def calculate_mean_trajectories(self):

        # Get all cluster labels
        cluster_labels_all = [track.get_cluster() for track in self.trajectories.values()]

        # Get unique cluster labels
        unique_cluster_labels = np.unique(cluster_labels_all)

        for cluster_label in unique_cluster_labels:
            if cluster_label == str(-1):
                continue

            cluster_trajectories = [track for track in self.trajectories.values() if
                                    track.get_cluster() == cluster_label]
            cluster_mean_trajectory = self.dtw_mean_trajectory(cluster_trajectories)

            cls = cluster_trajectories[0].get_class()

            self.meanTrajectories[cluster_label] = Trajectory(None, cls, None, None, None,
                                                              cluster_mean_trajectory, None, None,
                                                              None, None, None,
                                                              None, self.perspectiveTransform)
            self.meanTrajectories[cluster_label].set_cluster_label(cluster_label)

    def dtw_mean_trajectory(self, trajectories):
        positions = [np.array(trajectory.get_world_positions()) for trajectory in trajectories]

        mean_trajectory = dtw_barycenter_averaging(positions, max_iter=50)

        return mean_trajectory

    def get_mean_trajectories(self):
        return self.meanTrajectories

    def get_trajectory_by_idVehicle(self, idVehicle):
        key = str(self.measurementID) + "_" + str(self.sensID) + "_" + str(self.record_id) + "_" + str(idVehicle)
        if key in self.trajectories.keys():
            return self.trajectories[key]
        else:
            return None

    def get_unique_clusters(self):
        cluster_labels_all = [track.get_cluster() for track in self.trajectories.values()]
        unique_cluster_labels = np.unique(cluster_labels_all)

        return unique_cluster_labels

    def release(self):

        self.trajectories = None
        self.perspectiveTransform = None
        self.framePlotter = None
        self.detectionZoneFilter = None
        self.meanTrajectories = None
        self.timeStamp_trajectory_df = None
        self.record_id = None
        self.recordName = None
        self.recordFolder = None

