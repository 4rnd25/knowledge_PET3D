"""
Created on Jan 29 2025 10:23

@author: ISAC - pettirsch
"""

import csv
import os
import pdb

import numpy as np
from sympy.physics.units import acceleration


class TrajectorySaver:
    def __init__(self, trajectory_saver_config, database_config, recordId,  enhanced=True, outputFolder=None,
                 filename=None, verbose=False):

        self.save_csv = trajectory_saver_config["save_csv"]
        self.save_database = trajectory_saver_config["save_database"]

        self.measSeries = database_config["measurementID"]
        self.sensorID = database_config["sensorID"]
        self.recordID = recordId

        self.enhanced = enhanced

        self.outputFolder = os.path.join(outputFolder, "Trajectories")
        if not os.path.exists(self.outputFolder):
            os.makedirs(self.outputFolder)

        if self.enhanced:
            self.filename_csv = filename.split(".")[0] + "_enhanced_trajectories.csv"
        else:
            self.filename_csv = filename.split(".")[0] + "_trajectories.csv"

        self.cls_id_dict = {"motorcycle": 110, "car": 210, "truck": 410, "bus": 610, "bicycle": 910, "person": 810,
                            "e-scooter": 920}

        if self.enhanced:
            self.columns = [
                "idMeasurementSeries", "idSensor", "idRecord", "FrameTimeStamp", "FrameTimeStamp_MicroSec",
                "idVehicle", "ObjectClass", "Length", "Width", "Height",
                "posXFit", "posYFit", "posZFit", "YawFit", "VxFit", "VyFit", "VzFit", "AxFit", "AyFit", "AzFit",
                "YawRateFit"]
        else:
            self.columns = [
                "idMeasurementSeries", "idSensor", "idRecord", "FrameTimeStamp", "FrameTimeStamp_MicroSec",
                "SystemTimeStamp",
                "SystemTimeStamp_MicroSec", "idVehicle", "ObjectClass", "Length", "Width", "Height", "posX", "posY", "posZ",
                "Yaw", "Vx", "Vy", "Vz", "Ax", "Ay", "Az", "YawRate", "posXFit", "posYFit", "posZFit", "YawFit",
                "VxFit", "VyFit", "VzFit", "AxFit", "AyFit", "AzFit", "YawRateFit"
            ]

        self.fileCreated = False

        self.verbose = verbose

    def save_matched_trajectories(self, matched_trajectories):

        for track in matched_trajectories:
            # Get Track
            # track = tracks[track_id]

            # Preprocess world positions
            worldPositions = track.get_world_positions()
            worldPositions_arr = np.array(worldPositions)
            posX = list(worldPositions_arr[:, 0])
            posY = list(worldPositions_arr[:, 1])
            posZ = list(worldPositions_arr[:, 2])

            yaws = track.get_yaws()

            # Get cls
            obj_cls = track.get_voted_class()
            obj_cls = [self.cls_id_dict[obj_cls]] * len(worldPositions)

            # Get dims
            obj_dims = track.get_mean_dimensions()
            obj_dims = [obj_dims] * len(worldPositions)

            # get timestamps
            timestamps_video = track.get_timestamps_video()
            timestamps_video_microsec = [ts.microsecond for ts in timestamps_video]

            timestamps_system = track.get_timestamps_system()
            timestamps_system_microsec = [ts.microsecond for ts in timestamps_system]

            v_xs, v_ys, v_zs = track.get_velocities()
            a_xs, a_ys, a_zs = track.get_accelerations()
            yaw_rates = track.get_yaw_rates()

            measIDs = [self.measSeries] * len(worldPositions)
            sensorIDs = [self.sensorID] * len(worldPositions)
            recordIDs = [self.recordID] * len(worldPositions)
            track_ids = [track_id] * len(worldPositions)

            # Fitted values
            worldPositions_fitted = track.get_world_positions_fitted()
            posXFit = list(worldPositions_fitted[:, 0])
            posYFit = list(worldPositions_fitted[:, 1])
            posZFit = list(worldPositions_fitted[:, 2])

            vxFit, vyFit, vzFit = track.get_velocities_fitted()
            axFit, ayFit, azFit = track.get_accelerations_fitted()

            yawFit = list(track.get_yaws_fitted())
            yaw_rates_fit = track.get_yaw_rates_fitted()

            # Save to csv
            if self.save_csv:
                self.save_csv_file(measIDs, sensorIDs, recordIDs, timestamps_video, timestamps_video_microsec,
                                   timestamps_system, timestamps_system_microsec, track_ids, obj_cls, obj_dims, posX,
                                   posY, posZ,
                                   yaws, v_xs, v_ys, v_zs, a_xs, a_ys, a_zs, yaw_rates, posXFit, posYFit, posZFit,
                                   yawFit, vxFit, vyFit, vzFit, axFit, ayFit, azFit, yaw_rates_fit)

            if self.save_database:
                self.save_to_database(track)


    def save_trajectories(self, tracks, finished_track_ids):

        for track_id in finished_track_ids:
            # Get Track
            track = tracks[track_id]

            # Preprocess world positions
            worldPositions = track.get_world_positions()
            worldPositions_arr = np.array(worldPositions)
            posX = list(worldPositions_arr[:, 0])
            posY = list(worldPositions_arr[:, 1])
            posZ = list(worldPositions_arr[:, 2])

            yaws = track.get_yaws()

            # Get cls
            obj_cls = track.get_voted_class()
            obj_cls = [self.cls_id_dict[obj_cls]] * len(worldPositions)

            # Get dims
            obj_dims = track.get_mean_dimensions()
            obj_dims = [obj_dims] * len(worldPositions)

            # get timestamps
            timestamps_video = track.get_timestamps_video()
            timestamps_video_microsec = [ts.microsecond for ts in timestamps_video]

            timestamps_system = track.get_timestamps_system()
            timestamps_system_microsec = [ts.microsecond for ts in timestamps_system]

            v_xs, v_ys, v_zs = track.get_velocities()
            a_xs, a_ys, a_zs = track.get_accelerations()
            yaw_rates = track.get_yaw_rates()

            measIDs = [self.measSeries] * len(worldPositions)
            sensorIDs = [self.sensorID] * len(worldPositions)
            recordIDs = [self.recordID] * len(worldPositions)
            track_ids = [track_id] * len(worldPositions)

            # Fitted values
            worldPositions_fitted = track.get_world_positions_fitted()
            posXFit = list(worldPositions_fitted[:, 0])
            posYFit = list(worldPositions_fitted[:, 1])
            posZFit = list(worldPositions_fitted[:, 2])

            vxFit, vyFit, vzFit = track.get_velocities_fitted()
            axFit, ayFit, azFit = track.get_accelerations_fitted()

            yawFit = list(track.get_yaws_fitted())
            yaw_rates_fit = track.get_yaw_rates_fitted()

            # Save to csv
            if self.save_csv:
                self.save_csv_file(measIDs, sensorIDs, recordIDs, timestamps_video, timestamps_video_microsec,
                                   timestamps_system, timestamps_system_microsec, track_ids, obj_cls, obj_dims, posX,
                                   posY, posZ,
                                   yaws, v_xs, v_ys, v_zs, a_xs, a_ys, a_zs, yaw_rates, posXFit, posYFit, posZFit,
                                   yawFit, vxFit, vyFit, vzFit, axFit, ayFit, azFit, yaw_rates_fit)

            if self.save_database:
                self.save_to_database(track)

    def save_csv_file(self, measIDs, sensorIDs, recordIDs, timestamps_video, timestamps_video_microsec, timestamps_system,
                       timestamps_system_microsec, track_ids, obj_cls, obj_dims, posX, posY, posZ, yaws, v_xs, v_ys, v_zs,
                       a_xs, a_ys, a_zs, yaw_rates, posXFit, posYFit, posZFit, yawFit, vxFit, vyFit, vzFit, axFit, ayFit,
                       azFit, yaw_rates_fit):

        if not self.fileCreated:
            with open(os.path.join(self.outputFolder, self.filename_csv), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.columns)
            self.fileCreated = True


        with open(os.path.join(self.outputFolder, self.filename_csv), mode='a', newline='') as file:
            writer = csv.writer(file)
            for i in range(len(measIDs)):
                writer.writerow(
                    [measIDs[i], sensorIDs[i], recordIDs[i], timestamps_video[i], timestamps_video_microsec[i],
                     timestamps_system[i], timestamps_system_microsec[i], track_ids[i], obj_cls[i], obj_dims[i][0],
                     obj_dims[i][1], obj_dims[i][2], posX[i], posY[i], posZ[i], yaws[i], v_xs[i], v_ys[i], v_zs[i],
                     a_xs[i], a_ys[i], a_zs[i], yaw_rates[i], posXFit[i], posYFit[i], posZFit[i], yawFit[i], vxFit[i],
                     vyFit[i], vzFit[i], axFit[i], ayFit[i], azFit[i], yaw_rates_fit[i]])

    def create_file(self):
        if not self.fileCreated:
            with open(os.path.join(self.outputFolder, self.filename_csv), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.columns)
            self.fileCreated = True
            print("File created: ", os.path.join(self.outputFolder, self.filename_csv))

    def check_processed(self):
        if os.path.exists(os.path.join(self.outputFolder, self.filename_csv)):
            return True
        return False

    def save_csv_file_enhanced(self, measIDs, sensorIDs, recordIDs, timestamps_video, timestamps_video_microsec, track_ids,
                          obj_cls, obj_dims, posX, posY, posZ, yaws, v_xs, v_ys, v_zs, a_xs, a_ys, a_zs, yaw_rates):

        if not self.fileCreated:
            with open(os.path.join(self.outputFolder, self.filename_csv), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.columns)
            self.fileCreated = True
            print("File created: ", os.path.join(self.outputFolder, self.filename_csv))

        with open(os.path.join(self.outputFolder, self.filename_csv), mode='a', newline='') as file:
            writer = csv.writer(file)
            for i in range(len(measIDs)):
                try:
                    writer.writerow(
                        [measIDs[i], sensorIDs[i], recordIDs[i], timestamps_video[i], timestamps_video_microsec[i],
                         track_ids[i], obj_cls[i], obj_dims[i][0], obj_dims[i][1], obj_dims[i][2], posX[i], posY[i], posZ[i],
                         yaws[i], v_xs[i], v_ys[i], v_zs[i], a_xs[i], a_ys[i], a_zs[i], yaw_rates[i]])
                except:
                    pdb.set_trace()

    def save_enhanced_trajectories(self, trajectories):

        for traj in trajectories:

            # Preprocess world positions
            worldPositions = traj.get_world_positions()
            worldPositions_arr = np.array(worldPositions)
            posX = list(worldPositions_arr[:, 0])
            posY = list(worldPositions_arr[:, 1])
            posZ = list(worldPositions_arr[:, 2])

            yaws = traj.get_yaws()

            # Get cls
            obj_cls = traj.get_class()
            obj_cls = [self.cls_id_dict[obj_cls]] * len(worldPositions)

            # Get dims
            obj_dims = traj.get_dimensions()
            obj_dims = [obj_dims] * len(worldPositions)

            # get timestamps
            timestamps_video = traj.get_timestamps_video()
            timestamps_video_microsec = traj.get_microseconds_video()

            velocities = traj.get_velocities()
            v_xs = velocities[:, 0]
            v_ys = velocities[:, 1]
            v_zs = velocities[:, 2]

            accelerations = traj.get_accelerations()
            a_xs = accelerations[:, 0]
            a_ys = accelerations[:, 1]
            a_zs = accelerations[:, 2]

            yaw_rates = traj.get_yaw_rates()

            track_id = traj.get_idVehicle()

            measIDs = [self.measSeries] * len(worldPositions)
            sensorIDs = [self.sensorID] * len(worldPositions)
            recordIDs = [self.recordID] * len(worldPositions)
            track_ids = [track_id] * len(worldPositions)

            # Save to csv
            if self.save_csv:
                self.save_csv_file_enhanced(measIDs, sensorIDs, recordIDs, timestamps_video, timestamps_video_microsec,
                                   track_ids, obj_cls, obj_dims, posX, posY, posZ, yaws, v_xs, v_ys, v_zs, a_xs,
                                            a_ys, a_zs, yaw_rates)

            if self.save_database:
                self.save_to_database(track)


    def save_enhanced_tracks(self, tracks):

        for track_id, track in tracks.items():

            # Get Track
            worldPositions = track.get_world_positions()
            worldPositions_arr = np.array(worldPositions)
            posX = list(worldPositions_arr[:, 0])
            posY = list(worldPositions_arr[:, 1])
            posZ = list(worldPositions_arr[:, 2])

            yaws = track.get_yaws()

            # Get cls
            obj_cls = track.get_voted_class()
            obj_cls = [self.cls_id_dict[obj_cls]] * len(worldPositions)

            # Get dims
            obj_dims = track.get_mean_dimensions()
            obj_dims = [obj_dims] * len(worldPositions)

            # get timestamps
            timestamps_all_video =  track.get_timestamps_video()
            timestamps_video = []
            timestamps_video_microsec = []
            for timestamp in timestamps_all_video:
                timestamps_video_microsec.append(timestamp.microsecond)
                timestamps_video.append(timestamp.replace(microsecond=0))

            velocities = track.get_velocities()
            v_xs = velocities[0]
            v_ys = velocities[1]
            v_zs = velocities[2]

            accelerations = track.get_accelerations()
            a_xs = accelerations[0]
            a_ys = accelerations[1]
            a_zs = accelerations[2]

            yaw_rates = track.get_yaw_rates()

            track_id = track.get_idVehicle()

            measIDs = [self.measSeries] * len(worldPositions)
            sensorIDs = [self.sensorID] * len(worldPositions)
            recordIDs = [self.recordID] * len(worldPositions)
            track_ids = [track_id] * len(worldPositions)

            # Save to csv
            if self.save_csv:
                self.save_csv_file_enhanced(measIDs, sensorIDs, recordIDs, timestamps_video, timestamps_video_microsec,
                                   track_ids, obj_cls, obj_dims, posX, posY, posZ, yaws, v_xs, v_ys, v_zs, a_xs,
                                            a_ys, a_zs, yaw_rates)

            if self.save_database:
                self.save_to_database(track)
        self.create_file()

    def restart(self, recordID=None,outputFolder=None, filename = None):

        self.recordID = recordID

        self.outputFolder = os.path.join(outputFolder, "Trajectories")
        if not os.path.exists(self.outputFolder):
            os.makedirs(self.outputFolder)
        if self.enhanced:
            self.filename_csv = filename.split(".")[0] + "_enhanced_trajectories.csv"
        else:
            self.filename_csv = filename.split(".")[0] + "_trajectories.csv"

        self.fileCreated = False


