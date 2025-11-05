"""
Created on May 29 2025 14:32

@author: ISAC - pettirsch
"""

from datetime import timedelta
from datetime import datetime

from CommonTools.VideoReader.videoreader import VideoReader
from Videoprocessing.OutputVideoWriting.conflictvideowriter import ConflictVideoWriter
from TrajectoryProcessing.TrajectoryReader.trajectorymanager import TrajectoryManager
from CommonTools.PerspectiveTransform.perspectiveTransform import PerspectiveTransform

class ConflictVideoCreator:
    def __init__(self, config, verbose=False):
        """
        Initialize the ConflictVideoCreator.

        Parameters:
            video_reader (VideoReader): An instance of VideoReader to read video frames.
            outputFolder (str): The folder where the output video will be saved.
            verbose (bool): If True, print additional information.
        """
        self.config = config
        self.verbose = verbose

    def create_conflict_video(self, recordName, recordID, conflict_indicator, idVehicle1,
                              idVehicle2, class1, class2, cluster1, cluster2, conflictVideoPath, timeStamp, timeStamp_micro,
                              auto_maneuver, value, save_images=False):

        # Set startRecordID and endRecordID in config
        self.config['database_config']['startrecordID'] = recordID
        self.config['database_config']['endrecordID'] = recordID

        # Load video reader
        video_reader = VideoReader(self.config['input_config'], self.config['conflict_config']['buffer_duration'],
                                   self.config['database_config'])

        # Initialize the ConflictVideoWriter
        conflictVideoWriter = ConflictVideoWriter(self.config['output_config']['output_folder'], video_reader.getFilename(),
                                                  self.config['output_video_config'], video_reader.fps,
                                                  video_reader.get_frame_size(), conflictWriter=True, verbose=self.verbose)

        # Intialize perstrans
        persTrans = PerspectiveTransform(
            calibrationPath=self.config["calibration_config"]["calibration_matrix_file"],
            triangulationFacesPath=self.config["calibration_config"]["calibration_faces_file"],
            triangulationPointsPath=self.config["calibration_config"]["calibration_points_file"],
            calibration_type=self.config["calibration_config"]["calibration_type"],
            imageSize=video_reader.get_frame_size(), verbose=self.verbose)

        # Load trajectories
        trajectory_manager = TrajectoryManager(self.config['output_config']['output_folder'], video_reader.getFilename(),
                                               self.config['database_config'], recordID,
                                               videoprocessing_config=None,
                                               imgsize=video_reader.get_frame_size(),
                                               database=False, lean=False, verbose=self.verbose)

        # Get involved trajectores
        involved_trajectory_1 = trajectory_manager.get_trajectory_by_idVehicle(idVehicle1)
        involved_trajectory_2 = trajectory_manager.get_trajectory_by_idVehicle(idVehicle2)
        involved_trajectories = [involved_trajectory_1, involved_trajectory_2]


        # Get video snippet
        start_time_traj_1 = involved_trajectory_1.get_time_stamps()[0].replace(
            microsecond=involved_trajectory_1.get_time_stamps_microsec()[0])
        end_time_traj_1 = involved_trajectory_1.get_time_stamps()[-1].replace(
            microsecond=involved_trajectory_1.get_time_stamps_microsec()[-1])
        start_time_traj_2 = involved_trajectory_2.get_time_stamps()[0].replace(
            microsecond=involved_trajectory_2.get_time_stamps_microsec()[0])
        end_time_traj_2 = involved_trajectory_2.get_time_stamps()[-1].replace(
            microsecond=involved_trajectory_2.get_time_stamps_microsec()[-1])

        # Get start and end time of video snippet
        start_time_snippet = min(start_time_traj_1, start_time_traj_2)
        end_time_snippet = max(end_time_traj_1, end_time_traj_2)

        # Check if time is above 20s
        if (end_time_snippet - start_time_snippet) > timedelta(seconds=20):
            # Get interaction time
            timeStamp = datetime.strptime(timeStamp, '%Y-%m-%d %H:%M:%S')
            interaction_time = timeStamp.replace(microsecond=timeStamp_micro)

            start_time = interaction_time - timedelta(seconds=10)
            start_time_new = max(start_time_snippet, start_time)

            end_time = interaction_time + timedelta(seconds=10)
            end_time_new = min(end_time_snippet, end_time)

            start_time_snippet = start_time_new
            end_time_snippet = end_time_new
        video_snippet = video_reader.get_video_snippet(start_time_snippet, end_time_snippet)

        # Set label
        if isinstance(timeStamp, datetime):
            interaction_time = timeStamp
        else:
            # Convert timeStamp string to datetime object
            if timeStamp_micro is not None:
                interaction_time = datetime.strptime(timeStamp, '%Y-%m-%d %H:%M:%S')

        value_rounded = round(value, 2)
        label = f"{interaction_time.strftime('%Y%m%dT%H%M%S')}_{auto_maneuver}_{class1}_{idVehicle1}_{cluster1}_{class2}_{idVehicle2}_{cluster2}_PET3D_{value_rounded}"

        # Write video
        if video_snippet is not None:
            conflictVideoWriter.write_video(video_snippet, involved_trajectories, persTrans, label,
                                            colors=self.config["videoprocessing_config"]["plotting_config"][
                                                "colors"],
                                            video_output_path=conflictVideoPath, save_images = save_images)



