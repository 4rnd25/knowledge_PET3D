"""
Created on Feb 11 2025 11:20

@author: ISAC - pettirsch
"""

import argparse
import os
import csv
from datetime import datetime

from TrajectoryProcessing.TrajectoryManager.trajectorymanager import TrajectoryManager
from TrajectoryProcessing.TrajectoryClusterer.trajectoryclusterer import TrajectoryClusterer
from TrajectoryProcessing.TrajectoryPlotter.trajectoryplotter import TrajectoryPlotter
from CommonTools.PerspectiveTransform.perspectiveTransform import PerspectiveTransform

from CommonTools.VideoReader.videoreader import VideoReader
from CommonUtils.ConfigUtils.read_config import load_config

def build_time_csv_path(config_path, output_folder):
    cfg_base = os.path.splitext(os.path.basename(config_path))[0]
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    fname = f"{ts}_time_measurement_clustering_{cfg_base}.csv"
    return os.path.join(output_folder, fname)


def main(config, start_rec_id, end_rec_id, plot_type='Both', database=False, plot_clusters_seperate=False,
         verbose=False):

    cluster_record = False

    recordIDlist = list(range(start_rec_id, end_rec_id + 1))

    # Get alphapetic order of folders in output folder
    # Get all folders in output folder
    folders = [f for f in os.listdir(config['output_config']['output_folder']) if
               os.path.isdir(os.path.join(config['output_config']['output_folder'], f))]
    # Sort folders
    folders.sort()

    for recordID in recordIDlist:
        if recordID == start_rec_id:
            config['database_config']['startrecordID'] = recordID
            config['database_config']['endrecordID'] = recordID
            # Read in all trajectory
            if plot_type == 'Both' or plot_type == 'Camera':
                # Create video reader
                video_reader = VideoReader(config['input_config'], config['conflict_config']['buffer_duration'],
                                           config['database_config'])
                persTrans = PerspectiveTransform(
                    calibrationPath=config["videoprocessing_config"]["calibration_config"]["calibration_matrix_file"],
                    triangulationFacesPath=config["videoprocessing_config"]["calibration_config"]["calibration_faces_file"],
                    triangulationPointsPath=config["videoprocessing_config"]["calibration_config"]["calibration_points_file"],
                    calibration_type=config["videoprocessing_config"]["calibration_config"]["calibration_type"],
                    imageSize=video_reader.get_frame_size(), create_bottom_map=False, verbose=verbose)
                frame = video_reader.get_next_frame()
            else:
                video_reader = None
            filename = video_reader.getFilename()
            foldername = filename.split('.')[0]
            folder_idx = folders.index(foldername)
            # Create trajectory clusterer
            trajectory_clusterer = TrajectoryClusterer(config['cluster_config'],
                                                       cluster_mean_path=config['output_config']['output_folder'],
                                                       meas_id=config['database_config']['measurementID'],
                                                       sens_id=config['database_config']['sensorID'],
                                                       persTrans=persTrans,
                                                       verbose=verbose)
        else:
            folder_idx = (folder_idx + 1) % len(folders)
            foldername = folders[folder_idx]
            filename = foldername + '.mov'

        print("Handle recordID: ", recordID, " in folder: ", foldername, " missing recordIDs: ",
              len(recordIDlist) - recordIDlist.index(recordID) - 1)

        # Check if recordID is first recordID
        if recordID == start_rec_id:
            # Create Trajectory reader
            trajectory_manager = TrajectoryManager(config['output_config']['output_folder'], filename,
                                                   config['database_config'], recordID,
                                                   videoprocessing_config=config["videoprocessing_config"],
                                                   imgsize=video_reader.get_frame_size(),
                                                   database=database, verbose=verbose)
        else:
            if cluster_record:
                # Load recordID from database
                trajectory_manager.load_recordID(filename, recordID, clear=True)
            else:
                # Load recordID from file
                trajectory_manager.load_recordID(filename, recordID)

        if cluster_record:
            # Create Trajectory plotter
            print("Trajectory Plotts are saved in: ", filename)
            trajectory_plotter = TrajectoryPlotter(config['output_config']['output_folder'], filename,
                                                   config["videoprocessing_config"]['plotting_config'],
                                                   verbose=verbose)

            # Start processing
            print("Start clustering")
            if video_reader is not None:
                frame = video_reader.get_next_frame()
            else:
                frame = None
            num_cluster, timing = trajectory_clusterer.cluster_trajectories(trajectory_manager.get_all_trajectories(), frame)
            print("Clustering finished")

            # Update csv
            trajectory_manager.update_csv(filename, recordID)

            # Plot clusters
            print("Plot overview")
            if video_reader is None:
                trajectory_plotter.plot_all_trajectories(trajectory_manager.get_all_trajectories(), None, plot_type,
                                                         mean_trajectories=trajectory_manager.get_mean_trajectories(),
                                                         cluster=True, num_cluster=num_cluster)
            else:
                if frame is not None:
                    trajectory_plotter.plot_all_trajectories(trajectory_manager.get_all_trajectories(), frame,
                                                             plot_type,
                                                             mean_trajectories=trajectory_manager.get_mean_trajectories(),
                                                             cluster=True, num_cluster=num_cluster)

            # Plot trajectories seperately
            if plot_clusters_seperate:
                print("Plot clusters seperate")
                unique_cluster_labels = trajectory_manager.get_unique_clusters()
                for cluster_label in unique_cluster_labels:
                    trajectories = [track for track in trajectory_manager.get_all_trajectories() if
                                    track.get_cluster() == cluster_label]
                    frame = video_reader.get_next_frame()
                    trajectory_plotter.plot_single_cluster(trajectories, cluster_label, frame)

    if not cluster_record:
        # Create Trajectory plotter
        print("Trajectory Plotts are saved in: ", filename)
        trajectory_plotter = TrajectoryPlotter(config['output_config']['output_folder'], filename,
                                               config["videoprocessing_config"]['plotting_config'],
                                               verbose=verbose)

        # Start processing
        print("Start clustering")
        frame = video_reader.get_next_frame()
        num_cluster, timing = trajectory_clusterer.cluster_trajectories(trajectory_manager.get_all_trajectories(), frame)
        print("Clustering finished")

        # Write clusters to csv files
        for recordID in recordIDlist:
            trajectory_manager.update_csv(filename, recordID)

        # Plot clusters
        print("Plot overview")
        if video_reader is None:
            trajectory_plotter.plot_all_trajectories(trajectory_manager.get_all_trajectories(), None, plot_type,
                                                     mean_trajectories=trajectory_manager.get_mean_trajectories(),
                                                     cluster=True, num_cluster = num_cluster)
        else:
            if frame is None:
                frame = video_reader.get_next_frame()
            if frame is not None:
                trajectory_plotter.plot_all_trajectories(trajectory_manager.get_all_trajectories(), frame, plot_type,
                                                         mean_trajectories=trajectory_manager.get_mean_trajectories(),
                                                         cluster=True, num_cluster = num_cluster)

        # Plot trajectories seperately
        print("Plot clusters seperate")
        if plot_clusters_seperate:
            unique_cluster_labels = trajectory_manager.get_unique_clusters()
            for cluster_label in unique_cluster_labels:
                trajectories = [track for track in trajectory_manager.get_all_trajectories() if
                                track.get_cluster() == cluster_label]
                frame = video_reader.get_next_frame()
                trajectory_plotter.plot_single_cluster(trajectories, cluster_label, frame)

    # --- Write timing CSV ---
    num_records = len(recordIDlist)
    csv_path = build_time_csv_path(args.config, config['output_config']['output_folder'])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["Step", "Number of Records", "Time", ""])
        writer.writerow(["clustering", num_records, f"{timing.get('clustering', 0.0):.6f}", ""])
        writer.writerow(["assign_cluster", num_records, f"{timing.get('assign_cluster', 0.0):.6f}", ""])
    print("⏱Timing CSV written to:", csv_path)


    # Release video reader
    if video_reader is not None:
        video_reader.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../Configs/Location_A.yaml',
                        help='Path to the config file')
    parser.add_argument('--plot', type=str, default='Both', help='Plot type: Both, Camera, World')
    parser.add_argument('--database', action='store_true', help='Use database')
    parser.add_argument('--verbose', action='store_true', help='Print debug information')
    parser.add_argument('--startRecID', type=int, default=1, help='Start record ID')
    parser.add_argument('--endRecID', type=int, default=137, help='End record ID')
    parser.add_argument('--plot_clusters_seperate', action='store_true', help='Plot clusters seperate')
    args = parser.parse_args()

    # Read config
    config = load_config(args.config)

    # Start main function
    main(config, args.startRecID, args.endRecID, args.plot, args.database, args.plot_clusters_seperate, args.verbose)
