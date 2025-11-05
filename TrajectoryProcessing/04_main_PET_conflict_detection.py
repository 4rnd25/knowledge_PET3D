"""
Created on May 07 2025 07:46

@author: ISAC - pettirsch
"""

import argparse
import os
import pandas as pd
import time
import csv
from datetime import datetime
from datetime import timedelta

from CommonUtils.ConfigUtils.read_config import load_config
from CommonTools.VideoReader.videoreader import VideoReader
from TrajectoryProcessing.TrajectoryManager.trajectorymanager import TrajectoryManager
from TrajectoryProcessing.InteractionDetector.pet3D_interactiondetector import Pet3DInteractionDetector
from TrajectoryProcessing.InteractionDetector.pet2D_interactiondetector import Pet2DInteractionDetector
from TrajectoryProcessing.TrajectoryPlotter.trajectoryplotter import TrajectoryPlotter
from Videoprocessing.OutputVideoWriting.conflictvideowriter import ConflictVideoWriter
from CommonTools.PerspectiveTransform.perspectiveTransform import PerspectiveTransform
from TrajectoryProcessing.Conflictsaver.conflictsaver import ConflictSaver

def build_time_csv_path(config_path, output_folder):
    cfg_base = os.path.splitext(os.path.basename(config_path))[0]
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    fname = f"{ts}_time_measurement_conflict_detection_{cfg_base}.csv"
    return os.path.join(output_folder, fname)

def main(config, start_rec_id, end_rec_id, plot_type='Both', database=False, save_video=False,
         verbose=False, config_path = None):
    recordIDlist = list(range(start_rec_id, end_rec_id + 1))

    # Get alphapetic order of folders in output folder
    # Get all folders in output folder
    folders = [f for f in os.listdir(config['output_config']['output_folder']) if
               os.path.isdir(os.path.join(config['output_config']['output_folder'], f))]
    # Sort folders
    folders.sort()

    # Adapt config
    config['database_config']['startrecordID'] = start_rec_id
    config['database_config']['endrecordID'] = end_rec_id

    # Load maneuver df
    meas_id = config['database_config']['measurementID']
    sens_id = config['database_config']['sensorID']
    maneuver_path = os.path.join(config['output_config']['output_folder'], "ManeuverClasses", f'ManeuverClasses_{meas_id}_{sens_id}.csv')
    maneuver_df = pd.read_csv(maneuver_path)

    # Create Interaction detectors
    pet2D_3_detector = Pet2DInteractionDetector(threshold=3, maneuver_df=maneuver_df, verbose=verbose)
    pet3D_3_detector = Pet3DInteractionDetector(threshold = 3,maneuver_df=maneuver_df, verbose=verbose)
    conflictDetectors =  [pet3D_3_detector] #[pet2D_3_detector, pet3D_3_detector]

    # Load video reader
    video_reader = VideoReader(config['input_config'], config['conflict_config']['buffer_duration'],
                               config['database_config'])

    # Create conflictsaver
    conflict_saver = ConflictSaver(
        os.path.join(config['output_config']['output_folder'], video_reader.getFilename().split('.')[0]),
        filename=video_reader.getFilename().split('.')[0],
        indicators=[indicator.get_name() + "_" + str(indicator.get_threshold()) + "s" for indicator in
                    conflictDetectors],
        measId=config['database_config']['measurementID'], sensID=config['database_config']['sensorID'],
        recId=start_rec_id, verbose=verbose)

    # Create trajectory manager
    trajectory_manager = TrajectoryManager(config['output_config']['output_folder'], video_reader.getFilename(),
                                           config['database_config'], start_rec_id,
                                           videoprocessing_config=None,
                                           imgsize=video_reader.get_frame_size(),
                                           database=database, lean=False, verbose=verbose)

    # Create trajectory plotter
    trajectory_plotter = TrajectoryPlotter(config['output_config']['output_folder'], video_reader.getFilename(),
                                           config["videoprocessing_config"]['plotting_config'],
                                           verbose=verbose)

    if save_video:
        # Create PerspectiveTransform object
        persTrans = PerspectiveTransform(config["videoprocessing_config"]["calibration_config"],
                                         imgSize=video_reader.get_frame_size(),
                                         databaseConfig=config["database_config"], recordID=start_rec_id,
                                         verbose=verbose, onlyWorldCamProj=True, bottom_map=False)
        # Create ConflictVideoWriter object
        conflictVideoWriter = ConflictVideoWriter(config['output_config']['output_folder'], video_reader.getFilename(),
                                                  config['output_video_config'], video_reader.fps,
                                                  video_reader.get_frame_size(), conflictWriter=True, verbose=verbose)

    # rows for timing CSV: [RecordID, Num Conflicts, Time]
    timing_rows = []

    for recordID in recordIDlist:

        # Adapt config to current recordID
        config['database_config']['startrecordID'] = recordID

        # Update folder and filename
        if save_video:
            # Update PerspectiveTransform object with new recordID
            persTrans.update_recordID(recordID, reloadCalibration=False)

            # Update Videoreader
            if video_reader.get_recordID() != recordID:
                video_reader.restart()
            filename = video_reader.getFilename()
            foldername = filename.split('.')[0]
            folder_idx = folders.index(foldername)

        else:
            if not recordID == start_rec_id:
                folder_idx = (folder_idx + 1) % len(folders)
                foldername = folders[folder_idx]
                filename = foldername + '.mov'
            else:
                filename = video_reader.getFilename()
                foldername = filename.split('.')[0]
                folder_idx = folders.index(foldername)

        print("Handle recordID: ", recordID, " in folder: ", foldername, " missing recordIDs: ",
              len(recordIDlist) - recordIDlist.index(recordID) - 1)

        # Update ConflictSaver object
        conflict_saver.update_record(recordID, os.path.join(config['output_config']['output_folder'], foldername),
                                     filename=foldername)

        # Update TrajectoryManager object
        trajectory_manager.update_record(filename=filename, recordID=recordID)

        # Update TrajectoryPlotter object
        trajectory_plotter.update_record(filename=filename)

        # Detect interactions
        t0 = time.perf_counter()
        num_conflicts_record = 0
        for detector in conflictDetectors:
            interactions = detector.detect_interactions(trajectory_manager.get_all_trajectories())

            num_conflicts_record += len(interactions)

            if len(interactions) == 0:
                continue

            plot_path = conflict_saver.get_plot_output_folder(
                detector.get_name() + "_" + str(detector.get_threshold()) + "s")
            video_path = conflict_saver.get_video_output_folder(
                detector.get_name() + "_" + str(detector.get_threshold()) + "s")

            # Save interactions
            for interaction in interactions:
                # Save interaction to conflict saver
                conflict_saver.save_conflict(interaction)

                exclude_from_video = ["Following", "Merging", "Diverging"]
                if interaction.maneuverType in exclude_from_video:
                    continue

                # Save conflict plot
                involved_trajectory_1 = trajectory_manager.get_trajectory_by_idVehicle(interaction.idVehicle1)
                involved_trajectory_2 = trajectory_manager.get_trajectory_by_idVehicle(interaction.idVehicle2)
                label1 = "ID: {}, Class: {}".format(involved_trajectory_1.idVehicle, involved_trajectory_1.get_class())
                label2 = "ID: {}, Class: {}".format(involved_trajectory_2.idVehicle, involved_trajectory_2.get_class())
                involved_trajectories = [involved_trajectory_1, involved_trajectory_2]
                label = interaction.get_label()
                conflict_time = interaction.get_conflict_time()
                trajectory_plotter.plot_conflict(involved_trajectories, label, conflict_time, label1, label2,
                                                 outputfolder=plot_path)

                if save_video:
                    # Get video snipped
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
                        interaction_time = interaction.get_conflict_time()

                        start_time = interaction_time - timedelta(seconds=10)
                        start_time_new = max(start_time_snippet, start_time)

                        end_time = interaction_time + timedelta(seconds=10)
                        end_time_new = min(end_time_snippet, end_time)

                        start_time_snippet = start_time_new
                        end_time_snippet = end_time_new

                    video_snippet = video_reader.get_video_snippet(start_time_snippet, end_time_snippet)

                    # Write video
                    if video_snippet is not None:
                        conflictVideoWriter.write_video(video_snippet, involved_trajectories, persTrans, label,
                                                        colors=config["videoprocessing_config"]["plotting_config"][
                                                            "colors"],
                                                        video_output_path=video_path)

        elapsed = time.perf_counter() - t0
        timing_rows.append([recordID, num_conflicts_record, elapsed])

        # Save conflits to csv
        conflict_saver.save_to_csv()

    # --- write timing CSV with totals row ---
    total_records = len(recordIDlist)
    total_conflicts = sum(row[1] for row in timing_rows)
    total_time = sum(row[2] for row in timing_rows)

    csv_path = build_time_csv_path(config_path or "", config['output_config']['output_folder'])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)  # default delimiter ','
        writer.writerow(["RecordID", "Num Conflicts", "Time"])
        for rid, nconf, tsec in timing_rows:
            writer.writerow([rid, nconf, f"{tsec:.6f}"])
        # totals row: RecordID=total number of records, Num Conflicts=total conflicts, Time=total seconds
        writer.writerow([total_records, total_conflicts, f"{total_time:.6f}"])
    print(f"Conflict-detection timing CSV written to: {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../Configs/Location_A.yaml',
                        help='Path to the config file')
    parser.add_argument('--plot', type=bool, default=False, help='Plot type: Both, Camera, World')
    parser.add_argument('--save_video', type=bool, default=False, help='Save video')
    parser.add_argument('--database', action='store_true', help='Use database')
    parser.add_argument('--verbose', action='store_true', help='Print debug information')
    parser.add_argument('--startRecID', type=int, default=1, help='Start record ID')
    parser.add_argument('--endRecID', type=int, default=135, help='End record ID')
    args = parser.parse_args()

    # Read config
    config = load_config(args.config)

    # Start main function
    args.plot_clusters_seperate = True
    args.verbose = True
    main(config, args.startRecID, args.endRecID, args.plot, args.database, args.save_video, args.verbose,
         config_path=args.config)
