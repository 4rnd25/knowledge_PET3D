"""
Created on Jun 09 2025 18:15

@author: ISAC - pettirsch
"""

import argparse
import os

from CommonUtils.ConfigUtils.read_config import load_config
from CommonTools.VideoReader.videoreader import VideoReader
from Conflictprocessing.ConflictManager.conflictmanager import ConflictManager
from Conflictprocessing.ConflictVideoCreator.conflictVideoCreator import ConflictVideoCreator
from Conflictprocessing.ConflictPlotter.conflictPlotter import ConflictPlotter

def main(config, recordID, idVeh1, idVeh2, outputpath, degree, verbose):

    # Get folders list
    folders = [f for f in os.listdir(config['output_config']['output_folder']) if
               os.path.isdir(os.path.join(config['output_config']['output_folder'], f))]
    folders.sort()

    # Adapt configs
    config['database_config']['startrecordID'] = recordID
    config['database_config']['endrecordID'] = recordID
    recordIDlist = [recordID]

    # Load video reader
    video_reader = VideoReader(config['input_config'], config['conflict_config']['buffer_duration'],
                               config['database_config'])
    filename = video_reader.getFilename()
    foldername = filename.split('.')[0]
    folder_idx_start = folders.index(foldername)

    # Create ConflictVideoCreator
    conflict_video_creator = ConflictVideoCreator(config=config, verbose=verbose)

    # Create conflictplotter
    conflict_plotter = ConflictPlotter(config=config, verbose=verbose)

    # Create ConflictManager
    conflictManager = ConflictManager(config['output_config']['output_folder'], recordIDlist,
                                      folders[folder_idx_start:folder_idx_start + len(recordIDlist)],
                                      load_all=True, verbose=verbose)

    # Get all conflicts with filter
    filtered_conflicts = conflictManager.get_filtered_conflicts(
        filterIndicator="PET3D",
        filterClass1="all",
        filterClass2="all",
        filterCluster1="all",
        filterCluster2="all",
        Auto_Type="all",
        auto_rule_flag="all",
        value=3,
        LOF1=0,
        LOF2=0,
        LOFOR=False,
        idVehicle1 = idVeh1,
        idVehicle2 = idVeh2
    )


    # Check if filtered conflicts are empty
    if filtered_conflicts.empty:
        print(f"No conflicts found for vehicles {idVeh1} and {idVeh2} in record {recordID}.")
        return

    # Create conflict video for each conflict
    for idx, conflict in filtered_conflicts.iterrows():
        outputpath = os.path.join(outputpath, "Shaddow_Knowledge_PET_{}_{}_{}_{}_Record_{}".format(
            conflict['Vehicle_Class_1'],
            conflict['idVehicle1'],
            conflict['Vehicle_Class_2'],
            conflict['idVehicle2'],
            recordID
        ))
        outputpath_video = os.path.join(outputpath, "Video_Images")
        outputpath_plot = os.path.join(outputpath, "Plots")
        outputpath_plot_ts = os.path.join(outputpath, "Plots_per_Timestamp")
        # Create folder if it does not exist
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        if not os.path.exists(outputpath_video):
            os.makedirs(outputpath_video)
        if not os.path.exists(outputpath_plot):
            os.makedirs(outputpath_plot)
        if not os.path.exists(outputpath_plot_ts):
            os.makedirs(outputpath_plot_ts)

        # Plot conflict
        # conflict_time_3d = conflict['FrameTimeStamp']
        # Set microseconds
        # conflict_time_3d = conflict_time_3d.replace(microsecond=conflict['FrameTimeStamp_MicroSec'])
        conflict_plotter.svgplot_conflict_per_timestamp(recordName = foldername,
            recordID=recordID,
            conflict_indicator="Knowledge_PET",
            idVehicle1=conflict['idVehicle1'],
            idVehicle2=conflict['idVehicle2'],
            class1=conflict['Vehicle_Class_1'],
            class2=conflict['Vehicle_Class_2'],
            cluster1=conflict['Vehicle_Cluster_1'],
            cluster2=conflict['Vehicle_Cluster_2'],
            conflictVideoPath=outputpath_plot_ts,
            timeStamp=conflict['FrameTimeStamp'],
            timeStamp_micro=conflict['FrameTimeStamp_MicroSec'],
            auto_maneuver=conflict['Auto_Type'],
            value=conflict['Value'],
            degree = degree)
        conflict_plotter.svgplot_conflict(outputpath=outputpath_plot, recordID=recordID, idVehicle1 = conflict['idVehicle1'],
                                       idVehicle2 = conflict['idVehicle2'])

        # Create conflict video
        conflict_video_creator.create_conflict_video(
            recordName = foldername,
            recordID=recordID,
            conflict_indicator="Knowledge_PET",
            idVehicle1=conflict['idVehicle1'],
            idVehicle2=conflict['idVehicle2'],
            class1=conflict['Vehicle_Class_1'],
            class2=conflict['Vehicle_Class_2'],
            cluster1=conflict['Vehicle_Cluster_1'],
            cluster2=conflict['Vehicle_Cluster_2'],
            conflictVideoPath=outputpath_video,
            timeStamp=conflict['FrameTimeStamp'],
            timeStamp_micro=conflict['FrameTimeStamp_MicroSec'],
            auto_maneuver=conflict['Auto_Type'],
            value=conflict['Value'],
            save_images = True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../Configs/Location_B.yaml',
                        help='Path to the config file')
    parser.add_argument('--RecordID', type=int, default=101658, help='Start record ID')
    parser.add_argument('--idVeh1', type=int, default=571, help='End record ID')
    parser.add_argument('--idVeh2', type=int, default=570, help='End record ID')
    parser.add_argument('--verbose', action='store_true', help='Print debug information')
    parser.add_argument('--degree', type=int, default=340, help='Degree for plot')
    args = parser.parse_args()

    output_path = "/data/"

    # Read config
    config = load_config(args.config)

    # Start main function
    main(config, args.RecordID, args.idVeh1, args.idVeh2, output_path, args.degree, args.verbose)