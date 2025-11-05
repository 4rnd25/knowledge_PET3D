"""
Created on May 07 2025 07:46

@author: ISAC - pettirsch
"""

import argparse
import os
import pandas as pd

import numpy as np
from CommonUtils.ConfigUtils.read_config import load_config
from CommonTools.VideoReader.videoreader import VideoReader
from Conflictprocessing.ConflictManager.conflictmanager import ConflictManager
from Conflictprocessing.ConflictVideoPlayer.conflictvideoplayer import ConflictVideoPlayer


def main(config, startRecID, endRecID, database, verbose):
    option_1 = {
        "name": "PET2D 1s",
        "indicator": "PET2D",
        "auto_rule_flag": "all",
        "threshold": 1,
        "lof_1": 0,
        "lof_2": 0
    }

    option_2 = {
        "name": "PET3D 1s",
        "indicator": "PET3D",
        "auto_rule_flag": "all",
        "threshold": 1,
        "lof_1": 0,
        "lof_2": 0
    }

    option_3 = {
        "name": "PET3D-Knowledge",
        "indicator": "PET3D",
        "threshold": 3,
        "auto_rule_flag": False,
        "lof_1": 0,
        "lof_2": 1.5 # Second arriving vehicle
    }

    option_4 = {
        "name": "PET3D-Knowledge LSTM",
        "indicator": "PET3D",
        "threshold": 3,
        "auto_rule_flag": False,
        "lof_1": 0,
        "lof_2": 0,
        "LOF2_LSTM": 1.5  # Second arriving vehicle
    }

    options = [option_1, option_2, option_3, option_4]

    # Create df with one columns [Type, PET2D 1s, PET3D  1s, PET3D-Knowledge]
    overview_df = pd.DataFrame()

    # Create recordlist
    recordIDlist = list(range(startRecID, endRecID + 1))

    # Get alphapetic order of folders in output folder
    # Get all folders in output folder
    folders = [f for f in os.listdir(config['output_config']['output_folder']) if
               os.path.isdir(os.path.join(config['output_config']['output_folder'], f))]
    # Sort folders
    folders.sort()

    # Adapt config
    config['database_config']['startrecordID'] = startRecID
    config['database_config']['endrecordID'] = endRecID

    # Load video reader
    video_reader = VideoReader(config['input_config'], config['conflict_config']['buffer_duration'],
                               config['database_config'])
    filename = video_reader.getFilename()
    foldername = filename.split('.')[0]
    folder_idx_start = folders.index(foldername)
    folder_idx_end = folder_idx_start + len(recordIDlist) - 1

    # Create ConflictManager
    conflictManager = ConflictManager(config['output_config']['output_folder'], recordIDlist,
                                      folders[folder_idx_start:folder_idx_end + 1],
                                      load_all=True, verbose=verbose)

    types = conflictManager.get_all_unique_types()
    #types = ["Turn-left-across-path", "Turn-right-across-path", "Crossing"]
    # types = ["Merging"]
    types.append("all")


    for type in types:
        row = {"Type": type}
        for option in options:
            # Get all conflicts with filter

            rule_flag = option['auto_rule_flag']
            lof1 = option["lof_1"]
            lof2 = option["lof_2"]

            if 'LOF2_LSTM' in option:
                filtered_conflicts = conflictManager.get_filtered_conflicts(
                    filterIndicator=option['indicator'],
                    filterClass1="all",
                    filterClass2="all",
                    filterCluster1="all",
                    filterCluster2="all",
                    Auto_Type=type,
                    auto_rule_flag=rule_flag,
                    value=option["threshold"],
                    LOF1=lof1,
                    LOF2=lof2,
                    LOF2_LSTM=option['LOF2_LSTM']
                )
            else:
                filtered_conflicts = conflictManager.get_filtered_conflicts(
                    filterIndicator=option['indicator'],
                    filterClass1="all",
                    filterClass2="all",
                    filterCluster1="all",
                    filterCluster2="all",
                    Auto_Type=type,
                    auto_rule_flag=rule_flag,
                    value=option["threshold"],
                    LOF1=lof1,
                    LOF2=lof2
                )
            # Get number of conflicts
            if type != "all":
                tp_pd = filtered_conflicts[filtered_conflicts['Manual_Conflict_Check'] == True]
                row[option['name'] + " - TP"] = len(tp_pd)
                fp_pd = filtered_conflicts[filtered_conflicts['Manual_Conflict_Check'] == False]
                row[option['name'] + " - FP"] = len(fp_pd)
                # get all conflicts with manual check empty
                not_checked_pd = filtered_conflicts[filtered_conflicts['Manual_Conflict_Check'].isna()]
                row[option['name'] + " - Not checked"] = len(not_checked_pd)
            else:
                row[option['name'] + " - TP"] = sum(overview_df[option['name'] + " - TP"])
                row[option['name'] + " - FP"] = sum(overview_df[option['name'] + " - FP"])
                row[option['name'] + " - Not checked"] = sum(overview_df[option['name'] + " - Not checked"])

        overview_df = overview_df._append(row, ignore_index=True)

    # Save overview_df to csv
    overview_df.to_csv(os.path.join(config['output_config']['output_folder'], "detected_conflict_overview_with_LSTM.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../Configs/Location_A.yaml',
                        help='Path to the config file')
    parser.add_argument('--database', action='store_true', help='Use database')
    parser.add_argument('--verbose', action='store_true', help='Print debug information')
    parser.add_argument('--startRecID', type=int, default=1, help='Start record ID')
    parser.add_argument('--endRecID', type=int, default=855, help='End record ID')
    args = parser.parse_args()

    # Read config
    config = load_config(args.config)

    # Start main function
    args.verbose = True
    main(config, args.startRecID, args.endRecID, args.database, args.verbose)
