"""
Created on May 07 2025 07:46

@author: ISAC - pettirsch
"""

import argparse
import os

import numpy as np
from CommonUtils.ConfigUtils.read_config import load_config
from CommonTools.VideoReader.videoreader import VideoReader
from Conflictprocessing.ConflictManager.conflictmanager import ConflictManager
from Conflictprocessing.ConflictVideoPlayer.conflictvideoplayer import ConflictVideoPlayer
from Conflictprocessing.ConflictVideoCreator.conflictVideoCreator import ConflictVideoCreator
import pandas as pd


def main(config, startRecID, endRecID, database, filterIndicator, filterClass1,
         filterClass2, filterCluster1, filterCluster2, auto_type, value, rule_flag, lof1, lof2, LOFOR, skip_checked, verbose):
    """
    Main function for conflict filtering.
    """
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

    # Create ConflictVideoCreator
    conflict_video_creator = ConflictVideoCreator(config=config,
                                                  verbose=verbose)

    # Create Conflictplayer
    conflict_video_player = ConflictVideoPlayer(superoutputFolder=config['output_config']['output_folder'],
                                                verbose=verbose, conflictVideoCreator = conflict_video_creator)

    # Get all conflicts with filter
    #rule_flag = "all"
    filtered_conflicts = conflictManager.get_filtered_conflicts(
        filterIndicator=filterIndicator,
        filterClass1=filterClass1,
        filterClass2=filterClass2,
        filterCluster1=filterCluster1,
        filterCluster2=filterCluster2,
        Auto_Type= auto_type,
        auto_rule_flag=rule_flag,
        value = value,
        LOF1=lof1,
        LOF2=lof2,
        LOF2_LSTM=0,
        LOFOR=LOFOR
    )

    number_no_following = len(filtered_conflicts[filtered_conflicts['Auto_Type'] != 'Following'])

    # Ignore following and diverging
    filtered_conflicts = filtered_conflicts[filtered_conflicts['Auto_Type'] != 'Following']
    filtered_conflicts = filtered_conflicts[filtered_conflicts['Auto_Type'] != 'Diverging']

    number_no_following = len(filtered_conflicts[filtered_conflicts['Auto_Type'] != 'Following'])

    print("number of filtered conflicts: ", len(filtered_conflicts))
    print("number of filtered conflicts without following: ", number_no_following)


    # Iterate over all recordIds in filtered conflicts
    for idx, recordId in enumerate(filtered_conflicts['idRecord']):

        print(f"Check {idx + 1} of {len(filtered_conflicts)}")

        if skip_checked:
            # Check if Manual_Type, Manual_Rule_Flag, Manual_Conflict_Check is not empty
            if not pd.isna(filtered_conflicts.iloc[idx]['Manual_Type']) and \
                    not pd.isna(filtered_conflicts.iloc[idx]['Manual_Rule_Flag']) and \
                    not pd.isna(filtered_conflicts.iloc[idx]['Manual_Conflict_Check']):
                continue

        # Skip is lof2 is >= 1.5
        # if filtered_conflicts.iloc[idx]['LOF2_LSTM'] >= 1.5:
        #     continue
        # Skip if manual check is False
        if filtered_conflicts.iloc[idx]['Manual_Conflict_Check'] == False:
            continue

        # RecordName
        record_idx = recordIDlist.index(recordId)
        recordName = folders[folder_idx_start + record_idx]

        # Get conflict indicator idVehicle 1 and idVehicle 2 and class1 and class2
        conflict_indicator = filtered_conflicts.iloc[idx]['Indicator']
        idVehicle1 = filtered_conflicts.iloc[idx]['idVehicle1']
        idVehicle2 = filtered_conflicts.iloc[idx]['idVehicle2']
        class1 = filtered_conflicts.iloc[idx]['Vehicle_Class_1']
        class2 = filtered_conflicts.iloc[idx]['Vehicle_Class_2']
        # Get cluster1 and cluster2
        cluster1 = int(filtered_conflicts.iloc[idx]['Vehicle_Cluster_1'])
        cluster2 = int(filtered_conflicts.iloc[idx]['Vehicle_Cluster_2'])

        value = filtered_conflicts.iloc[idx]['Value']
        maneuver = filtered_conflicts.iloc[idx]['Auto_Type']
        rule_flag = filtered_conflicts.iloc[idx]['Auto_Rule_Flag']

        print(f"Record: {recordName}, Conflict: {conflict_indicator}, Maneuver: {maneuver},  Value: {value}, Class 1: {class1}, Class 2: {class2}, Rule: {rule_flag}")
        print(f"Record-ID: {recordId}, Vehicle 1 ID: {idVehicle1}, Vehicle 2 ID: {idVehicle2}")


        lof1 = filtered_conflicts.iloc[idx]['LOF1']
        lof2 = filtered_conflicts.iloc[idx]['LOF2']
        lof2_LSTM = filtered_conflicts.iloc[idx]['LOF2_LSTM']
        print(f"LOF1: {lof1}, LOF2: {lof2}")
        print(f"LOF2_LSTM: {lof2_LSTM}")

        # Print Manual_Conflict_Check
        print(f"Manual_Conflict_Check: {filtered_conflicts.iloc[idx]['Manual_Conflict_Check']}, Manual_Type: {filtered_conflicts.iloc[idx]['Manual_Type']}, Manual_Rule_Flag: {filtered_conflicts.iloc[idx]['Manual_Rule_Flag']}")



        # Print Cluster
        print(f"Cluster 1: {cluster1}, Cluster 2: {cluster2}")

        conflict_flag, maneuver, rule_flag = conflict_video_player.show_conflict(recordName, conflict_indicator,
                                                                                 idVehicle1, idVehicle2, class1, class2,
                                                                                 cluster1, cluster2, filtered_conflicts.iloc[idx]["idRecord"],
                                                                                 filtered_conflicts.iloc[idx]["FrameTimeStamp"],
                                                                                 filtered_conflicts.iloc[idx]["FrameTimeStamp_MicroSec"],
                                                                                 filtered_conflicts.iloc[idx]['Value'], maneuver
                                                                                 )

        if conflict_flag is not None:
            # Load conflict df
            record_folder = os.path.join(config['output_config']['output_folder'], str(recordName))
            record_conflict_folder = os.path.join(record_folder, "Conflicts")
            csv_path = os.path.join(record_conflict_folder, str(recordName) + '_Conflicts.csv')
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file {csv_path} does not exist.")
            record_conflict_df = pd.read_csv(csv_path)
            # Get conflict row
            conflict_row = record_conflict_df.loc[
                (record_conflict_df['Indicator'] == conflict_indicator) &
                (record_conflict_df['idVehicle1'] == idVehicle1) &
                (record_conflict_df['idVehicle2'] == idVehicle2)
            ]
            # Set Manual_Conflict_Check, Manual_Type, Manual_Rule_Flag
            if not conflict_row.empty:
                record_conflict_df.at[conflict_row.index[0], 'Manual_Conflict_Check'] = conflict_flag
                record_conflict_df.at[conflict_row.index[0], 'Manual_Type'] = maneuver
                record_conflict_df.at[conflict_row.index[0], 'Manual_Rule_Flag'] = rule_flag
                # Save conflict df
                record_conflict_df.to_csv(csv_path, index=False)

                # Set Manual_Conflict_Check, Manual_Type, Manual_Rule_Flag in filtered_conflicts
                filtered_conflicts.at[idx, 'Manual_Conflict_Check'] = conflict_flag
                filtered_conflicts.at[idx, 'Manual_Type'] = maneuver
                filtered_conflicts.at[idx, 'Manual_Rule_Flag'] = rule_flag


        print(f"Conflict flag: {conflict_flag}, Maneuver: {maneuver}, Rule: {rule_flag}")

    # Filter filtered conflicts for Manual_Conflict_Check == True
    filtered_conflicts = filtered_conflicts[filtered_conflicts['Manual_Conflict_Check'] == True]
    # Print number of filtered conflicts
    print("number of filtered conflicts after manual check: ", len(filtered_conflicts))


    # Save filtered conflicts as csv
    filtered_conflicts.to_csv(os.path.join(config['output_config']['output_folder'], 'Filtered_Conflicts.csv'),
                             index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../Configs/Location_B.yaml',
                        help='Path to the config file')
    parser.add_argument('--database', action='store_true', help='Use database')
    parser.add_argument('--verbose', action='store_true', help='Print debug information')
    parser.add_argument('--startRecID', type=int, default=101310, help='Start record ID')  # 101312 #3305
    parser.add_argument('--endRecID', type=int, default=102085, help='End record ID')
    parser.add_argument('--filterIndicator', type=str, default='PET3D', help='Filter class')
    parser.add_argument('--filterClass1', type=str, default='all', help='Filter class') ##Bicycle
    parser.add_argument('--filterClass2', type=str, default='all', help='Filter class')
    parser.add_argument('--filterCluster1', type=str, default='all', help='Filter cluster')
    parser.add_argument('--filterCluster2', type=str, default='2', help='Filter cluster')
    parser.add_argument('--auto_type', type=str, default='all', help='Auto type')
    parser.add_argument('--auto_rule_flag', type=str, default='False', help='Auto rule flag') # false
    parser.add_argument('--LOF1', type=float, default=0, help='Filter auto type')
    parser.add_argument('--LOF2', type=float, default=1.5, help='Filter auto type') #1.5
    parser.add_argument('--LOFOR', type=bool, default=False, help='Auto rule flag')
    parser.add_argument('--value', type=float, default=3, help='Filter auto type')
    parser.add_argument('--skip_checked', type=bool, default=False, help='Auto rule flag')
    args = parser.parse_args()

    # Read config
    config = load_config(args.config)

    # Start main function
    args.verbose = True
    main(config, args.startRecID, args.endRecID, args.database, args.filterIndicator, args.filterClass1,
         args.filterClass2, args.filterCluster1, args.filterCluster2, args.auto_type, args.value, args.auto_rule_flag,
         args.LOF1, args.LOF2, args.LOFOR, args.skip_checked, args.verbose)
