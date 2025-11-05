"""
Created on May 15 2025 10:56

@author: ISAC - pettirsch
"""
import pdb

import pandas as pd
import argparse
import os
import time
import csv
from datetime import datetime

from CommonUtils.ConfigUtils.read_config import load_config
from CommonTools.VideoReader.videoreader import VideoReader

def build_time_csv_path(config_path, output_folder):
    cfg_base = os.path.splitext(os.path.basename(config_path))[0]
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    fname = f"{ts}_time_measurement_rule_assignment_filtering_{cfg_base}.csv"
    return os.path.join(output_folder, fname)

def main(config, startRecID, endRecID, value, verbose, config_path = None):
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


    rule_path = os.path.join(config['output_config']['output_folder'], 'Rules', 'Filtered_Conflicts_Rules.csv')
    # Load rules from csv
    if os.path.exists(rule_path):
        rules = pd.read_csv(rule_path)
    else:
        raise FileNotFoundError(f"Rules file {rule_path} does not exist.")

    # Iterate over all recordIds in filtered conflicts
    timing_rows = []
    for idx, recordId in enumerate(recordIDlist):

        t0 = time.perf_counter()

        # RecordName
        record_idx = recordIDlist.index(recordId)
        recordName = folders[folder_idx_start + record_idx]

        print("Handle record: ", recordName)
        print("Record {} of {}".format(idx + 1, len(recordIDlist)))

        # Load record conflict df
        record_folder = os.path.join(config['output_config']['output_folder'], str(recordName))
        record_conflict_folder = os.path.join(record_folder, "Conflicts")
        csv_path = os.path.join(record_conflict_folder, str(recordName) + '_Conflicts.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file {csv_path} does not exist.")
        record_conflict_df = pd.read_csv(csv_path)

        # Iterate over all rows
        for i, row in record_conflict_df.iterrows():
            #Check indicator
            cluster1 = row['Vehicle_Cluster_1']
            cluster2 = row['Vehicle_Cluster_2']

            if row['Auto_Type'] == 'Following':
                rule_flag = True
            elif row['Auto_Type'] == 'Diverging':
                rule_flag = True

            # conflict combination
            elif cluster1 == cluster2:
                rule_flag = True
            else:
                cluster_1 = cluster1
                cluster_2 = cluster2

                rule1 = rules[(rules['cluster_A'] == cluster_1) & (
                            rules['cluster_B'] == cluster_2)]
                rule2 = rules[(rules['cluster_A'] == cluster_2) & (
                            rules['cluster_B'] == cluster_1)]

                if rule1.empty and rule2.empty:
                    indicator = row["Indicator"]
                    print(
                        f"Indicator {indicator} Conflict string {cluster_1} and {cluster_2} not found in rules.")
                    continue
                elif rule1.empty:
                    rule = rule2
                    priority_veh_1 = rule['B_as_vehicle1'].values[0]
                    priority_veh_2 = rule['A_as_vehicle1'].values[0]
                elif rule2.empty:
                    rule = rule1
                    priority_veh_1 = rule['A_as_vehicle1'].values[0]
                    priority_veh_2 = rule['B_as_vehicle1'].values[0]

                if priority_veh_1 > priority_veh_2:
                    rule_flag = True
                elif priority_veh_1 < priority_veh_2:
                    if priority_veh_2 / (priority_veh_1 + priority_veh_2) < 0.66:
                        rule_flag = True
                    else:
                        rule_flag = False

            elapsed = time.perf_counter() - t0
            num_conflicts_record = int(len(record_conflict_df))
            timing_rows.append([recordId, num_conflicts_record, elapsed])

            # Fill column "Auto_Rule_Flag" for each row
            row['Auto_Rule_Flag'] = rule_flag
            record_conflict_df.at[i, 'Auto_Rule_Flag'] = rule_flag


        # Save record conflict df
        record_conflict_df.to_csv(csv_path, index=False)

    # --- write timing CSV with totals row ---
    total_records = len(recordIDlist)
    total_conflicts = sum(row[1] for row in timing_rows)
    total_time = sum(row[2] for row in timing_rows)

    csv_path = build_time_csv_path(config_path or "", config['output_config']['output_folder'])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["record id", "num conflicts", "time"])
        for rid, nconf, tsec in timing_rows:
            writer.writerow([rid, nconf, f"{tsec:.6f}"])
        # totals row per your spec
        writer.writerow([total_records, total_conflicts, f"{total_time:.6f}"])
    print(f"Rule-assignment timing CSV written to: {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../Configs/Location_A.yaml',
                        help='Path to the config file')
    parser.add_argument('--verbose', action='store_true', help='Print debug information')
    parser.add_argument('--startRecID', type=int, default=1, help='Start record ID')  # 101312 #3305
    parser.add_argument('--endRecID', type=int, default=135, help='End record ID')
    parser.add_argument('--value', type=float, default=3, help='Filter auto type')
    args = parser.parse_args()

    # Read config
    config = load_config(args.config)

    # Start main function
    args.verbose = True
    main(config, args.startRecID, args.endRecID, args.value, args.verbose, args.config)