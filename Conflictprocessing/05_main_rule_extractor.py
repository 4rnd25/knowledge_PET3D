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

import numpy as np
from CommonUtils.ConfigUtils.read_config import load_config
from CommonTools.VideoReader.videoreader import VideoReader
from Conflictprocessing.ConflictManager.conflictmanager import ConflictManager
from Conflictprocessing.ConflictVideoPlayer.conflictvideoplayer import ConflictVideoPlayer

def build_time_csv_path(config_path, output_folder):
    """Create output CSV path with timestamp and config name."""
    cfg_base = os.path.splitext(os.path.basename(config_path))[0]
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    fname = f"{ts}_time_measurement_rule_extraction_filtering_{cfg_base}.csv"
    return os.path.join(output_folder, fname)

def main(config, startRecID, endRecID, filterIndicator, value, verbose, config_path = None):
    """
    Main function for conflict filtering.
    """

    # Start timing
    t0 = time.perf_counter()

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

    # Get all conflicts with filter
    filtered_conflicts = conflictManager.get_filtered_conflicts(
        filterIndicator=filterIndicator,
        value = value
    )

    # Build pair key based on cluster - cluster combination
    filtered_conflicts['pair_key'] = filtered_conflicts.apply(
        lambda r: tuple(sorted([r['Vehicle_Cluster_1'], r['Vehicle_Cluster_2']])), axis=1)

    # — 3) group by that pair, and 4) count how often each appears in position 1 —
    rows = []
    for (cluster_a, cluster_b), grp in filtered_conflicts.groupby('pair_key'):
        rows.append({
            'cluster_A': cluster_a,
            'cluster_B': cluster_b,
            'A_as_vehicle1': int((grp['Vehicle_Cluster_1'] == cluster_a).sum()),
            'B_as_vehicle1': int((grp['Vehicle_Cluster_1'] == cluster_b).sum()),
            'total_pairs': len(grp)
        })

    result = pd.DataFrame(rows)

    # End timing
    elapsed = time.perf_counter() - t0
    num_rules = len(result)

    # Save result as csv
    rule_folder = os.path.join(config['output_config']['output_folder'], 'Rules')
    if not os.path.exists(rule_folder):
        os.makedirs(rule_folder)
    rule_path = os.path.join(config['output_config']['output_folder'], 'Rules', 'Filtered_Conflicts_Rules.csv')
    result.to_csv(rule_path, index=False)

    # --- Save timing CSV ---
    csv_path = build_time_csv_path(config_path or "", config['output_config']['output_folder'])
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_rules", "time"])
        writer.writerow([num_rules, f"{elapsed:.6f}"])
    print(f"Timing CSV written to: {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../Configs/Location_A.yaml',
                        help='Path to the config file')
    parser.add_argument('--verbose', action='store_true', help='Print debug information')
    parser.add_argument('--startRecID', type=int, default=1, help='Start record ID')  # 101312 #3305
    parser.add_argument('--endRecID', type=int, default=135, help='End record ID')
    parser.add_argument('--filterIndicator', type=str, default='PET3D', help='Filter class')
    parser.add_argument('--value', type=float, default=3, help='Filter auto type')
    args = parser.parse_args()

    # Read config
    config = load_config(args.config)

    # Start main function
    args.verbose = True
    main(config, args.startRecID, args.endRecID, args.filterIndicator, args.value, args.verbose)
