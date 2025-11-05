"""
Created on Oct 8 2025 10:56

@author: ISAC - pettirsch
"""

import argparse
import os
import pdb

import pandas as pd

import numpy as np
from CommonUtils.ConfigUtils.read_config import load_config
from CommonTools.VideoReader.videoreader import VideoReader
from sklearn.neighbors import LocalOutlierFactor
from Conflictprocessing.ConflictManager.conflictmanager import ConflictManager
from Conflictprocessing.ConflictVideoPlayer.conflictvideoplayer import ConflictVideoPlayer
from Conflictprocessing.ConflictTrajectorySaver.conflictTrajectorySaver import ConflictTrajectorySaver
from TrajectoryTransformer.LatentSpaceCalculator.latent_space_calculator import LatentSpaceCalculator
from TrajectoryTransformer.LatentSpaceCalculator.latent_space_calculator_lstm import LatentSpaceCalculatorLSTM

from sklearn.metrics.pairwise import cosine_distances

import matplotlib.pyplot as plt

def find_conflict_csv(output_root: str, foldername: str) -> str:
    """
    Return the existing conflict CSV path for this folder, accepting both
    '_Conflicts.csv' and '_conflicts.csv' (or any extension, any case).
    """
    record_conflict_folder = os.path.join(output_root, foldername, "Conflicts")

    # preferred candidates first
    candidates = [
        os.path.join(record_conflict_folder, f"{foldername}_Conflicts.csv"),  # old naming
        os.path.join(record_conflict_folder, f"{foldername}_conflicts.csv"),  # new naming
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # last resort: any case/extension variant
    # (glob is case-sensitive on Linux; we look for *_onflicts.*
    # to match both Conflicts and conflicts)
    matches = glob.glob(os.path.join(record_conflict_folder, f"{foldername}_*onflicts.*"))
    if matches:
        return matches[0]

    # not found
    raise FileNotFoundError(
        f"Conflict CSV not found under {record_conflict_folder}. "
        f"Tried: {candidates} and glob '{foldername}_*onflicts.*'"
    )

def main(config, startRecID, endRecID, filterIndicator, value, verbose):
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

    # Create ConflictTrajectorySaver
    conflictTrajectorySaver = ConflictTrajectorySaver(config['output_config']['output_folder'],
                                                      enhancer_config = config["videoprocessing_config"]['trajectory_enhancer_config'],
                                                      verbose=verbose)

    # Initialize LatentSpaceCalculator
    latent_space_calculator = LatentSpaceCalculatorLSTM(config['transformer_config'], verbose=verbose)

    # Clean all LOF
    for numRec, idRecord in enumerate(recordIDlist):
        print(f"Cleaning LOF for idRecord {idRecord} {numRec} / {len(recordIDlist)}")
        offset_start = idRecord - startRecID
        foldername = folders[folder_idx_start + offset_start]
        conflict_file = find_conflict_csv(config['output_config']['output_folder'], foldername)
        conflict_df = pd.read_csv(conflict_file, sep=',')

        # Set all LOF1, LOF2, is_anomaly1, is_anomaly2 to NaN
        conflict_df['LOF1_LSTM'] = np.nan
        conflict_df['LOF2_LSTM'] = np.nan
        conflict_df['is_anomaly1_LSTM'] = np.nan
        conflict_df['is_anomaly2_LSTM'] = np.nan

        # Save conflict_df
        conflict_df.to_csv(conflict_file, sep=',', index=False)

    # Get all conflicts with filter
    filtered_conflicts = conflictManager.get_filtered_conflicts(
        filterIndicator=filterIndicator,
        value=value
    )

    # Build pair key based on cluster - cluster combination
    filtered_conflicts['pair_key'] = filtered_conflicts.apply(
        lambda r: tuple(sorted([r['Vehicle_Cluster_1'], r['Vehicle_Cluster_2']])), axis=1)

    # — 3) group by that pair, and 4) count how often each appears in position 1 —
    rows = []
    measID = config['database_config']['measurementID']
    sensID = config['database_config']['sensorID']
    recID = config['database_config']['startrecordID']

    done_combinations = []

    for (cluster_a, cluster_b), grp in filtered_conflicts.groupby('pair_key'):

        all_scores_df = pd.DataFrame(columns=[
            "idMeasurementSeries", "idSensor", "idRecord", "idVehicle", "ObjectClass", "Cluster", "LOF",
            "is_anomaly", "conflicting_cluster"
        ])

        print(f"Processing conflict {cluster_a} - {cluster_b} with {len(grp)} rows")

        grp_cluster_com = grp.copy()

        ### Check if a or b is -1.0
        if cluster_a == -1.0 or cluster_b == -1.0:
            print(f"Skipping conflict {cluster_a} - {cluster_b} because one of the clusters is -1.0")
            continue

        if cluster_a == cluster_b:
            print(f"Skipping conflict {cluster_a} - {cluster_b} because clusters are equal")
            continue

        if grp.empty:
            print(f"Skipping empty group for {cluster_a} - {cluster_b}")
            continue

        # Check if a and b are already done
        if (cluster_a, cluster_b) in done_combinations or (cluster_b, cluster_a) in done_combinations:
            print(f"Skipping already done combination {cluster_a} - {cluster_b}")
            continue
        done_combinations.append((cluster_a, cluster_b))

        conflictTrajectorySaver.add_conflict(cluster_a, cluster_b)

        rule_options = [False] #[True, False]
        for rule_option in rule_options:
            grp_rule_flag = grp_cluster_com[grp_cluster_com['Auto_Rule_Flag'] == rule_option]
            for cluster in [cluster_a, cluster_b]:

                if rule_option == False:
                    title = "Conflict_between_{}_and_{}".format(cluster_a, cluster_b) + "_" + str(cluster) + "_only_rule_flag_false"
                else:
                    title = "Conflict_between_{}_and_{}".format(cluster_a, cluster_b) + "_" + str(cluster) + "_only_rule_flag_true"

                # Check cluster by cluster
                # Get all unique vehicle classes of cluster_a in grp
                vehicle_classes_cl1 = grp.loc[grp['Vehicle_Cluster_1'] == cluster, 'Vehicle_Class_1'].unique()
                vehicle_classes_cl2 = grp.loc[grp['Vehicle_Cluster_2'] == cluster, 'Vehicle_Class_2'].unique()
                all_classes = np.concatenate([vehicle_classes_cl1, vehicle_classes_cl2])
                vehicle_classes = np.unique(all_classes)

                for vehicle_class in vehicle_classes:
                    if rule_option==False:
                        title = "Conflict_between_{}_and_{}".format(cluster_a, cluster_b) + "_" + str(
                            cluster) + "_only_rule_flag_false"
                    else:
                        title = "Conflict_between_{}_and_{}".format(cluster_a, cluster_b) + "_" + str(cluster) + "_only_rule_flag_true"
                    title += "_" + str(vehicle_class)

                    # Get all rows of grp where vehicle_cluster1 = cluster_a and vehicle_class1 = vehicle_class or vehicle_cluster2 = cluster_a and vehicle_class2 = vehicle_class
                    grp_filtered = grp_rule_flag[
                        ((grp_rule_flag['Vehicle_Cluster_1'] == cluster) & (grp_rule_flag['Vehicle_Class_1'] == vehicle_class)) |
                        ((grp_rule_flag['Vehicle_Cluster_2'] == cluster) & (grp_rule_flag['Vehicle_Class_2'] == vehicle_class))]
                    if len(grp_filtered) <= 20:
                        title = "Conflict_between_{}_and_{}".format(cluster_a, cluster_b) + "_" + str(
                            cluster) + "_all_rule_flags"
                        title += "_" + str(vehicle_class)
                        grp_filtered = grp_cluster_com[
                            ((grp_cluster_com['Vehicle_Cluster_1'] == cluster) & (grp_cluster_com['Vehicle_Class_1'] == vehicle_class)) |
                            ((grp_cluster_com['Vehicle_Cluster_2'] == cluster) & (grp_cluster_com['Vehicle_Class_2'] == vehicle_class))]
                        if len(grp_filtered) <= 20:
                            print(f"Skipping conflict {cluster_a} - {cluster_b} because not enough points")
                            continue
                        elif len(grp_filtered) > 20:
                            n_neighbors = 20
                    elif len(grp_filtered) >20:
                        n_neighbors = 20

                    # Iterate through all conflicts
                    for index, row in grp_filtered.iterrows():
                        if recID != row['idRecord'] or startRecID == row['idRecord']:
                            # Load Trajectory file
                            recID = row['idRecord']
                            offset_start = row['idRecord'] - startRecID
                            foldername = folders[folder_idx_start + offset_start]
                            trajectory_path = os.path.join(config['output_config']['output_folder'], foldername,
                                                           "Trajectories",
                                                           foldername + "_enhanced_trajectories.csv")
                            trajectory_df = pd.read_csv(trajectory_path, sep=',')

                        # Get idVehicle
                        if row['Vehicle_Cluster_1'] == cluster and row['Vehicle_Class_1'] == vehicle_class:
                            idVehicle = row['idVehicle1']
                        elif row['Vehicle_Cluster_2'] == cluster and row['Vehicle_Class_2'] == vehicle_class:
                            idVehicle = row['idVehicle2']
                        else:
                            raise ValueError(
                                f"Vehicle class and cluster do not match for vehicle {vehicle_class} and cluster {cluster}"
                            )

                        # Get trajectory
                        df_vehicle = trajectory_df.loc[(trajectory_df['idVehicle'] == idVehicle)]

                        # Add trajectory
                        conflictTrajectorySaver.add_trajectory_1(df_vehicle)

                    # Get df with all trajectories involved in this conflict with specific cluster and class
                    traj_df = conflictTrajectorySaver.traj_1_df
                    if len(traj_df) <= 20:
                        print(f"Skipping conflict {cluster_a} - {cluster_b} {cluster} {vehicle_class} because not enough points")
                        continue

                    # Calc LOF for traj_1
                    keys = traj_df[['idRecord', 'idVehicle']].drop_duplicates()
                    emb_list = []
                    for _, row in keys.iterrows():
                        rec, vid = row['idRecord'], row['idVehicle']
                        df = traj_df[
                            (traj_df['idRecord'] == rec) &
                            (traj_df['idVehicle'] == vid)
                            ]
                        z = latent_space_calculator.embed_single_trajectory(df)  # same helper as before
                        emb_list.append(z)
                    embs = np.stack(emb_list, axis=0)  # (N, D)

                    # Save embeddings
                    df_embs = pd.DataFrame(
                        embs,
                        columns=[f'emb_{i}' for i in range(embs.shape[1])]
                    )
                    df_embs[['idRecord', 'idVehicle']] = keys.reset_index(drop=True)

                    # 2) Save to disk
                    embedding_folder = os.path.join(config['output_config']['output_folder'], "Embeddings")
                    if not os.path.exists(embedding_folder):
                        os.makedirs(embedding_folder)
                    title_embeddings = "Embeddings_LSTM_" + title + ".parquet"
                    embedding_path = os.path.join(embedding_folder, title_embeddings)
                    df_embs.to_parquet(embedding_path, index=False)

                    conflicting_cluster = cluster_a if cluster == cluster_b else cluster_b

                    # Calc LOF
                    n_samples = embs.shape[0]


                    if n_samples > n_neighbors:
                        # 1) instantiate LOF
                        lof = LocalOutlierFactor(
                            n_neighbors=n_neighbors,
                            metric='cosine',
                            novelty=False
                        )

                        # 2) fit
                        lof.fit(embs)

                        # 3) compute scores
                        raw_scores = lof.negative_outlier_factor_  # shape (N,)
                        lof_scores = -raw_scores
                        threshold = 1.5
                        is_anomaly = lof_scores > threshold

                    else:
                        # not enough points → give “no anomaly” defaults
                        lof_scores = np.zeros(n_samples)
                        is_anomaly = np.zeros(n_samples, dtype=bool)

                    conflictTrajectorySaver.fill_scores_df_1(lof_scores, is_anomaly, keys, conflicting_cluster)

                    conflictTrajectorySaver.plot_trajectories_1(f"LSTM_{title}")

                    scores_df = conflictTrajectorySaver.scores_df_1
                    all_scores_df = pd.concat([all_scores_df, scores_df], ignore_index=True)

                    conflictTrajectorySaver.save(f"LSTM_{title}", None)
                    conflictTrajectorySaver.clear()

        # Update conflicts df
        unique_idRecords = all_scores_df['idRecord'].unique()

        for numRec, idRecord in enumerate(recordIDlist):
            print(f"Updating conflicts for idRecord {idRecord} {numRec} / {len(recordIDlist)}")
            offset_start = idRecord - startRecID
            foldername = folders[folder_idx_start + offset_start]
            conflict_file = find_conflict_csv(config['output_config']['output_folder'], foldername)
            conflict_df = pd.read_csv(conflict_file, sep=',')

            # Load scores for record
            record_df = all_scores_df[all_scores_df['idRecord'] == idRecord]

            # Iterate through all idVehicles
            for index, row_rec in record_df.iterrows():
                # Check if in conflict_df is a row with idVehicle1 == row_rec['idVehicle'] and Vehicle_Cluster_1 == row_rec['Cluster'] and Vehicle_Cluster_2 = = row_rec['conflicting_cluster']
                mask = (
                        (conflict_df['idVehicle1'] == row_rec['idVehicle']) &
                        (conflict_df['Vehicle_Cluster_1'] == row_rec['Cluster']) &
                        (conflict_df['Vehicle_Cluster_2'] == row_rec['conflicting_cluster'])
                )
                if mask.any():
                    conflict_df.loc[mask, 'LOF1_LSTM'] = row_rec['LOF']
                    conflict_df.loc[mask, 'is_anomaly1_LSTM'] = row_rec['is_anomaly']

                mask = (
                        (conflict_df['idVehicle2'] == row_rec['idVehicle']) &
                        (conflict_df['Vehicle_Cluster_2'] == row_rec['Cluster']) &
                        (conflict_df['Vehicle_Cluster_1'] == row_rec['conflicting_cluster'])
                )
                if mask.any():
                    conflict_df.loc[mask, 'LOF2_LSTM'] = row_rec['LOF']
                    conflict_df.loc[mask, 'is_anomaly2_LSTM'] = row_rec['is_anomaly']

            # Save conflict_df
            conflict_df.to_csv(conflict_file, sep=',', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../Configs/Location_A.yaml',
                        help='Path to the config file')
    parser.add_argument('--verbose', action='store_true', help='Print debug information')
    parser.add_argument('--startRecID', type=int, default=1, help='Start record ID')  # 101312 #3305
    parser.add_argument('--endRecID', type=int, default=855, help='End record ID')
    parser.add_argument('--filterIndicator', type=str, default='PET3D', help='Filter class')
    parser.add_argument('--value', type=float, default=3, help='Filter auto type')
    args = parser.parse_args()

    # Read config
    config = load_config(args.config)

    # Start main function
    args.verbose = True
    main(config, args.startRecID, args.endRecID, args.filterIndicator, args.value, args.verbose)
