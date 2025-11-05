"""
Created on May 15 2025 15:25

@author: ISAC - pettirsch
"""
import pdb

import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection
from matplotlib.lines import Line2D
import numpy as np
import cv2
import pdb

class ConflictTrajectorySaver:
    def __init__(self, outputfolder = None, enhancer_config = None, verbose = False):

        self.outputfolder = os.path.join(outputfolder, "Conflict_Trajectories")
        if not os.path.exists(self.outputfolder):
            os.makedirs(self.outputfolder)

        self.traj_columns = [
            "idMeasurementSeries", "idSensor", "idRecord", "FrameTimeStamp", "FrameTimeStamp_MicroSec",
            "idVehicle", "ObjectClass", "Length", "Width", "Height",
            "posXFit", "posYFit", "posZFit", "YawFit", "VxFit", "VyFit", "VzFit", "AxFit", "AyFit", "AzFit",
            "YawRateFit", "Cluster"]

        self.scores_columns= [
            "idMeasurementSeries", "idSensor", "idRecord", "idVehicle", "ObjectClass", "Cluster", "LOF",
            "is_anomaly", "conflicting_cluster"
        ]

        self.final_min_duration = enhancer_config["final_min_duration"] * 30
        self.final_min_distance = enhancer_config["final_min_distance"]
        self.final_yaw_fluctuation = enhancer_config["finral_yaw_fluctuation"]

        self.verbose = verbose

    def add_conflict(self, cluster_class_1, cluster_class_2):

        # Create output folder
        foldername = f"{cluster_class_1}-{cluster_class_2}"
        self.curr_conflict_folder = os.path.join(self.outputfolder, foldername)
        if not os.path.exists(self.curr_conflict_folder):
            os.makedirs(self.curr_conflict_folder)

        # Create empty dfs
        self.create_empty_dfs()

    def create_empty_dfs(self):

        # Create empty df with traj_columns
        self.traj_1_df = pd.DataFrame(columns=self.traj_columns)
        self.traj_2_df = pd.DataFrame(columns=self.traj_columns)

        # Score df
        self.scores_df_1 = pd.DataFrame(columns=self.scores_columns)
        self.scores_df_2 = pd.DataFrame(columns=self.scores_columns)

    def add_trajectory_1(self, traj_df):

        # Check if self.traj_1_df has a row with measId, sensID, recID, vehID equal to traj_df row 0
        if not traj_df.empty and not self.traj_1_df.empty:
            # pull the 4 keys out of the first row
            row0 = traj_df.iloc[0]
            measId = row0['idMeasurementSeries']
            sensID = row0['idSensor']
            recID = row0['idRecord']
            vehID = row0['idVehicle']

            # build a boolean mask over all rows of traj_1_df
            mask = (
                    (self.traj_1_df['idMeasurementSeries'] == measId) &
                    (self.traj_1_df['idSensor'] == sensID) &
                    (self.traj_1_df['idRecord'] == recID) &
                    (self.traj_1_df['idVehicle'] == vehID)
            )

            # if any row matches, bail out
            if mask.any():
                return

        # Remove last and first 10 elements
        traj_df = traj_df.iloc[10:-10]

        # Check if the trajectories are valid
        duration_1 = len(traj_df)
        if duration_1 < self.final_min_duration:
            return

        # Check if the trajectories are valid
        start_pos = traj_df.iloc[0]['posXFit'], traj_df.iloc[0]['posYFit']
        end_pos = traj_df.iloc[-1]['posXFit'], traj_df.iloc[-1]['posYFit']
        distance = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        if distance< self.final_min_distance:
            return

        # Check if the trajectories are valid
        # Normalize the yaw to range -pi to pi
        angles = traj_df['YawFit'].values  # assumed in radians
        unwrapped = np.unwrap(angles)  # removes jumps > π by adding/subtracting 2π
        yaw_fluctuation = unwrapped.max() - unwrapped.min()
        if yaw_fluctuation > self.final_yaw_fluctuation:
            return

        # Add trajectory to df
        self.traj_1_df = pd.concat([self.traj_1_df, traj_df], ignore_index=True)

    def add_trajectory_2(self, traj_df):

        # Check if self.traj_2_df has a row with measId, sensID, recID, vehID equal to traj_df row 0
        if not traj_df.empty and not self.traj_2_df.empty:
            # pull the 4 keys out of the first row
            row0 = traj_df.iloc[0]
            measId = row0['idMeasurementSeries']
            sensID = row0['idSensor']
            recID = row0['idRecord']
            vehID = row0['idVehicle']

            # build a boolean mask over all rows of traj_2_df
            mask = (
                    (self.traj_2_df['idMeasurementSeries'] == measId) &
                    (self.traj_2_df['idSensor'] == sensID) &
                    (self.traj_2_df['idRecord'] == recID) &
                    (self.traj_2_df['idVehicle'] == vehID)
            )

            # if any row matches, bail out
            if mask.any():
                return

        # Add trajectory to df
        self.traj_2_df = pd.concat([self.traj_2_df, traj_df], ignore_index=True)

    def fill_scores_df_1(self, lof_scores, is_anomaly, keys, conflicting_cluster):
        # repead conflicting_cluster for all rows
        conflicting_cluster = [conflicting_cluster] * len(lof_scores)

        idMeasurementSeries = self.traj_1_df['idMeasurementSeries'].unique()[0]
        idSensor = self.traj_1_df['idSensor'].unique()[0]
        for i, (idRecord, idVehicle) in enumerate(keys[['idRecord', 'idVehicle']].values):
            # Get ObjectClass and Cluster from traj_1_df
            ObjectClass = self.traj_1_df.loc[
                (self.traj_1_df['idRecord'] == idRecord) &
                (self.traj_1_df['idVehicle'] == idVehicle), 'ObjectClass'].values[0]
            Cluster = self.traj_1_df.loc[
                (self.traj_1_df['idRecord'] == idRecord) &
                (self.traj_1_df['idVehicle'] == idVehicle), 'Cluster'].values[0]

            # Add to scores_df
            self.scores_df_1 = pd.concat([self.scores_df_1, pd.DataFrame({
                "idMeasurementSeries": [idMeasurementSeries],
                "idSensor": [idSensor],
                "idRecord": [idRecord],
                "idVehicle": [idVehicle],
                "ObjectClass": [ObjectClass],
                "Cluster": [Cluster],
                "LOF": [lof_scores[i]],
                "is_anomaly": [is_anomaly[i]],
                "conflicting_cluster": [conflicting_cluster[i]]
            })], ignore_index=True)

    def fill_scores_df_2(self, lof_scores, is_anomaly, keys):
        idMeasurementSeries = self.traj_2_df['idMeasurementSeries'].unique()[0]
        idSensor = self.traj_2_df['idSensor'].unique()[0]
        for i, (idRecord, idVehicle) in enumerate(keys[['idRecord', 'idVehicle']].values):
            # Get ObjectClass and Cluster from traj_2_df
            ObjectClass = self.traj_2_df.loc[
                (self.traj_2_df['idRecord'] == idRecord) &
                (self.traj_2_df['idVehicle'] == idVehicle), 'ObjectClass'].values[0]
            Cluster = self.traj_2_df.loc[
                (self.traj_2_df['idRecord'] == idRecord) &
                (self.traj_2_df['idVehicle'] == idVehicle), 'Cluster'].values[0]

            # Add to scores_df
            self.scores_df_2 = pd.concat([self.scores_df_2, pd.DataFrame({
                "idMeasurementSeries": [idMeasurementSeries],
                "idSensor": [idSensor],
                "idRecord": [idRecord],
                "idVehicle": [idVehicle],
                "ObjectClass": [ObjectClass],
                "Cluster": [Cluster],
                "LOF": [lof_scores[i]],
                "is_anomaly": [is_anomaly[i]]
            })], ignore_index=True)


    def plot_trajectories_1(self, title):
        """
        trajectories_df: DataFrame containing posXFit,posYFit,FrameTimeStamp,FrameTimeStamp_MicroSec,idRecord,idVehicle
        scores_df:      DataFrame with idRecord,idVehicle,is_anomaly
        subplot:        an identifier (e.g. int) used in filenames
        frame:          optional OpenCV image to overlay; if given, saved as Trajectories_{subplot}.png
        """

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = 11  # Base font size
        plt.rcParams["axes.labelsize"] = 11  # Axis label size
        plt.rcParams["axes.titlesize"] = 11  # Title size
        plt.rcParams["legend.fontsize"] = 11  # Legend font size
        plt.rcParams["xtick.labelsize"] = 11  # Tick label sizes
        plt.rcParams["ytick.labelsize"] = 11

        # 1) start a 3D figure
        # fig = plt.figure()
        fig = plt.figure(figsize=(7, 4.5)) #6.3
        ax = fig.add_subplot(111, projection='3d')

        first = True

        # 2) loop over each unique trajectory
        for _, row in self.scores_df_1.iterrows():
            rec      = row['idRecord']
            vid      = row['idVehicle']
            is_anom  = row['is_anomaly']

            sub = self.traj_1_df[
                (self.traj_1_df['idRecord'] == rec) &
                (self.traj_1_df['idVehicle'] == vid)
            ]
            if sub.empty:
                continue

            # absolute X/Y
            x = sub['posXFit'].values
            y = sub['posYFit'].values

            # time in seconds from first timestamp
            ts    = pd.to_datetime(sub['FrameTimeStamp'], errors='coerce')
            micro = pd.to_numeric(sub['FrameTimeStamp_MicroSec'], errors='coerce').fillna(0)
            full  = ts + pd.to_timedelta(micro, unit='us')
            t     = (full - full.iloc[0]).dt.total_seconds().values

            # pick color
            color = 'red' if is_anom else 'green'

            # plot
            ax.plot(x, y, t, color=color, linewidth=1)

        # 4) style the axes
        ax.set_xlabel('\nX [m]')
        ax.set_ylabel('\n\nY [m]')
        ax.set_zlabel('\nTime [s] from start\n\n')

        # Better label positioning with set_label_coords
        # ax.xaxis.set_label_coords(0.5, -0.15)
        # ax.yaxis.set_label_coords(-0.15, 0.5)
        # ax.zaxis.set_label_coords(0.5, 0.98)

        # ax.xaxis.set_label_coords(0.5, -0.25)  # move X label further down
        # ax.yaxis.set_label_coords(-0.3, 0.55)  # move Y label further left

        ax.ticklabel_format(style='plain', axis='x')  # or 'useOffset=False'
        ax.ticklabel_format(style='plain', axis='y')

        # Optional: reduce tick label size for extra margin
        # ax.tick_params(axis='both', which='major', labelsize=8)
        # ax.tick_params(axis='z', which='major', labelsize=8)

        # legend
        legend_elems = [
            Line2D([0],[0], color='green', lw=2, label='Normal'),
            Line2D([0],[0], color='red',   lw=2, label='Anomaly')
        ]
        ax.legend(handles=legend_elems, loc='upper left')

        # 5) render figure to an image and save
        # Adjust layout margins to avoid label clipping
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        fig.savefig(os.path.join(self.curr_conflict_folder, f"Trajectories_{title}.png"),
                    dpi=300, bbox_inches=None)
        fig.savefig(os.path.join(self.curr_conflict_folder, f"Trajectories_{title}.pdf"),
                    dpi=300, format='pdf')
        fig.savefig(os.path.join(self.curr_conflict_folder, f"Trajectories_{title}.svg"),
                    dpi=300, format='svg')
        # print path
        if 1==1:# self.verbose:
            print(f"Saved Trajectories_{title}.png/pdf/svg to {self.curr_conflict_folder}")
        plt.close(fig)

    def plot_trajectories_2(self, title):
        """
        trajectories_df: DataFrame containing posXFit,posYFit,FrameTimeStamp,FrameTimeStamp_MicroSec,idRecord,idVehicle
        scores_df:      DataFrame with idRecord,idVehicle,is_anomaly
        subplot:        an identifier (e.g. int) used in filenames
        frame:          optional OpenCV image to overlay; if given, saved as Trajectories_{subplot}.png
        """
        # 1) start a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 2) loop over each unique trajectory
        for _, row in self.scores_df_2.iterrows():
            rec      = row['idRecord']
            vid      = row['idVehicle']
            is_anom  = row['is_anomaly']

            sub = self.traj_2_df[
                (self.traj_2_df['idRecord'] == rec) &
                (self.traj_2_df['idVehicle'] == vid)
            ]
            if sub.empty:
                continue

            # absolute X/Y
            x = sub['posXFit'].values
            y = sub['posYFit'].values

            # time in seconds from first timestamp
            ts    = pd.to_datetime(sub['FrameTimeStamp'], errors='coerce')
            micro = pd.to_numeric(sub['FrameTimeStamp_MicroSec'], errors='coerce').fillna(0)
            full  = ts + pd.to_timedelta(micro, unit='us')
            t     = (full - full.iloc[0]).dt.total_seconds().values

            # pick color
            color = 'red' if is_anom else 'green'

            # plot
            ax.plot(x, y, t, color=color, linewidth=1)

        # 4) style the axes
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_zlabel('Time (s) from start')
        ax.set_title('Conflict Trajectories (3D) — LOF anomalies in red')

        # legend
        legend_elems = [
            Line2D([0],[0], color='green', lw=2, label='Normal'),
            Line2D([0],[0], color='red',   lw=2, label='Anomaly')
        ]
        ax.legend(handles=legend_elems, loc='upper left')

        # 5) render figure to an image and save
        fig.savefig(os.path.join(self.curr_conflict_folder, f"Trajectories_{title}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    def clear(self):
        # Clear the current conflict data
        self.traj_1_df = pd.DataFrame(columns=self.traj_columns)
        self.traj_2_df = pd.DataFrame(columns=self.traj_columns)
        self.scores_df_1 = pd.DataFrame(columns=self.scores_columns)
        self.scores_df_2 = pd.DataFrame(columns=self.scores_columns)

    def save(self, title1, title2):
        # Save the trajectories and scores to CSV files
        if title1 is not None:
            self.traj_1_df.to_csv(os.path.join(self.curr_conflict_folder, f"Trajectory_1_{title1}.csv"), index=False)
            self.scores_df_1.to_csv(os.path.join(self.curr_conflict_folder, f"Scores_1_{title1}.csv"), index=False)
        if title2 is not None:
            self.traj_2_df.to_csv(os.path.join(self.curr_conflict_folder, f"Trajectory_2_{title2}.csv"), index=False)
            self.scores_df_2.to_csv(os.path.join(self.curr_conflict_folder, f"Scores_2_{title2}.csv"), index=False)



def circular_range_deg(angles_deg):
    # Convert to radians and wrap into [0, 2π)
    angles = np.deg2rad(angles_deg) % (2 * np.pi)
    # Sort and append first+2π to close the loop
    sorted_a = np.sort(angles)
    extended = np.concatenate([sorted_a, sorted_a[:1] + 2 * np.pi])
    # Compute gaps between successive points
    gaps = np.diff(extended)
    # Largest gap is where our minimal spanning arc does NOT cover
    max_gap = np.max(gaps)
    # The minimal circular range is 2π minus that largest gap
    return np.rad2deg(2 * np.pi - max_gap)




