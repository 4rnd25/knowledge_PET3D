"""
Created on May 07 2025 07:52

@author: ISAC - pettirsch
"""

import os
import pandas as pd
import shutil


class ConflictSaver:
    def __init__(self, record_output_folder=None, filename=None, indicators=[], measId=None, sensID=None, recId=None,
                 verbose=False):

        self.record_output_folder = record_output_folder
        self.filename = filename + "_Conflicts.csv"
        self.indicator_names = indicators
        self.measId = measId
        self.sensID = sensID
        self.recId = recId
        self.verbose = verbose

        self.conflict_output_path = os.path.join(self.record_output_folder, "Conflicts")

        # Remove the old conflict output folder if it exists
        if os.path.exists(self.conflict_output_path):
            shutil.rmtree(self.conflict_output_path)
            if self.verbose:
                print(f"Removed old conflict output folder: {self.conflict_output_path}")

        if not os.path.exists(self.conflict_output_path):
            os.makedirs(self.conflict_output_path)
            if self.verbose:
                print(f"Created conflict output folder: {self.conflict_output_path}")

        # Create outputfolders for each indicator
        self.indicator_outputfolder = [os.path.join(self.conflict_output_path, indicator) for indicator in
                                       self.indicator_names]
        for indicator_path in self.indicator_outputfolder:
            if not os.path.exists(indicator_path):
                os.makedirs(indicator_path)
                if self.verbose:
                    print(f"Created indicator output folder: {indicator_path}")

            # Create subfolders for videos and plots
            video_output_path = os.path.join(indicator_path, "Conflict_Videos")
            if not os.path.exists(video_output_path):
                os.makedirs(video_output_path)
                if self.verbose:
                    print(f"Created video output folder: {video_output_path}")

            plot_output_path = os.path.join(indicator_path, "Conflict_Plots")
            if not os.path.exists(plot_output_path):
                os.makedirs(plot_output_path)
                if self.verbose:
                    print(f"Created plot output folder: {plot_output_path}")

        # Create df for saving conflicts
        self.csv_output_path = os.path.join(self.conflict_output_path, self.filename)

        self.load_csv()

    def save_conflict(self, conflict):
        # Build the new row dict
        new_row = {
            'idMeasurementSeries': self.measId,
            'idSensor': self.sensID,
            'idRecord': self.recId,
            'FrameTimeStamp': conflict.timeStampVideo,
            'FrameTimeStamp_MicroSec': conflict.timeStampVideoMicrosec,
            'TimeStamp': conflict.timeStamp,
            'TimeStamp_MicroSec': conflict.timeStampMicrosec,
            'Indicator': conflict.indicator,
            'idVehicle1': conflict.idVehicle1,
            'idVehicle2': conflict.idVehicle2,
            'Vehicle_Class_1': conflict.vehicle_class_1,
            'Vehicle_Class_2': conflict.vehicle_class_2,
            'Vehicle_Cluster_1': conflict.vehicle_cluster_1,
            'Vehicle_Cluster_2': conflict.vehicle_cluster_2,
            'posX': conflict.posX,
            'posY': conflict.posY,
            'Value': conflict.value,
            'Auto_Type': conflict.maneuverType,
            'Auto_Rule_Flag': conflict.ruleFlag,
            'LOF1': conflict.LOFVeh1,
            'LOF2': conflict.LOFVeh2,
            'Manual_Type': conflict.maneuverType_manual,
            'Manual_Rule_Flag': conflict.ruleFlag_manual,
            'Manual_Conflict_Check': conflict.checked_manual
        }

        # Check if a row with *all* these same values already exists
        if self.df.empty:
            exists = False
        else:
            # start with a mask of all True, then AND across each column
            mask = pd.Series([True] * len(self.df), index=self.df.index)
            for col, val in new_row.items():
                mask &= (self.df[col] == val)
            exists = mask.any()

        if not exists:
            # append the new row
            new_df = pd.DataFrame([new_row])
            self.df = pd.concat([self.df, new_df], ignore_index=True)
        else:
            # optionally log or handle duplicates
            print(
                f"Conflict already recorded; skipping duplicate (idMeasurementSeries={self.measId}, FrameTimeStamp={conflict.timeStampVideo})")

    def update_record(self, recordId, record_output_folder, filename):
        # Update the record output folder and filename
        self.recId = recordId
        self.record_output_folder = record_output_folder
        self.filename = filename + "_Conflicts.csv"
        self.conflict_output_path = os.path.join(self.record_output_folder, "Conflicts")

        if os.path.exists(self.conflict_output_path):
            # Remove the old conflict output folder
            shutil.rmtree(self.conflict_output_path)


        if not os.path.exists(self.conflict_output_path):
            os.makedirs(self.conflict_output_path)
            if self.verbose:
                print(f"Created conflict output folder: {self.conflict_output_path}")

        # Create outputfolders for each indicator
        self.indicator_outputfolder = [os.path.join(self.conflict_output_path, indicator) for indicator in
                                       self.indicator_names]
        for indicator_path in self.indicator_outputfolder:
            if not os.path.exists(indicator_path):
                os.makedirs(indicator_path)
                if self.verbose:
                    print(f"Created indicator output folder: {indicator_path}")

            # Create subfolders for videos and plots
            video_output_path = os.path.join(indicator_path, "Conflict_Videos")
            if not os.path.exists(video_output_path):
                os.makedirs(video_output_path)
                if self.verbose:
                    print(f"Created video output folder: {video_output_path}")

            plot_output_path = os.path.join(indicator_path, "Conflict_Plots")
            if not os.path.exists(plot_output_path):
                os.makedirs(plot_output_path)
                if self.verbose:
                    print(f"Created plot output folder: {plot_output_path}")

        self.csv_output_path = os.path.join(self.conflict_output_path, self.filename)
        # Remove old csv
        if os.path.exists(self.csv_output_path):
            os.remove(self.csv_output_path)
            if self.verbose:
                print(f"Removed old CSV file: {self.csv_output_path}")
        self.load_csv()

    def load_csv(self):
        # Check if file exists else create
        if os.path.exists(self.csv_output_path):
            self.df = pd.read_csv(self.csv_output_path)
            if self.verbose:
                print(f"Loaded existing conflict file: {self.csv_output_path}")
        else:
            # Define the column names
            columns = [
                'idMeasurementSeries',
                'idSensor',
                'idRecord',
                'FrameTimeStamp',
                'FrameTimeStamp_MicroSec',
                'TimeStamp',
                'TimeStamp_MicroSec',
                'Indicator',
                'idVehicle1',
                'idVehicle2',
                'Vehicle_Class_1',
                'Vehicle_Class_2',
                'posX',
                'posY',
                'Value',
                'Auto_Type',
                'Auto_Rule_Flag',
                'LOF1',
                'LOF2',
                'Manual_Type',
                'Manual_Rule_Flag',
                'Manual_Conflict_Check'
            ]
            # Create an empty DataFrame with those columns
            self.df = pd.DataFrame(columns=columns)


    def get_plot_output_folder(self, indicator_name):
        idx_indicator = self.indicator_names.index(indicator_name)
        indicator_output_path = self.indicator_outputfolder[idx_indicator]
        plot_output_path = os.path.join(indicator_output_path, "Conflict_Plots")
        return plot_output_path

    def get_video_output_folder(self, indicator_name):
        idx_indicator = self.indicator_names.index(indicator_name)
        indicator_output_path = self.indicator_outputfolder[idx_indicator]
        video_output_path = os.path.join(indicator_output_path, "Conflict_Videos")
        return video_output_path

    def save_to_csv(self):
        # Save the DataFrame to a CSV file
        self.df.to_csv(self.csv_output_path, index=False)
        if self.verbose:
            print(f"Saved conflicts to CSV file: {self.csv_output_path}")
