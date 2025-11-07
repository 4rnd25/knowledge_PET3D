"""
Created on Jan 14 2025 11:18

@author: ISAC - pettirsch
"""

import cv2
import time
from datetime import datetime
from datetime import timedelta
import os
import numpy as np


class VideoReader:
    def __init__(self, input_config, buffer_duration, database_config=None):
        self.source_type = input_config['source_type']
        self.source_path = input_config['source_path']
        self.max_duration = input_config['max_video_duration']
        self.projectName = input_config['filename']
        self.fileName = None

        # if csv source type, additional parameters
        if self.source_type == "csv":
            self.csv_file_path = input_config['csv_file_path']

        self.database_config = database_config

        self.buffer_duration = buffer_duration
        self.cap = None
        self.fps = 30  # Default value
        self.start_time = None
        self._initialize_reader()
        #self.buffer = deque(maxlen=int(self.buffer_duration * self.fps))
        self.finished = False


    def _initialize_reader(self):
        if self.source_type in ["file", "rtsp"]:
            if self.source_type == "file":
                filename = os.path.basename(self.source_path)
                inputFilesplit = filename.split("_")
                timestamp = None
                for i in range(len(InputFileSplit)):
                    date_string = InputFileSplit[i].split("-")[0]
                    try:
                        # Get date and time from filename
                        timestamp = datetime.datetime.strptime(date_string, '%Y%m%dT%H%M%S_')
                        break
                    except:
                        pass
                    # Get date and time from filename
                if not timestamp:
                    raise Exception('Could not find a timestamp in ' + Filename)
            else:
                timestamp = datetime.now().strftime("%Y%m%dT%H%M%S_")
            self.fileName = timestamp + self.projectName
            self.cap = cv2.VideoCapture(self.source_path)
            if self.cap.isOpened():
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or self.fps
            # Create time
            self.start_time = datetime.strptime(timestamp, '%Y%m%dT%H%M%S_')
        elif self.source_type == "folder":
            self.file_list = sorted(os.listdir(self.source_path))
            self.current_file_idx = 0
            self.fileName = self.file_list[self.current_file_idx]
            # Get timestamp from filename
            inputFilesplit = self.fileName.split("_")
            timestamp = None
            for i in range(len(InputFileSplit)):
                date_string = InputFileSplit[i].split("-")[0]
                try:
                    # Get date and time from filename
                    timestamp = datetime.datetime.strptime(date_string, '%Y%m%dT%H%M%S_')
                    break
                except:
                    pass
                # Get date and time from filename
            if not timestamp:
                raise Exception('Could not find a timestamp in ' + Filename)
            self.cap = cv2.VideoCapture(os.path.join(self.source_path, self.file_list[self.current_file_idx]))
            if self.cap.isOpened():
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or self.fps
            self.start_time = datetime.strptime(timestamp, '%Y%m%dT%H%M%S_')
        elif self.source_type == "database":
            assert self.database_config is not None, "Database config is missing"
            import SQLConnectionClass
            self.sqlconnection = SQLConnectionClass.SQLConnection(HostName=self.database_config["database_ip"],
                                                                  Port=self.database_config["database_port"],
                                                                  User=self.database_config["database_user"],
                                                                  Password=self.database_config["database_password"],
                                                                  Schema=self.database_config["database_name"])
            self.recordList = list(
                range(self.database_config["startrecordID"], self.database_config["endrecordID"] + 1))
            self.current_record_idx = 0
            filePath = self.sqlconnection.SearchVideoForRecordID(measID=self.database_config["measurementID"],
                                                          sensID=self.database_config["sensorID"],
                                                          recID=self.recordList[self.current_record_idx],
                                                          searchFolder=self.source_path)
            self.start_time = self.sqlconnection.LoadRecordStartTime(idMeasurementSeries=self.database_config["measurementID"],
                                                                    idSensor=self.database_config["sensorID"],
                                                                    idRecord=self.recordList[self.current_record_idx])
            self.fileName = os.path.basename(filePath)
            self.cap = cv2.VideoCapture(filePath)
            if self.cap.isOpened():
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or self.fps
        elif self.source_type == "csv":
            assert self.csv_file_path is not None, "CSV file path is missing"
            self.csv_datetime_format = "%Y-%m-%d %H:%M:%S.%f"  # Adjust based on your CSV format
            # Read the csv file and store as df
            import pandas as pd
            df = pd.read_csv(self.csv_file_path)
            if "RecordID" not in df.columns or "RecordFrameStartTime" not in df.columns:
                raise ValueError(
                    f"CSV must contain columns 'RecordID' and 'RecordFrameStartTime' (got {df.columns.tolist()})")

            start_id = self.database_config["startrecordID"]
            end_id = self.database_config["endrecordID"]
            df = df[(df["RecordID"] >= start_id) & (df["RecordID"] <= end_id)].copy()

            def extract_timestamp(v):
                """
                Handles cases where v is:
                - a numpy array of shape (1,) with dtype object
                - a plain string
                - already a datetime
                """
                if isinstance(v, np.ndarray):
                    # assume something like array(['2023-05-08 13:50:39'], dtype=object)
                    if v.size > 0:
                        v = v.item()  # get the scalar from the array
                    else:
                        return None
                if isinstance(v, (list, tuple)):
                    return v[0] if v else None
                return v

            df["RecordFrameStartTime"] = df["RecordFrameStartTime"].apply(extract_timestamp)

            # Now safely convert to datetime
            df["__ts"] = pd.to_datetime(df["RecordFrameStartTime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

            # Build the pattern
            df["__datepattern"] = df["__ts"].dt.strftime("%Y%m%dT%H%M%S")

            df = df.sort_values("RecordID")
            self._csv_df = df.reset_index(drop=True)
            self._csv_rows = self._csv_df.to_dict(orient="records")
            self.current_record_idx = 0
            if not self._csv_rows:
                raise ValueError("No CSV rows after filtering by RecordID range.")

            # Get filepath and filename
            row = self._csv_rows[self.current_record_idx]
            ts_dt = row["__ts"].to_pydatetime()
            self.start_time = ts_dt
            datepattern = row["__datepattern"]

            # Search for vile containing the datepattern
            matched_files = [f for f in os.listdir(self.source_path) if datepattern in f]
            if not matched_files:
                raise FileNotFoundError(f"No video file found in {self.source_path} for datepattern {datepattern}")
            filePath = os.path.join(self.source_path, matched_files[0])
            self.fileName = os.path.basename(filePath)
            self.cap = cv2.VideoCapture(filePath)
            if self.cap.isOpened():
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or self.fps
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")

    def get_next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            if self.source_type == "rtsp":
                raise Exception("Failed to read frame from RTSP stream")
            else:
                if self.source_type == "folder":
                    if self.current_file_idx >= len(self.file_list) - 1:
                        self.finished = True
                elif self.source_type == "file":
                    self.finished = True
                elif self.source_type == "database":
                    if self.current_record_idx >= len(self.recordList) - 1:
                        self.finished = True
                elif self.source_type == "csv":
                    if self.current_record_idx >= len(self._csv_rows) - 1:
                        self.finished = True
            return None

        timestamp = time.time()
        #self.buffer.append((timestamp, frame))
        return frame

    # def get_buffer(self):
    #     return list(self.buffer)

    def should_restart(self, start_time):
        if self.source_type == "rtsp":
            return time.time() - start_time >= self.max_duration
        return False

    def restart(self):
        self.cap.release()
        #self.buffer.clear()
        if self.source_type == "folder" and self.current_file_idx < len(self.file_list) - 1:
            self.current_file_idx += 1
            self.fileName = self.file_list[self.current_file_idx]
            # Get timestamp from filename
            inputFilesplit = self.fileName.split("_")
            timestamp = None
            for i in range(len(InputFileSplit)):
                date_string = InputFileSplit[i].split("-")[0]
                try:
                    # Get date and time from filename
                    timestamp = datetime.datetime.strptime(date_string, '%Y%m%dT%H%M%S_')
                    break
                except:
                    pass
                # Get date and time from filename
            if not timestamp:
                raise Exception('Could not find a timestamp in ' + Filename)
            self.start_time = datetime.strptime(timestamp, '%Y%m%dT%H%M%S_')
            self.cap = cv2.VideoCapture(os.path.join(self.source_path, self.file_list[self.current_file_idx]))
        elif self.source_type == "database" and self.current_record_idx < len(self.recordList) - 1:
            self.current_record_idx += 1
            filePath = self.sqlconnection.SearchVideoForRecordID(measID=self.database_config["measurementID"],
                                                          sensID=self.database_config["sensorID"],
                                                          recID=self.recordList[self.current_record_idx],
                                                          searchFolder=self.source_path)
            self.fileName = os.path.basename(filePath)
            self.start_time = self.sqlconnection.LoadRecordStartTime(idMeasurementSeries=self.database_config["measurementID"],
                                                                    idSensor=self.database_config["sensorID"],
                                                                    idRecord=self.recordList[self.current_record_idx])
            self.cap = cv2.VideoCapture(filePath)
        elif self.source_type == "csv" and self.current_record_idx < len(self._csv_rows) - 1:
            self.current_record_idx += 1
            row = self._csv_rows[self.current_record_idx]
            ts_dt = row["__ts"].to_pydatetime()
            self.start_time = ts_dt
            datepattern = row["__datepattern"]

            # Search for vile containing the datepattern
            matched_files = [f for f in os.listdir(self.source_path) if datepattern in f]
            if not matched_files:
                raise FileNotFoundError(f"No video file found in {self.source_path} for datepattern {datepattern}")
            filePath = os.path.join(self.source_path, matched_files[0])
            self.fileName = os.path.basename(filePath)
            self.cap = cv2.VideoCapture(filePath)
        else:
            if self.source_type == "database" or self.source_type == "csv":
                self.current_record_idx += 1
                if self.current_record_idx >= len(self.recordList) - 1:
                    self.finished = True
            self._initialize_reader()

    def is_done(self):
        if self.source_type == "rtsp":
            return False
        else:
            return self.finished

    def getFilename(self):
        return self.fileName

    def get_recordID(self):
        if self.source_type == "database":
            return self.recordList[self.current_record_idx]
        elif self.source_type == "csv":
            row = self._csv_rows[self.current_record_idx]
            return row["RecordID"]
        return None

    def get_frame_size(self):
        if self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (640, 480)  # Default frame size

    def release(self):
        self.cap.release()
        if self.source_type == "database":
            self.sqlconnection.__del__()

    def get_video_snippet(self, start_time, end_time):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, (start_time - self.start_time).total_seconds() * 1000)
        frames = []
        n = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            timestamp = start_time + n* timedelta(seconds=1/self.fps)
            frames.append((timestamp, frame))
            if timestamp >= end_time:
                break
            n+= 1
        return frames

    def get_start_time(self):
        return self.start_time

    def get_num_records(self):
        if self.source_type == "database":
            return len(self.recordList)
        elif self.source_type == "csv":
            return len(self._csv_rows)
        return 1