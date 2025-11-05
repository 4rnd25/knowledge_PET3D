"""
Created on Jan 14 2025 13:29

@author: ISAC - pettirsch
"""

import cv2
import os


class VideoWriter:
    def __init__(self, base_file_Name, super_output_folder, video_output_config, fps, frame_size, conflictWriter=False,
                 verbose=False):

        if not conflictWriter:
            self.base_file_Name = base_file_Name
            if "." in self.base_file_Name:
                self.base_file_Name = self.base_file_Name.split(".")[0]
            self.mother_output_path = super_output_folder
            self.save_video = video_output_config['create_output_video']
            self.save_image = video_output_config['create_output_images']

            self.videoOutputPath = None
            self.imageOutputPath = None

            if "Prefix_Filename" in video_output_config:
                self.prefix = video_output_config['Prefix_Filename']
            else:
                self.prefix = ''
            self.create_output_folder()
        else:
            self.base_file_Name = None
            self.mother_output_path = None
            self.save_video = True
            self.save_image = False

            self.videoOutputPath = None
            self.imageOutputPath = None

        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        self.frame_num = 0
        self.verbose = verbose

    def create_output_folder(self):
        pass

    def initialize_writer(self):
        if self.writer is None:
            self.writer = cv2.VideoWriter(
                self.videoOutputPath + ".mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                #self.fps,
                self.frame_size
            )

    def release(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        if self.verbose:
            print(f"VideoWriter released for {self.output_path}")

    def restart(self, super_output_folder):
        self.release()
        self.frame_num = 0
        self.mother_output_path = super_output_folder
        self.create_output_folder()
