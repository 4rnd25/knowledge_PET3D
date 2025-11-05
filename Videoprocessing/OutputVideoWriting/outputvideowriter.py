"""
Created on Jan 15 2025 08:24

@author: ISAC - pettirsch
"""
import os.path

import cv2

from Videoprocessing.OutputVideoWriting.video_writer import VideoWriter


class OutputVideoWriter(VideoWriter):

    def write_frame(self, frame, frame_id):
        if self.save_video:
            self.initialize_writer()
            self.writer.write(frame)
        if self.save_image:
            cv2.imwrite(self.imageOutputPath + "_{}".format(frame_id) + ".jpg", frame)
            if self.verbose:
                print("Frame written to output file")
        if self.verbose:
            print("Frame written to output video")
        self.frame_num += 1

    def create_output_folder(self):
        if self.save_image:
            self.imageOutputPath = os.path.join(self.mother_output_path, "Images")
            if not os.path.exists(self.imageOutputPath):
                os.makedirs(self.imageOutputPath)
                if self.verbose:
                    print(f"Image output folder created at {self.imageOutputPath}")
            self.imageOutputPath = os.path.join(self.imageOutputPath, self.prefix + self.base_file_Name)
        if self.save_video:
            self.videoOutputPath = os.path.join(self.mother_output_path, "Video")
            if not os.path.exists(self.videoOutputPath):
                os.makedirs(self.videoOutputPath)
                if self.verbose:
                    print(f"Video output folder created at {self.videoOutputPath}")
            self.videoOutputPath = os.path.join(self.videoOutputPath, self.prefix + self.base_file_Name)
