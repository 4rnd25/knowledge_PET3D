"""
Created on Jan 15 2025 08:24

@author: ISAC - pettirsch
"""

import os
import cv2
import torch
from Videoprocessing.OutputVideoWriting.video_writer import VideoWriter

from Videoprocessing.Utils.Cuboid_calc.calc_3d_corners import get_3d_corners
from Videoprocessing.Utils.Plotting_Utils.plot_3d_box import plot_3d_corners


class ConflictVideoWriter(VideoWriter):

    def write_video(self, frames, trajectories, perspective_transform, label, colors,
                    video_output_path, save_images=False):

        self.colors_thermal = {}
        for key in colors.keys():
            self.colors_thermal[key] = [colors[key][2], colors[key][1], colors[key][0]]

        self.videoOutputPath = os.path.join(video_output_path, label)
        self.writer = None
        self.initialize_writer()

        keypoints = []
        for trajectory in trajectories:
            keypoints.append([])

        for timestamp, frame in frames:
            frame, keypoints = self.draw_trajectories(frame, timestamp, trajectories, perspective_transform, keypoints)

            # Remove overlay
            # Define the overlay area coordinates (adjust as needed)
            overlay_top_left = (0, 0)
            overlay_bottom_right = (290, 20)  # Fine-tune if the overlay size differs

            # Draw a black rectangle to cover the overlay
            cv2.rectangle(frame, overlay_top_left, overlay_bottom_right, (0, 0, 0), thickness=-1)

            if save_images:
                image_name = os.path.join(video_output_path, "{}_{}.jpg".format(label, self.frame_num))
                cv2.imwrite(image_name, frame)
            self.writer.write(frame)
            self.frame_num += 1

        self.writer.release()

    def draw_trajectories(self, frame, timestamp, trajectories, perspective_transform, keypoints):


        for i, trajectory in enumerate(trajectories):


            position, yaw = trajectory.get_position_and_yaw(timestamp)


            length = trajectory.get_length()
            width = trajectory.get_width()
            height = trajectory.get_height()
            cls = trajectory.get_class()
            if trajectory.idVehicle == 150:
                cls = 'bicycle'

            if cls == 'bicycle':
                self.colors_thermal[cls] = [49, 15, 162]  # BGR = red

            if len(keypoints[i]) > 0:
                overlay = frame.copy()
                for keypoint_image in keypoints[i]:
                    # Not filled circle
                    # print("ignore")
                    overlay = cv2.circle(overlay, (keypoint_image[0], keypoint_image[1]), 5,
                                       self.colors_thermal[cls],
                                       2)
                alpha = 0.5
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            if position is None:
                continue

            bc_3d = torch.tensor(
                [position[0], position[1], position[2]], dtype=torch.float64).unsqueeze(0)
            corners_3d = get_3d_corners(bc_3d, torch.tensor(
                [length, width, height],
                dtype=torch.float64).unsqueeze(0),
                                        torch.tensor(yaw, dtype=torch.float64))[0, :, :]
            corners_3d_img = perspective_transform.worldToPixel(corners_3d.cpu().numpy())
            bc_3d_img = perspective_transform.worldToPixel(bc_3d.cpu().numpy())

            frame = plot_3d_corners(corners_3d_img, frame, label="", color=self.colors_thermal[cls],
                            line_thickness=2)
            # Fill circle
            bc_3d_img = bc_3d_img.astype(int)
            frame = cv2.circle(frame, (bc_3d_img[0], bc_3d_img[1]), 5,
                                 self.colors_thermal[cls],
                                    -1)

            frame_save = frame.copy()



            keypoints[i].append(bc_3d_img)

        return frame, keypoints






    def write_buffer_to_video(self, buffer, conflict_name):
        self.create_specific_output_folder(conflict_name)
        if self.save_video:
            self.initialize_writer()
        frame_num = 0
        for timestamp, frame in buffer:
            if self.save_video:
                self.writer.write(frame)
            if self.save_image:
                cv2.imwrite(self.imageOutputPath + "_{}".format(frame_num) + ".jpg", frame)
                if self.verbose:
                    print("Frame written to output file")
            frame_num += 1
        self.writer.release()
        self.writer = None
        if self.verbose:
            print("Conflict buffer written to video")
        # self.buffer.clear()

    def create_output_folder(self):
        self.output_path = os.path.join(self.mother_output_path, "Conflicts")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            if self.verbose:
                print(f"Conflict output folder created at {self.output_path}")

    def create_specific_output_folder(self, conflict_name):
        if self.save_video:
            self.videoOutputPath = os.path.join(self.output_path, conflict_name, "Video")
        if self.save_image:
            self.imageOutputPath = os.path.join(self.output_path, conflict_name, "Images")

        if not os.path.exists(self.videoOutputPath):
            os.makedirs(self.videoOutputPath)
            if self.verbose:
                print(f"Conflict output folder created at {self.videoOutputPath}")

        if not os.path.exists(self.imageOutputPath):
            os.makedirs(self.imageOutputPath)
            if self.verbose:
                print(f"Conflict image output folder created at {self.imageOutputPath}")

        self.videoOutputPath = os.path.join(self.videoOutputPath, conflict_name)
        self.imageOutputPath = os.path.join(self.imageOutputPath, conflict_name)
        if self.verbose:
            print(f"Conflict output folder created at {self.videoOutputPath}")
            print(f"Conflict image output folder created at {self.imageOutputPath}")
