"""
Created on Feb 05 2025 11:24

@author: ISAC - pettirsch
"""

from collections import deque
import numpy as np
from shapely.geometry import Polygon
import time
import datetime


class PETCalculator:
    def __init__(self, threshold, frame_rate=30):
        self.threshold = threshold
        self.frame_rate = frame_rate    # Update rate (30 FPS)
        self.max_frames = int(self.threshold * frame_rate)  # Max frames to store

        self.history = {}  # Store past occupied areas per vehicle
        self.historic_tracks = {}  # Store past tracks per vehicle
        self.vehicle_id_baned = {}
        self.last_time_seen = {}

    def process_frame(self, vehicles, curr_time_stamp_video, curr_time_stamp_system):
        """Update history and check for interactions for all vehicles in the current frame."""
        interactions = []
        for vehicle_id in vehicles.keys():
            vehicle = vehicles[vehicle_id]
            position = vehicle["position"][-1]  # Latest position
            length, width, yaw = vehicle["length"], vehicle["width"], vehicle["yaw"][-1]

            # Update history with the latest position
            self.update_history(vehicle_id, position, length, width, yaw)
            self.historic_tracks[vehicle_id] = vehicle

            # Check for interaction with past areas of other vehicles
            curr_veh, interacting_veh = self.check_interaction(vehicle_id, position, length, width, yaw)
            if curr_veh is not None:
                if vehicles[curr_veh]['cls'] != self.historic_tracks[interacting_veh]['cls']:
                    print(f"Conflict detected between vehicles {curr_veh} and {interacting_veh} at {curr_time_stamp_video}")
                    print(f"Classes: {vehicles[curr_veh]['cls']} and {self.historic_tracks[interacting_veh]['cls']}")

            self.last_time_seen[vehicle_id] = curr_time_stamp_video

        # Remove vehicles that have not been seen for a while
        keys_to_remove = []
        for vehicle_id in self.last_time_seen.keys():
            duration = (curr_time_stamp_video  - self.last_time_seen[vehicle_id]).total_seconds()
            if duration > self.threshold:
                self.history.pop(vehicle_id, None)
                self.vehicle_id_baned.pop(vehicle_id, None)
                keys_to_remove.append(vehicle_id)
                self.historic_tracks.pop(vehicle_id, None)

        for key in keys_to_remove:
            self.last_time_seen.pop(key, None)

        return interactions

    def get_bottom_bbox(self, position, length, width, yaw):
        """
        Computes the bottom bounding box in 2D (x, y) from a 3D bounding box representation.

        :param position: (x, y, z) bottom center in world coordinates
        :param dims: (length, width, height)
        :param yaw: rotation around Z-axis (yaw in radians)
        :return: Shapely Polygon representing the rotated 2D bounding box
        """

        x, y, z = position  # Extract bottom center coordinates

        # Compute bottom four corners (following the structure of get_3d_corners)
        half_l, half_w = length / 2, width / 2
        corners = np.array([
            [half_l, -half_w],  # front-right-bottom
            [half_l, half_w],  # front-left-bottom
            [-half_l, half_w],  # back-left-bottom
            [-half_l, -half_w]  # back-right-bottom
        ])

        # Create a rotation matrix for yaw
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])

        # Rotate the corners
        rotated_corners = np.dot(corners, rotation_matrix.T)

        # Translate to world coordinates
        rotated_corners[:, 0] += x  # Shift x
        rotated_corners[:, 1] += y  # Shift y

        # Create the polygon
        bottom_bbox = Polygon(rotated_corners)

        return bottom_bbox

    def update_history(self, vehicle_id, position, length, width, yaw):
        """Update past position history of a vehicle with an OBB."""
        if vehicle_id not in self.history:
            self.history[vehicle_id] = deque(maxlen=self.max_frames)

        obb = self.get_bottom_bbox(position, length, width, yaw)
        self.history[vehicle_id].append(obb)

    def check_interaction(self, current_vehicle_id, current_position, length, width, yaw):
        """Check if the current vehicle's OBB interacts with past OBBs of other vehicles."""
        curr_obb = self.get_bottom_bbox(current_position, length, width, yaw)

        if not current_vehicle_id in self.vehicle_id_baned:
            self.vehicle_id_baned[current_vehicle_id] = []

        for vehicle_id, past_obbs in self.history.items():
            if vehicle_id == current_vehicle_id:
                continue  # Skip self-check

            if vehicle_id in self.vehicle_id_baned[current_vehicle_id]:
                continue

            for past_obb in past_obbs:
                if curr_obb.intersects(past_obb):
                    self.vehicle_id_baned[current_vehicle_id].append(vehicle_id)
                    return current_vehicle_id, vehicle_id  # Interaction detected

        return None, None  # No interaction


