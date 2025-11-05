"""
Created on Jan 17 2025 09:38

@author: ISAC - pettirsch
"""
import torch
import numpy as np

class FrameObject:
    def __init__(self, frame_id, bbox_2d, keypoint_image, keypoint_world, score, class_name, dimensions, yaw):
        self.frame_id = frame_id
        self.bbox_2d = torch.tensor(bbox_2d).detach().cpu().numpy()  # Array (xmin, ymin, xmax, ymax)
        self.keypoint_image =  torch.tensor(keypoint_image).detach().cpu().numpy()  # Array (x, y)
        self.keypoint_world = torch.tensor(keypoint_world).detach().cpu().numpy()   # Array (x, y, z)
        self.score = score.detach().cpu().numpy()
        self.class_name = class_name
        self.dimensions = torch.tensor(dimensions).detach().cpu().numpy()  # Tuple (length, width, height)
        self.yaw = yaw.clone().detach().cpu().numpy() % (2 * np.pi)
        # self.yaw = torch.tensor(yaw).clone().detach().cpu().numpy() % (2 * np.pi)

    def to_dict(self):
        return {
            "frame_id": self.frame_id,
            "bbox_2d": self.bbox_2d,
            "keypoint_image": self.keypoint_image,
            "keypoint_world": self.keypoint_world,
            "score": self.score,
            "class_name": self.class_name,
            "dimensions": self.dimensions,
            "yaw": self.yaw,
        }

    def get_world_center(self):
        return self.keypoint_world

    def get_dimensions(self):
        return self.dimensions

    def get_class_name(self):
        return self.class_name

    def get_yaw(self):
        return self.yaw

    def get_confidence(self):
        return self.score

    def reset(self):
        """
        Resets all attributes of the FrameObject to None.
        """
        self.frame_id = None
        self.bbox_2d = None
        self.keypoint_image = None
        self.keypoint_world = None
        self.score = None
        self.class_name = None
        self.dimensions = None
        self.yaw = None