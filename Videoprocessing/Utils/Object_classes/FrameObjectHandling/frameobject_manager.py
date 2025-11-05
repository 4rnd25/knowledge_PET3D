"""
Created on Jan 17 2025 08:06

@author: ISAC - pettirsch
"""
import torch

class FrameObjectManager:
    def __init__(self, verbose=False):
        self.verbose = verbose

        self.frame_objects = []

    def add_frame_object(self, frame_object):
        self.frame_objects.append(frame_object)

    def add_frame_objects(self, frame_objects):
        self.frame_objects.extend(frame_objects)

    def get_frame_objects(self):
        return self.frame_objects

    def get_number_of_frame_objects(self):
        return len(self.frame_objects)

    def reset(self):
        self.release()

    def release(self):
        self.frame_objects = []
        self.verbose = None
