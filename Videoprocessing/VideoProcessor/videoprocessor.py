"""
Created on Jan 14 2025 11:36

@author: ISAC - pettirsch
"""

import numpy as np
import time

from Videoprocessing.Detection.detection_manager import DetectionManager
from CommonTools.Filtering.detection_zone_filter import DetectionZoneFilter
from CommonTools.PerspectiveTransform.perspectiveTransform import PerspectiveTransform
from Videoprocessing.Utils.Object_classes.FrameObjectHandling.frameobject_manager import FrameObjectManager
from Videoprocessing.Matching.matching_cost_calculator import MatchingCostCalculator
from Videoprocessing.Matching.object_matcher import ObjectMatcher
from Videoprocessing.Tracking.object_tracker import ObjectTracker
from CommonTools.Filtering.track_lifecylce_filter import TrackLifeCycleFilter
from Videoprocessing.Plotting.plotter import Plotter


class VideoProcessor:
    def __init__(self, config, imgsize=(640, 480), databaseConfig=None, recordID=None, fps=30, verbose=False):
        self.curr_frame = None

        self.perspectiveTransform = PerspectiveTransform(
            calibrationPath=config["calibration_config"]["calibration_matrix_file"],
            triangulationFacesPath=config["calibration_config"]["calibration_faces_file"],
            triangulationPointsPath=config["calibration_config"]["calibration_points_file"],
            calibration_type=config["calibration_config"]["calibration_type"],
            imageSize=imgsize, verbose=verbose)

        self.detectionZoneFilter = DetectionZoneFilter(config["detection_zone_config"], self.perspectiveTransform)

        self.detector = DetectionManager(config["detection_config"], self.perspectiveTransform,
                                         verbose=verbose)
        self.frameObjectManager = FrameObjectManager(verbose=verbose)

        self.matchingCostCalculator = MatchingCostCalculator(matching_config=config["matching_config"], verbose=verbose)
        self.objectMatcher = ObjectMatcher(matching_config=config["matching_config"], verbose=verbose)

        self.tracker = ObjectTracker(verbose=verbose)
        self.trackLifecycleFilter = TrackLifeCycleFilter(config["track_lifecycle_config"], fps=fps, verbose=verbose)

        imageBoarders, worldBoarders = self.getDetectionZoneBoarders()
        self.plotter = Plotter(config['plotting_config'], imageBoarders, worldBoarders, verbose=verbose)

        self.verbose = verbose
        self.fps = fps

    def process_frame(self, frame, frame_id, curr_time_stamp_video, curr_time_stamp_system):
        self.curr_frame = frame

        # Handle Detections
        start_det = time.time()
        detections = self.detector.process_frame(frame, frame_id, detectionZoneFilter=self.detectionZoneFilter)
        self.frameObjectManager.add_frame_objects(detections)
        end_det = time.time()
        # print("Detection Time: ", end_det - start_det)

        # Update Predictions for Tracks
        self.tracker.update_predictions()

        # Match Detections and Tracks
        start_match = time.time()
        cost_matrix, track_idx_list = self.matchingCostCalculator.calculate_matching_cost(self.frameObjectManager,
                                                                            self.tracker)
        matched_objects, unmatched_dets, unmatched_tracks = self.objectMatcher.match_objects(cost_matrix, track_idx_list)
        end_match = time.time()
        # print("Matching Time: ", end_match - start_match)

        # Update Tracks with new Detections
        start_track = time.time()
        self.tracker.update(matched_objects, unmatched_dets, unmatched_tracks,
                            self.frameObjectManager.get_frame_objects(),
                            frame_id, curr_time_stamp_video, curr_time_stamp_system, self.perspectiveTransform,
                            cost_matrix, detectionZoneFilter=self.detectionZoneFilter)
        end_track = time.time()
        # print("Tracking Time: ", end_track - start_track)

        active_and_candidate_tracks, finished_tracks, candidate_track_ids = self.trackLifecycleFilter.filter_tracks(
            self.tracker.get_tracks(),
            frame_id,
            detectionZoneFilter=self.detectionZoneFilter)

        # Set tracks
        self.tracker.setTracks(active_and_candidate_tracks.copy())

        # Get active tracks
        active_tracks = self.tracker.get_active_tracks_as_dict(candidate_track_ids)

        # Plot Output
        start_plot = time.time()
        self.curr_frame = self.plotter.plot_frame(self.curr_frame,
                                                  active_tracks,
                                                  self.perspectiveTransform)
        end_plot = time.time()
        # print("Plotting Time: ", end_plot - start_plot)

        self.frameObjectManager.reset()

        return active_tracks, finished_tracks

    def stop_all_tracks(self):
        active_tracks, finished_track_ids = self.trackLifecycleFilter.stop_tracks(self.tracker.get_tracks())
        return active_tracks, finished_track_ids

    def get_processed_frame(self):
        return self.curr_frame

    def restart(self, config, imgsize=(640, 480), databaseConfig=None, recordID=None):
        self.release()
        self.perspectiveTransform = PerspectiveTransform(calibrationPath=config["calibration_config"]["calibration_matrix_file"],
                                                         triangulationFacesPath=config["calibration_config"]["calibration_faces_file"],
                                                         triangulationPointsPath=config["calibration_config"]["calibration_points_file"],
                                                         calibration_type=config["calibration_config"]["calibration_type"],
                                                         imageSize=imgsize,verbose=self.verbose)

        self.detector = DetectionManager(config["detection_config"], self.perspectiveTransform, verbose=self.verbose)
        self.frameObjectManager = FrameObjectManager(verbose=self.verbose)

        self.matchingCostCalculator = MatchingCostCalculator(matching_config=config["matching_config"],
                                                             verbose=self.verbose)
        self.objectMatcher = ObjectMatcher(matching_config=config["matching_config"], verbose=self.verbose)

        self.tracker = ObjectTracker(verbose=self.verbose)
        self.trackLifecycleFilter = TrackLifeCycleFilter(config["track_lifecycle_config"], fps=self.fps, verbose=self.verbose)

        imageBoarders, worldBoarders = self.getDetectionZoneBoarders()
        self.plotter = Plotter(config['plotting_config'], imageBoarders, worldBoarders, verbose=self.verbose)

    def release(self):
        self.curr_frame = None
        self.perspectiveTransform = None
        self.detector = None
        self.frameObjectManager = None
        self.matchingCostCalculator = None
        self.objectMatcher = None
        self.tracker = None
        self.conflit_detector = None
        self.plotter = None

    def getDetectionZoneBoarders(self):
        imageBoarders = self.detectionZoneFilter.getImageBorders()
        if imageBoarders is not None:
            worldBoarders = [self.perspectiveTransform.pixelToStreePlane(np.asarray(imageBoarders_curr)) for
                             imageBoarders_curr in
                             imageBoarders]
        else:
            worldBoarders = None
        return imageBoarders, worldBoarders
