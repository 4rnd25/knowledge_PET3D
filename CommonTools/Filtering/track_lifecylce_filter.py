"""
Created on Jan 29 2025 10:23

@author: ISAC - pettirsch
"""
import numpy as np

class TrackLifeCycleFilter:
    def __init__(self, track_lifecycle_config, fps, verbose=False):

        self.track_initiation_threshold = track_lifecycle_config["track_initiation_threshold"]
        self.track_termination_threshold = track_lifecycle_config["track_termination_threshold"]

        self.verbose = verbose

    def filter_tracks(self, tracks, frame_id, detectionZoneFilter):

        active_and_candidate_tracks = {}
        finished_tracks = {}
        candidate_track_ids = []
        ids_to_remove = []

        for track_id, track in tracks.items():

            # Check birth
            # if track.get_track_duration() <= 10: # self.track_initiation_threshold:
            #     if not track.detected_last_frame():
            #         ids_to_remove.append(track_id)
            #         continue
            #     else:
            #         if track.get_track_duration() == self.track_initiation_threshold:
            #             active_and_candidate_tracks[track_id] = track
            #             continue
            #         else:
            #             active_and_candidate_tracks[track_id] = track
            #             candidate_track_ids.append(track_id)
            #             continue
            if track.get_track_duration() < 10:
                active_and_candidate_tracks[track_id] = track
                candidate_track_ids.append(track_id)
                continue

            if track.get_track_duration() == 10:
                if track.get_share_detected() < 0.5:
                    ids_to_remove.append(track_id)
                    continue

            # Check death
            if track.get_not_detected_counter() >= self.track_termination_threshold:
                finished_tracks[track_id] = track
                continue

            if track.get_non_valid_counter() >= self.track_termination_threshold:
                finished_tracks[track_id] = track
                continue

            if track.get_track_duration() > 110 and track.get_track_duration() < 300:
                # check number of banned points
                if track.get_banned_counter() >= 100:
                    ids_to_remove.append(track_id)
                    continue

            # Set all other tracks to active
            active_and_candidate_tracks[track_id] = track

        # Remove tracks
        for track_id in ids_to_remove:
            del tracks[track_id]

        return active_and_candidate_tracks, finished_tracks, candidate_track_ids

    def stop_tracks(self, tracks):

        active_tracks = {}
        finished_tracks = {}

        for track_id, track in tracks.items():

            # Check birth
            if track.get_track_duration() <= self.track_initiation_threshold:
                if track.get_track_duration() == self.track_initiation_threshold:
                    finished_tracks[track_id] = track
                    continue
                else:
                    continue

            # Check death
            if track.get_not_detected_counter() >= self.track_termination_threshold:
                finished_tracks[track_id] = track
                continue

            # Set all other tracks to active
            finished_tracks[track_id] = track

        return active_tracks, finished_tracks


    # def adapt_finished_tracks(self, tracks, finished_track_ids):
    #
    #     for track_id in finished_track_ids:
    #         # Get Track
    #         track = tracks[track_id]
    #
    #         # Remove last non valid elements
    #         track.remove_last_non_valid_elements()
    #
    #         # Remove first meter and last meter
    #         # track.remove_boarders()
    #
    #     return tracks, finished_track_ids