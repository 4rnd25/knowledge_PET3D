"""
Created on Feb 05 2025 08:20

@author: ISAC - pettirsch
"""

from CommonTools.ConflictDetector.Indicator.petcalculator import PETCalculator

class ConflictDetector:
    def __init__(self, config, database_config, record_id, verbose=False):
        self.conflict_indicator = config[
            'conflict_indicator']  # Dict with key = indicator name, value = indicator thresh

        self.save_video = config['create_output_video']
        self.save_images = config['create_output_images']
        self.save_csv = config['save_csv']
        self.save_database = config['save_database']
        self.buffer_duration = config['buffer_duration']

        self.database_config = database_config
        self.record_id = record_id
        self.verbose = verbose

        self.conflictDetector = {}
        self.init_conflict_detector()

    def init_conflict_detector(self):
        for conf_indicator in self.conflict_indicator.keys():
            if conf_indicator == "PET":
                self.conflictDetector[conf_indicator] = PETCalculator(self.conflict_indicator[conf_indicator],
                                                                      frame_rate = 30)

    def detect_conflicts(self, active_tracks, curr_time_stamp_video, curr_time_stamp_system):
        conflicts = []
        for conf_indicator in self.conflict_indicator.keys():
            curr_conflicts = self.conflictDetector[conf_indicator].process_frame(active_tracks, curr_time_stamp_video,
                                                                     curr_time_stamp_system)
            conflicts.extend(curr_conflicts)

        return conflicts
