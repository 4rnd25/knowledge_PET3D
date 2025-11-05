"""
Created on May 08 2025 11:18

@author: ISAC - pettirsch
"""
import os

from CommonTools.VideoReader.videoreader import VideoReader

class FileListCreator:

    def __init__(self, mother_dataset, verbose=False):
        self.mother_dataset = mother_dataset
        self.verbose = verbose

    def create_file_list(self, config, startRecID, endRecID):
        # Create recordlist
        recordIDlist = list(range(startRecID, endRecID + 1))

        # Get alphapetic order of folders in output folder
        # Get all folders in output folder
        folders = [f for f in os.listdir(config['output_config']['output_folder']) if
                   os.path.isdir(os.path.join(config['output_config']['output_folder'], f))]
        # Sort folders
        folders.sort()

        # Adapt config
        config['database_config']['startrecordID'] = startRecID
        config['database_config']['endrecordID'] = endRecID

        # Load video reader
        video_reader = VideoReader(config['input_config'], config['conflict_config']['buffer_duration'],
                                   config['database_config'])
        filename = video_reader.getFilename()
        foldername = filename.split('.')[0]
        folder_idx_start = folders.index(foldername)
        folder_idx_end = folder_idx_start + len(recordIDlist) - 1
        folder_idx_list = list(range(folder_idx_start, folder_idx_end + 1))

        # Create file list
        file_list = []
        for idx, record_id in enumerate(recordIDlist):
            folder_idx = folder_idx_list[idx]
            if mother_dataset is not None:
                folder_path = os.path.join(folders[folder_idx], "Trajectories")
            else:
                folder_path = os.path.join(config['output_config']['output_folder'], folders[folder_idx], "Trajectories")
            file_path = os.path.join(folder_path, folders[folder_idx] + "_enhanced_trajectories.csv")
            file_list.append(file_path)

        return file_list



