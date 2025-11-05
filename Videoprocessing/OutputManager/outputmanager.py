"""
Created on Jan 15 2025 10:15

@author: ISAC - pettirsch
"""

import os

class OutputManager:
    def __init__(self, output_folder, base_file_name, prefix = '', verbose=False):
        self.super_output_folder = output_folder
        self.base_file_Name = base_file_name
        if "." in self.base_file_Name:
            self.base_file_Name = self.base_file_Name.split(".")[0]
        self.prefix = prefix
        self.output_path = None
        self.verbose = verbose
        self.create_output_folder()

    def create_output_folder(self):
        if self.prefix != '':
            folder_name = self.prefix + self.base_file_Name
        else:
            folder_name = self.base_file_Name
        self.output_path = os.path.join(self.super_output_folder, folder_name)
        # Create output folder if not exists
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            if self.verbose:
                print(f"Output folder created at {self.output_path}")

    def get_output_path(self):
        return self.output_path

    def restart(self, base_file_name):
        self.base_file_Name = base_file_name
        if "." in self.base_file_Name:
            self.base_file_Name = self.base_file_Name.split(".")[0]
        self.create_output_folder()
        if self.verbose:
            print(f"Output folder restarted at {self.output_path}")


