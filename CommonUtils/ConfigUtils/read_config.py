"""
Created on Jan 14 2025 11:16

@author: ISAC - pettirsch
"""

import yaml

def load_config(config_path="config.yaml"):
    """
    Loads the configuration file in YAML format.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration data as a Python dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config