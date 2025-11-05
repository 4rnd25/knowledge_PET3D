# Knowledge PET3D

**Repository for the paper:**  
*Knowledge PET3D: An interpretable framework for 3D near-miss detection in thermal traffic video.*

📄 **Link to paper:** _[to be added]_  

---

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data](#data)
4. [Overview of the Pipeline](#overview-of-the-pipeline)
5. [Video Processing](#video-processing)
6. [Trajectory Clustering](#trajectory-clustering)
7. [Maneuver Classification](#maneuver-classification)
8. [Interaction Detection](#interaction-detection)
9. [Rule Extraction](#rule-extraction)
10. [Rule Assignment](#rule-assignment)
11. [Transformer Training](#transformer-training)
12. [Anomaly Detection](#anomaly-detection)
13. [Conflict Filtering and Reporting](#conflict-filtering-and-reporting)
14. [Acknowledgements](#acknowledgements)
15. [License](#license)

---

## Introduction
This repository contains the official implementation of the paper  
**“Knowledge PET3D: An Interpretable Framework for 3D Near-Miss Detection in Thermal Traffic Video.”**  

If you reference this repository in your research, please cite the corresponding paper.

---

## Installation
This project was developed using **Python 3.11**.  
All required packages are listed in `requirements.txt`.

To ensure proper module imports, include this repository in your `PYTHONPATH`.  
Additionally, for video processing, include the directory  
`$PATH_TO_REPO/Videoprocessing/Utils/YoloDetectorUtils`.

---

## Data
The dataset used in the paper is available [here](_add_link_).  
It includes:

- Six example videos per location  
- CSV files mapping record IDs to video start times  
- Enhanced trajectories covering the entire experimental period  
- Detected conflicts  

### Configuration examples
- **Location A:** `Configuration/Location_A.yaml`  
  - `RecID_Start: 1`  
  - `RecID_first24hours: 135`  
  - `RecID_end: 855`  

- **Location B:** `Configuration/Location_B.yaml`  
  - `RecID_Start: 101310`  
  - `RecID_first24hours: 101444`  
  - `RecID_end: 102085`

Ensure the `output_folder` parameter in the configuration file points to the correct dataset directory, e.g.:  
`Knowledge_PET3D_3D_Traffic_Trajectories_test/Location_A_Trajectories_and_Conflicts` or  
`Knowledge_PET3D_3D_Traffic_Trajectories_test/Location_B_Trajectories_and_Conflicts`.

---

## Overview of the Pipeline
1. Video Processing – `Videoprocessing/01_main_video_trajectory_extraction.py`  
2. Trajectory Clustering – `TrajectoryProcessing/02_main_trajectory_clustering.py`  
3. Maneuver Classification – `TrajectoryProcessing/03_main_maneuver_classification.py`  
4. Interaction Detection – `TrajectoryProcessing/04_main_PET_conflict_detection.py`  
5. Rule Extraction – `Conflictprocessing/05_main_rule_extractor.py`  
6. Rule Assignment – `Conflictprocessing/06_main_rule_assigner.py`  
7. Anomaly Detection – `Conflictprocessing/07a_main_LOF_calculator.py` or `Conflictprocessing/07b_main_LOF_calculator_LSTM.py`

---

## Video Processing
To extract trajectories, run `Videoprocessing/01_main_video_trajectory_extraction.py`.  
Arguments: configuration file (`Configurations/*`), `startRecID`, and `endRecID`.

Detection zones can be adapted in `Videoprocessing/DetectionZoneDefinition`.  
Model weights are located at `Videoprocessing/Detection/Detectors/projnet_cls_7_IDetectMon3D_ISAC_ALL_Extended.pt`.  
Object size configuration is in `Videoprocessing/Detection/Detectors/object_size_config.yaml`.  
Ensure that calibration and road surface paths in the configuration are correctly linked to the dataset.

---

## Trajectory Clustering
To cluster trajectories, run `TrajectoryProcessing/02_main_trajectory_clustering.py`.  
Arguments: configuration file (`Configurations/*`), `startRecID`, and `endRecID`.

If the output path already contains `cluster_means`, the script assigns clusters.  
Otherwise, it performs clustering.  
It is recommended to use one day of data to create cluster means and assign clusters to all subsequent trajectories.

---

## Maneuver Classification
To create a maneuver DataFrame, run `TrajectoryProcessing/03_main_maneuver_classification.py`.  
Arguments: configuration file (`Configurations/*`).  
Cluster means must be available beforehand.

---

## Interaction Detection
To detect interactions based on 2D or 3D PET, run `TrajectoryProcessing/04_main_PET_conflict_detection.py`.  
Arguments: configuration file (`Configurations/*`), `startRecID`, and `endRecID`.  
Cluster means and maneuver classifications must be available in advance.

---

## Rule Extraction
To extract traffic rules from detected conflicts, run `Conflictprocessing/05_main_rule_extractor.py`.  
Arguments: configuration file (`Configurations/*`), `startRecID`, and `endRecID`.

---

## Rule Assignment
To assign the extracted traffic rules, run `Conflictprocessing/06_main_rule_assigner.py`.  
Arguments: configuration file (`Configurations/*`), `startRecID`, and `endRecID`.

---

## Transformer Training
To train or evaluate the Transformer or LSTM model for trajectory latent-space representation, use  
`TrajectoryTransformer/train_trajectory_transformer.py` or `TrajectoryTransformer/evaluate_trajectory_transformer.py`.  

To reproduce the paper results, split the train and validation sets according to  
`TrajectoryTransformer/train_file_list.txt` and `TrajectoryTransformer/val_file_list.txt`.

The weights used in the paper can be found in `TrajectoryTransformer`.

---

## Anomaly Detection
To detect evasive actions using the Transformer or LSTM model, run  
`Conflictprocessing/07a_main_LOF_calculator.py` or `Conflictprocessing/07b_main_LOF_calculator_LSTM.py`.  
Arguments: configuration file (`Configurations/*`), `startRecID`, and `endRecID`.  
Ensure the trajectory transformer model path is correctly set.

---

## Conflict Filtering and Reporting
The following scripts provide examples of conflict filtering and reporting:  
`Conflictprocessing/main_conflict_filtering.py`,  
`Conflictprocessing/main_conflict_overview.py`, and  
`Conflictprocessing/main_conflict_report_creator.py`.  

The last script creates conflict reports including video snippets and visualizations.

---

## Acknowledgements
This repository includes parts of the following third-party module:

- **YOLOv7**  
  - Modified files: `Videoprocessing/Utils/YoloDetectorUtils`  
  - Repository: [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)  
  - License: GPL-3.0 (see `yolov7/LICENSE.md` for details)

---

## License
This repository is released under the **GPL License**.  
See the [LICENSE](./LICENSE) file for details.

