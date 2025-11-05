"""
Created on Jan 14 2025 13:08

@author: ISAC - pettirsch
"""

import torch

from Videoprocessing.Utils.YoloDetectorUtils.utils.general import scale_coords
from Videoprocessing.Utils.Object_classes.FrameObjectHandling.frameobject import FrameObject


class DetectionManager:
    def __init__(self, detection_config, perspectiveTransform, verbose=False):

        # Intialize the detection model
        if detection_config['detector'] == 'Monocular3DTorch':
            from Videoprocessing.Detection.Detectors.monocular3dtorch import Monocular3DTorch
            self.detection_model = Monocular3DTorch(detection_config["model_path"], detection_config["score_threshold"],
                                                    detection_config["nms_iou_threshold"],
                                                    detection_config["image_size"],
                                                    bottom_map=perspectiveTransform.get_bottom_map(),
                                                    cam_pos=perspectiveTransform.getCameraPosition(),
                                                    obj_size_config_path=detection_config["object_size_config"],
                                                    verbose=verbose)
        else:
            raise NotImplementedError("Detector not implemented")

        self.score_threshold = detection_config["score_threshold"]
        self.verbose = verbose
        self.names = self.detection_model.get_names()

    def process_frame(self, frame, frame_id, detectionZoneFilter=None):

        curr_detections = self.detection_model.detect(frame.copy())

        frame_objects = self.createFrameObjects(curr_detections, frame_id, detectionZoneFilter,
                                                detImgshape=self.detection_model.get_image_size(),
                                                imgShape=(frame.shape[1], frame.shape[0]))

        return frame_objects

    def createFrameObjects(self, curr_detections, frame_id, detectionZoneFilter=None,
                           detectionOverlapFilter = None, detImgshape=(640, 640),
                           imgShape=(640, 480)):
        """

        :param curr_detections: List of n,15 Tensor [xyxy, conf, cls, kpt_img_x, kpt_img_y, kpt_world_x, kpt_world_y, kpt_world_z, length, width, height, yaw]
        :return:
        """

        frame_objects = []

        # Process detections
        for i, det in enumerate(curr_detections):  # detections per image
            if len(det):

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(detImgshape, det[:, :4], imgShape).round()

                # Filter out detections with low confidence
                det = det[det[:, 4] > self.score_threshold]

                # Sort by confidence
                det = det[det[:, 4].argsort(descending=True)]

                # Write results
                curr_det_idx = 0
                for *xyxy, conf, cls, kpt_img_x, kpt_img_y, kpt_world_x, kpt_world_y, kpt_world_z, length, width, height, yaw in reversed(
                        det):

                    if detectionZoneFilter is not None:
                        if not detectionZoneFilter.validDetection(
                                (kpt_img_x.detach().cpu().numpy(), kpt_img_y.detach().cpu().numpy())):
                            continue

                    # Normalize yaw
                    normalized_yaw = (yaw + torch.pi) % (2 * torch.pi) - torch.pi

                    frame_objects.append(
                        FrameObject(frame_id, xyxy, (kpt_img_x, kpt_img_y), (kpt_world_x, kpt_world_y, kpt_world_z),
                                    conf, self.names[int(cls)], (length, width, height), normalized_yaw))

                    curr_det_idx += 1

        return frame_objects
