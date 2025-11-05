"""
Created on Jan 15 2025 13:24

@author: ISAC - pettirsch
"""

import torch
import yaml
import numpy as np

from Videoprocessing.Utils.YoloDetectorUtils.torch_utils import select_device
from Videoprocessing.Utils.YoloDetectorUtils.models.experimental import attempt_load
from Videoprocessing.Utils.YoloDetectorUtils.utils.general import check_img_size
from Videoprocessing.Utils.YoloDetectorUtils.datasets import letterbox
from Videoprocessing.Utils.YoloDetectorUtils.utils.torch_utils import time_synchronized
from Videoprocessing.Utils.YoloDetectorUtils.utils.general import non_max_suppression_3d
from Videoprocessing.Utils.Cuboid_calc.calc_3d_boxes import calc_3d_output



class Monocular3DTorch:

    def __init__(self, model_path, score_threshold, nms_iou_threshold, imgsz, bottom_map, cam_pos,
                 obj_size_config_path=None, verbose=False):

        # Initialize
        self.device = select_device('0')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.verbose = verbose

        # Load model
        self.model = attempt_load(model_path, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
        self.img_size = imgsz

        if self.half:
            self.model.half()  # to FP16
        self.model.eval()

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # Load bottom map and cam pos
        self.bottom_map = torch.from_numpy(bottom_map).to(self.device)
        self.cam_pos = torch.from_numpy(cam_pos).to(self.device)

        # Load object size config
        with open(obj_size_config_path, 'r') as f:
            obj_size_config = yaml.safe_load(f)
        self.obj_size_config_tensor = torch.tensor(
            [[obj_size_config['objects'][key]['length'], obj_size_config['objects'][key]['width'],
              obj_size_config['objects'][key]['height']] for key in
             obj_size_config['objects'].keys()])

    def detect(self, img0):

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred, _ = self.model(img, augment=False)
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression_3d(pred, self.score_threshold, self.nms_iou_threshold, labels=[], multi_label=False)
        t3 = time_synchronized()

        # Calc boxes
        pred_3d = calc_3d_output(pred, self.bottom_map, self.cam_pos, cls_mean_lookup=self.obj_size_config_tensor,
                                 img_shape=(img.shape[2], img.shape[3]))
        t4 = time_synchronized()

        if self.verbose:
            print("Time for inference: {} ms".format((1E3 * (t2 - t1))))
            print("Time for NMS: {} ms".format((1E3 * (t3 - t2))))
            print("Time for box calculation: {} ms".format((1E3 * (t4 - t3))))

        return pred_3d

    def get_names(self):
        return self.names

    def get_image_size(self):
        return (self.img_size, self.img_size)
