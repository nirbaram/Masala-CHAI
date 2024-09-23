"""
Inference of YoloV8 component detection model
"""

import numpy as np
import torch
import os

from models.experimental import attempt_load
#from utils.datasets import letterbox
from utils.yolov8_utils.dataloaders import letterbox
from utils.yolov8_utils.general import check_img_size, non_max_suppression, scale_coords
from utils.yolov8_utils.torch_utils import select_device



from models.common import DetectMultiBackend

def detect(img_in):
    """Runs YOLOv8 model to detect bounding boxes and classes of components present in the circuit

    Args:
        img_in (numpy array): input image

    Returns:
        det (numpy array): bounding boxes and classes
    """
    path = os.path.dirname(__file__)
    with torch.no_grad():

        weights, imgsz = './trained_checkpoints/yolov8_checkpoint.pt', 640

        device = select_device('cpu')
        conf_thres = 0.55
        iou_thres = 0.55
        classes = None
        agnostic_nms = False
        augment = False

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
        imgsz = check_img_size(imgsz, s=model.stride)

        # resizing image
        img = torch.zeros((3, imgsz, imgsz), device=device)  # init img
        img = letterbox(img_in, new_shape=imgsz)[0]

        # BGR to RGB, to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1)  
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0 
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # Inference
        pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        det = pred[0]
        # scaling coordinates back to original image size
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_in.shape).round()
        det = det.cpu().numpy()
    
    return det
