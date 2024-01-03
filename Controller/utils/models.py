import os
import cv2
import torch
import numpy as np
from PIL import Image
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt



def create_predictor(model_path, out_dir):
  '''
  Creates a DefaultPredictor object from a saved model
  '''
  cfg = get_cfg()
  cfg.INPUT.MASK_FORMAT = "bitmask"
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
  cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
  cfg.DATASETS.TRAIN = ()
  cfg.DATASETS.TEST = ()
  cfg.DATALOADER.NUM_WORKERS = 8
  cfg.SOLVER.IMS_PER_BATCH = 8
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster (default: 512)
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (tree)
  cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
  cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
  cfg.MODEL.MASK_ON = True

  cfg.OUTPUT_DIR = out_dir
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_path)
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

  predictor_synth = DefaultPredictor(cfg)

  return predictor_synth



def get_predictions(image, predictor):
  '''
  Makes a forward pass on the image through the model to yield predictions
  '''
  return predictor(image)



def save_mask(predictions):
  '''
  The mask of interest among all the predicted masks. This function takes in all the predictions and returns the largest mask
  '''
  masks = predictions["instances"].pred_masks
  sizes = []

  for mask in masks:
    mask = mask.cpu().numpy().astype(np.int32)
    mask_size = cv2.countNonZero(mask)
    sizes.append(mask_size)

  sizes = np.array(sizes)
  index = np.argmax(sizes)
  mask_oi = masks[index]

  return mask_oi.cpu().numpy()