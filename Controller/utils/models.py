import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

import segment_anything as sam




def create_trunk_predictor(model_path):
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

  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_path)
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

  predictor_synth = DefaultPredictor(cfg)

  return predictor_synth



def get_trunk_predictions(image, predictor):
  '''
  Makes a forward pass on the image through the model to yield predictions
  '''
  return predictor(image)



def save_trunk_mask(predictions):
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



def create_sam_predictor() -> sam.SamPredictor:
  '''
  Predict an image mask using the SAM predictor object
  '''
  model_path = 'assets/models/sam_vit_h_4b8939.pth'
  sam_checkpoint = model_path
  model_type = "vit_h"

  device = "cuda" if torch.cuda.is_available() else "cpu"

  sam_model = sam.sam_model_registry[model_type](checkpoint=sam_checkpoint)
  sam_model.to(device=device)

  predictor = sam.SamPredictor(sam_model)

  return predictor



def predict_sam_mask(predictor: sam.predictor.SamPredictor):
  '''
  Runs an inference on the SAM model to generate predictions based on the prompts given
  '''
  h, w = predictor.original_size
  input_point = np.array([[int(w/2), int(h/2)]])
  input_label = np.array([1])

  masks, scores, logits = predictor.predict(
      point_coords = input_point,
      point_labels = input_label,
      multimask_output=True
  )

  return masks, scores, logits