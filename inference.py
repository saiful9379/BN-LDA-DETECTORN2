import os
import json
import glob
import cv2
import numpy as np
import torch 
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from utility.visulization import draw_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

DEBUG = True

classes = ["paragraph", "text_box", "image", "table"]

CLASS_DICT = {0: 'paragraph', 1: 'text_box', 2: 'image', 3: 'table'}

COLORS = {
    0: (255, 0, 0), 
    1: (0, 255, 0), 
    2: (0, 0, 255), 
    3: (0, 255, 255)
    }

def masks_processing(mask, img_dim):
    H, W = img_dim[0], img_dim[1]
    mask = mask* 255
    mask  = mask.astype('uint8')
    mask = cv2.resize(mask, (W, H))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Extract the contour points from the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.reshape(-1, 2)
    contour_points = contour_points.reshape((-1, 1, 2))

    allx_ally = [(i[0][0], i[0][1]) for i in contour_points.tolist()]

    return allx_ally

def get_polygon_of_masks(masks, img_dim):
    img_dims = [img_dim]*len(masks)
    p_polygon = list(map(masks_processing, masks, img_dims))
    return p_polygon


def get_configuration(model_path, config_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_path))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.INPUT.MIN_SIZE_TEST=1024
    cfg.INPUT.MAX_SIZE_TEST=2048
    cfg.TEST.DETECTIONS_PER_IMAGE = 600
    cfg.MODEL.WEIGHTS = model_path
    
    return cfg

def model_loading(cfg):
    predictor = DefaultPredictor(cfg)
    return predictor

def prediction(predictor, img, file_name="unknown.jpg", output_dir = "logs"):
    H, W, _ = img.shape

    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1],
                scale=0.5,
                instance_mode=ColorMode.IMAGE_BW
                )
    predictions = outputs["instances"].to("cpu")
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

    boxes = boxes.tensor.detach().numpy().tolist()

    for box, _cls in zip(boxes, classes):
        box = [int(i) for i in box]
        cv2.putText(img, CLASS_DICT[_cls], (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                        COLORS[_cls], 1, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (box[0], box[1]),(box[2], box[3]),  COLORS[_cls], 2)
    if predictions.has("pred_masks"):
        masks = np.asarray(predictions.pred_masks)
        masks = [mask.astype(int) for mask in masks]

        polyon_coordinates = get_polygon_of_masks(masks, img_dim=[H, W])

        img = draw_mask(img, polyon_coordinates, classes)
    cv2.imwrite(os.path.join(output_dir, file_name), img)

if __name__ == "__main__":
    from tqdm import tqdm 
    config_path = "./configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    model_path = "/media/sayan/hdd1/CV/BN-LDA-DETECTORN2/model/model_final.pth"
    image_path = "/media/sayan/hdd1/CV/BN-LDA-DETECTORN2/image"

    output_dir = "logs"
    os.makedirs(output_dir, exist_ok=True)

    print("Model Loading .........")
    cfg = get_configuration(model_path, config_path)
    model = model_loading(cfg)
    print("Done")

    image_files = glob.glob(image_path+"/*")
    for i in tqdm(range(len(image_files))):
        file_ = image_files[i]
        file_name = os.path.basename(file_)
        img = cv2.imread(file_)
        prediction(model, img, file_name, output_dir) 