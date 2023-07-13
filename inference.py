import os
import json
import cv2
import numpy as np 
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode


classes = ["paragraph", "text_box", "image", "table"]

CLASS_DICT = {0: 'paragraph', 1: 'text_box', 2: 'image', 3: 'table'}

COLORS = {
    0: (255, 0, 0), 
    1: (0, 255, 0), 
    2: (0, 0, 255), 
    3: (0, 255, 255)
    }


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

    print(img.shape)

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

    print("keypoints :", keypoints)

    if predictions.has("pred_masks"):
        masks = np.asarray(predictions.pred_masks)
        print(masks)
        # masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]


    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image_out = out.get_image()[:, :, ::-1]

    print("after shape", image_out.shape)
    # plt.show()
    cv2.imwrite(os.path.join(output_dir, file_name), img)

if __name__ == "__main__":

    config_path = "./configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    model_path = "/media/sayan/hdd1/CV/BN-LDA-DETECTORN2/model/model_final.pth"
    image_path = "/media/sayan/hdd1/CV/BN-LDA-DETECTORN2/image/1cb63ad5-a85c-4969-a5cb-4a95723409ba.png"

    output_dir = "logs"
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.basename(image_path)
    img = cv2.imread(image_path)
    cfg = get_configuration(model_path, config_path)
    model = model_loading(cfg)
    prediction(model, img, file_name, output_dir) 