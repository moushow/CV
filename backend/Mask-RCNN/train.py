import json
import os

import mrcnn.model as modellib
import numpy as np
import skimage.draw
from keras.callbacks import History
from mrcnn import utils
from mrcnn.config import Config
from skimage.io import imread


class CustomMaskRCNN(modellib.MaskRCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = History()

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers):
        if not hasattr(self, 'keras_model'):
            self.set_log_dir()
            self.keras_model = self.build(learning_rate=learning_rate)
        callbacks = [self.history]
        super().train(train_dataset, val_dataset, learning_rate, epochs, layers, custom_callbacks=callbacks)


class TrainConfig(Config):
    NAME = "tumor_detect"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # background + 1 class of tumor
    DETECTION_MIN_CONFIDENCE = 0.7
    LEARNING_RATE = 0.001
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5


config = TrainConfig()


class TumorDataset(utils.Dataset):
    def load_brain_tumor_images(self, dataset_dir, subset):
        self.add_class("tumor", 1, "tumor")
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations = json.load(open(os.path.join(dataset_dir, 'annotations.json')))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]
        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions']]
            image_path = os.path.join(dataset_dir, a['filename'])
            image = imread(image_path)
            height, width = image.shape[:2]
            self.add_image("tumor", image_id=a['filename'], path=image_path,
                           width=width, height=height, polygons=polygons)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "tumor":
            return super().load_mask(image_id)
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "tumor":
            return info["path"]
        else:
            return super().image_reference(image_id)


# Load datasets

DATASET_DIR = './brain-tumor-segmentation/brain_tumor_data/'

dataset_train = TumorDataset()
dataset_train.load_brain_tumor_images(DATASET_DIR, "train")
dataset_train.prepare()

dataset_val = TumorDataset()
dataset_val.load_brain_tumor_images(DATASET_DIR, "val")
dataset_val.prepare()

# Create and train model

ROOT_PATH = '../Brain MRI segmentation/'
ROOT_DIR = os.path.abspath("./Mask_RCNN")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

model = CustomMaskRCNN(mode="training", config=config, model_dir=ROOT_DIR)
model.load_weights(COCO_MODEL_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=40,
            layers='heads')
model_path = os.path.join(ROOT_DIR, "mask_rcnn_tumor.h5")
model.keras_model.save_weights(model_path)
