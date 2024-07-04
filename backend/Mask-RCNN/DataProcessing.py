import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
from IPython.display import clear_output
from skimage.io import imread, imsave

ROOT_PATH = '../Brain MRI segmentation/'
mask_files = glob.glob(ROOT_PATH + '*/*_mask*')
image_files = [file.replace('_mask', '') for file in mask_files]

def diagnosis(mask_path):
    return 1 if np.max(cv2.imread(mask_path)) > 0 else 0

files_df = pd.DataFrame({
    "image_path": image_files,
    "mask_path": mask_files,
    "diagnosis": [diagnosis(x) for x in mask_files]
})

#  Create directories and split data

os.makedirs('brain-tumor-segmentation/brain_tumor_data/train', exist_ok=True)
os.makedirs('brain-tumor-segmentation/brain_tumor_data/val', exist_ok=True)
os.makedirs('brain-tumor-segmentation/brain_tumor_data/test', exist_ok=True)

# Sample 70% of the data for training
train_df = files_df.sample(frac=0.7, random_state=42)
remaining_df = files_df.drop(train_df.index)

# Sample 50% of the remaining 30% data for validation (this will be 10% of the total original data)
val_df = remaining_df.sample(frac=0.5, random_state=42)

# The rest of the data (10% of the total original data) will be used for testing
test_df = remaining_df.drop(val_df.index)

print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

def save_images_and_annotations(df, folder):
    annotations = {}
    for idx, row in df.iterrows():
        image_path = row['image_path']
        mask_path = row['mask_path']
        image = imread(image_path)
        mask = imread(mask_path, as_gray=True)
        new_image_path = f'brain-tumor-segmentation/brain_tumor_data/{folder}/{os.path.basename(image_path)}'
        new_mask_path = f'brain-tumor-segmentation/brain_tumor_data/{folder}/{os.path.basename(mask_path)}'
        imsave(new_image_path, image)
        imsave(new_mask_path, mask)
        regions = []
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if contour.size >= 6:
                contour = contour.squeeze()
                all_points_x = contour[:, 0].tolist()
                all_points_y = contour[:, 1].tolist()
                regions.append({
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": all_points_x,
                        "all_points_y": all_points_y
                    },
                    "region_attributes": {}
                })
        annotations[os.path.basename(image_path)] = {
            "filename": os.path.basename(image_path),
            "size": os.path.getsize(image_path),
            "regions": regions,
            "file_attributes": {}
        }
    with open(f'brain-tumor-segmentation/brain_tumor_data/{folder}/annotations.json', 'w') as f:
        json.dump(annotations, f)

save_images_and_annotations(train_df, 'train')
save_images_and_annotations(val_df, 'val')
save_images_and_annotations(test_df, 'test')
clear_output()
print("Data preparation complete.")