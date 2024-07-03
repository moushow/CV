import os
import random
from glob import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

# 定义 Dice 系数
def dice_coef(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

# 定义 IoU 系数
def iou_coef(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    sum = K.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou

# 创建数据框
def create_df(data_dir):
    images_paths = []
    masks_paths = glob(f'{data_dir}/*/*_mask*')

    for i in masks_paths:
        images_paths.append(i.replace('_mask', ''))

    df = pd.DataFrame(data={'images_paths': images_paths, 'masks_paths': masks_paths})
    return df.sample(n=500, random_state=42)

# 创建数据生成器
def create_gens(df, aug_dict):
    img_size = (128, 128)
    batch_size = 16

    img_gen = ImageDataGenerator(**aug_dict)
    msk_gen = ImageDataGenerator(**aug_dict)

    image_gen = img_gen.flow_from_dataframe(df, x_col='images_paths', class_mode=None, color_mode='rgb', target_size=img_size,
                                            batch_size=batch_size, save_to_dir=None, save_prefix='image', seed=1)

    mask_gen = msk_gen.flow_from_dataframe(df, x_col='masks_paths', class_mode=None, color_mode='grayscale', target_size=img_size,
                                            batch_size=batch_size, save_to_dir=None, save_prefix='mask', seed=1)
    gen = zip(image_gen, mask_gen)

    for (img, msk) in gen:
        img = img / 255
        msk = msk / 255
        msk[msk > 0.5] = 1
        msk[msk <= 0.5] = 0

        yield (img, msk)

# 显示图像
def show_images(images, masks):
    plt.figure(figsize=(12, 12))
    for i in range(9):  # 只显示9张图像
        plt.subplot(3, 3, i+1)
        img_path = images[i]
        mask_path = masks[i]
        # 读取图像并转换为 RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 读取掩码
        mask = cv2.imread(mask_path)
        # 显示图像和掩码
        plt.imshow(image)
        plt.imshow(mask, alpha=0.4)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# 加载数据集
data_dir = r"C:\Users\Lenovo\Desktop\Brain MRI segmentation"

df = create_df(data_dir)

# 分割数据框
def split_df(df):
    train_df, dummy_df = train_test_split(df, train_size=0.8)
    valid_df, test_df = train_test_split(dummy_df, train_size=0.5)
    return train_df, valid_df, test_df

_, _, test_df = split_df(df)

# 加载模型
model = load_model(r"D:\IDEAProject\CV\unet.h5", custom_objects={'dice_loss': dice_coef, 'iou_coef': iou_coef, 'dice_coef': dice_coef})

# 评估模型
test_gen = create_gens(test_df, aug_dict={})
ts_length = len(test_df)
test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80]))
test_steps = ts_length // test_batch_size

test_score = model.evaluate(test_gen, steps=test_steps, verbose=1)

print("测试损失: ", test_score[0])
print("测试精度: ", test_score[1])
print("测试IoU: ", test_score[2])
print("测试Dice: ", test_score[3])

# 预测并可视化结果
for _ in range(10):
    index = np.random.randint(1, len(test_df.index))
    img = cv2.imread(test_df['images_paths'].iloc[index])
    img = cv2.resize(img, (128, 128))
    img = img / 255
    img = img[np.newaxis, :, :, :]

    predicted_img = model.predict(img)

    plt.figure(figsize=(12, 12))

    plt.subplot(1, 3, 1)
    plt.imshow(np.squeeze(img))
    plt.axis('off')
    plt.title('原始图像')

    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(cv2.imread(test_df['masks_paths'].iloc[index])))
    plt.axis('off')
    plt.title('原始掩码')

    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(predicted_img) > 0.5)
    plt.title('预测结果')
    plt.axis('off')

    plt.show()
