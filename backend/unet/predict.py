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
    img_size = (256, 256)
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
data_dir = r"F:\BaiduNetdiskDownload\Brain MRI segmentation\Brain MRI segmentation"

df = create_df(data_dir)

# 分割数据框
def split_df(df):
    train_df, dummy_df = train_test_split(df, train_size=0.8)
    valid_df, test_df = train_test_split(dummy_df, train_size=0.5)
    return train_df, valid_df, test_df

_, _, test_df = split_df(df)

# 加载模型
model = load_model(r"F:\cv\CV\enhanced_unet.h5", custom_objects={'dice_loss': dice_coef, 'iou_coef': iou_coef, 'dice_coef': dice_coef})

# 评估模型
test_gen = create_gens(test_df, aug_dict={})
ts_length = len(test_df)
test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80]))
test_steps = ts_length // test_batch_size

test_score = model.evaluate(test_gen, steps=test_steps, verbose=1)

print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])
print("Test IoU: ", test_score[2])
print("Test Dice: ", test_score[3])

# 定义 dsc_per_volume 函数
def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    def dsc(y_pred, y_true, smooth=100):
        y_pred_flatten = y_pred.flatten()
        y_true_flatten = y_true.flatten()
        intersection = np.sum(y_pred_flatten * y_true_flatten)
        union = np.sum(y_pred_flatten) + np.sum(y_true_flatten)
        return (2 * intersection + smooth) / (union + smooth)

    dsc_list = []
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for p in range(len(num_slices)):
        y_pred = np.array(validation_pred[index: index + num_slices[p]])
        y_true = np.array(validation_true[index: index + num_slices[p]])
        dsc_value = dsc(y_pred, y_true)
        print(f'Volume {p}: DSC = {dsc_value}')  # 输出每个体积的 DSC
        dsc_list.append(dsc_value)
        index += num_slices[p]
    return dsc_list

# 预测并收集结果
validation_pred = []
validation_true = []
patient_slice_index = []

for i, (img, mask) in enumerate(create_gens(test_df, aug_dict={})):
    if i >= test_steps:
        break
    preds = model.predict(img)
    validation_pred.extend(preds)
    validation_true.extend(mask)
    patient_slice_index.extend([(i, j) for j in range(mask.shape[0])])

# 调试输出，检查收集的数据
print(f'Collected {len(validation_pred)} predictions and {len(validation_true)} true masks.')

# 调试输出，检查每个体积的第一个预测和真实掩码
for i in range(5):  # 只检查前5个
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(validation_pred[i]), cmap='gray')
    plt.title(f'Predicted Volume {i}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(validation_true[i]), cmap='gray')
    plt.title(f'True Volume {i}')
    plt.axis('off')
    plt.show()

# 计算每个体积的 DSC
dsc_scores = dsc_per_volume(validation_pred, validation_true, patient_slice_index)

print(f'DSC per volume: {dsc_scores}')
print(f'Average DSC: {np.mean(dsc_scores):.4f}')

# 绘制 DSC 每体积的条形图
volume_labels = test_df['images_paths'].apply(lambda x: os.path.basename(os.path.dirname(x))).unique()
dsc_scores = dsc_scores[:len(volume_labels)]

# 计算平均 DSC
average_dsc = np.mean(dsc_scores)

# 绘制条形图
plt.figure(figsize=(12, 8))
bars = plt.barh(volume_labels, dsc_scores, color='skyblue')

# 添加平均值线
plt.axvline(x=average_dsc, color='red', linewidth=2, label=f'Average DSC = {average_dsc:.2f}')
plt.legend()

# 设置标签和标题
plt.xlabel('Dice Coefficient')
plt.ylabel('Volumes')
plt.title('Dice Coefficient per Volume')

# 添加每个条形的值标签
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
             ha='center', va='center')

# 显示图像
plt.tight_layout()
plt.show()

# 预测并可视化结果
for _ in range(10):
    index = np.random.randint(1, len(test_df.index))
    img = cv2.imread(test_df['images_paths'].iloc[index])
    img = cv2.resize(img, (256, 256))
    img = img / 255
    img = img[np.newaxis, :, :, :]

    predicted_img = model.predict(img)

    plt.figure(figsize=(12, 12))

    plt.subplot(1, 3, 1)
    plt.imshow(np.squeeze(img))
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(cv2.imread(test_df['masks_paths'].iloc[index])))
    plt.axis('off')
    plt.title('Original Mask')

    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(predicted_img) > 0.5)
    plt.title('Prediction')
    plt.axis('off')

    plt.show()

# 输出预测的准确率
accuracy = test_score[1]
print(f'Prediction Accuracy: {accuracy:.2%}')
