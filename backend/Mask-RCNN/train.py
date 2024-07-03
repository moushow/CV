import os

import albumentations as A
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# 禁用 albumentations 版本检查
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"


class BrainMRIDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.tif') and '_mask' not in f]
        self.mask_files = [f.replace('.tif', '_mask.tif') for f in self.image_files]

        # Ensure that all masks exist
        valid_image_files = []
        valid_mask_files = []
        for image_file, mask_file in zip(self.image_files, self.mask_files):
            if os.path.exists(os.path.join(data_dir, mask_file)):
                valid_image_files.append(image_file)
                valid_mask_files.append(mask_file)
            else:
                print(f"Mask file {mask_file} does not exist for image {image_file}")

        self.image_files = valid_image_files
        self.mask_files = valid_mask_files

        print(f"Found {len(self.image_files)} valid images and {len(self.mask_files)} valid masks")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        mask_path = os.path.join(self.data_dir, self.mask_files[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = torch.as_tensor(mask, dtype=torch.uint8)

        target = {}
        target["masks"] = mask.unsqueeze(0)
        target["boxes"] = self.get_bounding_box(mask)

        return image, target

    def get_bounding_box(self, mask):
        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)


def get_transforms(train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])


def collate_fn(batch):
    return tuple(zip(*batch))


def get_instance_segmentation_model(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def evaluate_model(model, data_loader, device):
    model.eval()
    metric_logger = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for target, output in zip(targets, outputs):
                boxes_true = target['boxes'].cpu()
                boxes_pred = output['boxes'].cpu()

                if boxes_pred.size(0) == 0:
                    continue

                iou_matrix = torch.ops.torchvision.box_iou(boxes_true, boxes_pred)
                max_iou, _ = torch.max(iou_matrix, dim=1)
                metric_logger.append(max_iou.mean().item())

    print(f"Average IOU: {sum(metric_logger) / len(metric_logger):.4f}")


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_classes = 2
    data_dir = "D:\\IDEAProject\\CV\\backend\\Brain MRI segmentation\\TCGA_CS_4941_19960909"

    dataset = BrainMRIDataset(data_dir=data_dir, transforms=get_transforms(train=True))
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch #{epoch} loss: {losses.item():.4f}")

    torch.save(model.state_dict(), "mask_rcnn_model.pth")

    # Evaluate the model
    evaluate_model(model, data_loader, device)


if __name__ == "__main__":
    main()
