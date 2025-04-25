import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset as dataset
class CustomDataset(dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        mask = [cv2.imread(os.path.join(self.mask_dir, str(i), img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None]
                for i in range(self.num_classes)]
        mask = np.dstack(mask)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.astype('float32') / 255.0
        mask = mask.astype('float32') / 255.0
        img = img.transpose(2, 0, 1)  # From HWC to CHW
        mask = mask.transpose(2, 0, 1)  # From HWC to CHW

        return img, mask, img_id

# Define transformations
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Example usage
dataset = CustomDataset(img_ids=['image1', 'image2'], img_dir='path/to/images', mask_dir='path/to/masks',
                        img_ext='.png', mask_ext='.png', num_classes=3, transform=train_transform)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
