import cv2
import numpy as np
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, images_path, masks_path, size, geometric_transform=None, image_only_transform=None):
        super().__init__()
        self.images_path = images_path
        self.masks_path = masks_path
        self.geometric_transform = geometric_transform
        self.image_only_transform = image_only_transform
        self.n_samples = len(images_path)
        self.size = size

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        if self.geometric_transform is not None:
            augmentations = self.geometric_transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        if self.image_only_transform is not None:
            image = self.image_only_transform(image=image)["image"]

        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        image = np.transpose(image, (2, 0, 1))
        image = (image.astype(np.float32) / 255.0)

        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=0)
        mask = (mask.astype(np.float32) / 255.0)

        return image, mask

    def __len__(self):
        return self.n_samples
