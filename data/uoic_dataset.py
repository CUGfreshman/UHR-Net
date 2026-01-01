import os
import cv2
import numpy as np
import pickle
import random
import torch
from torch.utils.data import Dataset


def get_paste_position(dist_map, required_radius):
    candidates = np.argwhere(dist_map >= required_radius)
    if candidates.shape[0] == 0:
        return None
    chosen_index = random.randint(0, candidates.shape[0] - 1)
    return candidates[chosen_index]


def get_scaled_lesion_info(instance_info, dist_map):
    original_area = instance_info["area"]
    y1, x1, y2, x2 = instance_info["bbox"]
    original_h = y2 - y1
    original_w = x2 - x1

    abs_min_scale = np.sqrt(50 / (original_area + 1e-6))
    preferred_lower = max(0.25, abs_min_scale)
    preferred_upper = 0.75

    if preferred_lower < preferred_upper:
        current_upper = preferred_upper
        for _ in range(5):
            if preferred_lower >= current_upper:
                break
            scale = random.uniform(preferred_lower, current_upper)
            new_h, new_w = int(original_h * scale), int(original_w * scale)
            if new_h < 1 or new_w < 1:
                current_upper = scale
                continue
            required_radius = np.sqrt((new_h / 2) ** 2 + (new_w / 2) ** 2)
            paste_pos = get_paste_position(dist_map, required_radius)
            if paste_pos is not None:
                return {"scale": scale, "new_h": new_h, "new_w": new_w, "paste_pos": paste_pos}
            current_upper = scale

    fallback_upper = preferred_lower
    if abs_min_scale < fallback_upper:
        current_upper = fallback_upper
        for _ in range(5):
            if abs_min_scale >= current_upper:
                break
            scale = random.uniform(abs_min_scale, current_upper)
            new_h, new_w = int(original_h * scale), int(original_w * scale)
            if new_h < 1 or new_w < 1:
                current_upper = scale
                continue
            required_radius = np.sqrt((new_h / 2) ** 2 + (new_w / 2) ** 2)
            paste_pos = get_paste_position(dist_map, required_radius)
            if paste_pos is not None:
                return {"scale": scale, "new_h": new_h, "new_w": new_w, "paste_pos": paste_pos}
            current_upper = scale

    return None


class UOICDataset(Dataset):
    def __init__(self, images_path, masks_path, size, preprocessed_path, geometric_transform=None, image_only_transform=None, is_train=True):
        super().__init__()
        self.images_path = images_path
        self.masks_path = masks_path
        self.geometric_transform = geometric_transform
        self.image_only_transform = image_only_transform
        self.is_train = is_train
        self.n_samples = len(images_path)
        self.size = size
        self.uoic_metadata = None
        if preprocessed_path and os.path.exists(preprocessed_path):
            with open(preprocessed_path, "rb") as f:
                self.uoic_metadata = pickle.load(f)
        else:
            msg = preprocessed_path if preprocessed_path is not None else "None"
            print(f"Warning: metadata not found ({msg}); UO-IC copy-paste disabled.")

    def __getitem__(self, index):
        image_path = self.images_path[index]
        mask_path = self.masks_path[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        lesion_a_mask = np.zeros_like(mask, dtype=np.float32)
        lesion_b_mask = np.zeros_like(mask, dtype=np.float32)

        # UO-IC: geometry-aware copy-paste to form lesion A/B masks.
        if self.is_train and self.uoic_metadata:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            info = self.uoic_metadata.get(base_name)

            if info and info["instances"]:
                selected_instance = random.choice(info["instances"])
                cp_info = get_scaled_lesion_info(selected_instance, info["dist_map"])

                if cp_info:
                    lesion_a_mask = selected_instance["instance_mask"].astype(np.float32) * 255.0

                    y1, x1, y2, x2 = selected_instance["bbox"]
                    instance_img = image[y1:y2, x1:x2]
                    instance_mask = lesion_a_mask[y1:y2, x1:x2]

                    scaled_img = cv2.resize(instance_img, (cp_info["new_w"], cp_info["new_h"]), interpolation=cv2.INTER_AREA)
                    scaled_mask = cv2.resize(instance_mask, (cp_info["new_w"], cp_info["new_h"]), interpolation=cv2.INTER_NEAREST)

                    center_y, center_x = cp_info["paste_pos"]
                    h, w = cp_info["new_h"], cp_info["new_w"]
                    y1, x1 = center_y - h // 2, center_x - w // 2
                    y2, x2 = y1 + h, x1 + w

                    image_copy = image.copy()
                    mask_copy = mask.copy()

                    img_h, img_w, _ = image_copy.shape
                    paste_y1 = max(0, y1)
                    paste_x1 = max(0, x1)
                    paste_y2 = min(img_h, y2)
                    paste_x2 = min(img_w, x2)

                    offset_y1 = paste_y1 - y1
                    offset_x1 = paste_x1 - x1
                    offset_y2 = offset_y1 + (paste_y2 - paste_y1)
                    offset_x2 = offset_x1 + (paste_x2 - paste_x1)

                    if paste_y1 < paste_y2 and paste_x1 < paste_x2:
                        scaled_img_part = scaled_img[offset_y1:offset_y2, offset_x1:offset_x2]
                        scaled_mask_part = scaled_mask[offset_y1:offset_y2, offset_x1:offset_x2]

                        roi_img = image_copy[paste_y1:paste_y2, paste_x1:paste_x2]
                        bool_mask = scaled_mask_part > 0

                        roi_img[bool_mask] = scaled_img_part[bool_mask]
                        image_copy[paste_y1:paste_y2, paste_x1:paste_x2] = roi_img
                        mask_copy[paste_y1:paste_y2, paste_x1:paste_x2][bool_mask] = scaled_mask_part[bool_mask]
                        lesion_b_mask[paste_y1:paste_y2, paste_x1:paste_x2][bool_mask] = scaled_mask_part[bool_mask]

                        image = image_copy
                        mask = mask_copy

        if self.geometric_transform is not None:
            augmentations = self.geometric_transform(image=image, masks=[mask, lesion_a_mask, lesion_b_mask])
            image = augmentations["image"]
            mask, lesion_a_mask, lesion_b_mask = augmentations["masks"]

        if self.image_only_transform is not None:
            image = self.image_only_transform(image=image)["image"]

        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        lesion_a_mask = cv2.resize(lesion_a_mask, self.size, interpolation=cv2.INTER_NEAREST)
        lesion_b_mask = cv2.resize(lesion_b_mask, self.size, interpolation=cv2.INTER_NEAREST)

        image = (image.astype(np.float32) / 255.0)
        mask = (mask.astype(np.float32) / 255.0)
        lesion_a_mask = (lesion_a_mask.astype(np.float32) / 255.0)
        lesion_b_mask = (lesion_b_mask.astype(np.float32) / 255.0)

        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)
        lesion_a_mask = np.expand_dims(lesion_a_mask, axis=0)
        lesion_b_mask = np.expand_dims(lesion_b_mask, axis=0)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        lesion_a_mask = torch.from_numpy(lesion_a_mask).float()
        lesion_b_mask = torch.from_numpy(lesion_b_mask).float()

        return image, mask, lesion_a_mask, lesion_b_mask

    def __len__(self):
        return self.n_samples
