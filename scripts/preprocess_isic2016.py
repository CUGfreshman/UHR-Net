import os
import cv2
import numpy as np
import pickle
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

# UO-IC metadata builder: instance masks and distance map for geometry-aware copy-paste.
BASE_DIR = "/root/autodl-tmp/this/medical_SO_seg/projects/data"
DATASET_NAME = "ISIC-2016"
MIN_AREA_THRESHOLD = 15


def find_dataset_files(dataset_path):
    all_filenames = set()
    for split in ["train", "val", "test"]:
        split_path = os.path.join(dataset_path, f"{split}.txt")
        if not os.path.exists(split_path):
            print(f"Warning: missing {split_path}, skipping.")
            continue
        with open(split_path, "r") as f:
            filenames = {line.strip() for line in f if line.strip()}
            all_filenames.update(filenames)
    return sorted(list(all_filenames))


def process_dataset(dataset_path, output_path):
    def resolve_path(directory, base_name, extensions):
        for ext in extensions:
            candidate = os.path.join(directory, base_name + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    metadata = {}
    base_names = find_dataset_files(dataset_path)
    if not base_names:
        print("Error: no valid samples found.")
        return

    image_dir = os.path.join(dataset_path, "images")
    mask_dir = os.path.join(dataset_path, "masks")

    for base_name in tqdm(base_names, desc="Processing"):
        image_path = resolve_path(image_dir, base_name, [".png", ".jpg", ".jpeg", ".tif", ".tiff"])
        mask_path = resolve_path(mask_dir, base_name, [".png", ".jpg", ".jpeg", ".tif", ".tiff"])

        if image_path is None or mask_path is None:
            print(f"Warning: missing pair for {base_name}, skipping.")
            continue

        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            print(f"Warning: failed to read {mask_path}, skipping.")
            continue

        _, binary_mask = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY)

        initial_labels = label(binary_mask)
        clean_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        for prop in regionprops(initial_labels):
            if prop.area >= MIN_AREA_THRESHOLD:
                clean_mask[initial_labels == prop.label] = 255

        distance_map = distance_transform_edt((clean_mask == 0).astype(np.uint8))

        final_labels = label(clean_mask)
        lesion_instances = []
        for prop in regionprops(final_labels):
            y1, x1, y2, x2 = prop.bbox
            instance_mask = (final_labels == prop.label).astype(np.uint8)
            lesion_instances.append({
                "instance_mask": instance_mask,
                "bbox": (y1, x1, y2, x2),
                "area": prop.area,
            })

        metadata[base_name] = {"dist_map": distance_map, "instances": lesion_instances}

    with open(output_path, "wb") as f:
        pickle.dump(metadata, f)


if __name__ == "__main__":
    dataset_full_path = os.path.join(BASE_DIR, DATASET_NAME)
    output_meta_file = os.path.join(dataset_full_path, "preprocessed_metadata.pkl")

    if not os.path.isdir(dataset_full_path):
        print(f"Error: dataset path not found: {dataset_full_path}")
    else:
        process_dataset(dataset_full_path, output_meta_file)
