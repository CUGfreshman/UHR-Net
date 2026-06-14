import argparse
import os
import pickle

# UO-IC metadata builder: instance masks and distance map for geometry-aware copy-paste.
DEFAULT_DATA_ROOT = "data"
DEFAULT_DATASET_NAME = "Kvasir-SEG_maoBian_fixed"
MIN_AREA_THRESHOLD = 15


def parse_args():
    parser = argparse.ArgumentParser(description="Build UO-IC metadata for Kvasir-SEG.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Root directory containing dataset folders.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME, help="Dataset folder name under --data-root.")
    parser.add_argument("--output", default=None, help="Output metadata path. Defaults to dataset/preprocessed_metadata.pkl.")
    parser.add_argument("--min-area", type=int, default=MIN_AREA_THRESHOLD, help="Minimum connected-component area to keep.")
    return parser.parse_args()


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


def process_dataset(dataset_path, output_path, min_area_threshold=MIN_AREA_THRESHOLD):
    import cv2
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    from skimage.measure import label, regionprops
    from tqdm import tqdm

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
            if prop.area >= min_area_threshold:
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
    args = parse_args()
    dataset_full_path = os.path.join(args.data_root, args.dataset)
    output_meta_file = args.output or os.path.join(dataset_full_path, "preprocessed_metadata.pkl")

    if not os.path.isdir(dataset_full_path):
        print(f"Error: dataset path not found: {dataset_full_path}")
    else:
        process_dataset(dataset_full_path, output_meta_file, args.min_area)
