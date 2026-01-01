import os


def _resolve_with_ext(base_path, extensions):
    for ext in extensions:
        candidate = base_path + ext
        if os.path.exists(candidate):
            return candidate
    return base_path + extensions[0]


def load_names(path, file_path):
    with open(file_path, "r") as f:
        data = f.read().split("\n")[:-1]

    images = [
        _resolve_with_ext(os.path.join(path, "images", name), [".jpg", ".png", ".jpeg", ".tif", ".tiff"])
        for name in data
    ]
    masks = [
        _resolve_with_ext(os.path.join(path, "masks", name), [".png", ".jpg", ".jpeg", ".tif", ".tiff"])
        for name in data
    ]
    return images, masks


def load_data(path, val_name=None):
    train_names_path = f"{path}/train.txt"
    valid_names_path = f"{path}/val.txt" if val_name is None else f"{path}/val_{val_name}.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)

    return (train_x, train_y), (valid_x, valid_y)
