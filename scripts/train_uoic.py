import os
import sys
import random
import time
import datetime
import numpy as np
import albumentations as A
import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.utils import print_and_save, shuffling, epoch_time
from utils.metrics import DiceBCELoss
from data.io import load_data
from data.uoic_dataset import UOICDataset
from engine.uoic_engine import train, evaluate
from models.uoic_pretrain import UOICPretrainNet


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def my_seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    dataset_name = "Kvasir-SEG_maoBian_fixed"
    val_name = None

    seed = 7
    my_seeding(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    image_size = 256
    batch_size = 16
    num_epochs = 300
    lr = 1e-4
    early_stopping_patience = 100

    resume_path = None

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"stage1_{dataset_name}_{val_name}_lr{lr}_{current_time}"

    base_dir = "/root/autodl-tmp/this/medical_SO_seg/projects/data"
    data_path = os.path.join(base_dir, dataset_name)
    preprocessed_path = os.path.join(data_path, "preprocessed_metadata.pkl")

    save_dir = os.path.join("run_files", dataset_name, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    train_log_path = os.path.join(save_dir, "train_log.txt")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    with open(train_log_path, "w") as train_log:
        train_log.write("\n")

    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    hyperparameters_str = (
        f"Image Size: {image_size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
        f"Early Stopping Patience: {early_stopping_patience}\nSeed: {seed}\n"
    )
    print_and_save(train_log_path, hyperparameters_str)

    geometric_transform = A.Compose([
        A.Rotate(limit=90, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])

    image_only_transform = A.Compose([
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32),
    ])

    (train_x, train_y), (valid_x, valid_y) = load_data(data_path, val_name)
    train_x, train_y = shuffling(train_x, train_y)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    train_dataset = UOICDataset(
        images_path=train_x,
        masks_path=train_y,
        size=(image_size, image_size),
        preprocessed_path=preprocessed_path,
        geometric_transform=geometric_transform,
        image_only_transform=image_only_transform,
        is_train=True,
    )
    valid_dataset = UOICDataset(
        images_path=valid_x,
        masks_path=valid_y,
        size=(image_size, image_size),
        preprocessed_path=None,
        geometric_transform=None,
        image_only_transform=None,
        is_train=False,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        worker_init_fn=seed_worker,
        generator=g,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=seed_worker,
        generator=g,
    )

    device = torch.device("cuda")
    model = UOICPretrainNet()

    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=30, verbose=True)
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    data_str = f"Number of parameters: {num_params / 1000000}M\n"
    print_and_save(train_log_path, data_str)

    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss_dict, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[0] > best_valid_metrics:
            data_str = (
                f"Valid mIoU improved from {best_valid_metrics:2.4f} to {valid_metrics[0]:2.4f}. "
                f"Saving checkpoint: {checkpoint_path}"
            )
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[0]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0
        elif valid_metrics[0] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += (
            f"\tTrain Loss: {train_loss_dict.get('total', 0):.4f} - "
            f"mIoU: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - "
            f"Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        )
        data_str += (
            f"\tLoss Breakdown -> Mask: {train_loss_dict.get('mask', 0):.4f}, "
            f"Instance: {train_loss_dict.get('instance', 0):.4f}\n"
        )
        data_str += (
            f"\t Val. Loss: {valid_loss:.4f} - "
            f"mIoU: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - "
            f"Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        )
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = (
                "Early stopping: validation loss stops improving from last "
                f"{early_stopping_patience} continously.\n"
            )
            print_and_save(train_log_path, data_str)
            break
