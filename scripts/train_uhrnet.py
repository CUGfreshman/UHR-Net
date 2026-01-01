import os
import sys
import random
import cv2
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
from models.uhr_net import UHRNet
from utils.metrics import DiceBCELoss
from data.io import load_data
from data.segmentation_dataset import SegmentationDataset
from engine.uhrnet_engine import train, evaluate


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
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

    image_size = 256
    batch_size = 24
    num_epochs = 300
    lr = 1e-4
    lr_backbone = 1e-4
    early_stopping_patience = 100
    aux_loss_weight = 0.1

    pretrained_backbone = "/root/autodl-tmp/this/medical_SO_seg/projects/ablation4/run_files/Kvasir-SEG_maoBian_fixed/stage1_Kvasir-SEG_maoBian_fixed_None_lr0.0001_20251124-221820/checkpoint.pth"
    resume_path = None

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"{dataset_name}_{val_name}_lr{lr}_{current_time}"

    base_dir = "/root/autodl-tmp/this/medical_SO_seg/projects/data"
    data_path = os.path.join(base_dir, dataset_name)
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
        A.Rotate(limit=35, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
    ])

    image_only_transform = A.Compose([
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32),
    ])

    (train_x, train_y), (valid_x, valid_y) = load_data(data_path, val_name)
    train_x, train_y = shuffling(train_x, train_y)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    train_dataset = SegmentationDataset(
        train_x,
        train_y,
        (image_size, image_size),
        geometric_transform=geometric_transform,
        image_only_transform=image_only_transform,
    )
    valid_dataset = SegmentationDataset(
        valid_x,
        valid_y,
        (image_size, image_size),
        geometric_transform=None,
        image_only_transform=None,
    )

    g = torch.Generator()
    g.manual_seed(seed)

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
    model = UHRNet(use_ughr=True)

    if pretrained_backbone:
        saved_state = torch.load(pretrained_backbone, map_location="cpu")
        model_state = model.state_dict()
        backbone_prefixes = ("layer0", "layer1", "layer2", "layer3")
        filtered_state = {k: v for k, v in saved_state.items() if k in model_state and k.startswith(backbone_prefixes)}
        model_state.update(filtered_state)
        model.load_state_dict(model_state, strict=False)

    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint)

    model = model.to(device)

    param_groups = [
        {"params": [], "lr": lr_backbone},
        {"params": [], "lr": lr},
    ]

    for name, param in model.named_parameters():
        if name.startswith("layer0") or name.startswith("layer1") or name.startswith("layer2") or name.startswith("layer3"):
            param_groups[0]["params"].append(param)
        else:
            param_groups[1]["params"].append(param)

    assert len(param_groups[0]["params"]) > 0, "Layer group is empty!"
    assert len(param_groups[1]["params"]) > 0, "Rest group is empty!"

    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=30, threshold=1e-4, min_lr=1e-6, verbose=True
    )
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    data_str = f"Number of parameters: {num_params / 1000000}M\n"
    print_and_save(train_log_path, data_str)

    with open(os.path.join(save_dir, "train_log.csv"), "w") as f:
        f.write(
            "epoch,train_loss,train_mIoU,train_f1,train_recall,train_precision,valid_loss,valid_mIoU,valid_f1,valid_recall,valid_precision\n"
        )

    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            aux_loss_weight=aux_loss_weight,
        )
        valid_loss, valid_metrics = evaluate(
            model,
            valid_loader,
            loss_fn,
            device,
            aux_loss_weight=aux_loss_weight,
        )
        current_miou = float(valid_metrics[0])
        scheduler.step(current_miou)

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
            f"\tTrain Loss: {train_loss:.4f} - mIoU: {train_metrics[0]:.4f} - "
            f"F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        )
        data_str += (
            f"\t Val. Loss: {valid_loss:.4f} - mIoU: {valid_metrics[0]:.4f} - "
            f"F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        )
        print_and_save(train_log_path, data_str)

        with open(os.path.join(save_dir, "train_log.csv"), "a") as f:
            f.write(
                f"{epoch + 1},{train_loss},{train_metrics[0]},{train_metrics[1]},{train_metrics[2]},{train_metrics[3]},{valid_loss},{valid_metrics[0]},{valid_metrics[1]},{valid_metrics[2]},{valid_metrics[3]}\n"
            )

        if early_stopping_count == early_stopping_patience:
            data_str = (
                "Early stopping: validation loss stops improving from last "
                f"{early_stopping_patience} continously.\n"
            )
            print_and_save(train_log_path, data_str)
            break
