import argparse
import datetime
import os
import random
import sys
import time


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="Train UHR-Net for binary medical image segmentation.")
    parser.add_argument("--data-root", default="data", help="Root directory containing dataset folders.")
    parser.add_argument("--dataset", default="Kvasir-SEG_maoBian_fixed", help="Dataset folder name under --data-root.")
    parser.add_argument("--val-name", default=None, help="Validation split suffix, e.g. fold1 uses val_fold1.txt.")
    parser.add_argument("--output-root", default="run_files", help="Directory used for training logs and checkpoints.")
    parser.add_argument("--run-name", default=None, help="Optional run folder name. Defaults to <dataset>_<timestamp>.")
    parser.add_argument("--image-size", type=int, default=256, help="Square input size used for training and validation.")
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=300, help="Maximum number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for non-backbone parameters.")
    parser.add_argument("--lr-backbone", type=float, default=1e-4, help="Learning rate for ResNet backbone layers.")
    parser.add_argument("--early-stopping-patience", type=int, default=100, help="Stop after this many epochs without mIoU improvement.")
    parser.add_argument("--aux-loss-weight", type=float, default=0.1, help="Weight for the auxiliary coarse-mask loss.")
    parser.add_argument("--pretrained-backbone", default=None, help="Optional stage-1 checkpoint used to initialize backbone layers.")
    parser.add_argument("--resume", default=None, help="Optional UHR-Net checkpoint to resume from.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Training device.")
    parser.add_argument("--no-ughr", action="store_true", help="Disable UGHR blocks for ablation runs.")
    parser.add_argument("--num-prototypes", type=int, default=8, help="Number of foreground/background UGHR prototypes.")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of UGHR attention heads.")
    parser.add_argument("--logit-scale", type=float, default=1.0, help="UGHR attention logit scale.")
    return parser.parse_args()


def build_run_name(dataset_name, timestamp):
    return f"{dataset_name}_{timestamp}"


def my_seeding(seed, np, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    import numpy as np

    worker_seed = torch_initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_initial_seed():
    import torch

    return torch.initial_seed()


def resolve_device(torch, requested_device):
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def load_checkpoint_state(torch, path, map_location):
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def main():
    args = parse_args()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    import cv2
    import numpy as np
    import albumentations as A
    import torch
    from torch.utils.data import DataLoader

    from data.io import load_data
    from data.segmentation_dataset import SegmentationDataset
    from engine.uhrnet_engine import evaluate, train
    from models.uhr_net import UHRNet
    from utils.metrics import DiceBCELoss
    from utils.utils import epoch_time, print_and_save, shuffling

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    my_seeding(args.seed, np, torch)

    image_size = args.image_size
    dataset_name = args.dataset
    data_path = os.path.join(args.data_root, dataset_name)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = args.run_name or build_run_name(dataset_name, current_time)
    save_dir = os.path.join(args.output_root, dataset_name, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    train_log_path = os.path.join(save_dir, "train_log.txt")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    with open(train_log_path, "w") as train_log:
        train_log.write("\n")

    print_and_save(train_log_path, str(datetime.datetime.now()))
    print("")

    hyperparameters_str = (
        f"Image Size: {image_size}\nBatch Size: {args.batch_size}\nLR: {args.lr}\nEpochs: {args.epochs}\n"
        f"Backbone LR: {args.lr_backbone}\nEarly Stopping Patience: {args.early_stopping_patience}\n"
        f"Aux Loss Weight: {args.aux_loss_weight}\nSeed: {args.seed}\n"
        f"Data Path: {data_path}\n"
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

    (train_x, train_y), (valid_x, valid_y) = load_data(data_path, args.val_name)
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

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    device = resolve_device(torch, args.device)
    model = UHRNet(
        use_ughr=not args.no_ughr,
        num_prototypes=args.num_prototypes,
        num_heads=args.num_heads,
        logit_scale=args.logit_scale,
    )

    if args.pretrained_backbone:
        saved_state = load_checkpoint_state(torch, args.pretrained_backbone, map_location="cpu")
        model_state = model.state_dict()
        backbone_prefixes = ("layer0", "layer1", "layer2", "layer3")
        filtered_state = {
            key: value
            for key, value in saved_state.items()
            if key in model_state and key.startswith(backbone_prefixes)
        }
        model_state.update(filtered_state)
        model.load_state_dict(model_state, strict=False)
        print_and_save(train_log_path, f"Loaded backbone parameters from: {args.pretrained_backbone}")

    if args.resume:
        checkpoint = load_checkpoint_state(torch, args.resume, map_location="cpu")
        model.load_state_dict(checkpoint)
        print_and_save(train_log_path, f"Resumed model from: {args.resume}")

    model = model.to(device)

    param_groups = [
        {"params": [], "lr": args.lr_backbone},
        {"params": [], "lr": args.lr},
    ]

    for name, param in model.named_parameters():
        if name.startswith(("layer0", "layer1", "layer2", "layer3")):
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
    print_and_save(train_log_path, "Optimizer: Adam\nLoss: BCE Dice Loss\n")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_and_save(train_log_path, f"Number of parameters: {num_params / 1000000}M\n")

    with open(os.path.join(save_dir, "train_log.csv"), "w") as f:
        f.write(
            "epoch,train_loss,train_mIoU,train_f1,train_recall,train_precision,"
            "valid_loss,valid_mIoU,valid_f1,valid_recall,valid_precision\n"
        )

    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss, train_metrics = train(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            aux_loss_weight=args.aux_loss_weight,
        )
        valid_loss, valid_metrics = evaluate(
            model,
            valid_loader,
            loss_fn,
            device,
            aux_loss_weight=args.aux_loss_weight,
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
                f"{epoch + 1},{train_loss},{train_metrics[0]},{train_metrics[1]},"
                f"{train_metrics[2]},{train_metrics[3]},{valid_loss},{valid_metrics[0]},"
                f"{valid_metrics[1]},{valid_metrics[2]},{valid_metrics[3]}\n"
            )

        if early_stopping_count == args.early_stopping_patience:
            data_str = (
                "Early stopping: validation mIoU stops improving from last "
                f"{args.early_stopping_patience} epochs.\n"
            )
            print_and_save(train_log_path, data_str)
            break


if __name__ == "__main__":
    main()
