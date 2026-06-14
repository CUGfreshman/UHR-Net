import argparse
import os
import sys


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a UHR-Net checkpoint and write a test log.")
    parser.add_argument("--data-root", default="data", help="Root directory containing dataset folders.")
    parser.add_argument("--dataset", default="Kvasir-SEG_maoBian_fixed", help="Dataset folder name under --data-root.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained UHR-Net checkpoint.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--val-name", default=None, help="Validation split suffix, e.g. fold1 uses val_fold1.txt.")
    parser.add_argument("--log-path", default=None, help="Path to the test log. Defaults next to checkpoint.")
    parser.add_argument("--image-size", type=int, default=256, help="Square inference size.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary masks.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device.")
    parser.add_argument("--no-ughr", action="store_true", help="Instantiate UHR-Net without UGHR blocks.")
    parser.add_argument("--num-prototypes", type=int, default=8, help="Number of foreground/background UGHR prototypes.")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of UGHR attention heads.")
    parser.add_argument("--logit-scale", type=float, default=1.0, help="UGHR attention logit scale.")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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


def load_eval_split(dataset_path, split, val_name):
    from data.io import load_data, load_names

    if split == "train":
        (images, masks), _ = load_data(dataset_path, val_name)
        return images, masks
    if split == "val":
        _, (images, masks) = load_data(dataset_path, val_name)
        return images, masks

    names_path = os.path.join(dataset_path, "test.txt")
    return load_names(dataset_path, names_path)


def preprocess_image_for_model(cv2, np, torch, img_bgr, infer_size):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, infer_size, interpolation=cv2.INTER_LINEAR)
    arr = np.transpose(img_rgb, (2, 0, 1)) / 255.0
    arr = arr.astype(np.float32)[None, ...]
    return torch.from_numpy(arr)


def compute_iou(np, pred_bin_01, gt_bin_01):
    pred_bool = pred_bin_01.astype(bool)
    gt_bool = gt_bin_01.astype(bool)
    inter = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    if union == 0:
        return 1.0
    return inter / union


def _needs_sigmoid(torch, tensor):
    tensor_min = float(torch.amin(tensor).item())
    tensor_max = float(torch.amax(tensor).item())
    return (tensor_min < 0.0) or (tensor_max > 1.0)


def to_prob_2d(torch, mask_tensor):
    if mask_tensor.dim() == 3:
        prob = mask_tensor[0]
        if _needs_sigmoid(torch, prob):
            prob = torch.sigmoid(prob)
        return prob.clamp(0, 1)

    if mask_tensor.dim() != 4:
        raise ValueError(f"Unsupported mask tensor shape: {mask_tensor.shape}")

    batch_size, channels, _, _ = mask_tensor.shape
    assert batch_size == 1, "Batch size must be 1 for evaluation."

    if channels == 1:
        channel = mask_tensor[0, 0]
        if _needs_sigmoid(torch, channel):
            channel = torch.sigmoid(channel)
        return channel.clamp(0, 1)

    if channels == 2:
        softmax = torch.softmax(mask_tensor[0], dim=0)
        return softmax[1].clamp(0, 1)

    softmax = torch.softmax(mask_tensor[0], dim=0)
    foreground_prob = 1.0 - softmax[0]
    return foreground_prob.clamp(0, 1)


def main():
    args = parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm

    from models.uhr_net import UHRNet
    from utils.utils import seeding

    seeding(42)
    device = resolve_device(torch, args.device)
    print(f"[Info] Using device: {device}")

    model = UHRNet(
        use_ughr=not args.no_ughr,
        num_prototypes=args.num_prototypes,
        num_heads=args.num_heads,
        logit_scale=args.logit_scale,
    ).to(device)
    state = load_checkpoint_state(torch, args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dataset_path = os.path.join(args.data_root, args.dataset)
    eval_x, eval_y = load_eval_split(dataset_path, args.split, args.val_name)

    if args.log_path:
        log_path = args.log_path
    else:
        checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        log_path = os.path.join(checkpoint_dir, f"test_{args.split}.log")
    log_dir = os.path.dirname(os.path.abspath(log_path))
    if log_dir:
        ensure_dir(log_dir)

    infer_size = (args.image_size, args.image_size)
    iou_list = []
    skipped_count = 0
    log_lines = [
        "UHR-Net evaluation log",
        f"checkpoint: {os.path.abspath(args.checkpoint)}",
        f"dataset: {os.path.abspath(dataset_path)}",
        f"split: {args.split}",
        f"val_name: {args.val_name or ''}",
        f"image_size: {args.image_size}",
        f"threshold: {args.threshold}",
        f"device: {device}",
        "",
        "per_sample_iou:",
    ]

    print("[Info] Start inference...")
    for x_path, y_path in tqdm(list(zip(eval_x, eval_y)), desc="Processing"):
        image_bgr = cv2.imread(x_path, cv2.IMREAD_COLOR)
        gt_gray = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or gt_gray is None:
            skipped_count += 1
            warn_msg = f"[WARN] skip (read fail): img={x_path} gt={y_path}"
            print(warn_msg)
            log_lines.append(warn_msg)
            continue

        inp = preprocess_image_for_model(cv2, np, torch, image_bgr, infer_size).to(device)
        with torch.no_grad():
            out = model(inp)
        pred_logits = out[0] if isinstance(out, (tuple, list)) else out

        prob_2d = to_prob_2d(torch, pred_logits)

        pred_bin01_infer = (prob_2d.cpu().numpy() > args.threshold).astype(np.uint8)
        gt_resized = cv2.resize(gt_gray, infer_size, interpolation=cv2.INTER_NEAREST)
        gt_bin01_infer = (gt_resized > 127).astype(np.uint8)
        iou = compute_iou(np, pred_bin01_infer, gt_bin01_infer)
        iou_list.append(iou)

        stem = os.path.splitext(os.path.basename(y_path))[0]
        log_lines.append(f"{stem}\t{iou:.6f}")

    avg_iou = float(np.mean(iou_list)) if len(iou_list) > 0 else 0.0
    log_lines.extend([
        "",
        "summary:",
        f"samples: {len(iou_list)}",
        f"skipped: {skipped_count}",
        f"average_iou: {avg_iou:.6f}",
    ])
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print("\n[Done]")
    print(f"Test log saved to: {log_path}")
    print(f"Samples: {len(iou_list)}")
    print(f"Skipped: {skipped_count}")
    print(f"Average IoU: {avg_iou:.4f}")


if __name__ == "__main__":
    main()
