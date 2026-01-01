import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import cv2
from tqdm import tqdm
import torch

from models.uhr_net import UHRNet
from utils.utils import seeding
from data.io import load_data


NO_GOOD_IOU_THRESH = 0.6
INFER_SIZE = (256, 256)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def preprocess_image_for_model(img_bgr, infer_size):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, infer_size, interpolation=cv2.INTER_LINEAR)
    arr = np.transpose(img_rgb, (2, 0, 1)) / 255.0
    arr = arr.astype(np.float32)[None, ...]
    return torch.from_numpy(arr)


def compute_iou(pred_bin_01: np.ndarray, gt_bin_01: np.ndarray) -> float:
    pred_bool = pred_bin_01.astype(bool)
    gt_bool = gt_bin_01.astype(bool)
    inter = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    if union == 0:
        return 1.0
    return inter / union


def draw_overlay(image_bgr, pred_bin_255, gt_bin_255):
    vis = image_bgr.copy()
    _, p = cv2.threshold(pred_bin_255, 127, 255, cv2.THRESH_BINARY)
    _, g = cv2.threshold(gt_bin_255, 127, 255, cv2.THRESH_BINARY)
    p_cnts, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    g_cnts, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, p_cnts, -1, (0, 0, 255), 2)
    cv2.drawContours(vis, g_cnts, -1, (0, 255, 0), 2)
    return vis


def _needs_sigmoid(t: torch.Tensor) -> bool:
    tmin = float(torch.amin(t).item())
    tmax = float(torch.amax(t).item())
    return (tmin < 0.0) or (tmax > 1.0)


def to_prob_2d(mask_tensor: torch.Tensor) -> torch.Tensor:
    if mask_tensor.dim() == 3:
        prob = mask_tensor[0]
        if _needs_sigmoid(prob):
            prob = torch.sigmoid(prob)
        return prob.clamp(0, 1)

    if mask_tensor.dim() != 4:
        raise ValueError(f"Unsupported mask tensor shape: {mask_tensor.shape}")

    bsz, channels, _, _ = mask_tensor.shape
    assert bsz == 1, "Batch size must be 1 for visualization."

    if channels == 1:
        ch = mask_tensor[0, 0]
        if _needs_sigmoid(ch):
            ch = torch.sigmoid(ch)
        return ch.clamp(0, 1)

    if channels == 2:
        sm = torch.softmax(mask_tensor[0], dim=0)
        return sm[1].clamp(0, 1)

    sm = torch.softmax(mask_tensor[0], dim=0)
    fg_prob = 1.0 - sm[0]
    return fg_prob.clamp(0, 1)


if __name__ == "__main__":
    dataset_name = "Kvasir-SEG_maoBian_fixed"
    data_root = "/root/autodl-tmp/this/medical_SO_seg/projects/data"
    checkpoint_path = "/root/autodl-tmp/this/medical_SO_seg/projects/random_seed1/run_files/Kvasir-SEG_maoBian_fixed/Kvasir-SEG_maoBian_fixed_None_lr0.0001_20251105-085429/checkpoint.pth"

    seeding(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    model = UHRNet().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dataset_path = os.path.join(data_root, dataset_name)
    (_, _), (test_x, test_y) = load_data(dataset_path)

    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    out_root = os.path.join(ckpt_dir, "visualize_result")
    overlay_dir = os.path.join(out_root, "overlay")
    no_good_dir = os.path.join(out_root, "no_good")
    ensure_dir(overlay_dir)
    ensure_dir(no_good_dir)

    iou_list, no_good_count = [], 0

    print("[Info] Start inference & visualization...")
    for x_path, y_path in tqdm(list(zip(test_x, test_y)), desc="Processing"):
        image_bgr = cv2.imread(x_path, cv2.IMREAD_COLOR)
        gt_gray = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or gt_gray is None:
            print(f"[WARN] skip (read fail): img={x_path} gt={y_path}")
            continue

        height, width = image_bgr.shape[:2]

        inp = preprocess_image_for_model(image_bgr, INFER_SIZE).to(device)
        with torch.no_grad():
            out = model(inp)
        pred_logits = out[0] if isinstance(out, (tuple, list)) else out

        prob_2d = to_prob_2d(pred_logits)

        pred_bin01_infer = (prob_2d.cpu().numpy() > 0.5).astype(np.uint8)
        gt_resized = cv2.resize(gt_gray, INFER_SIZE, interpolation=cv2.INTER_NEAREST)
        gt_bin01_infer = (gt_resized > 127).astype(np.uint8)
        iou = compute_iou(pred_bin01_infer, gt_bin01_infer)
        iou_list.append(iou)

        pred_bin255_orig = cv2.resize(pred_bin01_infer * 255, (width, height), interpolation=cv2.INTER_NEAREST)
        gt_bin255_orig = cv2.resize(gt_bin01_infer * 255, (width, height), interpolation=cv2.INTER_NEAREST)
        overlay_img = draw_overlay(image_bgr, pred_bin255_orig, gt_bin255_orig)

        stem = os.path.splitext(os.path.basename(y_path))[0]
        img_ext = os.path.splitext(os.path.basename(x_path))[1]
        out_name = f"{stem}{img_ext}"

        cv2.imwrite(os.path.join(overlay_dir, out_name), overlay_img)

        if iou < NO_GOOD_IOU_THRESH:
            cv2.imwrite(os.path.join(no_good_dir, out_name), overlay_img)
            no_good_count += 1

    avg_iou = float(np.mean(iou_list)) if len(iou_list) > 0 else 0.0
    result_txt = os.path.join(out_root, "result.txt")
    with open(result_txt, "w") as f:
        f.write(f"Samples: {len(iou_list)}\n")
        f.write(f"Average IoU: {avg_iou:.4f}\n")
        f.write(f"No-good threshold: {NO_GOOD_IOU_THRESH}\n")
        f.write(f"No-good count: {no_good_count}\n")

    print("\n[Done]")
    print(f"Outputs saved to: {out_root}")
    print(" - overlay/: all visualizations")
    print(f" - no_good/: visualizations with IoU < {NO_GOOD_IOU_THRESH}")
    print(" - result.txt: summary")
