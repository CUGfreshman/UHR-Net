import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils import calculate_metrics


def masked_average_pooling(feature_map, mask):
    mask = (mask > 0).float()
    mask_sum = mask.sum() + 1e-6
    pooled = (feature_map * mask).sum(dim=[1, 2]) / mask_sum
    return pooled


def mine_hard_negative_feature(feature_map, pred_prob, gt_mask):
    background_mask = (gt_mask < 0.5).float()
    if background_mask.sum() == 0:
        return None

    weights = pred_prob * background_mask
    weight_sum = weights.sum() + 1e-8
    weighted_features = feature_map * weights
    return weighted_features.sum(dim=[1, 2]) / weight_sum


def train(model, loader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0.0
    epoch_loss_mask = 0.0
    epoch_loss_instance = 0.0
    epoch_jac = epoch_f1 = epoch_recall = epoch_precision = 0.0

    instance_loss_weight = 1.0
    temperature = 0.1

    for _, (x, y, lesion_a, lesion_b) in enumerate(tqdm(loader, desc="Training", total=len(loader))):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        lesion_a = lesion_a.to(device, dtype=torch.float32)
        lesion_b = lesion_b.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        mask_logits, uoic_feat = model(x)
        loss_mask = loss_fn(mask_logits, y)

        batch_pos_a = []
        batch_pos_b = []
        batch_neg = []

        for j in range(uoic_feat.shape[0]):
            if lesion_a[j].sum() > 0 and lesion_b[j].sum() > 0:
                batch_pos_a.append(masked_average_pooling(uoic_feat[j], lesion_a[j]))
                batch_pos_b.append(masked_average_pooling(uoic_feat[j], lesion_b[j]))

            pred_prob = torch.sigmoid(mask_logits[j])
            neg_feat = mine_hard_negative_feature(uoic_feat[j], pred_prob, y[j])
            if neg_feat is not None:
                batch_neg.append(neg_feat)

        loss_info_nce = torch.tensor(0.0).to(device)

        # UO-IC InfoNCE with lesion A/B positives and lesion-like background negatives.
        if batch_pos_a and batch_neg:
            pos_a = F.normalize(torch.stack(batch_pos_a), p=2, dim=1)
            pos_b = F.normalize(torch.stack(batch_pos_b), p=2, dim=1)
            neg = F.normalize(torch.stack(batch_neg), p=2, dim=1)

            l_pos = torch.einsum("nc,nc->n", [pos_a, pos_b]).unsqueeze(-1)
            l_neg = torch.einsum("nc,kc->nk", [pos_a, neg])
            logits = torch.cat([l_pos, l_neg], dim=1) / temperature
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
            loss_info_nce = F.cross_entropy(logits, labels)

        loss = loss_mask + (instance_loss_weight * loss_info_nce)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_loss_mask += loss_mask.item()
        epoch_loss_instance += loss_info_nce.item() if isinstance(loss_info_nce, torch.Tensor) else loss_info_nce

        batch_jac, batch_f1, batch_recall, batch_precision = [], [], [], []
        mask_pred_prob = torch.sigmoid(mask_logits)
        for yt, yp in zip(y, mask_pred_prob):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])
        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    num_batches = len(loader)
    epoch_loss /= num_batches
    epoch_loss_mask /= num_batches
    epoch_loss_instance /= num_batches
    epoch_jac /= num_batches
    epoch_f1 /= num_batches
    epoch_recall /= num_batches
    epoch_precision /= num_batches

    loss_dict = {
        "total": epoch_loss,
        "mask": epoch_loss_mask,
        "instance": epoch_loss_instance,
    }

    return loss_dict, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


def evaluate(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for _, (x, y, _, _) in enumerate(tqdm(loader, desc="Evaluation", total=len(loader))):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            mask_logits, _ = model(x)
            loss_mask = loss_fn(mask_logits, y)
            epoch_loss += loss_mask.item()

            batch_jac, batch_f1, batch_recall, batch_precision = [], [], [], []
            for yt, yp in zip(y, torch.sigmoid(mask_logits)):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])
            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss = epoch_loss / len(loader)
        epoch_jac = epoch_jac / len(loader)
        epoch_f1 = epoch_f1 / len(loader)
        epoch_recall = epoch_recall / len(loader)
        epoch_precision = epoch_precision / len(loader)
        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]
