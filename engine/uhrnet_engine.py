import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils import calculate_metrics


def train(model, loader, optimizer, loss_fn, device, aux_loss_weight=0.1):
    model.train()

    epoch_loss = 0.0
    epoch_jac = epoch_f1 = epoch_recall = epoch_precision = 0.0

    for _, (x, y) in enumerate(tqdm(loader, desc="Training", total=len(loader))):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)

        refined_prob, coarse_prob = model(x)

        eps = 1e-6
        refined_logits = torch.logit(refined_prob.clamp(min=eps, max=1 - eps))
        coarse_up = F.interpolate(coarse_prob, size=y.shape[2:], mode="bilinear", align_corners=False)
        coarse_logits = torch.logit(coarse_up.clamp(min=eps, max=1 - eps))

        main_loss = loss_fn(refined_logits, y)
        aux_loss = loss_fn(coarse_logits, y)
        total_loss = main_loss + aux_loss_weight * aux_loss

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

        batch_jac, batch_f1, batch_recall, batch_precision = [], [], [], []
        for yt, yp in zip(y, refined_prob):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])
        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss /= len(loader)
    epoch_jac /= len(loader)
    epoch_f1 /= len(loader)
    epoch_recall /= len(loader)
    epoch_precision /= len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


def evaluate(model, loader, loss_fn, device, aux_loss_weight=0.1):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for _, (x, y) in enumerate(tqdm(loader, desc="Evaluation", total=len(loader))):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            refined_prob, coarse_prob = model(x)

            eps = 1e-6
            refined_logits = torch.logit(refined_prob.clamp(min=eps, max=1 - eps))
            coarse_up = F.interpolate(coarse_prob, size=y.shape[2:], mode="bilinear", align_corners=False)
            coarse_logits = torch.logit(coarse_up.clamp(min=eps, max=1 - eps))

            main_loss = loss_fn(refined_logits, y)
            aux_loss = loss_fn(coarse_logits, y)
            loss = main_loss + aux_loss_weight * aux_loss

            epoch_loss += loss.item()

            batch_jac, batch_f1, batch_recall, batch_precision = [], [], [], []
            for yt, yp in zip(y, refined_prob):
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
