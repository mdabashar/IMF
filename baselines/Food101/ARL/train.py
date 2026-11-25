import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F
import os
import warnings

warnings.filterwarnings("ignore")
import json
import numpy as np
import argparse
import random
from sklearn.metrics import f1_score, average_precision_score
from model import *
from sklearn.metrics import accuracy_score, f1_score

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
        
def run_training_loop(
    model,
    train_loader,
    val_loader,          # 👉 原 test_loader → 改名更清晰
    test_loader,         # 👉 原 test_real_loader
    train_fn,
    optimizer,
    n_epochs,
    device,
    scheduler=None,
    model_name="model"
):
    """
    通用训练循环（支持多模态 + MMPareto + 多验证集）

    train_fn:
        (epoch, loader, model, optimizer) -> model, scores

    validate_fn:
        (epoch, loader, model, device) -> acc, f1_macro, f1_weighted
    """

    best_acc = 0
    best_metrics = {
        "acc": 0,
        "f1_macro": 0,
        "f1_weighted": 0
    }

    for epoch in range(n_epochs):

        print(f"\n===== Epoch {epoch} =====")

        # =========================
        # scheduler（建议放step后面更标准，但保留你原逻辑）
        # =========================
        if scheduler is not None:
            scheduler.step()

        # =========================
        # train
        # =========================
        model, scores = train_fn(
            epoch,
            train_loader,
            model,
            optimizer,
            device=device
        )

        # =========================
        # validation (in-domain)
        # =========================
        acc, f1_macro, f1_weighted = validate(
            epoch,
            val_loader,
            model,
            device
        )

        print(f"[Val] Acc: {acc:.4f}, Macro-F1: {f1_macro:.4f}, Weighted-F1: {f1_weighted:.4f}")

        # =========================
        # save best（基于 val）
        # =========================
        if acc >= best_acc:
            best_acc = acc
            best_metrics.update({
                "acc": acc,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted
            })

            # 👉 如果你之后要开，可以直接用
            # best_path = f"{model_name}_best.pth"
            # torch.save(model.state_dict(), best_path)

        # =========================
        # imbalance analysis（你这个非常好）
        # =========================
        strong = max(scores, key=scores.get)
        weak = min(scores, key=scores.get)

        ratio = scores[strong] / (scores[weak] + 1e-8)

        print(f"[Imbalance] Scores: {scores}")
        print(f"[Imbalance] Strong: {strong}, Weak: {weak}, Ratio: {ratio:.3f}")

        # =========================
        # test (out-of-domain / real)
        # =========================
        acc_test, f1_macro_test, f1_weighted_test = validate(
            epoch,
            test_loader,
            model,
            device
        )

        print(f"[Test] Acc: {acc_test:.4f}, Macro-F1: {f1_macro_test:.4f}, Weighted-F1: {f1_weighted_test:.4f}")

    print("\n===== Training Finished =====")
    print(f"Best Val Acc: {best_metrics['acc']:.4f}")
    print(f"Best Val Macro-F1: {best_metrics['f1_macro']:.4f}")
    print(f"Best Val Weighted-F1: {best_metrics['f1_weighted']:.4f}")

    return best_metrics

def compute_arl_weights(out_dict):
    """
    ARL: entropy + magnitude
    支持 N modalities
    """
    weights = {}

    entropies = {}
    magnitudes = {}

    for m, out in out_dict.items():
        prob = torch.softmax(out, dim=1)

        # entropy
        H = (-prob * torch.log(prob + 1e-12)).sum(dim=1).mean()

        # magnitude
        M = torch.mean(torch.abs(out))

        entropies[m] = H
        magnitudes[m] = M

    # normalize entropy
    total_H = sum(entropies.values())
    for m in entropies:
        entropies[m] = entropies[m] / (total_H + 1e-6)

    # inverse entropy
    inv_H = {m: 1.0 / (entropies[m] + 1e-6) for m in entropies}

    # combine
    raw = {
        m: inv_H[m] * magnitudes[m]
        for m in out_dict
    }

    # normalize
    Z = sum(raw.values())
    for m in raw:
        weights[m] = raw[m] / (Z + 1e-6)

    return weights

def train_arl_epoch(
    epoch,
    train_loader,
    model,
    optimizer,
    device="cuda"
):
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("Start training ... ")

    total_loss = 0
    total_uni_loss = {m: 0 for m in model.modalities}

    for step, batch in enumerate(train_loader):

        *modal_inputs, y = batch
        y = y.to(device)

        inputs = {
            m: modal_inputs[i].float().to(device)
            for i, m in enumerate(model.modalities)
        }

        optimizer.zero_grad()

        # =========================
        # forward
        # =========================
        out_mm, out_dict = model(inputs)

        # =========================
        # losses
        # =========================
        loss_mm = criterion(out_mm, y)
        losses = {
            m: criterion(out_dict[m], y)
            for m in model.modalities
        }

        # =========================
        # ARL: magnitude
        # =========================
        magnitudes = {}
        for m in model.modalities:
            magnitudes[m] = torch.mean(torch.abs(out_dict[m]), dim=0).sum().item()

        # normalize magnitude
        mag_sum = sum(magnitudes.values()) + 1e-8
        for m in magnitudes:
            magnitudes[m] /= mag_sum

        # =========================
        # ARL: entropy
        # =========================
        entropies = {}
        for m in model.modalities:
            prob = torch.softmax(out_dict[m], dim=1)
            H = torch.sum(-prob * torch.log(prob + 1e-16), dim=1).mean()
            entropies[m] = H.item()

        # normalize entropy
        H_sum = sum(entropies.values()) + 1e-8
        for m in entropies:
            entropies[m] /= H_sum

        # clamp（和原论文一样）
        for m in entropies:
            entropies[m] = max(entropies[m], 0.3)

        # =========================
        # inverse entropy
        # =========================
        inv_H = {
            m: 1.0 / entropies[m]
            for m in model.modalities
        }

        # normalize
        inv_sum = sum(inv_H.values()) + 1e-8
        for m in inv_H:
            inv_H[m] /= inv_sum

        # =========================
        # final ARL weight
        # =========================
        # 👉 cross-style（跟原论文一样）
        weights = {}
        mods = model.modalities

        for i, m in enumerate(mods):
            other_m = mods[1 - i] if len(mods) == 2 else m  # 多模态简单fallback
            weights[m] = magnitudes[other_m] * inv_H[m]

        # normalize
        w_sum = sum(weights.values()) + 1e-8
        for m in weights:
            weights[m] /= w_sum

        # =========================
        # loss（保持原ARL逻辑）
        # =========================
        loss_uni = sum(losses.values())

        loss = loss_mm + loss_uni

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=40, norm_type=2)

        optimizer.step()

        total_loss += loss.item()

        for m in model.modalities:
            total_uni_loss[m] += losses[m].item()

        # =========================
        # debug
        # =========================
        if step % 50 == 0:
            print(f"[Epoch {epoch}] Step {step}")
            print(f"Entropy: {entropies}")
            print(f"Magnitude: {magnitudes}")
            print(f"ARL weights: {weights}")

    print(f"Epoch {epoch} finished | Loss: {total_loss / len(train_loader):.4f}")

    # =========================
    # scores（兼容你 run loop）
    # =========================
    scores = {
        m: total_uni_loss[m] / len(train_loader)
        for m in model.modalities
    }

    return model, scores
    

def validate(
    epoch,
    val_loader,
    model,
    device="cuda",
    merge_alpha=0.5,
):

    model.eval()

    pred_all, label_all = [], []

    # 每个模态单独记录
    pred_modalities = {m: [] for m in model.modalities}

    with torch.no_grad():
        for batch in val_loader:

            # -------------------------
            # unpack（支持N模态）
            # -------------------------
            *modal_inputs, y = batch
            y = y.to(device)

            inputs = {
                m: modal_inputs[i].float().to(device)
                for i, m in enumerate(model.modalities)
            }

            # =========================
            # forward（统一入口！）
            # =========================
            out_mm, out_dict = model(inputs)

            # =========================
            # fusion（baseline：直接用mm）
            # =========================
            # 👉 推荐用 multimodal output
            out = out_mm

            # 如果你想对齐之前：
            # out = (out_mm + sum(out_dict.values())) / (len(out_dict) + 1)

            pred = out.argmax(dim=1)

            pred_all.extend(pred.cpu().tolist())

            # label兼容
            if y.ndim > 1:
                label_all.extend(y.argmax(dim=1).cpu().tolist())
            else:
                label_all.extend(y.cpu().tolist())

            # =========================
            # per-modality prediction
            # =========================
            for m in model.modalities:
                pred_m = out_dict[m].argmax(dim=1)
                pred_modalities[m].extend(pred_m.cpu().tolist())

    # =========================
    # overall metrics
    # =========================
    acc = accuracy_score(label_all, pred_all)
    f1_macro = f1_score(label_all, pred_all, average="macro")
    f1_weighted = f1_score(label_all, pred_all, average="weighted")

    print(f"[Val] Acc: {acc:.4f}, Macro-F1: {f1_macro:.4f}, Weighted-F1: {f1_weighted:.4f}")

    # =========================
    # per-modality metrics
    # =========================
    for m in model.modalities:
        acc_m = accuracy_score(label_all, pred_modalities[m])
        f1_m = f1_score(label_all, pred_modalities[m], average="macro")

        print(f"[Modality] {m} Acc: {acc_m:.4f} | F1: {f1_m:.4f}")

    return acc, f1_macro, f1_weighted