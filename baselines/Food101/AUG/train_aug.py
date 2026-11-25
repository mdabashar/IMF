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
from model_aug import *
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
    test_loader,
    test_real_loader,
    train_fn,
    optimizer,
    n_epochs,
    device,
    scheduler=None,
    lam=1.0,
    model_name="model"
):
    """
    通用训练循环（支持多模态 imbalance + metrics）

    train_fn:
        输入 -> (epoch, loader, model, optimizer)
        输出 -> (model, scores_dict)

    validate_fn:
        必须返回:
        acc, f1_macro, f1_weighted
    """

    best_acc = 0
    best_metrics = {
        "acc": 0,
        "f1_macro": 0,
        "f1_weighted": 0
    }

    check = max(1, n_epochs // 10)

    for epoch in range(n_epochs):

        print(f"\n===== Epoch {epoch} =====")

        # -------------------------
        # scheduler
        # -------------------------
        if scheduler is not None:
            scheduler.step()

        # -------------------------
        # train
        # -------------------------
        model, scores = train_fn(
            epoch,
            train_loader,
            model,
            optimizer
        )

        # -------------------------
        # validation（必须返回3个指标）
        # -------------------------
        acc, f1_macro, f1_weighted = validate(epoch, test_loader, model, device)

        print(f"[Val] Acc: {acc:.4f}, Macro-F1: {f1_macro:.4f}, Weighted-F1: {f1_weighted:.4f}")

        # -------------------------
        # save best
        # -------------------------
        if acc >= best_acc:
            best_acc = acc
            best_metrics["acc"] = acc
            best_metrics["f1_macro"] = f1_macro
            best_metrics["f1_weighted"] = f1_weighted

            # print("🔥 Find a better model and save it!")
            # best_path = f"{model_name}_best_{acc:.4f}.pth"

            # torch.save({
            #     "model_state": model.state_dict(),
            #     "num_layers": {
            #         m: len(model.additional_layers[m])
            #         for m in model.modalities
            #     }
            # }, best_path)

        # -------------------------
        # imbalance analysis
        # -------------------------
        strong = max(scores, key=scores.get)
        weak = min(scores, key=scores.get)

        ratio = scores[strong] / (scores[weak] + 1e-8)

        print(f"Scores: {scores}")
        print(f"Strong: {strong}, Weak: {weak}, Ratio: {ratio:.3f}")

        
        acc_test, f1_macro_test, f1_weighted_test = validate(epoch, test_real_loader, model, device)

        print(f"[TEST] Acc: {acc_test:.4f}, Macro-F1: {f1_macro_test:.4f}, Weighted-F1: {f1_weighted_test:.4f}")
        
        # -------------------------
        # dynamic capacity trigger
        # -------------------------
        if hasattr(model, "add_layer"):

            if ((epoch + 1) % check == 0 or epoch == 0):

                if ratio > lam + 0.01:
                    print(f"👉 {weak} is weaker → add layer")
                    model.add_layer(weak)

                elif ratio < lam - 0.01:
                    print(f"👉 {strong} imbalance → add layer")
                    model.add_layer(strong)

        

    print("\n===== Training Finished =====")
    print(f"Best Acc: {best_metrics['acc']:.4f}")
    print(f"Best Macro-F1: {best_metrics['f1_macro']:.4f}")
    print(f"Best Weighted-F1: {best_metrics['f1_weighted']:.4f}")

    return best_metrics


# def train_modality_balance(epoch, train_loader, model, optimizer, device = "cuda", merge_alpha=0.5):

#     model.train()

#     tl, tm1, tm2 = Averager(), Averager(), Averager()
#     criterion = nn.CrossEntropyLoss().to(device)

#     score_m1, score_m2 = 0.0, 0.0


def train_modality_balance(epoch, train_loader, model, optimizer, device="cuda", merge_alpha=0.5):

    model.train()

    tl = Averager()
    modality_losses = {m: Averager() for m in model.modalities}
    scores = {m: 0.0 for m in model.modalities}

    criterion = nn.CrossEntropyLoss().to(device)

    for step, batch in enumerate(train_loader):

        # -------------------------
        # unpack（兼容2/3模态）
        # -------------------------
        *modal_inputs, y = batch

        y = y.to(device)

        inputs = {}
        for i, m in enumerate(model.modalities):
            inputs[m] = modal_inputs[i].float().to(device)

        # =========================
        # per-modality training（核心）
        # =========================
        outputs = {}

        for m in model.modalities:

            optimizer.zero_grad()

            # encoder
            feat = model.encoders[m](inputs[m])

            # classifier
            out, aux_fea, add_fea = model.classifier(feat, m)
            outputs[m] = out

            # loss
            if add_fea is None:
                loss_m = criterion(out, y).mean()
            else:
                y_onehot = F.one_hot(y, num_classes=aux_fea.shape[1]).float()
                kl = y_onehot * aux_fea.detach().softmax(1)
                # kl = y * aux_fea.detach().softmax(1)
                loss_m = (
                    criterion(out, y).mean()
                    + criterion(aux_fea, y).mean()
                    + criterion(add_fea, y).mean()
                    - 0.5 * criterion(add_fea, kl).mean()
                )

            loss_m.backward()
            optimizer.step()

            modality_losses[m].add(loss_m.item())

        # =========================
        # merge loss（仅用于logging）
        # =========================
        loss = sum(modality_losses[m].item() for m in model.modalities) / len(model.modalities)
        tl.add(loss)

        # =========================
        # confidence score（关键）
        # =========================
        for m in model.modalities:

            prob = F.softmax(outputs[m], dim=1)

            scores[m] += sum([
                prob[i][torch.argmax(y[i])]
                for i in range(prob.size(0))
            ])

        # -------------------------
        # logging
        # -------------------------
        if step % 50 == 0:
            loss_str = " | ".join([f"{m}: {modality_losses[m].item():.3f}" for m in model.modalities])
            print(f"[Epoch {epoch}] Loss: {tl.item():.3f} | {loss_str}")

    # =========================
    # summary
    # =========================
    print("====================================")
    loss_str = " | ".join([f"{m}: {modality_losses[m].item():.3f}" for m in model.modalities])
    print(f"Epoch {epoch}: Loss {tl.item():.3f}, {loss_str}")

    return model, scores

def validate(epoch, val_loader, model, device="cuda", merge_alpha=0.5):

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

            inputs = {}
            for i, m in enumerate(model.modalities):
                inputs[m] = modal_inputs[i].float().to(device)

            # -------------------------
            # forward
            # -------------------------
            outputs = {}

            for m in model.modalities:
                feat = model.encoders[m](inputs[m])
                out, _, _ = model.classifier(feat, m)
                outputs[m] = out

            # -------------------------
            # 融合（默认平均）
            # -------------------------
            out = sum(outputs.values()) / len(outputs)

            pred = out.argmax(dim=1)

            pred_all.extend(pred.cpu().tolist())

            # 兼容 one-hot / label
            if y.ndim > 1:
                label_all.extend(y.argmax(dim=1).cpu().tolist())
            else:
                label_all.extend(y.cpu().tolist())

            # -------------------------
            # 每个模态预测
            # -------------------------
            for m in model.modalities:
                pred_m = outputs[m].argmax(dim=1)
                pred_modalities[m].extend(pred_m.cpu().tolist())

    # =========================
    # overall metrics
    # =========================
    acc = accuracy_score(label_all, pred_all)
    f1_macro = f1_score(label_all, pred_all, average="macro")
    f1_weighted = f1_score(label_all, pred_all, average="weighted")

    print(f"[Val] Acc: {acc:.4f}, Macro-F1: {f1_macro:.4f}, Weighted-F1: {f1_weighted:.4f}")

    # =========================
    # per-modality metrics（论文加分）
    # =========================
    for m in model.modalities:
        f1_m = f1_score(label_all, pred_modalities[m], average="macro")
        print(f"[Modality] {m} F1: {f1_m:.4f}")

    return acc, f1_macro, f1_weighted