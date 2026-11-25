import argparse
import torch
import numpy as np
import random
import pickle
import os

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import *
from train import *
from sklearn.model_selection import StratifiedKFold

def get_kfold_loaders(args):

    m1, m2, m3, y = load_data()

    # =========================
    # 🔥 处理 text modality（统一维度）
    # =========================
    if len(m1.shape) == 3:
        m1 = m1.mean(axis=1)

    # =========================
    # 🔥 input dims（只算一次！）
    # =========================
    input_dims = {
        "m1": m1.shape[1],
        "m2": m2.shape[1],
        "m3": m3.shape[1],
    }

    num_classes = len(np.unique(y))

    print("\n📐 Input dims:", input_dims)
    print("🎯 Num classes:", num_classes)

    # =========================
    # K-Fold
    # =========================
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.fold_seed)
    X_idx = np.arange(len(y))

    # Hyperparameter tuning must not select on an outer CV test fold.  For
    # tuning jobs, create one deterministic stratified validation split from
    # the available training corpus.  Final reporting can still use 5-fold CV
    # after a configuration has been selected.
    if os.environ.get("TUNING_VALIDATION_ONLY") == "1":
        train_idx, val_idx = train_test_split(
            X_idx, test_size=0.2, stratify=y, random_state=42
        )
        train_dataset = DatasetBERT(
            m1[train_idx], m2[train_idx], m3[train_idx], y[train_idx]
        )
        val_dataset = DatasetBERT(
            m1[val_idx], m2[val_idx], m3[val_idx], y[val_idx]
        )
        return [(
            DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False),
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False),
        )], input_dims, num_classes

    fold_loaders = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_idx, y)):

        print(f"\n========== Fold {fold} ==========")

        m1_tr, m1_te = m1[train_idx], m1[test_idx]
        m2_tr, m2_te = m2[train_idx], m2[test_idx]
        m3_tr, m3_te = m3[train_idx], m3[test_idx]
        y_tr, y_te   = y[train_idx], y[test_idx]

        train_dataset = DatasetBERT(m1_tr, m2_tr, m3_tr, y_tr)
        test_dataset  = DatasetBERT(m1_te, m2_te, m3_te, y_te)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

        val_loader = train_loader  # training metrics only; never used for selection
        fold_loaders.append((train_loader, val_loader, test_loader))

    # ✅ ⭐ 返回所有你需要的
    return fold_loaders, input_dims, num_classes

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default="neurips2025")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--blr', type=float, default=1.0)
    parser.add_argument('--nepochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--merge_alpha', type=float, default=0.5)
    parser.add_argument('--functional_p', type=float, default=0.3)
    parser.add_argument('--length', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold_seed', type=int, default=99)
    parser.add_argument('--arl_uni_weight', type=float, default=1.0)
    parser.add_argument('--arl_entropy_floor', type=float, default=0.3)
    parser.add_argument('--arl_weight_temperature', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--no_validation', action='store_true')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    dataset = "../../../datasets/mustard/"

    m3 = np.load(dataset + "mustard_audio.npy").astype(np.float32)
    m2 = np.load(dataset + "mustard_video.npy").astype(np.float32)
    m1 = np.load(dataset + "mustard_text.npy").astype(np.float32)

    y = np.load(dataset + "mustard_label.npy")
    y = np.argmax(y, axis=1)

    print("========== DATA ==========")
    print("m1 (text): ", m1.shape)
    print("m2 (video):", m2.shape)
    print("m3 (audio):", m3.shape)
    print("y:", y.shape)

    return m1, m2, m3, y

def test_evaluate(loader, model, device="mps"):
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()

    preds, labels = [], []

    with torch.no_grad():
        for m1, m2, y in loader:

            m1 = m1.to(device)
            m2 = m2.to(device)
            y = y.to(device)

            inputs = {"m1": m1, "m2": m2}
            features = model(inputs)

            outputs = []
            for m in model.modalities:
                out, _, _ = model.classifier(features[m], m)
                outputs.append(out)

            out = sum(outputs) / len(outputs)

            pred = out.argmax(dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(y.argmax(dim=1).cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")

    return acc, f1_macro, f1_weighted


if __name__ == "__main__":

    args = parse_args()

    print("===== Start Experiment =====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    set_seed(args.seed)

    def train_fn(epoch, loader, model, optimizer, device):
        return train_arl_epoch(
            epoch,
            loader,
            model,
            optimizer,
            device=device,
            uni_weight=args.arl_uni_weight,
            entropy_floor=args.arl_entropy_floor,
            weight_temperature=args.arl_weight_temperature
        )
   

    all_metrics = {
    "acc": [],
    "f1_macro": [],
    "f1_weighted": []
    }

    fold_loaders, input_dims, num_classes = get_kfold_loaders(args)

    for fold, (train_loader, val_loader, test_loader) in enumerate(fold_loaders):

       print(f"\n🚀 Training Fold {fold}")

       seed = args.seed + fold
       torch.manual_seed(seed)
       np.random.seed(seed)

       model = MultiModalGBClassifier(
        modalities=["m1", "m2", "m3"],
        num_class=num_classes,
        hidden_dim=args.length,
        embed_dim=args.length,
        encoder_types={"m1": "mlp", "m2": "mlp", "m3": "mlp"},
        input_dims=input_dims,
        device=device
       )

       model.to(device)

       optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
       )

       best_metrics = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_fn=train_fn,
        optimizer=optimizer,
        n_epochs=args.nepochs,
        select_on_validation=not args.no_validation,
        device=device,
        model_name=args.model
       )

       for k in all_metrics.keys():
            all_metrics[k].append(best_metrics[k])


    print("\n================ FINAL RESULTS ================")

    for k, v in all_metrics.items():
       v = np.array(v)
       mean = v.mean()
       std = v.std()
       print(f"{k}: {mean:.4f} ± {std:.4f}")
