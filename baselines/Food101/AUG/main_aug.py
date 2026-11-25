import argparse
import torch
import numpy as np
import random
import pickle
import os

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import *
from train_aug import *

# 假设你已有
# DatasetBERT
# run_training_loop
# validate / test_eval
# train_modality_balance / train_boosting


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default="neurips2025")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--blr', type=float, default=1.0)
    parser.add_argument('--nepochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--merge_alpha', type=float, default=0.5)

    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    with open((str(__import__("pathlib").Path(__file__).resolve().parents[3] / "datasets" / "Food101") + "/" + 'text_train.pkl'), 'rb') as f:
        m2 = pickle.load(f)
    with open((str(__import__("pathlib").Path(__file__).resolve().parents[3] / "datasets" / "Food101") + "/" + 'image_train.pkl'), 'rb') as f:
        m1 = pickle.load(f)
    with open((str(__import__("pathlib").Path(__file__).resolve().parents[3] / "datasets" / "Food101") + "/" + 'label_train.pkl'), 'rb') as f:
        y = pickle.load(f)

    with open((str(__import__("pathlib").Path(__file__).resolve().parents[3] / "datasets" / "Food101") + "/" + 'text_test.pkl'), 'rb') as f:
        m2_test = pickle.load(f)
    with open((str(__import__("pathlib").Path(__file__).resolve().parents[3] / "datasets" / "Food101") + "/" + 'image_test.pkl'), 'rb') as f:
        m1_test = pickle.load(f)
    with open((str(__import__("pathlib").Path(__file__).resolve().parents[3] / "datasets" / "Food101") + "/" + 'label_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)

    return m1, m2, y, m1_test, m2_test, y_test


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

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print("Device:", device)

    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    set_seed(args.seed)

    # =========================
    # Load data
    # =========================
    m1, m2, y, m1_test, m2_test, y_test = load_data()

    X = np.arange(len(m1))

    # train / val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.1, random_state=args.seed
    )

    m1_train = np.array(m1)[X_train]
    m2_train = np.array(m2)[X_train]
    y_train = np.array(y)[X_train]

    m1_val = np.array(m1)[X_val]
    m2_val = np.array(m2)[X_val]
    y_val = np.array(y)[X_val]

    # =========================
    # Dataset
    # =========================
    train_dataset = DatasetBERT(m1_train, m2_train, y_train)
    val_dataset = DatasetBERT(m1_val, m2_val, y_val)
    test_dataset = DatasetBERT(np.array(m1_test), np.array(m2_test), np.array(y_test))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    
    model = MultiModalGBClassifier(
        modalities=["m1", "m2"],
        num_class=len(np.unique(y_train)),
        hidden_dim=512,
        embed_dim=256,
        encoder_types={"m1": "mlp", "m2": "mlp"},
        input_dims={"m1": m1_train.shape[1], "m2": m2_train.shape[1]},
        device=device
    )

    train_fn = lambda epoch, loader, model, optimizer: \
        train_modality_balance(epoch, loader, model, optimizer, device=device, merge_alpha=args.merge_alpha)

   
    # =========================
    # Optimizer
    # =========================
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-3
    )

    # =========================
    # Training
    # =========================
    best_metrics = run_training_loop(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,  
        test_real_loader = test_loader,
        train_fn=train_fn,
        optimizer=optimizer,
        n_epochs=args.nepochs,
        device=device,
        lam=args.lam,
        model_name=args.model
    )

    # # =========================
    # # Load best model（支持动态结构）
    # # =========================
    # best_model_path = f"{args.model}_best_{best_metrics['acc']:.4f}.pth"
    
    # checkpoint = torch.load(best_model_path, map_location=device)
    
    # # -------------------------
    # # 1️⃣ rebuild architecture（关键）
    # # -------------------------
    # if "num_layers" in checkpoint:
    #     for m in checkpoint["num_layers"]:
    #         num_layers = checkpoint["num_layers"][m]
    
    #         for _ in range(num_layers):
    #             model.add_layer(m)
    
    # # -------------------------
    # # 2️⃣ load weights
    # # -------------------------
    # model.load_state_dict(checkpoint["model_state"])
    
    # model.to(device)
    
    # print("✅ Loaded best model successfully!")

    # print("\n===== Final Test Evaluation =====")

    # acc, f1_macro, f1_weighted = test_evaluate(test_loader, model)

    # print(f"[TEST] Acc: {acc:.4f}")
    # print(f"[TEST] Macro-F1: {f1_macro:.4f}")
    # print(f"[TEST] Weighted-F1: {f1_weighted:.4f}")