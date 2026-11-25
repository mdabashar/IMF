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
    parser.add_argument('--nepochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--merge_alpha', type=float, default=0.5)
    parser.add_argument('--functional_p', type=float, default=0.3)
    parser.add_argument('--length', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    dataset = "../../../vist/datasets/task2/"

    # ----------------------------
    # train
    # ----------------------------
    txt = np.load(dataset + "train_text_embed.npy", allow_pickle=True)
    img = np.load(dataset + "train_img_vit.npy", allow_pickle=True)
    y   = np.int64(np.load(dataset + "train_labels_embed.npy", allow_pickle=True)).squeeze()

    # ----------------------------
    # dev / validation
    # ----------------------------
    txt_dev = np.load(dataset + "dev_text_embed.npy", allow_pickle=True)
    img_dev = np.load(dataset + "dev_img_vit.npy", allow_pickle=True)
    y_dev   = np.int64(np.load(dataset + "dev_labels.npy", allow_pickle=True)).squeeze()

    # ----------------------------
    # test
    # ----------------------------
    txt_test = np.load(dataset + "test_text_embed.npy", allow_pickle=True)
    img_test = np.load(dataset + "test_img_vit.npy", allow_pickle=True)
    y_test   = np.int64(np.load(dataset + "test_labels.npy", allow_pickle=True)).squeeze()

    txt = txt.mean(axis=1)
    txt_dev = txt_dev.mean(axis=1)
    txt_test = txt_test.mean(axis=1) 
    print("\n========== TRAIN ==========")
    print("txt:", txt.shape, txt.dtype)
    print("img:", img.shape, img.dtype)
    print("y  :", y.shape, y.dtype)

    print("\n========== DEV ==========")
    print("txt_dev:", txt_dev.shape, txt_dev.dtype)
    print("img_dev:", img_dev.shape, img_dev.dtype)
    print("y_dev  :", y_dev.shape, y_dev.dtype)

    print("\n========== TEST ==========")
    print("txt_test:", txt_test.shape, txt_test.dtype)
    print("img_test:", img_test.shape, img_test.dtype)
    print("y_test  :", y_test.shape, y_test.dtype)

    return (
        txt, img, y,
        txt_dev, img_dev, y_dev,
        txt_test, img_test, y_test
    )

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

    # =========================
    # Load data
    # =========================
    txt, img, y, txt_dev, img_dev, y_dev, txt_test, img_test, y_test = load_data()

    train_dataset = DatasetBERT(txt, img, y)
    val_dataset   = DatasetBERT(txt_dev, img_dev, y_dev)
    test_dataset  = DatasetBERT(txt_test, img_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size)
    if os.environ.get("TUNING_VALIDATION_ONLY") == "1":
        # Never inspect the real test partition while choosing hyperparameters.
        test_loader = val_loader
    
    model = MultiModalGBClassifier(
        modalities=["m1", "m2"],
        num_class=len(np.unique(y)),
        hidden_dim=512,
        embed_dim=256,
        encoder_types={"m1": "mlp", "m2": "mlp"},
        input_dims={"m1": txt.shape[1], "m2": img.shape[1]},
        device=device
    )

    def train_fn(epoch, loader, model, optimizer, device):
        return train_arl_epoch(
            epoch,
            loader,
            model,
            optimizer,
            device=device
        )
   
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
        val_loader=val_loader,  
        test_loader = test_loader,
        train_fn=train_fn,
        optimizer=optimizer,
        n_epochs=args.nepochs,
        device=device,
        model_name=args.model
    )
