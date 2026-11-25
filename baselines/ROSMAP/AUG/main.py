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
    parser.add_argument('--functional_p', type=float, default=0.3)
    parser.add_argument('--length', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--validation_fraction', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=1e-3)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    dataset = (str(__import__("pathlib").Path(__file__).resolve().parents[3] / "datasets" / "ROSMAP") + "/")

    # ----------------------------
    # train
    # ----------------------------
    m1 = pd.read_csv(dataset + "1_tr.csv").values
    m2 = pd.read_csv(dataset + "2_tr.csv").values
    m3 = pd.read_csv(dataset + "3_tr.csv").values
    y  = pd.read_csv(dataset + "labels_tr.csv").values.squeeze()

    if os.environ.get("EVAL_ON_VALIDATION") == "1":
        return m1, m2, m3, y, None, None, None, None

    # ----------------------------
    # test
    # ----------------------------
    m1_test = pd.read_csv(dataset + "1_te.csv").values
    m2_test = pd.read_csv(dataset + "2_te.csv").values
    m3_test = pd.read_csv(dataset + "3_te.csv").values
    y_test  = pd.read_csv(dataset + "labels_te.csv").values.squeeze()

    return m1, m2, m3, y, m1_test, m2_test, m3_test, y_test


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

    train_fn = lambda epoch, loader, model, optimizer: \
        train_modality_balance(epoch, loader, model, optimizer, device=device, merge_alpha=args.merge_alpha)

    # =========================
    # Load data
    # =========================
    m1, m2, m3, y, m1_test, m2_test, m3_test, y_test = load_data()
    if os.environ.get("EVAL_ON_VALIDATION") == "1":
        tr, va = train_test_split(np.arange(len(y)), test_size=args.validation_fraction, stratify=y, random_state=2026)
        m1_test, m2_test, m3_test, y_test = m1[va], m2[va], m3[va], y[va]
        m1, m2, m3, y = m1[tr], m2[tr], m3[tr], y[tr]

    # =========================
    # Dataset
    # =========================
    train_dataset = DatasetBERT(m1, m2, m3, y)
    test_dataset = DatasetBERT(np.array(m1_test), np.array(m2_test), np.array(m3_test), np.array(y_test))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


    N_RUNS = args.runs

    all_metrics = {
    "acc": [],
    "f1_macro": [],
    "f1_weighted": []
    }

    for run in range(N_RUNS):

       print(f"\n================ Run {run+1}/{N_RUNS} ================")

       seed = 42 + run
       torch.manual_seed(seed)
       np.random.seed(seed)
       random.seed(seed)

       model = MultiModalGBClassifier(
        modalities=["m1", "m2", "m3"],
        num_class=len(np.unique(y)),
        hidden_dim=args.length,
        embed_dim=args.length,
        encoder_types={"m1": "mlp", "m2": "mlp", "m3": "mlp"},
        input_dims={
            "m1": m1.shape[1],
            "m2": m2.shape[1],
            "m3": m3.shape[1]
        },
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
        val_loader=test_loader,
        test_loader=test_loader,
        train_fn=train_fn,
        optimizer=optimizer,
        n_epochs=args.nepochs,
        device=device,
        model_name=args.model
       )

       for k in all_metrics.keys():
            all_metrics[k].append(best_metrics[k])


    prefix = os.environ.get("RESULT_PREFIX", "strict_20run")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(f"{prefix}_metrics.csv", index=False)
    metrics_df.agg(["mean", "std"]).to_csv(f"{prefix}_summary.csv")
    print("\n================ FINAL RESULTS ================")

    for k, v in all_metrics.items():
       v = np.array(v)
       mean = v.mean()
       std = v.std(ddof=1)
       print(f"{k}: {mean:.4f} ± {std:.4f}")
 
