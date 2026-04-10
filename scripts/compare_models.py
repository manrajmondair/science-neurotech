#!/usr/bin/env python3
"""
compare_models.py

Trains and evaluates 5 neural decoding algorithms, then generates plots
to find the optimal speed-accuracy balance for BCI real-time deployment.

Algorithms
----------
1. LDA   – Linear Discriminant Analysis      (sklearn)
2. RF    – Random Forest                      (sklearn)
3. XGB   – XGBoost                            (xgboost)
4. MLP   – Multi-Layer Perceptron             (PyTorch)
5. LSTM  – Long Short-Term Memory             (PyTorch)

Output
------
model_comparison.png          – Speed/accuracy trade-offs + confusion matrices
model_comparison_dynamics.png – Learning curves, ROC, ITR (Information Transfer Rate)

Usage
-----
    python scripts/compare_models.py
    python scripts/compare_models.py --n-classes 4 --n-samples 8000 --epochs 80
"""

import argparse
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    n_channels=64,   # electrode channels
    n_seq=20,        # time bins per window (for LSTM)
    n_classes=8,     # controller states (one-hot output size)
    n_samples=4000,  # total labelled examples
    batch_size=64,
    epochs=50,
    seed=42,
)
PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]


# ─── Data Simulation ──────────────────────────────────────────────────────────

def simulate_neural_data(n_samples, n_channels, n_seq, n_classes, seed=42):
    """
    Generate mock neural spike-rate data with class-specific tuning curves.

    Returns
    -------
    X_flat   : (n_samples, n_channels)           – binned spike rates (flat)
    X_seq    : (n_samples, n_seq, n_channels)    – time-series for LSTM
    y        : (n_samples,)                       – integer class labels
    y_onehot : (n_samples, n_classes)             – one-hot labels
    """
    rng = np.random.default_rng(seed)

    # Each class has a distinct mean firing-rate profile across channels
    class_means = rng.uniform(0.1, 1.0, size=(n_classes, n_channels))

    y = rng.integers(0, n_classes, size=n_samples)

    # Flat features: Poisson spike counts, normalised to firing rate (spikes/bin)
    X_flat = np.array(
        [rng.poisson(class_means[c] * 10) / 10.0 for c in y], dtype=np.float32
    )

    # Sequential features: slow drift across time bins + noise
    X_seq = np.zeros((n_samples, n_seq, n_channels), dtype=np.float32)
    for i, c in enumerate(y):
        for t in range(n_seq):
            scale = 8.0 + t * 0.1  # slight temporal modulation
            X_seq[i, t] = (
                rng.poisson(class_means[c] * scale) / scale
                + rng.normal(0, 0.05, n_channels)
            )

    y_onehot = np.zeros((n_samples, n_classes), dtype=np.float32)
    y_onehot[np.arange(n_samples), y] = 1.0

    return X_flat, X_seq, y.astype(np.int64), y_onehot


# ─── PyTorch Model Definitions ────────────────────────────────────────────────

class MLPDecoder(nn.Module):
    """Three-layer fully-connected decoder."""

    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMDecoder(nn.Module):
    """Stacked LSTM decoder – uses only the last hidden state."""

    def __init__(
        self, in_features: int, hidden_size: int, n_layers: int, n_classes: int
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            in_features, hidden_size, n_layers,
            batch_first=True, dropout=0.2 if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (batch, seq, hidden)
        return self.head(out[:, -1])   # last timestep → logits


# ─── Training & Evaluation Helpers ───────────────────────────────────────────

def train_torch(model, X_tr, y_tr, epochs, batch_size, device):
    """Train a PyTorch model, return per-epoch loss list and elapsed seconds."""
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(
        torch.from_numpy(X_tr).to(device),
        torch.from_numpy(y_tr).to(device),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    losses = []
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
        scheduler.step()

    return losses, time.perf_counter() - t0


def eval_torch(model, X_te, y_te, device, n_timing=300):
    """Evaluate a PyTorch model; return acc, confusion matrix, probs, latency_ms."""
    model.eval()
    X_t = torch.from_numpy(X_te).to(device)

    with torch.no_grad():
        logits = model(X_t)
        preds = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    acc = accuracy_score(y_te, preds)
    cm = confusion_matrix(y_te, preds)

    # Single-sample latency (median over n_timing warm-up+timed runs)
    single = torch.from_numpy(X_te[:1]).to(device)
    with torch.no_grad():
        for _ in range(20):  # warm up
            model(single)
    times = []
    with torch.no_grad():
        for _ in range(n_timing):
            t0 = time.perf_counter()
            model(single)
            times.append(time.perf_counter() - t0)
    lat_ms = float(np.median(times)) * 1000

    return acc, cm, probs, lat_ms


def eval_sklearn(clf, X_te, y_te, n_timing=300):
    """Evaluate a sklearn model; return acc, confusion matrix, probs, latency_ms."""
    preds = clf.predict(X_te)
    probs = clf.predict_proba(X_te) if hasattr(clf, "predict_proba") else None
    acc = accuracy_score(y_te, preds)
    cm = confusion_matrix(y_te, preds)

    # Warm up
    for _ in range(20):
        clf.predict(X_te[:1])
    times = []
    for _ in range(n_timing):
        t0 = time.perf_counter()
        clf.predict(X_te[:1])
        times.append(time.perf_counter() - t0)
    lat_ms = float(np.median(times)) * 1000

    return acc, cm, probs, lat_ms


# ─── ITR Calculation ──────────────────────────────────────────────────────────

def itr_bits_per_min(acc: float, lat_ms: float, n_classes: int) -> float:
    """
    Nykopp information transfer rate (bits/min).

    B = log2(N) + p·log2(p) + (1-p)·log2((1-p)/(N-1))
    ITR = (60 000 / latency_ms) · B
    """
    if acc <= 0 or acc >= 1 or n_classes < 2:
        return 0.0
    p = np.clip(acc, 1e-9, 1 - 1e-9)
    b = np.log2(n_classes) + p * np.log2(p) + (1 - p) * np.log2(
        np.clip((1 - p) / (n_classes - 1), 1e-12, None)
    )
    return max(b, 0.0) * (60_000.0 / lat_ms)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(cfg: dict):
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Generate data ─────────────────────────────────────────────────────────
    print("Simulating neural data...")
    X_flat, X_seq, y, _ = simulate_neural_data(
        cfg["n_samples"], cfg["n_channels"], cfg["n_seq"], cfg["n_classes"], cfg["seed"]
    )

    Xf_tr, Xf_te, Xs_tr, Xs_te, y_tr, y_te = train_test_split(
        X_flat, X_seq, y, test_size=0.2, random_state=cfg["seed"], stratify=y
    )

    results = {}  # name → {acc, latency_ms, train_time, cm, probs, losses}

    # ── 1. LDA ────────────────────────────────────────────────────────────────
    print("\n[1/5] Training LDA...")
    t0 = time.perf_counter()
    lda = LinearDiscriminantAnalysis()
    lda.fit(Xf_tr, y_tr)
    train_time = time.perf_counter() - t0
    acc, cm, probs, lat = eval_sklearn(lda, Xf_te, y_te)
    results["LDA"] = dict(acc=acc, latency_ms=lat, train_time=train_time,
                          cm=cm, probs=probs, losses=None)
    print(f"  Acc={acc:.3f}  Latency={lat:.4f}ms  Train={train_time:.2f}s")

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    print("\n[2/5] Training Random Forest...")
    t0 = time.perf_counter()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, n_jobs=-1, random_state=cfg["seed"]
    )
    rf.fit(Xf_tr, y_tr)
    train_time = time.perf_counter() - t0
    acc, cm, probs, lat = eval_sklearn(rf, Xf_te, y_te)
    results["Random Forest"] = dict(acc=acc, latency_ms=lat, train_time=train_time,
                                    cm=cm, probs=probs, losses=None)
    print(f"  Acc={acc:.3f}  Latency={lat:.4f}ms  Train={train_time:.2f}s")

    # ── 3. XGBoost ────────────────────────────────────────────────────────────
    print("\n[3/5] Training XGBoost...")
    t0 = time.perf_counter()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=cfg["seed"],
        verbosity=0,
    )
    xgb_clf.fit(Xf_tr, y_tr)
    train_time = time.perf_counter() - t0
    acc, cm, probs, lat = eval_sklearn(xgb_clf, Xf_te, y_te)
    results["XGBoost"] = dict(acc=acc, latency_ms=lat, train_time=train_time,
                               cm=cm, probs=probs, losses=None)
    print(f"  Acc={acc:.3f}  Latency={lat:.4f}ms  Train={train_time:.2f}s")

    # ── 4. MLP ────────────────────────────────────────────────────────────────
    print("\n[4/5] Training MLP...")
    mlp = MLPDecoder(cfg["n_channels"], cfg["n_classes"])
    losses_mlp, train_time = train_torch(
        mlp, Xf_tr, y_tr, cfg["epochs"], cfg["batch_size"], device
    )
    acc, cm, probs, lat = eval_torch(mlp, Xf_te, y_te, device)
    results["MLP"] = dict(acc=acc, latency_ms=lat, train_time=train_time,
                          cm=cm, probs=probs, losses=losses_mlp)
    print(f"  Acc={acc:.3f}  Latency={lat:.4f}ms  Train={train_time:.2f}s")

    # ── 5. LSTM ───────────────────────────────────────────────────────────────
    print("\n[5/5] Training LSTM...")
    lstm = LSTMDecoder(cfg["n_channels"], hidden_size=128, n_layers=2,
                       n_classes=cfg["n_classes"])
    losses_lstm, train_time = train_torch(
        lstm, Xs_tr, y_tr, cfg["epochs"], cfg["batch_size"], device
    )
    acc, cm, probs, lat = eval_torch(lstm, Xs_te, y_te, device)
    results["LSTM"] = dict(acc=acc, latency_ms=lat, train_time=train_time,
                           cm=cm, probs=probs, losses=losses_lstm)
    print(f"  Acc={acc:.3f}  Latency={lat:.4f}ms  Train={train_time:.2f}s")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_comparison(results, y_te, cfg["n_classes"])
    _plot_dynamics(results, y_te, cfg["n_classes"])


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _plot_comparison(results: dict, y_te: np.ndarray, n_classes: int):
    """Figure 1: trade-off scatter, bar charts, confusion matrices."""
    names = list(results.keys())
    accs  = [results[n]["acc"]        for n in names]
    lats  = [results[n]["latency_ms"] for n in names]
    trains= [results[n]["train_time"] for n in names]

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle("Neural Decoder — Model Comparison", fontsize=16, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

    # ── 1. Accuracy vs Latency scatter (Pareto front) ─────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    for i, name in enumerate(names):
        ax1.scatter(lats[i], accs[i], s=220, color=PALETTE[i], zorder=5,
                    label=name, edgecolors="white", linewidths=1.5)
        ax1.annotate(name, (lats[i], accs[i]),
                     textcoords="offset points", xytext=(10, 4), fontsize=10)

    # Pareto front: models not dominated in both dimensions
    sorted_models = sorted(zip(lats, accs), key=lambda x: x[0])
    px, py, best_acc = [], [], -1
    for lat, acc in sorted_models:
        if acc > best_acc:
            px.append(lat); py.append(acc); best_acc = acc
    if len(px) > 1:
        ax1.step(px, py, where="post", color="gray", linestyle="--",
                 alpha=0.55, label="Pareto front", linewidth=1.5)

    mid_lat = float(np.median(lats))
    mid_acc = float(np.median(accs))
    ax1.axhline(mid_acc, color="gray", alpha=0.18, linewidth=0.9)
    ax1.axvline(mid_lat, color="gray", alpha=0.18, linewidth=0.9)
    ax1.fill_betweenx([mid_acc, 1.05], 0, mid_lat,
                      alpha=0.06, color="green", label="_nolegend_")
    ax1.text(0.02, (mid_acc + 1.05) / 2, "OPTIMAL\nZONE",
             fontsize=9, color="green", alpha=0.7, va="center",
             transform=ax1.get_yaxis_transform())

    ax1.set_xlabel("Single-sample inference latency (ms)", fontsize=11)
    ax1.set_ylabel("Test accuracy", fontsize=11)
    ax1.set_title("Speed vs Accuracy Trade-off  (upper-left = ideal)", fontsize=12)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_ylim(0, 1.08)
    ax1.grid(True, alpha=0.3)

    # ── 2. Accuracy bar chart ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    bars = ax2.bar(names, accs, color=PALETTE, edgecolor="white", linewidth=1.2)
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Test Accuracy", fontsize=12)
    ax2.tick_params(axis="x", rotation=30)
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{acc:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    # ── 3. Latency bar chart ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    bars2 = ax3.bar(names, lats, color=PALETTE, edgecolor="white", linewidth=1.2)
    ax3.set_ylabel("Latency (ms)", fontsize=11)
    ax3.set_title("Single-Sample Inference Latency\n(lower = faster decoding)", fontsize=11)
    ax3.tick_params(axis="x", rotation=30)
    for bar, lat in zip(bars2, lats):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                 f"{lat:.3f}", ha="center", va="bottom", fontsize=9)
    ax3.grid(True, axis="y", alpha=0.3)

    # ── 4. Training time bar chart ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    bars3 = ax4.bar(names, trains, color=PALETTE, edgecolor="white", linewidth=1.2)
    ax4.set_ylabel("Seconds", fontsize=11)
    ax4.set_title("Training Time", fontsize=12)
    ax4.tick_params(axis="x", rotation=30)
    for bar, t in zip(bars3, trains):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                 f"{t:.1f}s", ha="center", va="bottom", fontsize=9)
    ax4.grid(True, axis="y", alpha=0.3)

    # ── 5. Composite score  acc / log(1 + lat) ────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    raw_scores = [acc / np.log1p(lat) for acc, lat in zip(accs, lats)]
    scores_norm = np.array(raw_scores) / max(raw_scores)
    bars4 = ax5.bar(names, scores_norm, color=PALETTE, edgecolor="white", linewidth=1.2)
    best_i = int(np.argmax(scores_norm))
    bars4[best_i].set_edgecolor("gold")
    bars4[best_i].set_linewidth(3)
    ax5.set_ylim(0, 1.18)
    ax5.set_ylabel("Score (normalised)", fontsize=11)
    ax5.set_title("Composite Score\nacc / log(1 + latency_ms)", fontsize=11)
    ax5.tick_params(axis="x", rotation=30)
    for bar, s in zip(bars4, scores_norm):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{s:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax5.text(0.5, 1.10, f"★  Best: {names[best_i]}",
             ha="center", va="center", transform=ax5.transAxes,
             fontsize=10, color="goldenrod", fontweight="bold")
    ax5.grid(True, axis="y", alpha=0.3)

    # ── 6–8. Confusion matrices (top 3 by accuracy) ───────────────────────────
    top3 = sorted(range(len(accs)), key=lambda i: accs[i], reverse=True)[:3]
    for col, mi in enumerate(top3):
        ax = fig.add_subplot(gs[2, col])
        cm = results[names[mi]]["cm"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Confusion Matrix\n{names[mi]}  (acc={accs[mi]:.3f})", fontsize=10)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ticks = range(n_classes)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for r in range(n_classes):
            for c in range(n_classes):
                val = cm_norm[r, c]
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if val > 0.5 else "black")

    fig.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    print("\nSaved → model_comparison.png")


def _plot_dynamics(results: dict, y_te: np.ndarray, n_classes: int):
    """Figure 2: training loss curves, macro ROC, ITR bar chart."""
    names = list(results.keys())
    accs  = [results[n]["acc"]        for n in names]
    lats  = [results[n]["latency_ms"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("Training Dynamics & BCI Performance Metrics",
                 fontsize=14, fontweight="bold", y=1.02)

    # ── 1. Learning curves (PyTorch models only) ──────────────────────────────
    ax = axes[0]
    for name, color in zip(["MLP", "LSTM"], PALETTE[3:]):
        losses = results[name]["losses"]
        if losses:
            ax.plot(losses, label=name, color=color, linewidth=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross-entropy loss", fontsize=11)
    ax.set_title("Training Loss Curves\n(MLP & LSTM)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── 2. Macro-average ROC ──────────────────────────────────────────────────
    ax = axes[1]
    y_bin = label_binarize(y_te, classes=list(range(n_classes)))
    for i, name in enumerate(names):
        probs = results[name]["probs"]
        if probs is None:
            continue
        per_class_fpr, per_class_tpr = {}, {}
        macro_auc_vals = []
        for c in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
            per_class_fpr[c] = fpr
            per_class_tpr[c] = tpr
            macro_auc_vals.append(auc(fpr, tpr))
        macro_auc = float(np.mean(macro_auc_vals))
        all_fpr = np.unique(np.concatenate(list(per_class_fpr.values())))
        mean_tpr = np.mean(
            [np.interp(all_fpr, per_class_fpr[c], per_class_tpr[c]) for c in range(n_classes)],
            axis=0,
        )
        ax.plot(all_fpr, mean_tpr, color=PALETTE[i], lw=2,
                label=f"{name}  (AUC={macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_title("Macro-Average ROC Curves", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    # ── 3. Information Transfer Rate ─────────────────────────────────────────
    ax = axes[2]
    itrs = [itr_bits_per_min(acc, lat, n_classes) for acc, lat in zip(accs, lats)]
    bars = ax.bar(names, itrs, color=PALETTE, edgecolor="white", linewidth=1.2)
    best_i = int(np.argmax(itrs))
    bars[best_i].set_edgecolor("gold")
    bars[best_i].set_linewidth(3)
    ax.set_ylabel("ITR (bits / min)", fontsize=11)
    ax.set_title("Information Transfer Rate\nbalances accuracy AND speed", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    for bar, itr in zip(bars, itrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{itr:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(0.5, 0.97, f"★  Best ITR: {names[best_i]}",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=10, color="goldenrod", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig("model_comparison_dynamics.png", dpi=150, bbox_inches="tight")
    print("Saved → model_comparison_dynamics.png")
    plt.show()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare neural decoder models")
    parser.add_argument("--n-classes",  type=int, default=DEFAULTS["n_classes"])
    parser.add_argument("--n-channels", type=int, default=DEFAULTS["n_channels"])
    parser.add_argument("--n-seq",      type=int, default=DEFAULTS["n_seq"])
    parser.add_argument("--n-samples",  type=int, default=DEFAULTS["n_samples"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--epochs",     type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--seed",       type=int, default=DEFAULTS["seed"])
    args = parser.parse_args()

    cfg = {k.replace("-", "_"): v for k, v in vars(args).items()}
    main(cfg)
