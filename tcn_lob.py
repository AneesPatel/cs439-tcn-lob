import warnings
warnings.filterwarnings("ignore")
import random

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    jaccard_score,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


def generate_lob_data(
    n_snapshots=50_000,
    n_levels=5,
    tick_size=0.01,
    mid_price_init=100.0,
    price_vol=0.008,
    imbalance_signal_strength=0.025,
    imbalance_ar_rho=0.75,
    seed=SEED,
):
    rng = np.random.default_rng(seed)

    imbalance_series = np.empty(n_snapshots, dtype=np.float32)
    I = 0.0
    for t in range(n_snapshots):
        I = imbalance_ar_rho * I + rng.normal(0.0, np.sqrt(1 - imbalance_ar_rho**2))
        imbalance_series[t] = float(np.clip(I, -1.0, 1.0))

    mid_prices = np.empty(n_snapshots, dtype=np.float32)
    mid_prices[0] = mid_price_init
    idio_noise = rng.normal(0.0, price_vol, size=n_snapshots).astype(np.float32)
    for t in range(1, n_snapshots):
        mid_prices[t] = (
            mid_prices[t-1]
            + imbalance_signal_strength * imbalance_series[t-1]
            + idio_noise[t]
        )
    mid_prices = np.maximum(mid_prices, tick_size)

    raw_features = np.empty((n_snapshots, 4 * n_levels), dtype=np.float32)
    bid_vols_top = np.empty(n_snapshots, dtype=np.float32)
    ask_vols_top = np.empty(n_snapshots, dtype=np.float32)

    base_vol = 5.0
    vol_noise_sd = 1.5

    for t in range(n_snapshots):
        m = mid_prices[t]
        imb = imbalance_series[t]

        spread_scale = 1.0 + 0.5 * max(0.0, float(-imb))
        bid_spacings = np.abs(rng.normal(0.0, tick_size * spread_scale, size=n_levels)) + tick_size
        ask_spacings = np.abs(rng.normal(0.0, tick_size, size=n_levels)) + tick_size
        bid_prices = m - np.cumsum(bid_spacings)
        ask_prices = m + np.cumsum(ask_spacings)

        bid_vols = np.abs(rng.normal(base_vol * (1 + 0.6 * imb), vol_noise_sd, size=n_levels)).astype(np.float32)
        ask_vols = np.abs(rng.normal(base_vol * (1 - 0.6 * imb), vol_noise_sd, size=n_levels)).astype(np.float32)

        for lvl in range(n_levels):
            base = lvl * 2
            raw_features[t, base] = bid_prices[lvl]
            raw_features[t, base + 1] = bid_vols[lvl]
            base_ask = n_levels * 2 + lvl * 2
            raw_features[t, base_ask] = ask_prices[lvl]
            raw_features[t, base_ask + 1] = ask_vols[lvl]

        bid_vols_top[t] = bid_vols[0]
        ask_vols_top[t] = ask_vols[0]

    best_bid_col = 0
    best_ask_col = n_levels * 2

    spread = (raw_features[:, best_ask_col] - raw_features[:, best_bid_col]).reshape(-1, 1)

    denom = bid_vols_top + ask_vols_top + 1e-8
    imbalance = ((bid_vols_top - ask_vols_top) / denom).reshape(-1, 1)

    vw_mid = (
        (bid_vols_top * raw_features[:, best_ask_col] +
         ask_vols_top * raw_features[:, best_bid_col]) / denom
    ).reshape(-1, 1)

    X = np.concatenate([raw_features, spread, imbalance, vw_mid], axis=1)
    return X.astype(np.float32), mid_prices.astype(np.float32)


def build_labels(mid_prices, horizon=5, alpha=None):
    T = len(mid_prices)
    returns = mid_prices[horizon:] - mid_prices[:T - horizon]

    if alpha is None:
        alpha = np.percentile(np.abs(returns), 33)

    labels = np.where(
        returns > alpha, 2,
        np.where(returns < -alpha, 0, 1)
    ).astype(np.int64)

    print(f"[INFO] Label threshold alpha = {alpha:.5f}")
    unique, counts = np.unique(labels, return_counts=True)
    class_names = {0: "Down", 1: "Stationary", 2: "Up"}
    for u, c in zip(unique, counts):
        print(f"       {class_names[u]:>12s}: {c:6d}  ({100*c/len(labels):.1f}%)")

    return labels


def temporal_split(X, y, train_frac=0.70, val_frac=0.15):
    N = len(X)
    i_val = int(N * train_frac)
    i_test = int(N * (train_frac + val_frac))

    X_tr, y_tr = X[:i_val], y[:i_val]
    X_va, y_va = X[i_val:i_test], y[i_val:i_test]
    X_te, y_te = X[i_test:], y[i_test:]

    print(f"[INFO] Split sizes -- train: {len(X_tr)}, val: {len(X_va)}, test: {len(X_te)}")
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def normalize(X_tr, X_va, X_te):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)
    return X_tr_s, X_va_s, X_te_s, scaler


def make_windows(X, y, window_len=50):
    T, F = X.shape
    n_windows = T - window_len + 1
    X_win = np.lib.stride_tricks.sliding_window_view(X, (window_len, F))
    X_win = X_win[:, 0, :, :]
    y_win = y[window_len - 1:]
    assert len(X_win) == len(y_win) == n_windows
    return X_win.astype(np.float32), y_win


def to_dataloader(X_win, y_win, batch_size=128, shuffle=False):
    X_t = torch.from_numpy(X_win).permute(0, 2, 1)
    y_t = torch.from_numpy(y_win)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class _TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.3):
        super().__init__()
        pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def _trim(self, x, ref_len):
        return x[:, :, :ref_len]

    def forward(self, x):
        L = x.size(2)
        out = self.relu(self.bn1(self._trim(self.conv1(x), L)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self._trim(self.conv2(out), L)))
        out = self.dropout(out)
        return self.relu(out + self.residual_proj(x))


class TCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, kernel_size=3, dilations=None, n_classes=3, dropout=0.3):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32]

        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        blocks = []
        for d in dilations:
            blocks.append(_TCNBlock(hidden_channels, hidden_channels, kernel_size, d, dropout))
        self.tcn_blocks = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        h = self.input_proj(x)
        h = self.tcn_blocks(h)
        z = h[:, :, -1]
        return self.classifier(z)


class LSTMModel(nn.Module):
    def __init__(self, in_features, hidden_size=128, n_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])


def compute_class_weights(y_train, n_classes=3):
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = len(y_train) / (n_classes * counts)
    return torch.from_numpy(weights).to(DEVICE)


def train_model(model, train_loader, val_loader, class_weights, n_epochs=60, lr=1e-3, patience=15, lr_patience=7):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=lr_patience)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    val_losses = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss_sum += criterion(model(xb), yb).item() * len(yb)
        val_loss = val_loss_sum / len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | val_loss={val_loss:.4f} | best={best_val_loss:.4f} | patience={epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"[INFO] Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return val_losses


CLASS_NAMES = ["Down", "Stationary", "Up"]


def evaluate_model(model, test_loader, model_name="TCN"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(yb.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    mean_iou = jaccard_score(y_true, y_pred, average="macro")
    acc = (y_true == y_pred).mean()

    print(f"\n{'='*60}")
    print(f"  {model_name} Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Macro-F1   : {macro_f1:.4f}")
    print(f"  Mean IoU   : {mean_iou:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES)}")

    return y_true, y_pred


def plot_training_curve(val_losses, model_name="TCN-XLarge", path="training_curve.png"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(val_losses) + 1), val_losses, color="#4a90d9", linewidth=1.8)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation Cross-Entropy Loss", fontsize=11)
    ax.set_title(f"{model_name} -- Validation Loss Curve", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Training curve saved -> {path}")


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Confusion matrix saved -> {path}")


def extract_embeddings(model, test_loader):
    embeddings, labels_out = [], []
    captured = {}

    def _hook(module, inp, out):
        captured["embed"] = out.detach().cpu().numpy()

    handle = model.classifier[1].register_forward_hook(_hook)

    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            model(xb)
            embeddings.append(captured["embed"])
            labels_out.append(yb.numpy())

    handle.remove()
    return np.concatenate(embeddings), np.concatenate(labels_out)


def plot_pca_embeddings(embeddings, labels, path="pca_embeddings.png"):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=SEED)
    proj = pca.fit_transform(embeddings)
    ev = pca.explained_variance_ratio_

    palette = {0: "#e05252", 1: "#f5a623", 2: "#4a90d9"}

    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, name in zip([0, 1, 2], CLASS_NAMES):
        mask = labels == cls
        ax.scatter(proj[mask, 0], proj[mask, 1], c=palette[cls], s=6, alpha=0.5, label=name)

    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
    ax.set_title("PCA of TCN Learned Embeddings (Test Set)", fontsize=13)
    ax.legend(markerscale=3, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] PCA plot saved -> {path}")


def plot_shap_importance(model, X_te_win, feature_dim, n_background=50, n_explain=100, path="shap_importance.png"):
    try:
        import shap
    except ImportError:
        print("[WARN] shap not installed. Skipping SHAP plot.")
        return

    X_flat = X_te_win.mean(axis=1)
    bg = X_flat[:n_background]
    exp = X_flat[:n_explain]

    def predict_fn(x):
        x_t = torch.from_numpy(x).float().unsqueeze(2).to(DEVICE)
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(x_t), dim=-1).cpu().numpy()
        return probs

    explainer = shap.KernelExplainer(predict_fn, bg)
    shap_values = explainer.shap_values(exp, nsamples=30, silent=True)

    n_levels = (feature_dim - 3) // 4
    feat_names = []
    for i in range(1, n_levels + 1):
        feat_names += [f"BidPx_{i}", f"BidVol_{i}"]
    for i in range(1, n_levels + 1):
        feat_names += [f"AskPx_{i}", f"AskVol_{i}"]
    feat_names += ["Spread", "Imbalance", "VWMid"]

    stacked = np.stack([np.abs(sv) for sv in shap_values], axis=0)
    mean_abs_shap = stacked.mean(axis=(0, 1))

    n_top = min(15, len(feat_names))
    order = np.argsort(mean_abs_shap)[::-1][:n_top]
    plot_order = order[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [feat_names[i] for i in plot_order],
        mean_abs_shap[plot_order],
        color="#4a90d9"
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"SHAP Feature Importance (Top {n_top} Features)", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] SHAP importance plot saved -> {path}")


DILATION_CONFIGS = {
    "TCN-Small":  [1, 2],
    "TCN-Medium": [1, 2, 4],
    "TCN-Large":  [1, 2, 4, 8],
    "TCN-XLarge": [1, 2, 4, 8, 16, 32],
}


def run_ablation(train_loader, val_loader, test_loader, class_weights, in_channels):
    results = {}
    best_model = None
    best_val_losses = []

    for name, dilations in DILATION_CONFIGS.items():
        rf = 1 + sum((3 - 1) * d for d in dilations)
        print(f"\n[ABLATION] {name} | dilations={dilations} | RF={rf}")
        model = TCNModel(
            in_channels=in_channels,
            hidden_channels=64,
            kernel_size=3,
            dilations=dilations,
            dropout=0.3,
        ).to(DEVICE)
        val_losses = train_model(model, train_loader, val_loader, class_weights, n_epochs=60, patience=15)
        y_true, y_pred = evaluate_model(model, test_loader, model_name=name)
        results[name] = {
            "accuracy": float((y_true == y_pred).mean()),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "mean_iou": float(jaccard_score(y_true, y_pred, average="macro")),
            "rf": rf,
        }
        if name == "TCN-XLarge":
            best_model = model
            best_val_losses = val_losses

    return results, best_model, best_val_losses


def print_ablation_table(results):
    print(f"\n{'='*70}")
    print(f"  {'Model':<15} {'RF':>12} {'Accuracy':>10} {'Macro-F1':>10} {'Mean IoU':>10}")
    print(f"  {'-'*65}")
    for name, r in results.items():
        print(f"  {name:<15} {r['rf']:>12d} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['mean_iou']:>10.4f}")
    print(f"{'='*70}")


def plot_ablation(results, path="ablation_f1.png"):
    names = list(results.keys())
    f1s = [results[n]["macro_f1"] for n in names]
    rfs = [results[n]["rf"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, f1s, color="#4a90d9", edgecolor="white", linewidth=0.5)
    for bar, rf in zip(bars, rfs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"RF={rf}",
            ha="center", va="bottom", fontsize=9
        )
    ax.set_ylabel("Macro-averaged F1", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Ablation Study: Dilation Schedule vs. Macro-F1", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Ablation bar chart saved -> {path}")


def run_sklearn_baselines(X_tr_win, y_tr, X_te_win, y_te):
    X_tr_flat = X_tr_win.reshape(len(X_tr_win), -1)
    X_te_flat = X_te_win.reshape(len(X_te_win), -1)

    majority = int(np.bincount(y_tr).argmax())
    y_maj = np.full(len(y_te), majority)
    print("\n[BASELINE] Majority Class")
    print(classification_report(y_te, y_maj, target_names=CLASS_NAMES, zero_division=0))

    n_sub = min(5000, len(X_tr_flat))
    idx = np.random.choice(len(X_tr_flat), n_sub, replace=False)
    lr_model = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", n_jobs=-1)
    lr_model.fit(X_tr_flat[idx], y_tr[idx])
    y_lr = lr_model.predict(X_te_flat)
    print("[BASELINE] Logistic Regression")
    print(classification_report(y_te, y_lr, target_names=CLASS_NAMES))


def main():
    print("\n[STEP 1] Generating synthetic LOB data ...")
    X, mid_prices = generate_lob_data(n_snapshots=50_000)
    print(f"         X shape: {X.shape}")

    print("\n[STEP 2] Constructing directional labels ...")
    HORIZON = 5
    y = build_labels(mid_prices, horizon=HORIZON)
    X = X[:len(y)]

    print("\n[STEP 3] Temporal train / val / test split ...")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = temporal_split(X, y)

    print("\n[STEP 4] Normalizing (fit on train only -- no leakage) ...")
    X_tr_s, X_va_s, X_te_s, scaler = normalize(X_tr, X_va, X_te)

    print("\n[STEP 5] Creating sliding windows (L=50) ...")
    WINDOW = 50
    X_tr_win, y_tr_win = make_windows(X_tr_s, y_tr, window_len=WINDOW)
    X_va_win, y_va_win = make_windows(X_va_s, y_va, window_len=WINDOW)
    X_te_win, y_te_win = make_windows(X_te_s, y_te, window_len=WINDOW)
    print(f"         Train windows: {X_tr_win.shape}, Val: {X_va_win.shape}, Test: {X_te_win.shape}")

    train_loader = to_dataloader(X_tr_win, y_tr_win, shuffle=True)
    val_loader = to_dataloader(X_va_win, y_va_win)
    test_loader = to_dataloader(X_te_win, y_te_win)

    class_weights = compute_class_weights(y_tr_win)
    print(f"[INFO] Class weights: {class_weights.cpu().numpy()}")

    print("\n[STEP 8] Running sklearn baselines ...")
    run_sklearn_baselines(X_tr_win, y_tr_win, X_te_win, y_te_win)

    print("\n[STEP 9] Training LSTM baseline ...")
    in_ch = X_tr_win.shape[2]
    lstm = LSTMModel(in_features=in_ch).to(DEVICE)
    train_model(lstm, train_loader, val_loader, class_weights, n_epochs=60, patience=15)
    evaluate_model(lstm, test_loader, model_name="LSTM")

    print("\n[STEP 10] Running TCN ablation study ...")
    ablation_results, best_model, best_val_losses = run_ablation(
        train_loader, val_loader, test_loader, class_weights, in_ch
    )
    print_ablation_table(ablation_results)
    plot_ablation(ablation_results)

    print("\n[STEP 11] Full evaluation of TCN-XLarge ...")
    plot_training_curve(best_val_losses, model_name="TCN-XLarge")
    y_true, y_pred = evaluate_model(best_model, test_loader, model_name="TCN-XLarge")
    plot_confusion_matrix(y_true, y_pred, title="TCN-XLarge Confusion Matrix")

    print("\n[STEP 12] PCA visualisation of learned embeddings ...")
    embeddings, emb_labels = extract_embeddings(best_model, test_loader)
    plot_pca_embeddings(embeddings, emb_labels)

    print("\n[STEP 13] SHAP feature importance (may take a few minutes) ...")
    plot_shap_importance(best_model, X_te_win, feature_dim=in_ch)

    print("\n[DONE] All outputs saved to the current directory.")


if __name__ == "__main__":
    main()
