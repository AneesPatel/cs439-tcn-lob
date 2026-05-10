# -*- coding: utf-8 -*-
"""
tcn_lob.py
==========
High-Frequency Mid-Price Prediction in Limit Order Books
using Temporal Convolutional Networks (TCNs).

Fully self-contained: no external CSVs required.
Requires: torch, scikit-learn, numpy, matplotlib, seaborn, shap

Install deps:
    pip install torch scikit-learn numpy matplotlib seaborn shap
"""

# ─── Standard library ────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
import random

# ─── Numerical / ML ──────────────────────────────────────────────────────────
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

# ─── Deep learning ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── Visualisation ───────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # headless backend -- saves PNGs without a display
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ═════════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC LIMIT ORDER BOOK DATA GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

def generate_lob_data(
    n_snapshots: int = 50_000,
    n_levels: int = 5,
    tick_size: float = 0.01,
    mid_price_init: float = 100.0,
    price_vol: float = 0.008,
    imbalance_signal_strength: float = 0.012,
    imbalance_ar_rho: float = 0.75,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic LOB snapshots with embedded microstructure signal.

    SIGNAL DESIGN
    -------------
    A pure random-walk mid-price would contain zero predictive information,
    making it impossible to meaningfully compare model architectures.  We
    therefore embed a known microstructure mechanism into the generator:
    order-flow imbalance causally drives future price direction, consistent
    with Cont & de Larrard (2013) and the empirical findings of Cartea et
    al. (2015).  Specifically:

        I_t = rho * I_{t-1} + eta_t          (AR(1) imbalance, rho=0.75)
        m_{t+1} = m_t + gamma * I_t + eps_t  (imbalance-driven price)

    where gamma=0.012 >> sigma_eps=0.008 ensures a favourable signal-to-noise
    ratio.  The AR(1) structure means imbalance is autocorrelated over ~4
    time-steps, giving the TCN's receptive field a real target to exploit.

    Parameters
    ----------
    n_snapshots              : total time-steps
    n_levels                 : LOB depth per side (5 is standard)
    tick_size                : minimum price increment
    mid_price_init           : starting mid-price
    price_vol                : std-dev of idiosyncratic price noise (eps)
    imbalance_signal_strength: causal coefficient gamma
    imbalance_ar_rho         : AR(1) autocorrelation of imbalance

    Returns
    -------
    X : (n_snapshots, F) feature matrix  (F = 23 for n_levels=5)
    m : (n_snapshots,)  mid-price series
    """
    rng = np.random.default_rng(seed)

    # --- AR(1) order-imbalance process ---------------------------------------
    # I_t in [-1, 1]: positive means more bid-side volume (bullish pressure).
    imbalance_series = np.empty(n_snapshots, dtype=np.float32)
    I = 0.0
    for t in range(n_snapshots):
        I = imbalance_ar_rho * I + rng.normal(0.0, np.sqrt(1 - imbalance_ar_rho**2))
        imbalance_series[t] = float(np.clip(I, -1.0, 1.0))

    # --- Imbalance-driven mid-price process ----------------------------------
    # m_{t+1} = m_t + gamma * I_t + eps_t
    # The causal dependency gives models real signal to exploit.
    mid_prices = np.empty(n_snapshots, dtype=np.float32)
    mid_prices[0] = mid_price_init
    idio_noise = rng.normal(0.0, price_vol, size=n_snapshots).astype(np.float32)
    for t in range(1, n_snapshots):
        mid_prices[t] = mid_prices[t-1] + imbalance_signal_strength * imbalance_series[t-1] + idio_noise[t]
    mid_prices = np.maximum(mid_prices, tick_size)

    # --- LOB snapshot construction -------------------------------------------
    # Volumes are correlated with imbalance so the feature carries signal:
    # high imbalance_series[t] -> large bid_vol, small ask_vol.
    raw_features = np.empty((n_snapshots, 4 * n_levels), dtype=np.float32)
    bid_vols_top = np.empty(n_snapshots, dtype=np.float32)
    ask_vols_top = np.empty(n_snapshots, dtype=np.float32)

    base_vol     = 5.0
    vol_noise_sd = 1.5

    for t in range(n_snapshots):
        m   = mid_prices[t]
        imb = imbalance_series[t]

        bid_spacings = np.abs(rng.normal(0.0, tick_size, size=n_levels)) + tick_size
        ask_spacings = np.abs(rng.normal(0.0, tick_size, size=n_levels)) + tick_size
        bid_prices   = m - np.cumsum(bid_spacings)
        ask_prices   = m + np.cumsum(ask_spacings)

        # Volumes tilted by imbalance: positive imb -> higher bids, lower asks
        bid_vols = np.abs(rng.normal(base_vol * (1 + 0.6 * imb), vol_noise_sd, size=n_levels)).astype(np.float32)
        ask_vols = np.abs(rng.normal(base_vol * (1 - 0.6 * imb), vol_noise_sd, size=n_levels)).astype(np.float32)

        for lvl in range(n_levels):
            base = lvl * 2
            raw_features[t, base]     = bid_prices[lvl]
            raw_features[t, base + 1] = bid_vols[lvl]
            base_ask = n_levels * 2 + lvl * 2
            raw_features[t, base_ask]     = ask_prices[lvl]
            raw_features[t, base_ask + 1] = ask_vols[lvl]

        bid_vols_top[t] = bid_vols[0]
        ask_vols_top[t] = ask_vols[0]

    # --- Derived features ----------------------------------------------------
    best_bid_col = 0            # bid price, level 1
    best_ask_col = n_levels * 2  # ask price, level 1

    spread = (raw_features[:, best_ask_col] - raw_features[:, best_bid_col]).reshape(-1, 1)

    # Realised order imbalance from the volumes (recoverable by the model)
    denom     = bid_vols_top + ask_vols_top + 1e-8
    imbalance = ((bid_vols_top - ask_vols_top) / denom).reshape(-1, 1)

    vw_mid = (
        (bid_vols_top * raw_features[:, best_ask_col] +
         ask_vols_top * raw_features[:, best_bid_col]) / denom
    ).reshape(-1, 1)

    X = np.concatenate([raw_features, spread, imbalance, vw_mid], axis=1)
    return X.astype(np.float32), mid_prices.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# 2. LABEL CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════

def build_labels(
    mid_prices: np.ndarray,
    horizon: int = 5,
    alpha: float | None = None,
) -> np.ndarray:
    """
    Assign three-class directional labels based on future mid-price movement.

    Label encoding:
        0 -> Down       (m_{t+h} - m_t < -alpha)
        1 -> Stationary (|m_{t+h} - m_t| <= alpha)
        2 -> Up         (m_{t+h} - m_t > +alpha)

    Parameters
    ----------
    mid_prices : raw mid-price series of length T
    horizon    : look-ahead steps k
    alpha      : movement threshold; defaults to 33rd percentile of |returns|

    Returns
    -------
    labels : integer array of length T - horizon
    """
    T       = len(mid_prices)
    returns = mid_prices[horizon:] - mid_prices[:T - horizon]   # forward returns

    if alpha is None:
        alpha = np.percentile(np.abs(returns), 33)

    labels = np.where(
        returns > alpha, 2,           # Up
        np.where(returns < -alpha, 0, # Down
                 1)                   # Stationary
    ).astype(np.int64)

    print(f"[INFO] Label threshold alpha = {alpha:.5f}")
    unique, counts = np.unique(labels, return_counts=True)
    class_names = {0: "Down", 1: "Stationary", 2: "Up"}
    for u, c in zip(unique, counts):
        print(f"       {class_names[u]:>12s}: {c:6d}  ({100*c/len(labels):.1f}%)")

    return labels


# ═════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING -- NORMALIZATION & STRICT TEMPORAL SPLIT
# ═════════════════════════════════════════════════════════════════════════════

def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> tuple:
    """
    Split (X, y) in strict chronological order to prevent data leakage.

    WHY TEMPORAL ORDER MATTERS
    --------------------------
    Shuffling a time series before splitting allows future observations to
    appear in the training set.  A model trained on shuffled data can
    implicitly memorise future states, yielding unrealistically optimistic
    test metrics that will not generalise to live data.  Chronological
    splitting is the only valid protocol for time-series evaluation.

    Returns
    -------
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te)
    """
    N      = len(X)
    i_val  = int(N * train_frac)
    i_test = int(N * (train_frac + val_frac))

    X_tr, y_tr = X[:i_val],        y[:i_val]
    X_va, y_va = X[i_val:i_test],  y[i_val:i_test]
    X_te, y_te = X[i_test:],       y[i_test:]

    print(f"[INFO] Split sizes -- train: {len(X_tr)}, val: {len(X_va)}, test: {len(X_te)}")
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def normalize(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    X_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on training data ONLY and apply to all splits.

    DATA LEAKAGE PREVENTION
    -----------------------
    Fitting the scaler on the full dataset before splitting would encode
    test-set statistics (mean and variance) into the normalisation
    parameters used during training -- a subtle but impactful leakage.
    By calling `fit` exclusively on X_tr and `transform` on X_va / X_te,
    we ensure the model never observes any information from the future.
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)     # fit + transform on train only
    X_va_s = scaler.transform(X_va)         # transform only -- no leakage
    X_te_s = scaler.transform(X_te)         # transform only -- no leakage
    return X_tr_s, X_va_s, X_te_s, scaler


def make_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_len: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice a (T, F) sequence into overlapping (window_len, F) windows.

    The label for window [t-L+1 : t+1] is y[t] (label at the last timestep).
    Stride = 1 maximises sample count at the cost of temporal correlation
    between consecutive windows -- acceptable given our strict temporal split.
    """
    T, F = X.shape
    n_windows = T - window_len + 1
    X_win = np.lib.stride_tricks.sliding_window_view(X, (window_len, F))
    X_win = X_win[:, 0, :, :]   # shape: (n_windows, window_len, F)
    y_win = y[window_len - 1:]   # label at last position in each window
    assert len(X_win) == len(y_win) == n_windows
    return X_win.astype(np.float32), y_win


def to_dataloader(
    X_win: np.ndarray,
    y_win: np.ndarray,
    batch_size: int = 128,
    shuffle: bool = False,
) -> DataLoader:
    # (N, L, F) -> (N, F, L): PyTorch Conv1d expects (batch, channels, length)
    X_t = torch.from_numpy(X_win).permute(0, 2, 1)
    y_t = torch.from_numpy(y_win)
    ds  = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ═════════════════════════════════════════════════════════════════════════════
# 4. MODEL DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

class _TCNBlock(nn.Module):
    """
    One residual TCN block: two dilated causal Conv1d layers with BatchNorm
    and ReLU, plus a residual (skip) connection.

    Causal padding: for dilation d and kernel size k, we left-pad by
    (k-1)*d samples so the output length matches the input length and
    no future timestep is visible (causal mask).
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        dilation:     int,
        dropout:      float = 0.1,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation  # left-padding for causality

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=pad
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation, padding=pad
        )
        self.bn1     = nn.BatchNorm1d(out_channels)
        self.bn2     = nn.BatchNorm1d(out_channels)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1×1 convolution to project residual when channel dims differ
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def _trim(self, x: torch.Tensor, ref_len: int) -> torch.Tensor:
        """Remove excess right-padding introduced by causal padding."""
        return x[:, :, :ref_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L   = x.size(2)   # sequence length
        out = self.relu(self.bn1(self._trim(self.conv1(x), L)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self._trim(self.conv2(out), L)))
        out = self.dropout(out)
        return self.relu(out + self.residual_proj(x))


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for three-class LOB mid-price classification.

    Architecture
    ------------
    1. Input projection  : Conv1d(F -> hidden_channels, kernel=1)
    2. TCN blocks        : one block per dilation in `dilations`
    3. Global last-step  : take the hidden state at position t = L-1
    4. Classification    : Linear -> ReLU -> Linear -> 3 logits

    Dilated causal convolution (Equation 1 in the paper):
        y_t = Σ_{i=0}^{k-1} f(i) · x_{t - d·i}

    The dilation factor d doubles with each successive block, giving a
    receptive field of  1 + (k-1) * Σ d_m  that grows exponentially.
    """

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int = 64,
        kernel_size:     int = 3,
        dilations:       list[int] | None = None,
        n_classes:       int = 3,
        dropout:         float = 0.1,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32]

        # Input projection: maps F feature channels -> hidden_channels
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        # Stack of TCN residual blocks with increasing dilation
        blocks = []
        for d in dilations:
            blocks.append(
                _TCNBlock(hidden_channels, hidden_channels, kernel_size, d, dropout)
            )
        self.tcn_blocks = nn.Sequential(*blocks)

        # Classification MLP applied to the last temporal position
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, F, L)  -- channels-first format for Conv1d

        Returns
        -------
        logits : (batch, n_classes)
        """
        h = self.input_proj(x)     # (B, C_h, L)
        h = self.tcn_blocks(h)     # (B, C_h, L)
        z = h[:, :, -1]            # (B, C_h)  -- last temporal position
        return self.classifier(z)  # (B, 3)


class LSTMModel(nn.Module):
    """Single-layer LSTM baseline with identical classification head."""

    def __init__(self, in_features: int, hidden_size: int = 128, n_classes: int = 3):
        super().__init__()
        self.lstm       = nn.LSTM(in_features, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, F, L) from DataLoader; LSTM expects (B, L, F)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])


# ═════════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP WITH EARLY STOPPING & CLASS-WEIGHT BALANCING
# ═════════════════════════════════════════════════════════════════════════════

def compute_class_weights(y_train: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    """
    Inverse-frequency weighting to counter Stationary-class dominance.

    w_c = total_samples / (n_classes * count_c)

    Without balancing, the model tends to predict the majority class
    (Stationary) for most inputs, collapsing macro-F1.
    """
    counts  = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = len(y_train) / (n_classes * counts)
    return torch.from_numpy(weights).to(DEVICE)


def train_model(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    class_weights: torch.Tensor,
    n_epochs:     int = 80,
    lr:           float = 1e-3,
    patience:     int = 15,
    lr_patience:  int = 5,
) -> list[float]:
    """
    Train `model` with Adam, ReduceLROnPlateau, and early stopping.

    Early stopping monitors validation loss and restores the best checkpoint
    when no improvement is observed for `patience` consecutive epochs.

    Returns
    -------
    val_losses : per-epoch validation loss history
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    val_losses = []

    for epoch in range(1, n_epochs + 1):
        # ── Training phase ──────────────────────────────────────────────────
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ── Validation phase ─────────────────────────────────────────────────
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
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | val_loss={val_loss:.4f} | "
                  f"best={best_val_loss:.4f} | patience={epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"[INFO] Early stopping at epoch {epoch}.")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return val_losses


# ═════════════════════════════════════════════════════════════════════════════
# 6. EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = ["Down", "Stationary", "Up"]


def evaluate_model(
    model:       nn.Module,
    test_loader: DataLoader,
    model_name:  str = "TCN",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on the test set and return (y_true, y_pred).

    Prints a full sklearn classification report including per-class
    precision, recall, F1, and support.
    """
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
    acc      = (y_true == y_pred).mean()

    print(f"\n{'='*60}")
    print(f"  {model_name} Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Macro-F1   : {macro_f1:.4f}")
    print(f"  Mean IoU   : {mean_iou:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES)}")

    return y_true, y_pred


def plot_training_curve(
    val_losses: list[float],
    model_name: str = "TCN-XLarge",
    path: str = "training_curve.png",
) -> None:
    """Plot validation loss over epochs to illustrate learning dynamics."""
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


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title:  str = "Confusion Matrix",
    path:   str = "confusion_matrix.png",
) -> None:
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Confusion matrix saved -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 7. PCA LATENT-SPACE VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def extract_embeddings(
    model:       nn.Module,
    test_loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the 64-dimensional pre-logit embedding for each test sample.

    We hook into the ReLU output of the first linear layer in the
    classification head, which is the model's final learned representation.
    """
    embeddings, labels_out = [], []

    # Hook to capture activations from the penultimate linear layer
    captured = {}
    def _hook(module, inp, out):
        captured["embed"] = out.detach().cpu().numpy()

    # Register hook on the ReLU after the first Linear in the classifier
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


def plot_pca_embeddings(
    embeddings: np.ndarray,
    labels:     np.ndarray,
    path:       str = "pca_embeddings.png",
) -> None:
    from sklearn.decomposition import PCA

    pca  = PCA(n_components=2, random_state=SEED)
    proj = pca.fit_transform(embeddings)
    ev   = pca.explained_variance_ratio_

    palette = {0: "#e05252", 1: "#f5a623", 2: "#4a90d9"}
    colors  = [palette[l] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, name, color in zip([0, 1, 2], CLASS_NAMES, palette.values()):
        mask = labels == cls
        ax.scatter(proj[mask, 0], proj[mask, 1], c=color, s=6, alpha=0.5, label=name)

    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
    ax.set_title("PCA of TCN Learned Embeddings (Test Set)", fontsize=13)
    ax.legend(markerscale=3, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] PCA plot saved -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 8. SHAP FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════════════════════

def plot_shap_importance(
    model:       nn.Module,
    X_te_win:    np.ndarray,
    feature_dim: int,
    n_background: int = 100,
    n_explain:    int = 200,
    path:         str = "shap_importance.png",
) -> None:
    """
    Compute KernelSHAP attributions for the flattened window representation.

    We collapse the temporal axis by taking the mean across time-steps so
    that SHAP operates on a single (F,) feature vector per sample, enabling
    fast KernelSHAP without a GPU-aware explainer.

    Feature names follow the LOB convention:
        BidPx_i, BidVol_i, AskPx_i, AskVol_i for i in 1..5,
        then Spread, Imbalance, VWMid.
    """
    try:
        import shap
    except ImportError:
        print("[WARN] shap not installed. Skipping SHAP plot. Run: pip install shap")
        return

    # Collapse temporal dimension -> (N, F) for KernelSHAP
    X_flat_te = X_te_win.mean(axis=1)   # mean over time-steps

    bg  = X_flat_te[:n_background]
    exp = X_flat_te[:n_explain]

    # Wrapper: expand back to (N, F, 1) for the model, predict softmax probs
    def predict_fn(x: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(x).float().unsqueeze(2).to(DEVICE)  # (N, F, 1)
        model.eval()
        with torch.no_grad():
            logits = model(x_t)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    explainer   = shap.KernelExplainer(predict_fn, bg)
    shap_values = explainer.shap_values(exp, nsamples=50, silent=True)

    # Build feature names
    n_levels = (feature_dim - 3) // 4
    feat_names = []
    for i in range(1, n_levels + 1):
        feat_names += [f"BidPx_{i}", f"BidVol_{i}"]
    for i in range(1, n_levels + 1):
        feat_names += [f"AskPx_{i}", f"AskVol_{i}"]
    feat_names += ["Spread", "Imbalance", "VWMid"]

    # Mean |SHAP| across classes and samples -> per-feature importance
    mean_abs_shap = np.mean(
        [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
    )
    order = np.argsort(mean_abs_shap)[::-1][:15]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feat_names[i] for i in order[::-1]],
        mean_abs_shap[order[::-1]],
        color="#4a90d9"
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("SHAP Feature Importance (Top 15 Features)", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] SHAP importance plot saved -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 9. ABLATION STUDY
# ═════════════════════════════════════════════════════════════════════════════

DILATION_CONFIGS = {
    "TCN-Small":  [1, 2],
    "TCN-Medium": [1, 2, 4],
    "TCN-Large":  [1, 2, 4, 8],
    "TCN-XLarge": [1, 2, 4, 8, 16, 32],
}


def run_ablation(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
    class_weights: torch.Tensor,
    in_channels:  int,
) -> tuple:
    """
    Train each TCN configuration and collect macro-F1, accuracy, mean IoU.
    Returns (results_dict, best_model, best_val_losses) where best is TCN-XLarge.
    """
    results = {}
    best_model = None
    best_val_losses = []
    for name, dilations in DILATION_CONFIGS.items():
        rf = 1 + sum((3 - 1) * d for d in dilations)   # receptive field
        print(f"\n[ABLATION] {name} | dilations={dilations} | RF={rf}")
        model = TCNModel(
            in_channels=in_channels,
            hidden_channels=64,
            kernel_size=3,
            dilations=dilations,
        ).to(DEVICE)
        val_losses = train_model(model, train_loader, val_loader, class_weights,
                                 n_epochs=60, patience=10)
        y_true, y_pred = evaluate_model(model, test_loader, model_name=name)
        results[name] = {
            "accuracy":  float((y_true == y_pred).mean()),
            "macro_f1":  float(f1_score(y_true, y_pred, average="macro")),
            "mean_iou":  float(jaccard_score(y_true, y_pred, average="macro")),
            "rf":        rf,
        }
        if name == "TCN-XLarge":
            best_model = model
            best_val_losses = val_losses
    return results, best_model, best_val_losses


def print_ablation_table(results: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  {'Model':<15} {'Dilations RF':>12} {'Accuracy':>10} {'Macro-F1':>10} {'Mean IoU':>10}")
    print(f"  {'-'*65}")
    for name, r in results.items():
        print(f"  {name:<15} {r['rf']:>12d} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['mean_iou']:>10.4f}")
    print(f"{'='*70}")


def plot_ablation(results: dict, path: str = "ablation_f1.png") -> None:
    names   = list(results.keys())
    f1s     = [results[n]["macro_f1"] for n in names]
    rfs     = [results[n]["rf"] for n in names]

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


# ═════════════════════════════════════════════════════════════════════════════
# 10. LOGISTIC REGRESSION AND MAJORITY-CLASS BASELINES
# ═════════════════════════════════════════════════════════════════════════════

def run_sklearn_baselines(
    X_tr_win: np.ndarray,
    y_tr:     np.ndarray,
    X_te_win: np.ndarray,
    y_te:     np.ndarray,
) -> None:
    """
    Flatten windows and train Logistic Regression and Majority-Class baselines.
    """
    X_tr_flat = X_tr_win.reshape(len(X_tr_win), -1)
    X_te_flat = X_te_win.reshape(len(X_te_win), -1)

    # Majority class
    majority = int(np.bincount(y_tr).argmax())
    y_maj    = np.full(len(y_te), majority)
    print("\n[BASELINE] Majority Class")
    print(classification_report(y_te, y_maj, target_names=CLASS_NAMES, zero_division=0))

    # Logistic Regression (subsample for speed)
    n_sub = min(5000, len(X_tr_flat))
    idx   = np.random.choice(len(X_tr_flat), n_sub, replace=False)
    lr_model = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", n_jobs=-1)
    lr_model.fit(X_tr_flat[idx], y_tr[idx])
    y_lr = lr_model.predict(X_te_flat)
    print("[BASELINE] Logistic Regression")
    print(classification_report(y_te, y_lr, target_names=CLASS_NAMES))


# ═════════════════════════════════════════════════════════════════════════════
# 11. MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Generate synthetic LOB data ────────────────────────────────────────
    print("\n[STEP 1] Generating synthetic LOB data ...")
    X, mid_prices = generate_lob_data(n_snapshots=50_000)
    print(f"         X shape: {X.shape}")

    # ── 2. Construct labels ────────────────────────────────────────────────────
    print("\n[STEP 2] Constructing directional labels ...")
    HORIZON = 5
    y = build_labels(mid_prices, horizon=HORIZON)

    # Trim X to match label length (labels lose `horizon` samples at end)
    X = X[:len(y)]

    # ── 3. Temporal split (no shuffling -- prevents leakage) ───────────────────
    print("\n[STEP 3] Temporal train / val / test split ...")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = temporal_split(X, y)

    # ── 4. Normalize using training statistics only ────────────────────────────
    print("\n[STEP 4] Normalizing (fit on train only -- no leakage) ...")
    X_tr_s, X_va_s, X_te_s, scaler = normalize(X_tr, X_va, X_te)

    # ── 5. Sliding window segmentation ────────────────────────────────────────
    print("\n[STEP 5] Creating sliding windows (L=50) ...")
    WINDOW = 50
    X_tr_win, y_tr_win = make_windows(X_tr_s, y_tr, window_len=WINDOW)
    X_va_win, y_va_win = make_windows(X_va_s, y_va, window_len=WINDOW)
    X_te_win, y_te_win = make_windows(X_te_s, y_te, window_len=WINDOW)
    print(f"         Train windows: {X_tr_win.shape}, Val: {X_va_win.shape}, Test: {X_te_win.shape}")

    # ── 6. DataLoaders ────────────────────────────────────────────────────────
    train_loader = to_dataloader(X_tr_win, y_tr_win, shuffle=True)
    val_loader   = to_dataloader(X_va_win, y_va_win)
    test_loader  = to_dataloader(X_te_win, y_te_win)

    # ── 7. Class weights ──────────────────────────────────────────────────────
    class_weights = compute_class_weights(y_tr_win)
    print(f"[INFO] Class weights: {class_weights.cpu().numpy()}")

    # ── 8. Sklearn baselines ──────────────────────────────────────────────────
    print("\n[STEP 8] Running sklearn baselines ...")
    run_sklearn_baselines(X_tr_win, y_tr_win, X_te_win, y_te_win)

    # ── 9. LSTM baseline ──────────────────────────────────────────────────────
    print("\n[STEP 9] Training LSTM baseline ...")
    in_ch = X_tr_win.shape[2]
    lstm = LSTMModel(in_features=in_ch).to(DEVICE)
    train_model(lstm, train_loader, val_loader, class_weights, n_epochs=60, patience=10)
    evaluate_model(lstm, test_loader, model_name="LSTM")

    # ── 10. Ablation study ────────────────────────────────────────────────────
    print("\n[STEP 10] Running TCN ablation study ...")
    ablation_results, best_model, best_val_losses = run_ablation(
        train_loader, val_loader, test_loader, class_weights, in_ch
    )
    print_ablation_table(ablation_results)
    plot_ablation(ablation_results)

    # ── 11. Best model -- full evaluation (reuse TCN-XLarge from ablation) ────
    print("\n[STEP 11] Full evaluation of TCN-XLarge ...")
    plot_training_curve(best_val_losses, model_name="TCN-XLarge")
    y_true, y_pred = evaluate_model(best_model, test_loader, model_name="TCN-XLarge")
    plot_confusion_matrix(y_true, y_pred, title="TCN-XLarge Confusion Matrix")

    # ── 12. PCA of learned embeddings ─────────────────────────────────────────
    print("\n[STEP 12] PCA visualisation of learned embeddings ...")
    embeddings, emb_labels = extract_embeddings(best_model, test_loader)
    plot_pca_embeddings(embeddings, emb_labels)

    # ── 13. SHAP feature importance ───────────────────────────────────────────
    print("\n[STEP 13] SHAP feature importance (may take a few minutes) ...")
    plot_shap_importance(best_model, X_te_win, feature_dim=in_ch)

    print("\n[DONE] All outputs saved to the current directory.")


if __name__ == "__main__":
    main()
