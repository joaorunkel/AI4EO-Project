"""
ConvLSTM2D deforestation spread prediction.

Loads Sentinel-2 NDMI time series, trains a ConvLSTM to predict the next frame,
then generates a directional arrow map showing predicted deforestation spread.

Usage:  python train_convlstm.py
Output: model.pt, arrow_map.png
"""

import os
import numpy as np
import rasterio
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, uniform_filter

# ── Config ───────────────────────────────────────────────────────────────────
BANDS_DIR = "Data/bands"
PATCH_SIZE = 128
STRIDE = 64
SEQ_LEN = 4          # 3 input + 1 target
NAN_THRESH = 0.5     # skip patches with >50% NaN
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 80
PATIENCE = 10        # early stopping
DELTA_THRESH = -0.05 # NDMI drop threshold for deforestation
BLOCK_SIZE = 32      # arrow grid resolution
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BAD_SCL = {0, 1, 2, 3, 8, 9, 10, 11}


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_ndmi_series():
    """Load all dates, compute NDMI, return list of (ndmi, valid_mask, date_str)."""
    files = sorted(os.listdir(BANDS_DIR))
    date_files = defaultdict(dict)
    for f in files:
        if not f.endswith(".tif"):
            continue
        name = f.replace(".tif", "")
        date_str, band = name.rsplit("_", 1)
        date_files[date_str][band] = os.path.join(BANDS_DIR, f)

    all_dates = sorted(d for d in date_files if {"B08", "B11", "SCL"} <= date_files[d].keys())
    print(f"Found {len(all_dates)} dates with all bands")

    series = []
    for date_str in all_dates:
        with rasterio.open(date_files[date_str]["B08"]) as src:
            b08 = src.read(1).astype(np.float32)
        with rasterio.open(date_files[date_str]["B11"]) as src:
            b11 = src.read(1).astype(np.float32)
        with rasterio.open(date_files[date_str]["SCL"]) as src:
            scl = src.read(1)

        bad_mask = np.isin(scl, list(BAD_SCL)) | (b08 == 0)
        denom = b08 + b11
        ndmi = np.where((denom > 0) & ~bad_mask, (b08 - b11) / denom, np.nan)
        valid = ~np.isnan(ndmi)
        ndmi_filled = np.where(valid, ndmi, 0.0).astype(np.float32)
        series.append((ndmi_filled, valid, date_str))

    return series


def compute_time_deltas(date_strings):
    """Compute normalized days between consecutive dates."""
    dts = [datetime.strptime(d, "%Y-%m-%d") for d in date_strings]
    deltas = []
    for i in range(len(dts)):
        if i == 0:
            deltas.append(0.0)
        else:
            deltas.append((dts[i] - dts[i - 1]).days)
    deltas = np.array(deltas, dtype=np.float32)
    max_d = max(deltas.max(), 1.0)
    return deltas / max_d


def pad_to_multiple(arr, multiple):
    """Pad 2D array to nearest multiple of `multiple`."""
    h, w = arr.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return arr
    return np.pad(arr, ((0, ph), (0, pw)), mode="constant", constant_values=0)


# ── Dataset ──────────────────────────────────────────────────────────────────

class NDMIPatchDataset(Dataset):
    def __init__(self, ndmi_frames, valid_masks, time_deltas, augment=False):
        """
        ndmi_frames: list of (H, W) arrays, NaN→0
        valid_masks: list of (H, W) bool arrays
        time_deltas: (T,) normalized deltas
        """
        self.augment = augment
        self.samples = []

        T = len(ndmi_frames)
        H, W = ndmi_frames[0].shape

        # Pad all frames
        pad_mult = PATCH_SIZE
        frames_padded = [pad_to_multiple(f, pad_mult) for f in ndmi_frames]
        masks_padded = [pad_to_multiple(m.astype(np.float32), pad_mult).astype(bool)
                        for m in valid_masks]
        pH, pW = frames_padded[0].shape

        # Extract patch locations
        patch_rows = list(range(0, pH - PATCH_SIZE + 1, STRIDE))
        patch_cols = list(range(0, pW - PATCH_SIZE + 1, STRIDE))
        n_patches = len(patch_rows) * len(patch_cols)
        print(f"  Image padded to {pH}×{pW}, {n_patches} patches/frame "
              f"({len(patch_rows)}×{len(patch_cols)})")

        # Sliding window over time
        for t_start in range(T - SEQ_LEN + 1):
            t_end = t_start + SEQ_LEN
            for r in patch_rows:
                for c in patch_cols:
                    # Check NaN fraction across sequence
                    total_valid = 0
                    total_pixels = 0
                    for t in range(t_start, t_end):
                        patch_mask = masks_padded[t][r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                        total_valid += patch_mask.sum()
                        total_pixels += PATCH_SIZE * PATCH_SIZE
                    if total_valid / total_pixels < (1 - NAN_THRESH):
                        continue

                    # Store indices (lazy loading from arrays)
                    self.samples.append((t_start, t_end, r, c))

        # Keep references for __getitem__
        self._frames = frames_padded
        self._masks = masks_padded
        self._deltas = time_deltas
        print(f"  {len(self.samples)} samples total")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t_start, t_end, r, c = self.samples[idx]

        # Input: T-1 frames, each with 2 channels (NDMI + time_delta)
        input_seq = []
        for t in range(t_start, t_end - 1):
            ndmi_patch = self._frames[t][r:r+PATCH_SIZE, c:c+PATCH_SIZE]
            delta_ch = np.full_like(ndmi_patch, self._deltas[t])
            input_seq.append(np.stack([ndmi_patch, delta_ch], axis=0))  # (2, H, W)
        input_seq = np.stack(input_seq, axis=0)  # (T-1, 2, H, W)

        # Target: last frame NDMI
        target = self._frames[t_end - 1][r:r+PATCH_SIZE, c:c+PATCH_SIZE]
        target_mask = self._masks[t_end - 1][r:r+PATCH_SIZE, c:c+PATCH_SIZE]

        # Previous frame for deforestation weighting
        prev = self._frames[t_end - 2][r:r+PATCH_SIZE, c:c+PATCH_SIZE]

        if self.augment:
            # Random flip and rotation
            k = np.random.randint(4)
            flip_h = np.random.random() > 0.5
            flip_v = np.random.random() > 0.5

            def transform(arr):
                if arr.ndim == 3:  # (C, H, W)
                    arr = np.rot90(arr, k, axes=(1, 2)).copy()
                    if flip_h:
                        arr = arr[:, :, ::-1].copy()
                    if flip_v:
                        arr = arr[:, ::-1, :].copy()
                else:  # (H, W)
                    arr = np.rot90(arr, k).copy()
                    if flip_h:
                        arr = arr[:, ::-1].copy()
                    if flip_v:
                        arr = arr[::-1, :].copy()
                return arr

            input_seq = np.stack([transform(frame) for frame in input_seq])
            target = transform(target)
            target_mask = transform(target_mask.astype(np.float32)) > 0.5
            prev = transform(prev)

        return (torch.from_numpy(input_seq.copy()).float(),
                torch.from_numpy(target.copy()).float(),
                torch.from_numpy(target_mask.copy()).float(),
                torch.from_numpy(prev.copy()).float())


# ── ConvLSTM Model ───────────────────────────────────────────────────────────

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                               kernel_size, padding=pad)

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))


class ConvLSTMPredictor(nn.Module):
    def __init__(self, input_channels=2, hidden_dims=(32, 32)):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = input_channels
        for hd in hidden_dims:
            self.layers.append(ConvLSTMCell(in_dim, hd, kernel_size=3))
            in_dim = hd
        self.output_head = nn.Conv2d(hidden_dims[-1], 1, kernel_size=1)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        states = [layer.init_hidden(B, H, W, x.device) for layer in self.layers]

        for t in range(T):
            inp = x[:, t]  # (B, C, H, W)
            for i, layer in enumerate(self.layers):
                h, c = layer(inp, states[i])
                states[i] = (h, c)
                inp = h

        # Use last hidden state to predict
        pred = self.output_head(states[-1][0])  # (B, 1, H, W)
        return pred.squeeze(1)  # (B, H, W)


# ── Training ─────────────────────────────────────────────────────────────────

def masked_mse_loss(pred, target, mask, prev):
    """MSE on valid pixels, 2× weight where NDMI decreased (deforestation)."""
    diff = (pred - target) ** 2
    deforest_weight = torch.where(target < prev, 2.0, 1.0)
    weighted = diff * deforest_weight * mask
    n = mask.sum().clamp(min=1)
    return weighted.sum() / n


def train(model, train_loader, val_loader, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        n_batches = 0
        for inputs, targets, masks, prevs in train_loader:
            inputs, targets, masks, prevs = (
                inputs.to(DEVICE), targets.to(DEVICE),
                masks.to(DEVICE), prevs.to(DEVICE))
            optimizer.zero_grad()
            preds = model(inputs)
            loss = masked_mse_loss(preds, targets, masks, prevs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for inputs, targets, masks, prevs in val_loader:
                inputs, targets, masks, prevs = (
                    inputs.to(DEVICE), targets.to(DEVICE),
                    masks.to(DEVICE), prevs.to(DEVICE))
                preds = model(inputs)
                loss = masked_mse_loss(preds, targets, masks, prevs)
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(n_val, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} (best val={best_val:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


# ── Inference & Arrow Map ────────────────────────────────────────────────────

def predict_full_frame(model, ndmi_frames, valid_masks, time_deltas):
    """Run model on all patches for last 3 frames, reassemble with overlap averaging."""
    T_input = SEQ_LEN - 1  # 3

    # Take last T_input+1 frames (3 input + we need last observed for delta)
    frames = ndmi_frames[-(T_input):]
    masks = valid_masks[-(T_input):]

    H, W = frames[0].shape
    pad_mult = PATCH_SIZE
    frames_p = [pad_to_multiple(f, pad_mult) for f in frames]
    pH, pW = frames_p[0].shape

    # Build time delta channel for input frames
    deltas = time_deltas[-(T_input):]

    pred_sum = np.zeros((pH, pW), dtype=np.float64)
    pred_count = np.zeros((pH, pW), dtype=np.float64)

    patch_rows = list(range(0, pH - PATCH_SIZE + 1, STRIDE))
    patch_cols = list(range(0, pW - PATCH_SIZE + 1, STRIDE))

    model.eval()
    with torch.no_grad():
        for r in patch_rows:
            for c in patch_cols:
                input_seq = []
                for t in range(T_input):
                    ndmi_patch = frames_p[t][r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                    delta_ch = np.full_like(ndmi_patch, deltas[t])
                    input_seq.append(np.stack([ndmi_patch, delta_ch], axis=0))
                input_seq = np.stack(input_seq, axis=0)  # (T, 2, H, W)
                inp = torch.from_numpy(input_seq).float().unsqueeze(0).to(DEVICE)
                pred = model(inp).cpu().numpy()[0]  # (128, 128)
                pred_sum[r:r+PATCH_SIZE, c:c+PATCH_SIZE] += pred
                pred_count[r:r+PATCH_SIZE, c:c+PATCH_SIZE] += 1

    pred_count = np.maximum(pred_count, 1)
    predicted = (pred_sum / pred_count).astype(np.float32)

    # Crop back to original size
    return predicted[:H, :W]


def generate_arrow_map(predicted, last_observed, last_valid, output_path="arrow_map.png"):
    """Generate directional arrow map showing predicted deforestation spread."""
    delta = predicted - last_observed

    # Only consider valid pixels with significant NDMI drop
    deforest = (delta < DELTA_THRESH) & last_valid
    delta_masked = np.where(deforest, delta, 0.0)

    # Smooth for gradient computation
    delta_smooth = uniform_filter(delta_masked.astype(np.float64), size=15)

    # Compute gradient (direction of steepest NDMI decrease)
    grad_x = sobel(delta_smooth, axis=1)  # horizontal
    grad_y = sobel(delta_smooth, axis=0)  # vertical

    H, W = last_observed.shape

    # Downsample to block grid
    n_rows = H // BLOCK_SIZE
    n_cols = W // BLOCK_SIZE

    arrow_x = np.zeros((n_rows, n_cols))
    arrow_y = np.zeros((n_rows, n_cols))
    arrow_mag = np.zeros((n_rows, n_cols))
    block_delta = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            r0, r1 = i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE
            c0, c1 = j * BLOCK_SIZE, (j + 1) * BLOCK_SIZE
            block = deforest[r0:r1, c0:c1]
            if block.sum() < BLOCK_SIZE:  # need minimum deforested pixels
                continue
            gx = grad_x[r0:r1, c0:c1].mean()
            gy = grad_y[r0:r1, c0:c1].mean()
            mag = np.sqrt(gx**2 + gy**2)
            if mag > 1e-6:
                arrow_x[i, j] = gx / mag
                arrow_y[i, j] = -gy / mag  # flip y for image coords
                arrow_mag[i, j] = mag
            block_delta[i, j] = delta[r0:r1, c0:c1].mean()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # (a) Last observed NDMI
    im0 = axes[0].imshow(np.where(last_valid, last_observed, np.nan),
                         cmap="RdYlGn", vmin=-0.5, vmax=0.7)
    axes[0].set_title(f"Last Observed NDMI", fontsize=13)
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # (b) Predicted next NDMI
    im1 = axes[1].imshow(np.where(last_valid, predicted, np.nan),
                         cmap="RdYlGn", vmin=-0.5, vmax=0.7)
    axes[1].set_title("Predicted Next NDMI", fontsize=13)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # (c) Change map with arrows
    im2 = axes[2].imshow(np.where(last_valid, delta, np.nan),
                         cmap="RdBu", vmin=-0.3, vmax=0.3)
    axes[2].set_title("Predicted NDMI Change + Spread Direction", fontsize=13)
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="ΔNDMI")

    # Arrow overlay
    active = arrow_mag > 0
    if active.any():
        yi, xi = np.where(active)
        cx = xi * BLOCK_SIZE + BLOCK_SIZE // 2
        cy = yi * BLOCK_SIZE + BLOCK_SIZE // 2
        u = arrow_x[active]
        v = arrow_y[active]
        colors = -block_delta[active]  # higher = more deforestation
        colors = np.clip(colors / max(colors.max(), 1e-6), 0, 1)

        axes[2].quiver(cx, cy, u, v, colors, cmap="YlOrRd",
                       scale=25, width=0.003, headwidth=4,
                       clim=(0, 1), alpha=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Arrow map saved to {output_path}")
    plt.close(fig)

    return predicted, delta


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ConvLSTM2D Deforestation Spread Prediction")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading NDMI time series...")
    series = load_ndmi_series()
    if len(series) < SEQ_LEN:
        print(f"Need at least {SEQ_LEN} dates, got {len(series)}. Aborting.")
        return

    ndmi_frames = [s[0] for s in series]
    valid_masks = [s[1] for s in series]
    date_strings = [s[2] for s in series]
    time_deltas = compute_time_deltas(date_strings)

    print(f"  {len(series)} frames, shape {ndmi_frames[0].shape}")
    print(f"  Dates: {date_strings[0]} → {date_strings[-1]}")

    # Split: last 2 temporal windows for validation
    n_train_windows = len(series) - SEQ_LEN + 1 - 2
    n_val_windows = 2
    train_end = n_train_windows + SEQ_LEN - 1
    print(f"  Train windows: {n_train_windows}, Val windows: {n_val_windows}")

    # Build datasets
    print("\n[2/4] Building patch datasets...")
    print("  Training set:")
    train_ds = NDMIPatchDataset(
        ndmi_frames[:train_end], valid_masks[:train_end],
        time_deltas[:train_end], augment=True)
    print("  Validation set:")
    val_ds = NDMIPatchDataset(
        ndmi_frames[-(n_val_windows + SEQ_LEN - 1):],
        valid_masks[-(n_val_windows + SEQ_LEN - 1):],
        time_deltas[-(n_val_windows + SEQ_LEN - 1):], augment=False)

    if len(train_ds) == 0:
        print("No training samples! Check data quality. Aborting.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)

    # Build model
    print(f"\n[3/4] Training ConvLSTM on {DEVICE}...")
    model = ConvLSTMPredictor(input_channels=2, hidden_dims=(32, 32)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    best_val = train(model, train_loader, val_loader)
    torch.save(model.state_dict(), "model.pt")
    print(f"  Model saved to model.pt (best val loss: {best_val:.6f})")

    # Inference + arrow map
    print("\n[4/4] Generating predictions and arrow map...")
    predicted = predict_full_frame(model, ndmi_frames, valid_masks, time_deltas)
    generate_arrow_map(predicted, ndmi_frames[-1], valid_masks[-1])

    print("\nDone!")


if __name__ == "__main__":
    main()
