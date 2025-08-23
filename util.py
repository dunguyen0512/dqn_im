
import os
import sys
import logging
from datetime import datetime
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
CLASS_NAMES = ["H", "ITSC", "LD", "MF"]

__all__ = [
    "setup_logging",
    "save_frame",
    "save_render_figure",
    "init_confusion",
    "update_confusion",
    "confusion_accuracy",
    "save_confusion_csv",
    "save_confusion_png",
    "CLASS_NAMES", "_save_curves_and_csv","moving_average"
]





# ------------------------ LOGGING ------------------------
def moving_average(x, window=100):
    if len(x) == 0:
        return []
    import numpy as _np
    x = _np.asarray(x, dtype=float)
    if window <= 1:
        return x
    cumsum = _np.cumsum(_np.insert(x, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / float(window)
    # pad to same length by prefixing with first available average
    pad = [ma[0]] * (window - 1)
    return _np.concatenate([pad, ma])

# def moving_average(data, window=100):
#     if len(data) == 0:
#         return np.array([])  # nothing to smooth
#     if len(data) < window:
#         window = len(data)
#     ma = np.convolve(data, np.ones(window)/window, mode="valid")
#     pad = [ma[0]] * (window - 1)
#     return np.concatenate([pad, ma])

# def moving_average(data, window=100):
#     """Compute moving average with safe handling of short or empty data."""
#     if len(data) == 0:
#         return np.array([])
#     window = min(window, len(data))
#     # use "same" so output length = input length
#     ma = np.convolve(data, np.ones(window)/window, mode="same")
#     return ma

def _save_curves_and_csv(args, train_losses, ep_returns, ep_success, ep_avg_qs,ma_window,ma_ret,ma_succ):
    """
    Save all plots and CSVs: loss curve, avg Q per episode, reward/success curves.
    """
    import os, csv
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)

    # ----- Loss curve & CSV -----
    if len(train_losses) > 0:
        plt.figure(figsize=(4,4))
        plt.plot(train_losses)
        plt.xlabel('Training step')
        plt.ylabel('Loss')
        # plt.title('DQN Training Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=150)
        plt.close()

        with open(os.path.join(args.out_dir, "loss_history.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "loss"])
            for i, v in enumerate(train_losses, 1):
                w.writerow([i, v])

    # ----- Avg Q per episode -----
    if len(ep_avg_qs) > 0:
        plt.figure(figsize=(4,4))
        plt.plot(ep_avg_qs)
        plt.xlabel("Episode")
        plt.ylabel("Average Q")
        # plt.title("Average Q per Episode")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "avg_q_per_episode.png"), dpi=150)
        plt.close()

        with open(os.path.join(args.out_dir, "avg_q_per_episode.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "avg_q"])
            for i, q in enumerate(ep_avg_qs, 1):
                w.writerow([i, q])

    # ----- Reward & Success curves -----
    # ma_window = getattr(args, "reward_ma_window", 1000)
    # ma_ret = moving_average(ep_returns, window=ma_window) if len(ep_returns) else []
    # ma_succ = moving_average(ep_success, window=ma_window) if len(ep_success) else []

    # Normalize reward curve [0,1]
    if len(ma_ret) > 0:
        rmin, rmax = float(np.min(ma_ret)), float(np.max(ma_ret))
        ma_ret_norm = (ma_ret - rmin) / (rmax - rmin) if rmax > rmin else np.zeros_like(ma_ret)

        plt.figure(figsize=(4,4))
        plt.plot(ma_ret_norm)
        plt.xlabel('Episode')
        plt.ylabel('Training Reward (normalized)')
        # plt.title(f'Moving Avg Reward (window={ma_window})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'avg_training_reward_norm.png'), dpi=150)
        plt.close()

    # Success rate curve
    if len(ma_succ) > 0:
        plt.figure(figsize=(4,4))
        plt.plot(ma_succ)
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        # plt.title(f'Moving Avg Success (window={ma_window})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'avg_success_rate.png'), dpi=150)
        plt.close()

    # CSV with per-episode reward/success
    with open(os.path.join(args.out_dir, 'train_reward_success.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episode','ep_return','success','ma_return_norm','ma_success'])
        M = len(ep_returns)
        for i in range(M):
            mrn = float(ma_ret[i]) if i < len(ma_ret) else ""
            ms = float(ma_succ[i]) if i < len(ma_succ) else ""
            w.writerow([i+1, ep_returns[i], ep_success[i], mrn, ms])



def setup_logging(log_dir="Report", log_name_prefix="run"):
    """
    Create a logger that writes to both console and a timestamped file.
    Returns (logger, log_path).
    """
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{log_name_prefix}_{ts}.log")

    logger = logging.getLogger("specrl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # file handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging to {log_path}")
    return logger, log_path

# --------------------- IMAGE SAVERS ----------------------

def save_frame(obs, path, dpi=150):
    """
    Save a single observation array as an image.
    obs: (H,W) or (H,W,3), values in [0,1] or [0,255].
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = obs
    if isinstance(img, np.ndarray) and img.ndim in (2, 3):
        if img.dtype != np.uint8:
            # Normalize to 0..255 for saving
            img = np.clip(img, 0, 1) if img.max() <= 1.0 else img
            if img.max() <= 1.0:
                img = (img * 255.0).astype(np.uint8)
    else:
        raise ValueError(f"save_frame expected ndarray HxW or HxWx3, got {type(obs)} with shape {getattr(obs, 'shape', None)}")

    plt.figure(figsize=(4, 3))
    if img.ndim == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close()


def save_render_figure(image, mask, step, path, dpi=150):
    """
    Save the 3-panel figure exactly like your render():
      [Mask] [Image] [Image * Mask]
    - image: (H,W) or (H,W,3), values in [0,1] or [0,255]
    - mask: (H,W), typically 0/1
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Prepare image for display
    img = np.asarray(image)
    msk = np.asarray(mask)

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img_disp = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img_disp = img.astype(np.uint8)
    else:
        img_disp = img

    # Masked image: broadcast mask across channels if needed
    if img_disp.ndim == 3:
        masked = (img_disp * msk[..., None]).astype(np.uint8)
    else:
        masked = (img_disp * msk).astype(np.uint8)

    fig = plt.figure(figsize=(4, 3))
    plt.suptitle(f"Step {step}")

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow((msk * 255).astype(np.uint8),  interpolation="nearest")
    ax1.set_title("Mask")
    ax1.axis("off")

    ax2 = plt.subplot(1, 3, 2)
    if img_disp.ndim == 3:
        ax2.imshow(img_disp, interpolation="nearest")
    else:
        ax2.imshow(img_disp,  interpolation="nearest")
    ax2.set_title("Image")
    ax2.axis("off")

    ax3 = plt.subplot(1, 3, 3)
    if masked.ndim == 3:
        ax3.imshow(masked, interpolation="nearest")
    else:
        ax3.imshow(masked, interpolation="nearest")
    ax3.set_title("Unmasked")
    ax3.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1, dpi=dpi)
    plt.close(fig)

# ---------------- CONFUSION MATRIX TOOLS ----------------

def init_confusion(num_classes=4):
    return np.zeros((num_classes, num_classes), dtype=int)

def update_confusion(cm, y_true, y_pred):
    if 0 <= y_true < cm.shape[0] and 0 <= y_pred < cm.shape[1]:
        cm[y_true, y_pred] += 1

def confusion_accuracy(cm):
    total = cm.sum()
    return (np.trace(cm) / total) if total > 0 else 0.0

def save_confusion_csv(cm, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ",".join([""] + [f"pred_{name}" for name in CLASS_NAMES])
    lines = [header]
    for i, row in enumerate(cm):
        line = ",".join([f"true_{CLASS_NAMES[i]}"] + [str(int(v)) for v in row])
        lines.append(line)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def save_confusion_png(cm, path, title="Confusion Matrix"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    # ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel='True label',
        xlabel='Predicted label',
        title=title
    )

    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close(fig)
