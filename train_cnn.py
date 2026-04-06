"""
train_cnn.py
------------
Train CloudFeedbackCNN on the prepared tensor dataset.

Training set: GA8 + GA9 pooled (~1006 members). Pooling both PPE generations
forces the model to find CRE→feedback relationships that hold across different
parameter perturbations, reducing overfitting to GA8-specific spatial patterns.
CFMIP structural generalisation is the primary out-of-sample benchmark.

Per-pixel normalisation is fit on the pooled training set. Norm stats are saved
so the eval notebook and any future inference script reproduce the same preprocessing.

Saves to checkpoints/cnn/:
  norm_stats.npz              — X_mean, X_std, y_mean, y_std (fit on GA8+GA9)
  fold_01.pt … fold_N.pt      — best-val-loss checkpoint for each CV fold
  final.pt                    — model trained on all GA8+GA9
  y_pool_cv.npy               — held-out CV predictions for pooled set
  y_ga8.npy, y_ga9.npy        — ground truth (for diagnostic plots)
  y_ga8_pred.npy, y_ga9_pred.npy — final model predictions on each PPE (diagnostic)
  y_c2.npy, cfmip_fb.npy, cfmip_models.npy — OOS ground truth
  y_c2_pred.npy, y_cfmip_pred.npy           — OOS predictions
  ceres_final_pred.npy, ceres_fold_preds.npy — CERES constraint

Usage:
    python train_cnn.py [--epochs 1000] [--patience 50] [--folds 10]
                        [--lr 1e-3] [--batch_size 100] [--weight_decay 1e-4]
"""

import argparse
import importlib
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import model as _m
importlib.reload(_m)
from model import CloudFeedbackCNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     default=str(ROOT / "data"))
    p.add_argument("--ckpt_dir",     default=str(ROOT / "checkpoints" / "cnn"))
    p.add_argument("--folds",        type=int,   default=10)
    p.add_argument("--epochs",       type=int,   default=1000)
    p.add_argument("--patience",     type=int,   default=50)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--batch_size",   type=int,   default=100)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    return p.parse_args()


def train_model(X_tr, y_tr, X_val, y_val, y_mean, y_std, device, *,
                epochs, patience, lr, batch_size, weight_decay):
    def norm_y(y):
        return (y - y_mean) / y_std

    model = CloudFeedbackCNN().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ds  = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(norm_y(y_tr)))
    ldr = DataLoader(ds, batch_size=batch_size, shuffle=True)
    Xv  = torch.FloatTensor(X_val).to(device)
    yv  = torch.FloatTensor(norm_y(y_val)).to(device)

    best_loss, best_state, no_improve = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in ldr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.mse_loss(model(xb), yb).backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model(Xv), yv).item()
        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model


def predict(model, X, y_mean, y_std, device):
    model.eval()
    with torch.no_grad():
        p = model(torch.FloatTensor(X).to(device)).cpu().numpy()
    return (p * y_std + y_mean).astype(np.float32)


def main():
    args   = parse_args()
    data   = Path(args.data_dir)
    ckpt   = Path(args.ckpt_dir)
    ckpt.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else "cpu"
    )
    print(f"Device: {device}")

    # ── Load raw tensors ───────────────────────────────────────────────────────
    t = np.load(data / "tensors.npz", allow_pickle=True)
    X_ga8    = t["X_ga8"].astype(np.float32)
    y_ga8    = t["y_ga8"].astype(np.float32)
    X_ga9    = t["X_ga9"].astype(np.float32)
    y_ga9    = t["y_ga9"].astype(np.float32)
    y_c2     = t["y_c2"].astype(np.float32)
    cfmip_fb = t["cfmip_fb"].astype(np.float32)
    cfmip_models = t["cfmip_models"]

    # ── Pool GA8 + GA9 for training ───────────────────────────────────────────
    # Pooling both PPE generations forces the model to learn relationships that
    # hold across different parameter perturbations rather than GA8-specific
    # spatial correlations. CFMIP structural test is the primary OOS benchmark.
    X_pool = np.concatenate([X_ga8, X_ga9], axis=0)
    y_pool = np.concatenate([y_ga8, y_ga9], axis=0)
    print(f"Pooled training set: {len(y_pool)} members (GA8={len(y_ga8)}, GA9={len(y_ga9)})")

    # ── Per-pixel normalisation (fit on pooled training set) ──────────────────
    X_mean = X_pool.mean(axis=0, keepdims=True)
    X_std  = X_pool.std( axis=0, keepdims=True).clip(min=1e-6)
    y_mean = float(y_pool.mean())
    y_std  = float(y_pool.std())

    np.savez(ckpt / "norm_stats.npz",
             X_mean=X_mean, X_std=X_std,
             y_mean=np.array(y_mean), y_std=np.array(y_std))

    def normalise(X):
        return ((X.astype(np.float32) - X_mean) / X_std)

    X_pool_n  = normalise(X_pool)
    X_ga8_n   = normalise(X_ga8)
    X_ga9_n   = normalise(X_ga9)
    X_c2_n    = normalise(t["X_c2"])
    X_cfmip_n = normalise(t["X_cfmip"])
    X_ceres_n = normalise(t["X_ceres"])

    # ── 10-fold CV on pooled set ───────────────────────────────────────────────
    print(f"\n{args.folds}-fold CV on pooled GA8+GA9 ({len(y_pool)} members)...")
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    y_pool_cv = np.empty_like(y_pool)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_pool_n)):
        print(f"  Fold {fold+1}/{args.folds}  train={len(tr_idx)}  val={len(val_idx)}", flush=True)
        mod = train_model(
            X_pool_n[tr_idx], y_pool[tr_idx],
            X_pool_n[val_idx], y_pool[val_idx],
            y_mean, y_std, device,
            epochs=args.epochs, patience=args.patience,
            lr=args.lr, batch_size=args.batch_size,
            weight_decay=args.weight_decay,
        )
        y_pool_cv[val_idx] = predict(mod, X_pool_n[val_idx], y_mean, y_std, device)
        r2_fold = r2_score(y_pool[val_idx], y_pool_cv[val_idx])
        print(f"    R²={r2_fold:.3f}", flush=True)
        torch.save(mod.state_dict(), ckpt / f"fold_{fold+1:02d}.pt")

    r2_cv   = r2_score(y_pool, y_pool_cv)
    rmse_cv = np.sqrt(mean_squared_error(y_pool, y_pool_cv))
    print(f"\nPooled CV  R²={r2_cv:.3f}  RMSE={rmse_cv:.3f} W/m²")

    # ── Final model on all GA8+GA9 ─────────────────────────────────────────────
    print("\nTraining final model on all GA8+GA9...")
    rng = np.random.default_rng(0)
    val_mask = rng.random(len(y_pool)) < 0.15
    final_model = train_model(
        X_pool_n[~val_mask], y_pool[~val_mask],
        X_pool_n[val_mask],  y_pool[val_mask],
        y_mean, y_std, device,
        epochs=args.epochs, patience=args.patience,
        lr=args.lr, batch_size=args.batch_size,
        weight_decay=args.weight_decay,
    )
    torch.save(final_model.state_dict(), ckpt / "final.pt")

    # ── Diagnostic: final model fit on GA8 and GA9 separately ─────────────────
    y_ga8_pred = predict(final_model, X_ga8_n, y_mean, y_std, device)
    y_ga9_pred = predict(final_model, X_ga9_n, y_mean, y_std, device)
    r2_ga8 = r2_score(y_ga8, y_ga8_pred)
    r2_ga9 = r2_score(y_ga9, y_ga9_pred)
    print(f"\nFinal model on training data:")
    print(f"  GA8  R²={r2_ga8:.3f}  (in training set — not OOS)")
    print(f"  GA9  R²={r2_ga9:.3f}  (in training set — not OOS)")
    np.save(ckpt / "y_ga8_pred.npy", y_ga8_pred)
    np.save(ckpt / "y_ga9_pred.npy", y_ga9_pred)

    # ── Out-of-sample predictions ──────────────────────────────────────────────
    print("\nOut-of-sample evaluation:")
    oos = [
        ("CESM2", X_c2_n,    y_c2,     "y_c2_pred.npy"),
        ("CFMIP", X_cfmip_n, cfmip_fb, "y_cfmip_pred.npy"),
    ]
    for label, X_oos, y_oos, fname in oos:
        y_pred = predict(final_model, X_oos, y_mean, y_std, device)
        r2   = r2_score(y_oos, y_pred)
        rmse = np.sqrt(mean_squared_error(y_oos, y_pred))
        print(f"  {label:<8s}  R²={r2:.3f}  RMSE={rmse:.3f} W/m²")
        np.save(ckpt / fname, y_pred)

    # ── CERES constraint ───────────────────────────────────────────────────────
    ceres_final = predict(final_model, X_ceres_n, y_mean, y_std, device)[0]
    ceres_folds = []
    for i in range(args.folds):
        fm = CloudFeedbackCNN().to(device)
        fm.load_state_dict(torch.load(ckpt / f"fold_{i+1:02d}.pt",
                                      map_location=device, weights_only=True))
        ceres_folds.append(predict(fm, X_ceres_n, y_mean, y_std, device)[0])
    ceres_folds = np.array(ceres_folds, dtype=np.float32)

    np.save(ckpt / "ceres_final_pred.npy", np.array([ceres_final], dtype=np.float32))
    np.save(ckpt / "ceres_fold_preds.npy", ceres_folds)
    print(f"\nCERES constraint: {ceres_final:+.3f} W/m²")
    print(f"  Fold spread: {ceres_folds.min():+.3f} – {ceres_folds.max():+.3f} W/m²")
    print(f"  Mean ± std:  {ceres_folds.mean():+.3f} ± {ceres_folds.std():.3f} W/m²")

    # ── Ground-truth arrays (for eval notebook) ────────────────────────────────
    np.save(ckpt / "y_pool_cv.npy",    y_pool_cv)
    np.save(ckpt / "y_pool.npy",       y_pool)
    np.save(ckpt / "y_ga8.npy",        y_ga8)
    np.save(ckpt / "y_ga9.npy",        y_ga9)
    np.save(ckpt / "y_c2.npy",         y_c2)
    np.save(ckpt / "cfmip_fb.npy",     cfmip_fb)
    np.save(ckpt / "cfmip_models.npy", cfmip_models)
    print(f"\nAll outputs saved to {ckpt}/")


if __name__ == "__main__":
    main()
