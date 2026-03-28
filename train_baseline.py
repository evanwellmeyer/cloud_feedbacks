"""
train_baseline.py
-----------------
Ridge regression baseline trained on the prepared tensor dataset.
Saves fold predictions, final model artefacts, and ground-truth y arrays
to checkpoints/baseline/ so the eval notebook has everything it needs.

Usage:
    python train_baseline.py [--data_dir PATH] [--ckpt_dir PATH] [--folds 10]
"""

import argparse
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent


def flatten(X):
    """(n, 2, lat, lon) → (n, 2*lat*lon)"""
    return X.reshape(len(X), -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(ROOT / "data"))
    parser.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints" / "baseline"))
    parser.add_argument("--folds", type=int, default=10)
    args = parser.parse_args()

    data = Path(args.data_dir)
    ckpt = Path(args.ckpt_dir)
    ckpt.mkdir(parents=True, exist_ok=True)

    t = np.load(data / "tensors.npz", allow_pickle=True)

    X_ga8_raw    = flatten(t["X_ga8"])
    y_ga8        = t["y_ga8"]
    X_ga9_raw    = flatten(t["X_ga9"])
    y_ga9        = t["y_ga9"]
    X_c2_raw     = flatten(t["X_c2"])
    y_c2         = t["y_c2"]
    X_cfmip_raw  = flatten(t["X_cfmip"])
    cfmip_fb     = t["cfmip_fb"]
    X_ceres_raw  = flatten(t["X_ceres"])
    cfmip_models = t["cfmip_models"]

    # Index splitting SW vs LW features (lat * lon per channel)
    n_sw = t["X_ga8"].shape[2] * t["X_ga8"].shape[3]

    print(f"GA8: {X_ga8_raw.shape[0]} members, {X_ga8_raw.shape[1]} features")

    ALPHAS = np.logspace(-1, 8, 60)

    # ── 10-fold CV on GA8 ─────────────────────────────────────────────────────
    print(f"\n{args.folds}-fold CV on GA8...")
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    y_ga8_cv = np.empty_like(y_ga8)
    alphas_chosen = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_ga8_raw)):
        scaler = StandardScaler()
        X_tr  = scaler.fit_transform(X_ga8_raw[tr_idx])
        X_val = scaler.transform(X_ga8_raw[val_idx])
        ridge = RidgeCV(alphas=ALPHAS, fit_intercept=True)
        ridge.fit(X_tr, y_ga8[tr_idx])
        y_ga8_cv[val_idx] = ridge.predict(X_val)
        alphas_chosen.append(ridge.alpha_)
        r2_fold = r2_score(y_ga8[val_idx], y_ga8_cv[val_idx])
        print(f"  Fold {fold+1}/{args.folds}  alpha={ridge.alpha_:.1e}  R²={r2_fold:.3f}", flush=True)

    r2_cv   = r2_score(y_ga8, y_ga8_cv)
    rmse_cv = np.sqrt(mean_squared_error(y_ga8, y_ga8_cv))
    print(f"\nGA8 CV  R²={r2_cv:.3f}  RMSE={rmse_cv:.3f} W/m²")
    np.save(ckpt / "y_ga8_cv.npy", y_ga8_cv)

    # ── Final models on all GA8 ───────────────────────────────────────────────
    # Both-channel model
    scaler_both = StandardScaler().fit(X_ga8_raw)
    ridge_both  = RidgeCV(alphas=ALPHAS, fit_intercept=True).fit(
        scaler_both.transform(X_ga8_raw), y_ga8
    )
    print(f"Final model (both channels) alpha: {ridge_both.alpha_:.1e}")
    joblib.dump(scaler_both, ckpt / "scaler_both.pkl")
    joblib.dump(ridge_both,  ckpt / "model_both.pkl")

    # LW-only model (fair CESM2 comparison — SW channel is a proxy there)
    X_ga8_lw = X_ga8_raw[:, n_sw:]
    scaler_lw = StandardScaler().fit(X_ga8_lw)
    ridge_lw  = RidgeCV(alphas=ALPHAS, fit_intercept=True).fit(
        scaler_lw.transform(X_ga8_lw), y_ga8
    )
    print(f"Final model (LW only)       alpha: {ridge_lw.alpha_:.1e}")
    joblib.dump(scaler_lw, ckpt / "scaler_lw.pkl")
    joblib.dump(ridge_lw,  ckpt / "model_lw.pkl")

    # ── Out-of-sample predictions ─────────────────────────────────────────────
    print("\nOut-of-sample evaluation:")
    evals = [
        ("GA9",          X_ga9_raw,          y_ga9,    "y_ga9_pred.npy",      scaler_both, ridge_both),
        ("CESM2 (both)", X_c2_raw,           y_c2,     "y_c2_both_pred.npy",  scaler_both, ridge_both),
        ("CESM2 (LW)",   X_c2_raw[:, n_sw:], y_c2,     "y_c2_lw_pred.npy",   scaler_lw,   ridge_lw),
        ("CFMIP",        X_cfmip_raw,        cfmip_fb, "y_cfmip_pred.npy",    scaler_both, ridge_both),
    ]
    for label, X_oos, y_oos, fname, scaler, ridge in evals:
        y_pred = ridge.predict(scaler.transform(X_oos))
        r2   = r2_score(y_oos, y_pred)
        rmse = np.sqrt(mean_squared_error(y_oos, y_pred))
        print(f"  {label:<24s}  R²={r2:>7.3f}  RMSE={rmse:.3f} W/m²")
        np.save(ckpt / fname, y_pred)

    # ── CERES constraint ──────────────────────────────────────────────────────
    ceres_pred = ridge_both.predict(scaler_both.transform(X_ceres_raw))[0]
    np.save(ckpt / "ceres_pred.npy", np.array([ceres_pred], dtype=np.float32))
    print(f"\nCERES constraint: {ceres_pred:+.3f} W/m²  ({ceres_pred / 4:+.3f} W/m²/K)")

    # ── Ground-truth arrays (for eval notebook) ───────────────────────────────
    np.save(ckpt / "y_ga8.npy",      y_ga8)
    np.save(ckpt / "y_ga9.npy",      y_ga9)
    np.save(ckpt / "y_c2.npy",       y_c2)
    np.save(ckpt / "cfmip_fb.npy",   cfmip_fb)
    np.save(ckpt / "cfmip_models.npy", cfmip_models)
    print(f"\nAll outputs saved to {ckpt}/")


if __name__ == "__main__":
    main()
