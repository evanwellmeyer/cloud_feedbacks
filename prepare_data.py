"""
prepare_data.py
---------------
Regrid all CRE/feedback datasets to the HadGEM native 144×192 grid and save
raw (unnormalised) float32 tensors to data/tensors.npz.

Run once before any training script.  Training scripts apply their own
normalisation so each model family can do the right thing (per-pixel z-score
for the CNN, StandardScaler per-fold for ridge).

Usage:
    python prepare_data.py [--data_dir PATH]
"""

import argparse
import numpy as np
import xarray as xr
import xesmf as xe
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def build_grid(lat, lon):
    return xr.Dataset({"lat": ("lat", lat), "lon": ("lon", lon)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(DATA))
    args = parser.parse_args()
    data = Path(args.data_dir)

    # Build HadGEM target grid from the GA8 file
    _hg = xr.open_dataset(data / "hadgem_ga8_cre.nc")["cre"]
    hadgem_lat = _hg.latitude.values
    hadgem_lon = _hg.longitude.values
    ds_target  = build_grid(hadgem_lat, hadgem_lon)
    print(f"Target grid: {len(hadgem_lat)} lat × {len(hadgem_lon)} lon")

    regridder_cache = {}

    def get_regridder(src_lat, src_lon, label):
        if label not in regridder_cache:
            regridder_cache[label] = xe.Regridder(
                build_grid(src_lat, src_lon), ds_target,
                "bilinear", periodic=True, ignore_degenerate=True,
            )
            print(f"  Built regridder: {label}")
        return regridder_cache[label]

    def to_tensor(cre_nc, fb_nc, label):
        """Load CRE + feedback, regrid to HadGEM 144×192 if needed.
        Returns X (n, 2, lat, lon) float32 and y (n,) float32."""
        cre = xr.open_dataset(cre_nc)["cre"]
        fb  = xr.open_dataset(fb_nc)["delta_net_cre"].values.astype(np.float32)
        src_lat = cre.latitude.values
        src_lon = cre.longitude.values
        on_native = (
            len(src_lat) == len(hadgem_lat)
            and np.allclose(src_lat, hadgem_lat, atol=0.01)
            and np.allclose(src_lon, hadgem_lon, atol=0.01)
        )
        channels = []
        for ch in ("sw_cre", "lw_cre"):
            da = cre.sel(channel=ch)
            if on_native:
                arr = da.values.astype(np.float32)
            else:
                arr = get_regridder(src_lat, src_lon, label)(da).values.astype(np.float32)
            channels.append(arr)
        return np.stack(channels, axis=1), fb

    print("Loading GA8...")
    X_ga8, y_ga8 = to_tensor(data / "hadgem_ga8_cre.nc", data / "hadgem_ga8_fb.nc", "ga8")
    print("Loading GA9...")
    X_ga9, y_ga9 = to_tensor(data / "hadgem_ga9_cre.nc", data / "hadgem_ga9_fb.nc", "ga9")
    print("Loading CESM2...")
    X_c2,  y_c2  = to_tensor(data / "cesm2_cre.nc",      data / "cesm2_fb.nc",      "cesm2")

    print("Loading CFMIP...")
    cfmip_cre    = xr.open_dataset(data / "cfmip_cre.nc")["cre"]
    cfmip_fb     = xr.open_dataset(data / "cfmip_fb.nc")["delta_net_cre"].values.astype(np.float32)
    cfmip_models = np.array([str(m) for m in cfmip_cre.model.values])
    rg_cfmip     = get_regridder(cfmip_cre.latitude.values, cfmip_cre.longitude.values, "cfmip")
    X_cfmip = np.stack([
        np.stack([
            rg_cfmip(cfmip_cre.sel(channel=ch, model=mod)).values.astype(np.float32)
            for ch in ("sw_cre", "lw_cre")
        ], axis=0)
        for mod in cfmip_models
    ])  # (10, 2, 144, 192)

    print("Loading CERES...")
    ceres_cre = xr.open_dataset(data / "ceres_cre.nc")["cre"]
    rg_ceres  = get_regridder(ceres_cre.latitude.values, ceres_cre.longitude.values, "ceres")
    X_ceres   = np.stack([
        rg_ceres(ceres_cre.sel(channel=ch)).values.astype(np.float32)
        for ch in ("sw_cre", "lw_cre")
    ])[None]  # (1, 2, 144, 192)

    out = data / "tensors.npz"
    np.savez_compressed(
        out,
        X_ga8=X_ga8, y_ga8=y_ga8,
        X_ga9=X_ga9, y_ga9=y_ga9,
        X_c2=X_c2,   y_c2=y_c2,
        X_cfmip=X_cfmip, cfmip_fb=cfmip_fb,
        X_ceres=X_ceres,
        cfmip_models=cfmip_models,
        hadgem_lat=hadgem_lat,
        hadgem_lon=hadgem_lon,
    )
    print(f"\nSaved → {out}")
    print(f"  GA8  : {X_ga8.shape}  y {y_ga8.min():.2f}–{y_ga8.max():.2f} W/m²")
    print(f"  GA9  : {X_ga9.shape}  y {y_ga9.min():.2f}–{y_ga9.max():.2f} W/m²")
    print(f"  CESM2: {X_c2.shape}   y {y_c2.min():.2f}–{y_c2.max():.2f} W/m²")
    print(f"  CFMIP: {X_cfmip.shape}  ({len(cfmip_models)} models)")
    print(f"  CERES: {X_ceres.shape}")


if __name__ == "__main__":
    main()
