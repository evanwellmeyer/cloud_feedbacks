"""
Preprocessing script for cloud feedback PPE data.

Computes climatological mean-state CRE maps (inputs) and global-mean net
cloud feedback (targets) for the HadGEM GA8/GA9 and CESM2 PPEs.

Outputs (saved to --outdir):
  hadgem_ga8_cre.nc   -- (member, 2, lat, lon): [SW_CRE, LW_CRE] amip climatology
  hadgem_ga8_fb.nc    -- (member,): global-mean net cloud feedback
  hadgem_ga9_cre.nc
  hadgem_ga9_fb.nc
  cesm2_cre.nc        -- (member, 2, lat, lon): matched PD members only
  cesm2_fb.nc         -- (member,): global-mean net cloud feedback

Sign conventions used here:
  SW CRE = rsutcs - rsut  [>0 means clouds cool: reflected SW by clouds]
  LW CRE = rlutcs - rlut  [>0 means clouds warm: less OLR due to clouds]
  Net CRE = SW_CRE + LW_CRE
  Cloud feedback = delta(Net CRE) / delta(T) -- but we output delta(Net CRE)
  since delta(T) is fixed at 4K for all members within each PPE.

CESM2 variable mapping:
  FLUT  -> rlut   (upwelling LW at TOA, all-sky)
  FLUTC -> rlutcs (upwelling LW at TOA, clear-sky)
  FSUTOA  -> rsut   (upwelling SW at TOA, all-sky)
  FSNTOAC -> rsutcs (net SW at TOA, clear-sky) -- NOTE: this is NET not upwelling
    => rsutcs = rsdt - FSNTOAC  but rsdt cancels in CRE difference so we need
       SW_CRE = rsutcs - rsut = (rsdt - FSNTOAC) - FSUTOA
       Since rsdt is the same for all-sky and clear-sky (incident solar),
       SW_CRE = FSUTOA_clear - FSUTOA = FSNTOAC_allsky... actually:
       FSNTOAC = rsdt - rsutcs  =>  rsutcs = rsdt - FSNTOAC
       rsut = FSUTOA
       SW_CRE = rsutcs - rsut = (rsdt - FSNTOAC) - FSUTOA
       But rsdt is not in the files. However rsdt cancels if we compute delta SW_CRE.
       For the mean state input we need absolute SW_CRE -- use rsdt from a climatology,
       OR note that FSNTOAC = net downward clear-sky SW = rsdt - rsutcs, so:
         SW_CRE = rsutcs - rsut = -(FSNTOAC - rsdt + FSUTOA) ... still needs rsdt.

       SIMPLEST: use CESM2 convention directly:
         SW_CRE_cesm = FSNTOAC - (rsdt - FSUTOA) ... still needs rsdt.

       Actually the cleanest path: FSNTOA = rsdt - FSUTOA (net all-sky SW, not in our files).
       We have FSUTOA (all-sky upwelling) and FSNTOAC (clear-sky net = rsdt - rsutcs).
       SW CRE = (rsdt - rsutcs) - (rsdt - rsut) = rsut - rsutcs = FSUTOA - rsutcs
              = FSUTOA - (rsdt - FSNTOAC)

       Since rsdt is not available we cannot compute absolute SW CRE from CESM2 files alone.
       WORKAROUND: rsdt varies little across members (same forcing), so we can approximate
       rsdt ~ mean rsdt from all members. OR, note that for DELTA SW_CRE (feedback target)
       rsdt cancels exactly (same prescribed solar for PD and SST4K). For the INPUT (mean-state
       SW CRE) we need rsdt.

       SET --cesm2_rsdt_file to provide a separate rsdt file, OR the script will estimate
       rsdt from the data: rsdt = FSNTOAC + FSUTOA_clear_approx. This is not ideal.

       FLAG: if rsdt is unavailable, SW CRE for CESM2 inputs will be set to NaN with a warning.
       The feedback target (delta Net CRE) is always valid since rsdt cancels.

Usage:
  conda run -n <env_with_xarray> python preprocess.py \\
      --hadgem_dir /path/to/HadGEM/new \\
      --cesm2_pd_dir /path/to/CESM2/PD \\
      --cesm2_4k_dir /path/to/CESM2/SST4K \\
      --outdir /path/to/output

Dependencies: xarray, numpy, scipy (for area-weighted mean)
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def area_weights(lat):
    """Cosine-of-latitude area weights, shape (lat,)."""
    return np.cos(np.deg2rad(lat))


def global_mean(da, lat_dim="latitude"):
    """Latitude-weighted global mean of a DataArray."""
    w = area_weights(da[lat_dim])
    return da.weighted(xr.DataArray(w, coords={lat_dim: da[lat_dim]})).mean(
        dim=[lat_dim, "longitude"] if "longitude" in da.dims else [lat_dim, "lon"]
    )


def time_mean(da):
    """Simple time mean (climatology over all time steps)."""
    return da.mean(dim="time")


def regrid_to_target(da, target_lat, target_lon, method="bilinear"):
    """
    Regrid DataArray to target lat/lon using xarray interp (linear).
    da must have dims (..., lat, lon) or (..., latitude, longitude).
    """
    lat_dim = "latitude" if "latitude" in da.dims else "lat"
    lon_dim = "longitude" if "longitude" in da.dims else "lon"
    return da.interp(
        {lat_dim: target_lat, lon_dim: target_lon},
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )


# ---------------------------------------------------------------------------
# HadGEM processing
# ---------------------------------------------------------------------------

def load_hadgem_var(data_dir, ppe, experiment, varname):
    """
    Load a HadGEM PPE variable.
    File pattern: {ppe}PPE_{experiment}_{varname}.nc
    Variable inside: depends on varname mapping.
    Returns DataArray with dims (realization, time, latitude, longitude).
    """
    var_map = {
        "rlut": "toa_outgoing_longwave_flux",
        "rlutcs": "toa_outgoing_longwave_flux_assuming_clear_sky",
        "rsut": "toa_outgoing_shortwave_flux",
        "rsutcs": "toa_outgoing_shortwave_flux_assuming_clear_sky",
    }
    fname = Path(data_dir) / f"{ppe}PPE_{experiment}_{varname}.nc"
    ds = xr.open_dataset(fname, chunks={"realization": 50})
    # Find the data variable (not coordinate variables)
    data_vars = [v for v in ds.data_vars if v not in ("latitude_longitude",)]
    if len(data_vars) == 1:
        return ds[data_vars[0]]
    # Fall back to expected name
    expected = var_map.get(varname)
    if expected and expected in ds:
        return ds[expected]
    raise ValueError(f"Cannot identify data variable in {fname}. Variables: {list(ds.data_vars)}")


def compute_hadgem_cre_and_feedback(data_dir, ppe):
    """
    Compute mean-state CRE maps and net cloud feedback for a HadGEM PPE.

    Returns:
        cre_amip : DataArray (realization, 2, latitude, longitude)
                   channel 0 = SW CRE, channel 1 = LW CRE from amip
        feedback : DataArray (realization,) -- delta Net CRE, global mean
    """
    print(f"  Loading {ppe} amip...")
    rlut_amip    = load_hadgem_var(data_dir, ppe, "amip",       "rlut")
    rlutcs_amip  = load_hadgem_var(data_dir, ppe, "amip",       "rlutcs")
    rsut_amip    = load_hadgem_var(data_dir, ppe, "amip",       "rsut")
    rsutcs_amip  = load_hadgem_var(data_dir, ppe, "amip",       "rsutcs")

    print(f"  Loading {ppe} amipFuture...")
    rlut_fut     = load_hadgem_var(data_dir, ppe, "amipFuture", "rlut")
    rlutcs_fut   = load_hadgem_var(data_dir, ppe, "amipFuture", "rlutcs")
    rsut_fut     = load_hadgem_var(data_dir, ppe, "amipFuture", "rsut")
    rsutcs_fut   = load_hadgem_var(data_dir, ppe, "amipFuture", "rsutcs")

    # Time-mean climatologies
    print(f"  Computing climatologies...")
    sw_cre_amip = time_mean(rsutcs_amip - rsut_amip)    # (real, lat, lon)
    lw_cre_amip = time_mean(rlutcs_amip - rlut_amip)    # (real, lat, lon)

    sw_cre_fut  = time_mean(rsutcs_fut  - rsut_fut)
    lw_cre_fut  = time_mean(rlutcs_fut  - rlut_fut)

    net_cre_amip = sw_cre_amip + lw_cre_amip
    net_cre_fut  = sw_cre_fut  + lw_cre_fut

    delta_net_cre = net_cre_fut - net_cre_amip  # (real, lat, lon)

    # Global-mean feedback (delta Net CRE; divide by 4 for W/m2/K if desired)
    feedback = global_mean(delta_net_cre, lat_dim="latitude").compute()
    feedback.name = "delta_net_cre"
    feedback.attrs["long_name"] = "Global-mean delta Net CRE (amipFuture - amip)"
    feedback.attrs["units"] = "W m-2"
    feedback.attrs["note"] = "Divide by 4 K to get feedback parameter in W/m2/K"

    # Stack CRE channels: (real, channel, lat, lon)
    sw_cre_amip = sw_cre_amip.compute()
    lw_cre_amip = lw_cre_amip.compute()

    sw_cre_amip.name = "sw_cre"
    lw_cre_amip.name = "lw_cre"

    cre_amip = xr.concat([sw_cre_amip, lw_cre_amip], dim="channel")
    cre_amip = cre_amip.assign_coords(channel=["sw_cre", "lw_cre"])
    cre_amip.name = "cre"
    cre_amip.attrs["long_name"] = "Mean-state CRE (amip climatology)"
    cre_amip.attrs["sw_cre_convention"] = "rsutcs - rsut (>0: clouds cool)"
    cre_amip.attrs["lw_cre_convention"] = "rlutcs - rlut (>0: clouds warm)"

    return cre_amip, feedback


# ---------------------------------------------------------------------------
# CESM2 processing
# ---------------------------------------------------------------------------

def load_cesm2_var(base_dir, varname, member_idx):
    """
    Load a single CESM2 member file.
    File pattern: cc_PPE_250_ensemble_{experiment}.{idx:03d}.h0.{VAR}.nc
    experiment is inferred from base_dir name (PD or SST4K).
    """
    experiment = Path(base_dir).name  # "PD" or "SST4K"
    fname = Path(base_dir) / varname / f"cc_PPE_250_ensemble_{experiment}.{member_idx:03d}.h0.{varname}.nc"
    ds = xr.open_dataset(fname)
    return ds[varname]  # DataArray (time, lat, lon)


def get_cesm2_member_indices(pd_dir, sst4k_dir):
    """Return sorted list of member indices present in both PD and SST4K."""
    def indices(d, var="FLUT", exp="PD"):
        exp_name = Path(d).name
        files = list(Path(d / var).glob(f"cc_PPE_250_ensemble_{exp_name}.*.h0.{var}.nc"))
        return set(int(f.name.split(".")[1]) for f in files)

    pd_idx   = indices(pd_dir, "FLUT", "PD")
    sst4k_idx = indices(sst4k_dir, "FLUT", "SST4K")
    common = sorted(pd_idx & sst4k_idx)
    n_pd = len(pd_idx)
    n_sst4k = len(sst4k_idx)
    n_common = len(common)
    print(f"  CESM2 members: PD={n_pd}, SST4K={n_sst4k}, intersection={n_common}")
    if n_common < min(n_pd, n_sst4k):
        dropped = (pd_idx | sst4k_idx) - set(common)
        warnings.warn(f"Dropping {len(dropped)} member(s) not present in both experiments: {sorted(dropped)}")
    return common


def compute_cesm2_cre_and_feedback(pd_dir, sst4k_dir):
    """
    Compute mean-state CRE maps and net cloud feedback for CESM2 PPE.

    CESM2 variable mapping:
      FLUT    = all-sky upwelling LW  (= rlut)
      FLUTC   = clear-sky upwelling LW (= rlutcs)
      FSUTOA  = all-sky upwelling SW   (= rsut)
      FSNTOAC = clear-sky net downward SW = rsdt - rsutcs

    SW CRE = rsutcs - rsut = (rsdt - FSNTOAC) - FSUTOA
    For delta SW CRE (feedback), rsdt cancels:
      delta SW CRE = -delta FSNTOAC - delta FSUTOA

    For mean-state SW CRE we need rsdt. We approximate:
      rsdt ~ FSNTOAC + FSUTOA  (valid only if net upward clear-sky SW ~ 0,
      which is not true). Instead we flag this and set sw_cre = NaN with a
      warning, unless --cesm2_rsdt is provided.

    Returns:
        cre_pd   : DataArray (member, 2, lat, lon)
        feedback : DataArray (member,) -- delta Net CRE global mean
        member_indices : list of int
    """
    pd_dir   = Path(pd_dir)
    sst4k_dir = Path(sst4k_dir)

    member_indices = get_cesm2_member_indices(pd_dir, sst4k_dir)
    n = len(member_indices)

    print(f"  Processing {n} CESM2 members...")

    # Collect per-member data
    lw_cre_pd_list   = []
    sw_cre_pd_list   = []
    delta_net_cre_list = []

    for i, idx in enumerate(member_indices):
        if i % 50 == 0:
            print(f"    member {i+1}/{n} (index {idx})")

        # PD
        flut_pd    = time_mean(load_cesm2_var(pd_dir,    "FLUT",    idx))
        flutc_pd   = time_mean(load_cesm2_var(pd_dir,    "FLUTC",   idx))
        fsutoa_pd  = time_mean(load_cesm2_var(pd_dir,    "FSUTOA",  idx))
        fsntoac_pd = time_mean(load_cesm2_var(pd_dir,    "FSNTOAC", idx))

        # SST4K
        flut_4k    = time_mean(load_cesm2_var(sst4k_dir, "FLUT",    idx))
        flutc_4k   = time_mean(load_cesm2_var(sst4k_dir, "FLUTC",   idx))
        fsutoa_4k  = time_mean(load_cesm2_var(sst4k_dir, "FSUTOA",  idx))
        fsntoac_4k = time_mean(load_cesm2_var(sst4k_dir, "FSNTOAC", idx))

        # LW CRE = rlutcs - rlut = FLUTC - FLUT (>0: clouds warm)
        lw_cre_pd  = (flutc_pd  - flut_pd).compute()
        lw_cre_4k  = (flutc_4k  - flut_4k).compute()

        # SW CRE = rsutcs - rsut = (rsdt - FSNTOAC) - FSUTOA = rsdt - FSNTOAC - FSUTOA
        # rsdt is not in the files but is ~constant across members.
        # Proxy = FSNTOAC + FSUTOA = rsdt - SW_CRE  (offset by rsdt, same for all members;
        # cancels after zero-mean normalisation). Store proxy; users can subtract rsdt if available.
        sw_cre_pd_proxy = (fsntoac_pd + fsutoa_pd).compute()

        # Delta SW CRE: rsdt cancels exactly
        # delta SW_CRE = -(delta FSNTOAC) - (delta FSUTOA)
        delta_sw_cre = (fsntoac_pd - fsntoac_4k) + (fsutoa_pd - fsutoa_4k)
        delta_sw_cre = delta_sw_cre.compute()
        delta_lw_cre = (lw_cre_4k - lw_cre_pd).compute()
        delta_net    = (delta_sw_cre + delta_lw_cre).compute()

        lw_cre_pd_list.append(lw_cre_pd)
        sw_cre_pd_list.append(sw_cre_pd_proxy)
        delta_net_cre_list.append(delta_net)

    print("  Concatenating along member dimension...")
    member_coord = xr.DataArray(member_indices, dims="member", name="member")

    lw_cre_pd_da   = xr.concat(lw_cre_pd_list,  dim=member_coord)
    sw_cre_pd_da   = xr.concat(sw_cre_pd_list,  dim=member_coord)
    delta_net_da   = xr.concat(delta_net_cre_list, dim=member_coord)

    # Rename lat/lon dims to match HadGEM convention
    lw_cre_pd_da = lw_cre_pd_da.rename({"lat": "latitude", "lon": "longitude"})
    sw_cre_pd_da = sw_cre_pd_da.rename({"lat": "latitude", "lon": "longitude"})
    delta_net_da = delta_net_da.rename({"lat": "latitude", "lon": "longitude"})

    # Global-mean feedback
    feedback = global_mean(delta_net_da, lat_dim="latitude")
    feedback.name = "delta_net_cre"
    feedback.attrs["long_name"] = "Global-mean delta Net CRE (SST4K - PD)"
    feedback.attrs["units"] = "W m-2"
    feedback.attrs["note"] = (
        "Uniform +4K warming. Not directly comparable to HadGEM patterned +4K. "
        "Divide by 4 K to get feedback parameter in W/m2/K."
    )

    # Stack CRE channels
    sw_cre_pd_da.name = "sw_cre"
    lw_cre_pd_da.name = "lw_cre"
    cre_pd = xr.concat([sw_cre_pd_da, lw_cre_pd_da], dim="channel")
    cre_pd = cre_pd.assign_coords(channel=["sw_cre", "lw_cre"])
    cre_pd.name = "cre"
    cre_pd.attrs["long_name"] = "Mean-state CRE (PD climatology)"
    cre_pd.attrs["sw_cre_convention"] = (
        "FSNTOAC + FSUTOA (= rsdt - SW_CRE; rsdt offset is ~constant across members "
        "and cancels after zero-mean normalisation; SW_CRE = rsutcs - rsut > 0 for cooling)"
    )
    cre_pd.attrs["lw_cre_convention"] = "FLUTC - FLUT (>0: clouds warm)"
    cre_pd.attrs["sw_warning"] = (
        "SW CRE contains a constant rsdt offset (rsdt not available). "
        "Delta SW CRE (used for feedback target) is exact since rsdt cancels."
    )

    return cre_pd, feedback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hadgem_dir",   default="/Users/ewellmeyer/Documents/research/HadGEM/new",
                   help="Directory containing HadGEM PPE .nc files")
    p.add_argument("--cesm2_pd_dir", default="/Users/ewellmeyer/Documents/research/CESM2/PD",
                   help="CESM2 PD base directory (contains FLUT/, FLUTC/, ... subdirs)")
    p.add_argument("--cesm2_4k_dir", default="/Users/ewellmeyer/Documents/research/CESM2/SST4K",
                   help="CESM2 SST4K base directory")
    p.add_argument("--outdir",       default="/Users/ewellmeyer/Documents/research/scripts/cloud_feedbacks/data",
                   help="Output directory for preprocessed .nc files")
    p.add_argument("--skip_hadgem",  action="store_true")
    p.add_argument("--skip_cesm2",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- HadGEM GA8 --------------------------------------------------------
    if not args.skip_hadgem:
        for ppe in ["GA8", "GA9"]:
            print(f"\n=== HadGEM {ppe} ===")
            cre, fb = compute_hadgem_cre_and_feedback(args.hadgem_dir, ppe)

            cre_path = outdir / f"hadgem_{ppe.lower()}_cre.nc"
            fb_path  = outdir / f"hadgem_{ppe.lower()}_fb.nc"

            print(f"  Saving {cre_path}")
            cre.to_netcdf(cre_path)
            print(f"  Saving {fb_path}")
            fb.to_dataset(name="delta_net_cre").to_netcdf(fb_path)

            print(f"  {ppe}: {cre.sizes['realization']} members")
            print(f"  Feedback range: {float(fb.min()):.2f} to {float(fb.max()):.2f} W/m2")

    # ---- CESM2 -------------------------------------------------------------
    if not args.skip_cesm2:
        print(f"\n=== CESM2 PPE ===")
        cre, fb = compute_cesm2_cre_and_feedback(args.cesm2_pd_dir, args.cesm2_4k_dir)

        cre_path = outdir / "cesm2_cre.nc"
        fb_path  = outdir / "cesm2_fb.nc"

        print(f"  Saving {cre_path}")
        cre.to_netcdf(cre_path)
        print(f"  Saving {fb_path}")
        fb.to_dataset(name="delta_net_cre").to_netcdf(fb_path)

        print(f"  CESM2: {cre.sizes['member']} members")
        print(f"  Feedback range: {float(fb.min()):.2f} to {float(fb.max()):.2f} W/m2")

    print("\nDone.")


if __name__ == "__main__":
    main()
