"""
Preprocessing script for cloud feedback PPE data.

Computes climatological mean-state CRE maps (inputs) and global-mean net
cloud feedback (targets) for the HadGEM GA8/GA9 and CESM2 PPEs.

Outputs (saved to --outdir):
  hadgem_ga8_cre.nc   -- (realization, 2, lat, lon): [SW_CRE, LW_CRE] amip climatology
  hadgem_ga8_fb.nc    -- (realization,): global-mean net cloud feedback
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
  FSNTOAC -> clear-sky net downward SW at TOA = rsdt - rsutcs

  Therefore:
    rsutcs = rsdt - FSNTOAC
    SW_CRE = rsutcs - rsut = (rsdt - FSNTOAC) - FSUTOA

  The mean-state SW CRE needs rsdt, which is not present in these files. The script
  therefore stores the exact proxy

    FSNTOAC + FSUTOA = rsdt - SW_CRE

  which differs from SW_CRE by the nearly fixed incoming-solar offset rsdt.

  For the feedback target, rsdt cancels exactly:

    delta SW_CRE
      = [(rsdt - FSNTOAC_4K) - FSUTOA_4K] - [(rsdt - FSNTOAC_PD) - FSUTOA_PD]
      = (FSNTOAC_PD - FSNTOAC_4K) + (FSUTOA_PD - FSUTOA_4K)

  so the global-mean delta Net CRE target is exact for CESM2 even though the stored
  mean-state SW channel is an rsdt-offset proxy.

Usage:
  conda run -n <env_with_xarray> python preprocess.py \\
      --hadgem_dir /path/to/HadGEM/new \\
      --cesm2_pd_dir /path/to/CESM2/PD \\
      --cesm2_4k_dir /path/to/CESM2/SST4K \\
      --outdir /path/to/output

Dependencies: xarray, numpy, scipy (for area-weighted mean)
"""

import argparse
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


def sw_cre(rsutcs, rsut):
    """Shortwave cloud radiative effect using the project sign convention."""
    return rsutcs - rsut


def lw_cre(rlutcs, rlut):
    """Longwave cloud radiative effect using the project sign convention."""
    return rlutcs - rlut


def cesm2_sw_cre_proxy(fsntoac, fsutoa):
    """Exact CESM2 proxy equal to rsdt - SW_CRE when rsdt is unavailable."""
    return fsntoac + fsutoa


def cesm2_delta_sw_cre(fsntoac_pd, fsutoa_pd, fsntoac_4k, fsutoa_4k):
    """Exact CESM2 delta SW CRE; rsdt cancels between PD and SST4K."""
    return (fsntoac_pd - fsntoac_4k) + (fsutoa_pd - fsutoa_4k)


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
    ds = xr.open_dataset(fname, chunks={"realization": 50}, decode_timedelta=False)
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
    sw_cre_amip = time_mean(sw_cre(rsutcs_amip, rsut_amip))  # (real, lat, lon)
    lw_cre_amip = time_mean(lw_cre(rlutcs_amip, rlut_amip))  # (real, lat, lon)

    sw_cre_fut  = time_mean(sw_cre(rsutcs_fut, rsut_fut))
    lw_cre_fut  = time_mean(lw_cre(rlutcs_fut, rlut_fut))

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

    For delta SW CRE (feedback), rsdt cancels exactly:
      delta SW CRE = (FSNTOAC_PD - FSNTOAC_4K) + (FSUTOA_PD - FSUTOA_4K)

    For mean-state SW CRE we do not have rsdt, so we store the exact proxy
      FSNTOAC + FSUTOA = rsdt - SW_CRE
    which preserves all member-to-member structure up to the shared rsdt offset.

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
        lw_cre_pd  = lw_cre(flutc_pd, flut_pd).compute()
        lw_cre_4k  = lw_cre(flutc_4k, flut_4k).compute()

        # SW CRE = rsutcs - rsut = (rsdt - FSNTOAC) - FSUTOA = rsdt - FSNTOAC - FSUTOA
        # rsdt is not in the files but is ~constant across members.
        # Proxy = FSNTOAC + FSUTOA = rsdt - SW_CRE  (offset by rsdt, same for all members;
        # cancels after zero-mean normalisation). Store proxy; users can subtract rsdt if available.
        sw_cre_pd_proxy = cesm2_sw_cre_proxy(fsntoac_pd, fsutoa_pd).compute()

        # Delta SW CRE: rsdt cancels exactly
        delta_sw_cre = cesm2_delta_sw_cre(fsntoac_pd, fsutoa_pd, fsntoac_4k, fsutoa_4k)
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

# ---------------------------------------------------------------------------
# CFMIP processing
# ---------------------------------------------------------------------------

# Common 2.5-degree target grid for regridding CFMIP models before concatenation.
# HadGEM and CESM2 PPE outputs will also need regridding to this grid before
# cross-dataset comparisons or joint training (Stage 1d).
COMMON_LAT = np.arange(-88.75, 90.0, 2.5)   # 72 points
COMMON_LON = np.arange(0.0,   360.0, 2.5)   # 144 points

# Models present in both PD and amip-future4K (GFDL-CM4 excluded: no PD amip)
CFMIP_MODELS = [
    "BCC-CSM2-MR",
    "CanESM5",
    "CESM2",
    "CNRM-CM6-1",
    "E3SM-1-0",
    "HadGEM3-GC31-LL",
    "IPSL-CM6A-LR",
    "MIROC6",
    "MRI-ESM2-0",
    "TaiESM1",
]

# Common period shared by all 10 models
CFMIP_CLIM_SLICE = slice("1979-01", "2014-12")


def load_cfmip_var(data_dir, model, varname):
    """
    Load a CFMIP variable for one model, concatenating multiple files if present.
    File pattern: {var}_Amon_{model}_amip*_{variant}_{grid}_{dates}.nc
    Returns DataArray with dim (time, lat, lon).
    """
    files = sorted(Path(data_dir).glob(f"{varname}_Amon_{model}_*.nc"))
    if not files:
        raise FileNotFoundError(f"No files found for {varname} {model} in {data_dir}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=xr.SerializationWarning)
        warnings.filterwarnings("ignore", "In a future version of xarray the default value for data_vars",
                                category=FutureWarning)
        ds = xr.open_mfdataset(files, combine="by_coords", data_vars="all",
                               decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    da = ds[varname].sel(time=CFMIP_CLIM_SLICE)
    if da.sizes["time"] == 0:
        raise ValueError(f"No data in {CFMIP_CLIM_SLICE} for {varname} {model}")
    return da


def compute_cfmip_cre_and_feedback(pd_dir, future4k_dir, models=None):
    """
    Compute mean-state CRE maps and net cloud feedback for CFMIP models.

    Uses the common period 1979-2014 for all models.
    GFDL-CM4 is excluded (no PD amip run available).

    Returns:
        cre_pd   : DataArray (model, channel, lat, lon), channel = [sw_cre, lw_cre]
        feedback : DataArray (model,) -- global-mean delta Net CRE
    """
    if models is None:
        models = CFMIP_MODELS

    cre_list = []
    fb_list  = []

    for model in models:
        print(f"  {model}...")
        rlut_pd    = time_mean(load_cfmip_var(pd_dir,       model, "rlut"))
        rlutcs_pd  = time_mean(load_cfmip_var(pd_dir,       model, "rlutcs"))
        rsut_pd    = time_mean(load_cfmip_var(pd_dir,       model, "rsut"))
        rsutcs_pd  = time_mean(load_cfmip_var(pd_dir,       model, "rsutcs"))

        rlut_4k    = time_mean(load_cfmip_var(future4k_dir, model, "rlut"))
        rlutcs_4k  = time_mean(load_cfmip_var(future4k_dir, model, "rlutcs"))
        rsut_4k    = time_mean(load_cfmip_var(future4k_dir, model, "rsut"))
        rsutcs_4k  = time_mean(load_cfmip_var(future4k_dir, model, "rsutcs"))

        sw_cre_pd = sw_cre(rsutcs_pd, rsut_pd).compute()
        lw_cre_pd = lw_cre(rlutcs_pd, rlut_pd).compute()

        delta_sw  = sw_cre(rsutcs_4k, rsut_4k)   - sw_cre_pd
        delta_lw  = lw_cre(rlutcs_4k, rlut_4k)   - lw_cre_pd
        delta_net = (delta_sw + delta_lw).compute()

        # Rename lat/lon to common names if needed
        rename = {}
        if "lat" in sw_cre_pd.dims:
            rename["lat"] = "latitude"
        if "lon" in sw_cre_pd.dims:
            rename["lon"] = "longitude"
        if rename:
            sw_cre_pd = sw_cre_pd.rename(rename)
            lw_cre_pd = lw_cre_pd.rename(rename)
            delta_net = delta_net.rename(rename)

        fb = global_mean(delta_net, lat_dim="latitude")
        fb_list.append(float(fb))

        # Regrid to common 2.5° grid before concatenating across models
        sw_cre_pd = regrid_to_target(sw_cre_pd, COMMON_LAT, COMMON_LON).compute()
        lw_cre_pd = regrid_to_target(lw_cre_pd, COMMON_LAT, COMMON_LON).compute()

        cre_model = xr.concat([sw_cre_pd, lw_cre_pd], dim="channel")
        cre_list.append(cre_model)

        print(f"    delta Net CRE = {float(fb):.2f} W/m2")

    model_coord = xr.DataArray(models, dims="model", name="model")
    cre_da = xr.concat(cre_list, dim=model_coord)
    cre_da = cre_da.assign_coords(channel=["sw_cre", "lw_cre"])
    cre_da.name = "cre"
    cre_da.attrs["long_name"] = "Mean-state CRE (amip 1979-2014 climatology)"
    cre_da.attrs["sw_cre_convention"] = "rsutcs - rsut (>0: clouds cool)"
    cre_da.attrs["lw_cre_convention"] = "rlutcs - rlut (>0: clouds warm)"

    fb_da = xr.DataArray(fb_list, coords={"model": models}, dims="model", name="delta_net_cre")
    fb_da.attrs["long_name"] = "Global-mean delta Net CRE (amip-future4K minus amip, 1979-2014)"
    fb_da.attrs["units"] = "W m-2"
    fb_da.attrs["note"] = (
        "Patterned +4K warming (amipFuture protocol). "
        "Divide by 4 K to get feedback parameter in W/m2/K. "
        "GFDL-CM4 excluded (no PD amip run)."
    )

    return cre_da, fb_da


# ---------------------------------------------------------------------------
# CERES processing
# ---------------------------------------------------------------------------

def compute_ceres_cre(ceres_file):
    """
    Compute climatological mean CRE from CERES EBAF-TOA Ed4.2.1.

    Uses the full available record (all time steps in the file).
    Regrids to the common 2.5° grid to match CFMIP outputs.

    Variable mapping:
      toa_sw_all_mon  -> rsut
      toa_sw_clr_c_mon -> rsutcs
      toa_lw_all_mon  -> rlut
      toa_lw_clr_c_mon -> rlutcs

    Returns:
        cre : DataArray (channel, latitude, longitude), channel = [sw_cre, lw_cre]
    """
    ds = xr.open_dataset(ceres_file, mask_and_scale=True)

    # Compute time-mean CRE on native 1x1 grid
    sw = time_mean(ds["toa_sw_clr_c_mon"] - ds["toa_sw_all_mon"])   # SW CRE
    lw = time_mean(ds["toa_lw_clr_c_mon"] - ds["toa_lw_all_mon"])   # LW CRE

    # Rename to common dim names
    sw = sw.rename({"lat": "latitude", "lon": "longitude"})
    lw = lw.rename({"lat": "latitude", "lon": "longitude"})

    print(f"  SW CRE global mean: {float(sw.mean()):.1f} W/m2  (expect ~-47)")
    print(f"  LW CRE global mean: {float(lw.mean()):.1f} W/m2  (expect ~+26)")
    print(f"  Time period: {str(ds.time[0].values)[:10]} to {str(ds.time[-1].values)[:10]}"
          f"  ({ds.sizes['time']} months)")

    # Regrid to common 2.5° grid
    sw = regrid_to_target(sw, COMMON_LAT, COMMON_LON).compute()
    lw = regrid_to_target(lw, COMMON_LAT, COMMON_LON).compute()

    cre = xr.concat([sw, lw], dim="channel")
    cre = cre.assign_coords(channel=["sw_cre", "lw_cre"])
    cre.name = "cre"
    cre.attrs["long_name"] = "CERES EBAF-TOA Ed4.2.1 climatological mean CRE"
    cre.attrs["sw_cre_convention"] = "toa_sw_clr_c_mon - toa_sw_all_mon (>0 impossible globally; expect ~-47 W/m2)"
    cre.attrs["lw_cre_convention"] = "toa_lw_clr_c_mon - toa_lw_all_mon (>0: clouds warm; expect ~+26 W/m2)"
    cre.attrs["source_file"] = str(Path(ceres_file).name)

    return cre


# ---------------------------------------------------------------------------
# Argument parsing and main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hadgem_dir",      default="/Users/ewellmeyer/Documents/research/HadGEM/new",
                   help="Directory containing HadGEM PPE .nc files")
    p.add_argument("--cesm2_pd_dir",    default="/Users/ewellmeyer/Documents/research/CESM2/PD",
                   help="CESM2 PD base directory (contains FLUT/, FLUTC/, ... subdirs)")
    p.add_argument("--cesm2_4k_dir",    default="/Users/ewellmeyer/Documents/research/CESM2/SST4K",
                   help="CESM2 SST4K base directory")
    p.add_argument("--cfmip_pd_dir",    default="/Users/ewellmeyer/Documents/research/AMIP/PD/cfbvars",
                   help="CFMIP amip PD directory (flat, CMIP6-named .nc files)")
    p.add_argument("--cfmip_4k_dir",    default="/Users/ewellmeyer/Documents/research/AMIP/future4K/cfbvars",
                   help="CFMIP amip-future4K directory")
    p.add_argument("--outdir",          default="/Users/ewellmeyer/Documents/research/scripts/cloud_feedbacks/data",
                   help="Output directory for preprocessed .nc files")
    p.add_argument("--ceres_file",      default="/Users/ewellmeyer/Documents/research/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202601.nc",
                   help="CERES EBAF-TOA Ed4.2.1 netCDF file")
    p.add_argument("--skip_hadgem",     action="store_true")
    p.add_argument("--skip_cesm2",      action="store_true")
    p.add_argument("--skip_cfmip",      action="store_true")
    p.add_argument("--skip_ceres",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- HadGEM GA8 & GA9 --------------------------------------------------
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

    # ---- CFMIP -------------------------------------------------------------
    if not args.skip_cfmip:
        print(f"\n=== CFMIP models ===")
        cre, fb = compute_cfmip_cre_and_feedback(args.cfmip_pd_dir, args.cfmip_4k_dir)

        cre_path = outdir / "cfmip_cre.nc"
        fb_path  = outdir / "cfmip_fb.nc"

        print(f"  Saving {cre_path}")
        cre.to_netcdf(cre_path)
        print(f"  Saving {fb_path}")
        fb.to_dataset(name="delta_net_cre").to_netcdf(fb_path)

        print(f"  Models: {list(fb.model.values)}")
        print(f"  Feedback range: {float(fb.min()):.2f} to {float(fb.max()):.2f} W/m2")

    # ---- CERES -------------------------------------------------------------
    if not args.skip_ceres:
        print(f"\n=== CERES EBAF-TOA ===")
        cre = compute_ceres_cre(args.ceres_file)

        cre_path = outdir / "ceres_cre.nc"
        print(f"  Saving {cre_path}")
        cre.to_netcdf(cre_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
