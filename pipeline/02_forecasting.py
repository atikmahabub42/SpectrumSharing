#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
02_forecasting.py -- Stage 2 of the sense-forecast-share pipeline
=================================================================
Sub-channel occupancy forecasting on the PSD tier of the Fireworks dataset
(IEEE DataPort, DOI 10.21227/f2rv-ms26), labelled by the stage-1 detector.

LABELS (paper Sec. IV)
  Every band is split into C = 32 sub-channels; the excess e_{t,c} (dB) of a
  sub-channel over its noise floor (Q0.10 of the training span) becomes
  y_{t,c} = 1{e_{t,c} > eta_b}. eta_b is calibrated per band to OSCAR-Net
  (FedAvg) decisions from <results>/01_sensing/sensing_handoff/: the value in
  [0, 20] dB that agrees best (balanced accuracy) with the detector on the
  training span; bands without enough co-located I/Q recordings keep 6 dB.
  dataset_card.csv lists eta_b, the number of calibration points and the
  agreement of every band. --input psd uses a fixed eta = 6 dB instead.

MODELS (--family)
  gem   GEM (Gated Expert Mixture, paper Sec. VI, Algorithm 2) and its
        streaming drift-compensated variant GEM-DCS (gem_dcs); the DCS step
        gamma_b is chosen per band on validation from dcs_etas.
  sota  N-HiTS, PatchTST, iTransformer, Bi-ConvLSTM, SpectrumFM
  all   (default) both families, one after the other, in separate folders:
        <results>/02_forecasting       (SOTA; read by 03_sharing_baselines.py)
        <results>/02b_forecasting_gem  (GEM;  read by 03b_sharing_tide.py)
  Baselines persistence, seasonal naive and historical mean are always run.

PROTOCOL
  window W = 6 sweeps -> horizon H = 2; chronological 65/15/20 split inside
  32 time blocks per band; 3 seeds; 40 epochs (centralized) / 40 FedAvg
  rounds of 2 local epochs. Main federated setting: K = 8 non-IID temporal
  clients. Client sweep: K in {4, 8, 12, 16, 24, 32} x {non-IID, IID} x
  {FedAvg, FedProx}. Per-band parameters ('loc_' prefix) are never averaged.

OUTPUT (CSV)  forecast_overall(_agg).csv, forecast_per_horizon.csv,
  forecast_per_snr_*.csv, forecast_complexity.csv, fl_communication(_agg).csv,
  client_partitions.csv, dataset_card.csv, reference_baselines.csv and
  stage2/ (labels + forecasts of the best model per mode, for stage 3).

USAGE
  python 02_forecasting.py                                 # both families
  python 02_forecasting.py --family gem --quick            # smoke test
  python 02_forecasting.py --family sota --task 3          # one part of the grid
  python 02_forecasting.py --family gem --list-tasks
Finished parts are cached in <out>/parts_v2/; reruns resume.
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import hashlib
import json
import math
import os
import random
import sys
import time
import zlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("PyTorch is required:  pip install torch")

# ==========================================================================
#                                  CONFIG
# ==========================================================================
# Paths: --root / --out on the command line, or the environment variables
# SFS_DATA (export .h5 folder) and SFS_RESULTS (pipeline results folder).
DATA_ROOT = os.environ.get("SFS_DATA", os.path.join(".", "data", "fireworks_export"))
PIPELINE_ROOT = os.environ.get("SFS_RESULTS", os.path.join(".", "results"))
RESULTS = PIPELINE_ROOT

# per-family defaults (see --family)
FAMILIES = {
    "sota": {"forecast_models": [],
             "sota_models": ["nhits", "patchtst", "itransformer", "biconvlstm_s2s", "spectrumfm_f"],
             "sota_models_ablation": ["nhits", "patchtst", "itransformer", "biconvlstm_s2s", "spectrumfm_f"],
             "out_dir": os.path.join(PIPELINE_ROOT, "02_forecasting")},
    "gem": {"forecast_models": ["gem"], "sota_models": [], "sota_models_ablation": [],
            "out_dir": os.path.join(PIPELINE_ROOT, "02b_forecasting_gem")},
}

CONFIG: Dict = {
    # ---- input ---------------------------------------------------------------
    "input": "sensing",               # sensing (PSD labelled by the 01 detector) | psd (eta_db) | sensing_grid
    "sensing_dir": os.path.join(PIPELINE_ROOT, "01_sensing", "sensing_handoff"),
    "sensing_dir_fallbacks": [os.path.join(PIPELINE_ROOT, "01_sensing")],
    "sensing_source": "main",         # main (OSCAR-Net) | sota | energy
    "source_ablation": [],            # e.g. "auto" = also re-run with the sota / energy labels (not in the paper)
    "visit_busy_frac": 0.5,           # a recording is sensed busy when >= this share of its frames is H1
    "recording_gap_s": 1.0,           # I/Q frames further apart start a new recording
    "calib_eta_grid": (0.0, 20.0, 0.25),   # candidate thresholds (dB) for the per-band calibration
    "calib_max_lag_s": None,          # recording <-> sweep match; None = max(2 sweeps, 5 s)
    "calib_min_points": 12,           # fewer matched recordings in a band -> eta_db
    "tile_half_hz": None,             # None = half the spacing of the tile centres
    "min_tile_coverage": 0.5,         # sweeps with fewer observed tiles are gaps
    "data_root": DATA_ROOT,
    "source": "",                     # explicit session folder or .h5 ("" = auto-discovery)
    "prefer": "h5",                   # "h5" (gap-handled export) or "session" (raw DataPort release)
    "family": "all",                  # all | sota | gem
    "out_dir": FAMILIES["gem"]["out_dir"],
    "bands": [],                      # [] = every band in the data
    "exclude_bands": [],
    "devices": [],                    # [] = every radio (.h5 'radio' attribute), as dataset_arms "combined"
    # ---- sub-channels ------------------------------------------------------
    "subchannel_mode": "equal",       # "equal": C sub-channels per band | "fixed": subchannel_hz wide
    "subchannels_per_band": 32,
    "subchannel_hz": 2.5e6,           # used by "fixed"
    # ---- labels (identical to the 2.4 GHz runs) ----------------------------
    "noise_quantile": 0.10,
    "eta_db": 6.0,
    "snr_bin_start_db": 6.0,
    "snr_bin_width_db": 2.0,
    "snr_bin_max_db": 30.0,           # 2 dB bins 6-8 ... 28-30, then one "> 30" bin
    # ---- windows / split ---------------------------------------------------
    "lookback": 6, "horizon": 2, "stride": 1,    # W = 6 sweeps -> H = 2 (paper)
    "val_frac": 0.15, "test_frac": 0.20,    # chronological within each (time block, band), gap = W + H
    "split_blocks": 32,               # as the sensing pipeline; reduced automatically if windows do not fit
    "max_gap_steps": 3.0,             # windows spanning a gap > 3 x median step (dropped sweeps) are removed
    "auto_window": True, "min_train_windows": 64, "min_eval_windows": 8,
    "min_train_share": 0.55,          # realised train share of the windows (target 0.65; gaps cost windows)
    "window_fallbacks": [(32, 8), (24, 8), (24, 6), (16, 4), (12, 4), (8, 2), (6, 2), (4, 2), (4, 1)],
    "min_train_windows_floor": 24,    # used only when no window fits with min_train_windows
    "drop_short_scenarios": True,     # drop bands with too few sweeps instead of stopping
    "input_repr": "binary",
    "baselines": ["persistence", "seasonal_naive", "historical_mean"],
    # ---- models -------------------------------------------------------------
    "forecast_models": FAMILIES["gem"]["forecast_models"],   # set by --family
    "sota_models": [],
    "sota_models_ablation": [],
    "dcs": True,
    "dcs_etas": [0.0, 0.01, 0.03, 0.1, 0.3],
    "modes": ["local", "centralized", "fedavg", "fedprox"],
    # ---- client / partition grid --------------------------------------------
    "ab_clients": [4, 8, 12, 16, 24, 32], # the sensing pipeline's client counts
    "main_clients": 8,                # C of the main federated run (n_time_blocks_single_device)
    "ab_partitions": ["noniid", "iid"],   # + "band" (one band per client) as an extra control
    "min_per_class_per_client": 8,    # merge adjacent time stretches until both classes are held
    "ab_algos": ["fedavg", "fedprox"],
    "ab_reference_modes": ["centralized", "local"],
    "ab_stage2_algos": ["fedavg"],    # FL algorithm whose forecasts feed allocation / sharing
    "snr_horizons": [1, 2],
    "seeds": [0, 1, 2],
    "seed": 20260807,                 # same base seed as the sensing pipeline
    "epochs": 40, "batch_size": 256, "lr": 1e-3, "weight_decay": 1e-4,
    "patience": 6, "d_model": 64, "dropout": 0.1, "grad_clip": 1.0,
    "fl_rounds": 40, "fl_local_epochs": 2, "fedprox_mu": 0.01, "fl_patience": 8,
    "fl_norm": "groupnorm",           # BatchNorm -> GroupNorm in federated training (sensing pipeline)
    "target_pfa": 0.10,               # Neyman-Pearson operating point reported next to 0.5 / F1-opt
    "fm_pretrain_epochs": 10, "fm_mask_ratio": 0.30,
    "decision_threshold": 0.5,
    # ---- profiling -----------------------------------------------------------
    "prof_iters": 50, "prof_warmup": 5, "prof_batch": 256,
    "device": "auto",
    "resume": True,
    "verbose": True,
}

QUICK = {"epochs": 2, "fl_rounds": 2, "fl_local_epochs": 1, "fm_pretrain_epochs": 1,
         "snr_horizons": [1], "seeds": [0], "prof_iters": 3, "prof_warmup": 1, "patience": 1,
         "fl_patience": 1}

_LOG_FH = None


def log(m: str = "") -> None:
    if CONFIG.get("verbose", True):
        print(m, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(m + "\n")
        _LOG_FH.flush()


def set_seed(sd: int) -> None:
    random.seed(sd)
    np.random.seed(sd % (2 ** 32 - 1))
    torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_seed(cfg: Dict, *parts) -> int:
    key = "|".join(str(p) for p in parts)
    return int((int(cfg["seed"]) + zlib.crc32(key.encode())) % (2 ** 31 - 1))


def get_device(cfg: Dict) -> torch.device:
    if cfg["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])


# ==========================================================================
#            DATA: one multi-band session -> one "scenario" per BAND
# ==========================================================================
#  The Fireworks (mass-gathering) dataset is ONE 51-min session that sweeps
#  43 tiles of 20 MHz over seven bands (ISM_2400, the U-NII allocations and
#  the lower 6 GHz edge). Every band is treated as one scenario / one natural
#  federated client: the bands carry different traffic (Wi-Fi 2.4 GHz,
#  Wi-Fi 5 GHz, quiet DFS blocks, ...), i.e. real non-IID data.
#  Accepted inputs (auto-discovered under CONFIG["data_root"]):
#    * the IEEE DataPort release folder  (session_meta.json + sweeps_*.csv +
#      psd_*.f16; raw 1024-bin PSD rows, all seven bands)
#    * an export .h5 of the dataset (gap-handled /grid/<band>, or
#      /sweeps + /psd)
# ==========================================================================
_CSV_ALIASES = {
    "band": ["band_name", "band", "band_label"],
    "sweep": ["sweep", "sweep_idx", "sweep_index", "sweep_id"],
    "utc": ["utc_start", "utc", "t_start", "timestamp", "time_utc", "unix_time", "t_utc"],
    "f_lo": ["f_lo_hz", "f_lo", "flo_hz"],
    "f_hi": ["f_hi_hz", "f_hi", "fhi_hz"],
    "rate": ["samp_rate_hz", "sample_rate_hz", "samp_rate", "fs_hz"],
    "ctr": ["center_hz", "center_freq_hz", "fc_hz", "freq_hz"],
    "prow": ["psd_row", "psd_index", "row"],
}


def _is_session_dir(d: str) -> bool:
    return (os.path.isfile(os.path.join(d, "session_meta.json"))
            and bool(glob.glob(os.path.join(d, "sweeps_*.csv"))))


def _h5_has_bands(p: str) -> bool:
    try:
        import h5py
        with h5py.File(p, "r") as f:
            return ("grid" in f and len(f["grid"].keys()) > 0) or ("sweeps" in f and "psd" in f)
    except Exception:
        return False


def _dec(v):
    return v.decode("utf-8", "replace") if isinstance(v, (bytes, np.bytes_)) else str(v)


def _col(row_keys, which):
    for a in _CSV_ALIASES[which]:
        if a in row_keys:
            return a
    return None


def _session_keep_frac(meta: Dict) -> float:
    c = meta.get("config", meta)
    for k in ("keep_frac", "usable_frac", "keep_fraction", "crop_frac"):
        if k in c:
            try:
                return float(c[k])
            except (TypeError, ValueError):
                pass
    return 0.8


def _load_session_all(d: str) -> Dict:
    """Raw session (DataPort release layout): every dwell of every band."""
    meta = json.load(open(os.path.join(d, "session_meta.json")))
    cfgm = meta.get("config", {})
    bins = int(cfgm.get("psd_bins", meta.get("psd_bins", 1024)))
    keep = _session_keep_frac(meta)
    cols = {k: [] for k in _CSV_ALIASES}
    parts, off, n_sim = [], 0, 0
    for cp in sorted(glob.glob(os.path.join(d, "sweeps_*.csv"))):
        key = os.path.basename(cp)[len("sweeps_"):-len(".csv")]
        pp = os.path.join(d, f"psd_{key}.f16")
        base = off
        if os.path.exists(pp):
            raw = np.fromfile(pp, dtype=np.float16)
            nr = raw.size // bins
            parts.append(raw[:nr * bins].reshape(nr, bins).astype(np.float32))
            off += nr
        with open(cp, newline="") as fh:
            rd = csv.DictReader(fh)
            keys = rd.fieldnames or []
            m = {k: _col(keys, k) for k in _CSV_ALIASES}
            if m["prow"] is None or m["ctr"] is None or m["rate"] is None:
                return {"error": f"{os.path.basename(cp)} lacks psd_row / center_hz / samp_rate_hz "
                                 f"(columns: {keys[:25]})"}
            for i, r in enumerate(rd):
                if str(r.get("simulated", "0")).strip() not in ("", "0", "0.0", "False", "false"):
                    n_sim += 1
                    continue
                try:
                    pr = base + int(float(r[m["prow"]]))
                    ctr, rate = float(r[m["ctr"]]), float(r[m["rate"]])
                except (KeyError, ValueError, TypeError):
                    continue
                cols["prow"].append(pr); cols["ctr"].append(ctr); cols["rate"].append(rate)
                cols["band"].append(r[m["band"]] if m["band"] else "band")
                cols["sweep"].append(float(r[m["sweep"]]) if m["sweep"] and r[m["sweep"]] != "" else np.nan)
                cols["utc"].append(float(r[m["utc"]]) if m["utc"] and r[m["utc"]] != "" else float(i))
                lo = float(r[m["f_lo"]]) if m["f_lo"] and r[m["f_lo"]] != "" else ctr - keep * rate / 2
                hi = float(r[m["f_hi"]]) if m["f_hi"] and r[m["f_hi"]] != "" else ctr + keep * rate / 2
                cols["f_lo"].append(lo); cols["f_hi"].append(hi)
    if n_sim:
        return {"error": f"{n_sim} dwells flagged simulated=1 -> refused (no synthetic data)"}
    if not cols["prow"] or not parts:
        return {"error": "no usable sweeps rows / psd rows in the session folder"}
    dw = {k: np.asarray(v) for k, v in cols.items()}
    dw["prow"] = dw["prow"].astype(np.int64)
    if np.isnan(dw["sweep"].astype(float)).any():           # no sweep column: a new sweep starts when
        o = np.argsort(dw["utc"], kind="stable")            # a (band, tile) repeats in time order
        sw, cur, seen = np.empty(len(o)), 0, set()
        for j in o:
            t = (dw["band"][j], round(float(dw["ctr"][j]) / 1e3))
            if t in seen:
                cur += 1; seen = set()
            seen.add(t); sw[j] = cur
        dw["sweep"] = sw
        log("    no sweep column -> sweeps inferred from repeating tiles")
    return {"kind": "dwells", "dw": dw, "psd_rows": np.concatenate(parts, 0), "src": d, "meta": meta}


def _load_h5_all(path: str) -> Dict:
    """Export .h5 of the dataset: every /grid/<band> (gap-handled), or the
    per-dwell /sweeps + /psd of all bands."""
    try:
        import h5py
    except ImportError:
        sys.exit("h5py is required for .h5 input:  pip install h5py")
    with h5py.File(path, "r") as f:
        radio = f.attrs.get("radio", "unknown")                 # same attribute as the sensing pipeline
        radio = radio.decode() if isinstance(radio, bytes) else str(radio)
        if "sweeps" in f and "simulated" in f["sweeps"]:
            sim = np.asarray(f["sweeps"]["simulated"][()], dtype=np.float64)
            if np.nansum(sim) > 0:
                return {"error": f"{int(np.nansum(sim))} dwells flagged simulated=1 -> refused"}
        if "grid" in f and len(f["grid"].keys()):
            bands = {}
            for b in sorted(f["grid"].keys()):
                g = f["grid"][b]
                if not all(k in g for k in ("psd_db", "freq_hz", "utc")):
                    log(f"    /grid/{b} lacks psd_db/freq_hz/utc -> band skipped")
                    continue
                st = np.asarray(g["status"][()]) if "status" in g else None
                bands[b] = {"kind": "grid", "psd_bins": np.asarray(g["psd_db"][()], dtype=np.float32),
                            "freq_hz": np.asarray(g["freq_hz"][()], dtype=np.float64),
                            "utc": np.asarray(g["utc"][()], dtype=np.float64),
                            "filled_frac": float((st == 1).mean()) if st is not None else float("nan"),
                            "src": f"{path}::/grid/{b}"}
            return {"kind": "grids", "bands": bands, "src": path, "radio": radio}
        if "sweeps" not in f or "psd" not in f or "psd_db" not in f["psd"]:
            return {"error": f"needs /grid/<band> or /sweeps + /psd/psd_db; file has {sorted(f.keys())}"}
        gs = f["sweeps"]
        n = int(gs["sweep"].shape[0])
        num = lambda k: np.asarray(gs[k][()], dtype=np.float64).reshape(-1)[:n]
        need = ["sweep", "utc_start", "samp_rate_hz", "center_hz"]
        miss = [k for k in need if k not in gs]
        if miss:
            return {"error": f"/sweeps lacks {miss}"}
        rate, ctr = num("samp_rate_hz"), num("center_hz")
        lo = num("f_lo_hz") if "f_lo_hz" in gs else ctr - 0.4 * rate
        hi = num("f_hi_hz") if "f_hi_hz" in gs else ctr + 0.4 * rate
        bn = (np.array([_dec(x) for x in gs["band_name"][()]])[:n] if "band_name" in gs
              else np.array(["band"] * n))
        prow = (np.asarray(f["psd"]["row_index"][()], dtype=np.int64)[:n] if "row_index" in f["psd"]
                else num("psd_row").astype(np.int64))
        dw = {"band": bn, "sweep": num("sweep"), "utc": num("utc_start"), "f_lo": lo, "f_hi": hi,
              "rate": rate, "ctr": ctr, "prow": prow}
        return {"kind": "dwells", "dw": dw, "psd_rows": np.asarray(f["psd"]["psd_db"][()], dtype=np.float32),
                "src": path, "radio": radio}


def discover_sources(cfg: Dict) -> List[str]:
    """Same discovery as 01_sensing.py: every export .h5
    under ROOT/*/export/ and ROOT/*.h5 is POOLED (one file per session /
    radio). Only files that carry PSD data (/grid/<band> or /sweeps + /psd)
    can be forecast; I/Q-only exports are skipped with a message. If no such
    .h5 exists, the raw session folder(s) of the DataPort release are used."""
    root = cfg["data_root"]
    src = cfg.get("source") or ""
    if src:
        items = src if isinstance(src, (list, tuple)) else [s for s in str(src).split(";") if s]
        out = []
        for e in items:
            full = e if os.path.isabs(e) else os.path.join(root, e)
            if os.path.exists(full):
                out.append(full)
            else:
                log(f"  source {full} not found")
        if out:
            return out
    if not os.path.isdir(root):
        log(f"  data_root {root} does not exist")
        return []
    h5 = sorted(set(glob.glob(os.path.join(root, "*", "export", "*.h5")) + glob.glob(os.path.join(root, "*.h5"))))
    sess = [d for d in [root] + sorted(glob.glob(os.path.join(root, "*"))) if os.path.isdir(d) and _is_session_dir(d)]
    if not h5 and not sess:                                   # deeper layouts
        for dp, dn, fn in os.walk(root):
            if os.path.basename(dp).lower() in ("results", "cache", "stage2", "derived", "_frame_cache"):
                dn[:] = []
                continue
            if _is_session_dir(dp):
                sess.append(dp)
            h5 += [os.path.join(dp, f) for f in fn if f.lower().endswith(".h5")]
    psd_h5 = []
    for p in h5:
        if _h5_has_bands(p):
            psd_h5.append(p)
        else:
            log(f"  {os.path.basename(p)}: no /grid or /sweeps + /psd (I/Q-only export) -> not usable for forecasting")
    if psd_h5 and sess:
        use = psd_h5 if cfg.get("prefer", "h5") == "h5" else sess
        log(f"  both export .h5 and raw session folders found -> using the "
            f"{'export .h5' if use is psd_h5 else 'raw session'} files (prefer = {cfg.get('prefer', 'h5')})")
        return use
    return psd_h5 or sess


def _radio_of_meta(meta: Dict) -> str:
    for d in (meta, meta.get("config", {}), meta.get("hardware", {}) if isinstance(meta.get("hardware"), dict) else {}):
        for k in ("radio", "device", "radio_label", "hardware", "model"):
            v = d.get(k) if isinstance(d, dict) else None
            if isinstance(v, str) and v:
                return v
    return "unknown"


def load_bands(path: str, cfg: Dict) -> Tuple[Dict[str, Dict], str, str]:
    """-> ({band: source dict usable by to_subchannels}, radio, error)."""
    if os.path.isdir(path):
        if _is_session_dir(path):
            src = _load_session_all(path)
            radio = _radio_of_meta(src.get("meta", {})) if "error" not in src else "unknown"
        else:
            ex = [p for p in sorted(glob.glob(os.path.join(path, "**", "*.h5"), recursive=True),
                                    key=os.path.getmtime) if _h5_has_bands(p)]
            if not ex:
                return {}, "", "folder has neither a raw session (session_meta.json + sweeps_*.csv) nor an export .h5"
            src = _load_h5_all(ex[-1])
            radio = src.get("radio", "unknown")
    else:
        src = _load_h5_all(path)
        radio = src.get("radio", "unknown")
    if "error" in src:
        return {}, "", src["error"]
    if src["kind"] == "grids":
        out = src["bands"]
    else:
        dw = src["dw"]
        out = {}
        for b in sorted(set(dw["band"].tolist())):
            m = dw["band"] == b
            out[b] = {"kind": "dwells", "dw": {k: v[m] for k, v in dw.items()},
                      "psd_rows": src["psd_rows"], "src": f"{src['src']}::{b}"}
    want = [b for b in cfg.get("bands") or []]
    if want:
        miss = [b for b in want if b not in out]
        if miss:
            log(f"  requested band(s) {miss} not in {os.path.basename(path)} (has {sorted(out)})")
        out = {b: v for b, v in out.items() if b in want}
    for b in cfg.get("exclude_bands") or []:
        out.pop(b, None)
    return out, str(radio), ""


def pool_sources(paths: List[str], cfg: Dict) -> Dict[Tuple[str, str], List[Dict]]:
    """(radio, band) -> list of per-file sources (several sessions of the
    same radio are concatenated in time later)."""
    pool: Dict[Tuple[str, str], List[Dict]] = {}
    for p in paths:
        srcs, radio, err = load_bands(p, cfg)
        if err:
            log(f"  {p}: NOT USED ({err})")
            continue
        if cfg.get("devices") and radio not in cfg["devices"]:
            log(f"  {os.path.basename(p)} [{radio}] not in CONFIG['devices'] -> skipped")
            continue
        log(f"  {p}  [{radio}]  bands {sorted(srcs)}")
        for b, s in srcs.items():
            pool.setdefault((radio, b), []).append(s)
    return pool


def merge_parts(parts: List[Tuple[np.ndarray, np.ndarray, int]]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Concatenate the sub-channel series of several files of one (radio,
    band) in time; identical timestamps (re-exports) are kept once."""
    P = np.concatenate([p for p, _, _ in parts]); t = np.concatenate([t for _, t, _ in parts])
    ni = int(sum(n for _, _, n in parts))
    o = np.argsort(t, kind="stable")
    P, t = P[o], t[o]
    keep = np.concatenate([[True], np.diff(t) > 1e-6]) if t.size else np.zeros(0, bool)
    return P[keep], t[keep], ni


# ---------------------------------------------------------------------------
#   SPLIT: chronological 0.65 / 0.15 / 0.20 WITHIN every (time block, band),
#   exactly as make_splits() of 01_sensing.py, adapted to
#   windows: a window must lie inside one block, and train / val / test of a
#   block are separated by W + H steps (no shared step between the splits).
# ---------------------------------------------------------------------------
def window_starts(t: np.ndarray, W: int, H: int, cfg: Dict) -> np.ndarray:
    """Window starts; windows that span a recording gap (a step longer than
    max_gap_steps x the median step: dropped sweeps, separate sessions) are
    removed."""
    T = len(t)
    st = np.arange(0, max(0, T - W - H + 1), cfg["stride"])
    if st.size and cfg.get("max_gap_steps", 0) > 0 and T > 2:
        dt = np.diff(t)
        seg = np.concatenate([[0], np.cumsum(dt > cfg["max_gap_steps"] * float(np.median(dt)))])
        st = st[seg[st] == seg[st + W + H - 1]]
    return st


def split_windows(starts: np.ndarray, t: np.ndarray, tedges: np.ndarray, W: int, H: int, cfg: Dict):
    """-> train, val, test indices into `starts`, block of every window."""
    B = len(tedges) - 1
    bstep = np.clip(np.searchsorted(tedges, t, "right") - 1, 0, B - 1)
    ok = bstep[starts] == bstep[starts + W + H - 1]             # whole window inside one block
    blk = np.where(ok, bstep[starts], -1)
    gap = (W + H) // cfg["stride"] + (1 if (W + H) % cfg["stride"] else 0)
    tr, va, te = [], [], []
    for b in range(B):
        wb = np.flatnonzero(blk == b)
        m = len(wb)
        n_te = int(round(cfg["test_frac"] * m)); n_va = int(round(cfg["val_frac"] * m))
        n_tr = m - n_te - n_va - 2 * gap
        if n_tr < 1 or n_va < 1 or n_te < 1:
            continue                                          # block too short to hold three splits
        tr += wb[:n_tr].tolist()
        va += wb[n_tr + gap:n_tr + gap + n_va].tolist()
        te += wb[n_tr + n_va + 2 * gap:].tolist()
    return np.array(tr, int), np.array(va, int), np.array(te, int), blk


def time_block_edges(ts: List[np.ndarray], B: int) -> np.ndarray:
    """B contiguous time blocks with equal numbers of steps, pooled over all
    scenarios (np.array_split of the time order, as in make_splits)."""
    allt = np.sort(np.concatenate(ts))
    e = np.quantile(allt, np.linspace(0, 1, B + 1))
    e[0], e[-1] = allt[0] - 1e-6, allt[-1] + 1e-6
    return e


def _fits(ts: Dict[str, np.ndarray], W: int, H: int, B: int, mtw: int, cfg: Dict):
    te = time_block_edges(list(ts.values()), B)
    for t in ts.values():
        st = window_starts(t, W, H, cfg)
        tr, va, tt, _ = split_windows(st, t, te, W, H, cfg)
        share = len(tr) / max(len(tr) + len(va) + len(tt), 1)
        if (len(tr) < mtw or min(len(va), len(tt)) < cfg["min_eval_windows"]
                or share < cfg["min_train_share"]):
            return None                           # the W+H gaps of many short blocks would eat the
    return te                                     # training split (0.65 target -> keep >= min_train_share)


def plan_split(ts: Dict[str, np.ndarray], cfg: Dict) -> Tuple[int, int, int, np.ndarray, List[str]]:
    """Largest W / H (lookback/horizon, then window_fallbacks) and largest
    number of split blocks <= split_blocks for which every kept scenario has
    >= min_train_windows training windows (else >= min_train_windows_floor),
    >= min_eval_windows validation and test windows, and a train share >=
    min_train_share. Bands with FEW sweeps (the export revisits some bands
    less often) are dropped, shortest first, only when no setting fits them
    (drop_short_scenarios). -> W, H, B, time edges, dropped scenarios."""
    W0, H0 = cfg["lookback"], cfg["horizon"]
    cands = [(W0, H0)] + [(W, H) for W, H in cfg["window_fallbacks"] if W <= W0 and H <= H0 and (W, H) != (W0, H0)]
    if not cfg.get("auto_window", True):
        cands = cands[:1]
    floors = sorted({int(cfg["min_train_windows"]),
                     int(min(cfg.get("min_train_windows_floor", cfg["min_train_windows"]),
                             cfg["min_train_windows"]))}, reverse=True)
    lens = sorted({len(t) for t in ts.values()})
    thresholds = lens if cfg.get("drop_short_scenarios", True) else lens[:1]
    for thr in thresholds:                         # keep every scenario with >= thr steps
        sub_ts = {n: t for n, t in ts.items() if len(t) >= thr}
        for mtw in floors:
            for W, H in cands:
                for B in range(max(1, int(cfg["split_blocks"])), 0, -1):
                    te = _fits(sub_ts, W, H, B, mtw, cfg)
                    if te is None:
                        continue
                    dropped = [n for n in ts if n not in sub_ts]
                    if dropped:
                        log(f"  NOTE: {len(dropped)} scenario(s) with too few sweeps for any window setting "
                            f"-> dropped: " + ", ".join(f"{n} ({len(ts[n])} steps)" for n in dropped))
                    if (W, H) != (W0, H0):
                        log(f"  NOTE: too few sweeps for W={W0}, H={H0} -> using W={W}, H={H}")
                    if mtw != cfg["min_train_windows"]:
                        log(f"  NOTE: min_train_windows {cfg['min_train_windows']} -> {mtw} "
                            f"(min_train_windows_floor)")
                    if B != cfg["split_blocks"]:
                        log(f"  NOTE: split_blocks {cfg['split_blocks']} -> {B} (largest number of time blocks "
                            f"whose train / val / test segments still hold whole W+H windows)")
                    return W, H, B, te, dropped
    sys.exit(f"session too short for any window / block setting (steps per scenario "
             f"{sorted({len(t) for t in ts.values()})}); lower window_fallbacks / min_train_windows_floor")


def freq_span(src: Dict, band: str) -> Tuple[float, float]:
    if src["kind"] == "grid":
        f = src["freq_hz"]
        df = float(np.median(np.diff(f))) if f.size > 1 else 0.0
        return float(f.min() - df / 2), float(f.max() + df / 2)
    dw = src["dw"]
    return float(dw["f_lo"].min()), float(dw["f_hi"].max())


def band_edges(spans: Dict[str, Tuple[float, float]], cfg: Dict) -> Tuple[Dict[str, np.ndarray], int]:
    """Common number of sub-channels C for every band (the models, centralised
    pooling and FedAvg need the same C).
      'equal' : each band's measured span split into C = subchannels_per_band
                equal sub-channels (width differs per band, printed)
      'fixed' : subchannel_hz wide sub-channels; C = the largest count that
                fits in the narrowest band; a centred window in every band"""
    if cfg["subchannel_mode"] == "fixed":
        w = cfg["subchannel_hz"]
        C = int(min(math.floor((hi - lo) / w + 1e-9) for lo, hi in spans.values()))
        if C < 4:
            sys.exit(f"subchannel_mode 'fixed': the narrowest band holds only {C} sub-channels")
        return {b: lo + (hi - lo - C * w) / 2 + np.arange(C + 1) * w for b, (lo, hi) in spans.items()}, C
    C = int(cfg["subchannels_per_band"])
    return {b: np.linspace(lo, hi, C + 1) for b, (lo, hi) in spans.items()}, C


def to_subchannels(src: Dict, band: str, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """-> P[T, C] dB (linear-domain mean per sub-channel), utc[T], #interpolated."""
    C = len(edges) - 1
    if src["kind"] == "grid":
        P, f, t = src["psd_bins"], src["freq_hz"], src["utc"]
        which = np.searchsorted(edges, f, "right") - 1
        out = np.full((P.shape[0], C), np.nan, np.float32)
        lin = np.power(10.0, P.astype(np.float64) / 10.0)
        for c in range(C):
            m = which == c
            if m.any():
                with np.errstate(invalid="ignore"):
                    out[:, c] = 10 * np.log10(np.nanmean(lin[:, m], 1) + 1e-20)
        o = np.argsort(t)
        out, t = out[o], t[o]
    else:
        dw = src["dw"]
        m = dw["band"] == band
        dw = {k: v[m] for k, v in dw.items()}
        R = src["psd_rows"]
        sw, si = np.unique(dw["sweep"], return_inverse=True)
        out = np.full((len(sw), C), np.nan, np.float32)
        t = np.bincount(si, weights=dw["utc"]) / np.maximum(np.bincount(si), 1)
        for i in range(len(dw["sweep"])):
            pr = dw["prow"][i]
            if not 0 <= pr < R.shape[0]:
                continue
            row = R[pr]
            nb = row.size
            bf = dw["ctr"][i] + (np.arange(nb) - nb / 2.0) * (dw["rate"][i] / nb)
            k = (bf >= dw["f_lo"][i]) & (bf <= dw["f_hi"][i]) & np.isfinite(row)
            if k.sum() < 4:
                continue
            wc = np.searchsorted(edges, bf[k], "right") - 1
            lin = np.power(10.0, row[k].astype(np.float64) / 10.0)
            ok = (wc >= 0) & (wc < C)
            s = np.bincount(wc[ok], weights=lin[ok], minlength=C)
            n = np.bincount(wc[ok], minlength=C)
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.where(n > 0, 10 * np.log10(s / np.maximum(n, 1) + 1e-20), np.nan)
            out[si[i]] = np.fmax(out[si[i]], v)
        o = np.argsort(t)
        out, t = out[o], t[o]
    n_interp = int((~np.isfinite(out)).sum())
    for c in range(C):                      # residual holes: interpolate in time
        col = out[:, c]
        bad = ~np.isfinite(col)
        if bad.all():
            col[:] = np.nanmin(out) if np.isfinite(out).any() else -100.0
        elif bad.any():
            ok = np.flatnonzero(~bad)
            col[bad] = np.interp(np.flatnonzero(bad), ok, col[ok])
    return out, t, n_interp


def _runs(col: np.ndarray, val: int) -> List[int]:
    out, n = [], 0
    for v in col:
        if v == val:
            n += 1
        elif n:
            out.append(n); n = 0
    if n:
        out.append(n)
    return out


def build_scenario(name: str, P: np.ndarray, t: np.ndarray, edges: np.ndarray, cfg: Dict, n_interp: int,
                   src_path: str, radio: str, tedges: np.ndarray, calib=None) -> Tuple[Dict, Dict]:
    """Labels + windows + split of one scenario (band). The noise floor and
    the INR statistics use only the steps covered by TRAINING windows
    (no validation / test leakage)."""
    T, C = P.shape
    W, H = cfg["lookback"], cfg["horizon"]
    starts = window_starts(t, W, H, cfg)
    tr, va, te, blk = split_windows(starts, t, tedges, W, H, cfg)
    trmask = np.zeros(T, bool)
    for s0 in starts[tr]:
        trmask[s0:s0 + W + H] = True
    nf = np.quantile(P[trmask], cfg["noise_quantile"], axis=0)        # Q10, training steps only
    E = (P - nf[None, :]).astype(np.float32)                           # excess = SNR
    eta, cinfo = float(cfg["eta_db"]), {}
    if cfg["input"] == "sensing":                                      # threshold calibrated to the detector
        eta, cinfo = calibrate_eta(E, t, edges, trmask, calib, cfg)
    Y = (E > eta).astype(np.float32)
    occ_runs = [r for c in range(C) for r in _runs(Y[:, c].astype(int), 1)]
    idle_runs = [r for c in range(C) for r in _runs(Y[:, c].astype(int), 0)]
    s = Y.mean(1) - Y.mean()
    ac1 = float(np.corrcoef(s[:-1], s[1:])[0, 1]) if s.std() > 1e-9 else float("nan")
    step = float(np.median(np.diff(t))) if T > 1 else float("nan")
    idx_in = starts[:, None] + np.arange(W)[None, :]
    idx_out = starts[:, None] + W + np.arange(H)[None, :]
    Xsrc = (np.clip((E - eta) / 10.0, -2.0, 4.0).astype(np.float32)
            if cfg["input_repr"] == "excess" else Y)
    w = {"X": Xsrc[idx_in], "Xbin": Y[idx_in], "Y": Y[idx_out], "E": E[idx_out],
         "t0": t[starts + W], "start": starts, "tr": tr, "va": va, "te": te}
    w["period"] = estimate_period(Y[trmask], W)
    inr = np.maximum(np.power(10.0, E[trmask] / 10.0) - 1.0, 0.0)
    yb = Y[trmask] > 0.5
    w["inr_busy"] = np.array([inr[yb[:, c], c].mean() if yb[:, c].any() else 10.0 for c in range(C)], np.float32)
    w["inr_idle"] = np.array([inr[~yb[:, c], c].mean() if (~yb[:, c]).any() else 0.0 for c in range(C)], np.float32)
    ysplit = {k: w["Y"][w[k]] for k in ("tr", "va", "te")}
    sc = {"name": name, "P": P, "E": E, "Y": Y, "utc": t, "noise_floor": nf, "edges": edges, "T": T, "C": C,
          "step_s": step, "eta": eta}
    sc["card"] = {
        "scenario": name, "radio": radio, "source": src_path, "steps": T, "subchannels": C,
        "f_lo_MHz": edges[0] / 1e6, "f_hi_MHz": edges[-1] / 1e6,
        "subchannel_MHz": float(edges[1] - edges[0]) / 1e6,
        "step_s_median": step, "duration_min": float((t[-1] - t[0]) / 60.0),
        "cells_interpolated": n_interp, "windows_dropped_gap": int(T - W - H + 1 - len(starts)),
        "split_blocks": len(tedges) - 1, "windows_train": len(tr), "windows_val": len(va), "windows_test": len(te),
        "noise_quantile": cfg["noise_quantile"], "eta_db": eta,
        "label_source": (f"PSD, eta calibrated to {cfg.get('_sensing_desc', '')}" if cfg["input"] == "sensing"
                         else f"PSD, fixed eta {cfg['eta_db']:g} dB"),
        "noise_floor_mean_db": float(nf.mean()), "noise_floor_std_db": float(nf.std()),
        "positive_rate": float(Y.mean()),
        "positive_rate_train": float(ysplit["tr"].mean()) if len(tr) else float("nan"),
        "positive_rate_val": float(ysplit["va"].mean()) if len(va) else float("nan"),
        "positive_rate_test": float(ysplit["te"].mean()) if len(te) else float("nan"),
        "on_run_mean_steps": float(np.mean(occ_runs)) if occ_runs else 0.0,
        "off_run_mean_steps": float(np.mean(idle_runs)) if idle_runs else 0.0,
        "lag1_acf": ac1, "excess_max_db": float(E.max()),
        "cells_within_1dB_of_eta": float((np.abs(E - eta) < 1).mean()),
        **{k: v for k, v in cinfo.items() if k != "eta_db"},
    }
    log(f"  [{name}] T={T} steps (median {step:.2f} s/step, {sc['card']['duration_min']:.1f} min), "
        f"occupancy {Y.mean() * 100:.1f}% (train/val/test {sc['card']['positive_rate_train'] * 100:.1f}/"
        f"{sc['card']['positive_rate_val'] * 100:.1f}/{sc['card']['positive_rate_test'] * 100:.1f}%), "
        f"NF {nf.mean():.1f} dB, max SNR {E.max():.1f} dB, windows {len(tr)}/{len(va)}/{len(te)}, "
        f"seasonal period {w['period']}")
    if cinfo:
        log(f"      eta = {eta:g} dB  ({cinfo['calib_status']}; {cinfo['calib_points']} recordings, "
            f"agreement with the detector {cinfo['calib_ba']:.3f} vs {cinfo['calib_ba_at_default_eta']:.3f} "
            f"at {cfg['eta_db']:g} dB)")
    return sc, w


def estimate_period(Y: np.ndarray, max_lag: int) -> int:
    s = Y.mean(1)
    s = s - s.mean()
    if s.std() < 1e-9 or s.size < 8:
        return 1
    n = min(max_lag, s.size // 2)
    ac = np.nan_to_num(np.array([np.corrcoef(s[:-l], s[l:])[0, 1] for l in range(1, n)]))
    if ac.size < 2:
        return 1
    k = int(np.argmax(ac[1:]) + 2)
    return k if ac[k - 1] > 0.2 else 1


# ==========================================================================
#            REFERENCE BASELINES (for skill columns; identical to SOTA)
# ==========================================================================
def baseline_forecast(name: str, X: np.ndarray, H: int, period: int) -> np.ndarray:
    if name == "persistence":
        return np.repeat(X[:, -1:, :], H, axis=1)
    if name == "historical_mean":
        return np.repeat(X.mean(axis=1, keepdims=True), H, axis=1)
    if name == "seasonal_naive":
        m, W = max(1, period), X.shape[1]
        out = np.empty((X.shape[0], H, X.shape[2]), dtype=X.dtype)
        for h in range(H):
            out[:, h, :] = X[:, max(0, min(W - m + (h % m), W - 1)), :]
        return out
    raise ValueError(name)


MODEL_LABEL = {"persistence": "Persistence", "seasonal_naive": "Seasonal naive",
               "historical_mean": "Historical mean", "gem": "GEM", "gem_dcs": "GEM-DCS"}


# ==========================================================================
#                       SHARED BUILDING BLOCKS
# ==========================================================================
_SCALES = (1, 2, 4, 8, 16, 32)


def window_stats(xb: torch.Tensor, xe: torch.Tensor) -> torch.Tensor:
    """Per sub-channel statistics of the window -> [B, C, F].
    xb binary occupancy [B,W,C], xe margin (e-eta)/10 [B,W,C]."""
    W = xb.shape[1]
    ks = [k for k in _SCALES if k <= W]
    f = [xb[:, -k:].mean(1) for k in ks] + [xe[:, -k:].mean(1) for k in ks]
    same = (xb == xb[:, -1:]).flip(1).float()
    f.append(torch.cumprod(same, 1).sum(1) / W)                      # current run length
    f.append((xb[:, 1:] - xb[:, :-1]).abs().mean(1))                  # flip rate
    f.append(xe.std(1, unbiased=False))                               # margin volatility
    return torch.stack(f, -1)


def n_stats(W: int) -> int:
    return 2 * len([k for k in _SCALES if k <= W]) + 3


def base_logit(xb: torch.Tensor, H: int) -> torch.Tensor:
    """Logit of the Krichevsky-Trofimov smoothed window mean -> [B, H, C]."""
    W = xb.shape[1]
    p = (xb.sum(1) + 0.5) / (W + 1.0)
    return torch.logit(p)[:, None, :].expand(-1, H, -1)


class ScenarioCalib(nn.Module):
    """Per-scenario temperature and per-(horizon, sub-channel) bias. Its
    parameters start with 'loc_' and are NEVER averaged in federated training:
    each client keeps its own calibration (personalisation)."""

    def __init__(self, S, H, C):
        super().__init__()
        self.loc_bias = nn.Parameter(torch.zeros(S, H, C))
        self.loc_logtemp = nn.Parameter(torch.zeros(S))

    def forward(self, z, sid):
        return z * torch.exp(self.loc_logtemp[sid])[:, None, None] + self.loc_bias[sid]


def _zero_last(seq: nn.Sequential):
    last = [m for m in seq.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


# ==========================================================================
#                 PROPOSED MODEL: GEM / GEM-DCS  (paper Sec. VI)
# --------------------------------------------------------------------------
#  A sweep-level record of one evening holds a few hundred windows per band,
#  so variance, not capacity, is the difficulty. GEM computes the statistics
#  that have closed forms and learns only how to weight them:
#   1. PriorMemory: per-(band, sub-channel) busy / visit / 0->1 / 1->0 counts
#      accumulated from the TRAINING input windows (never from targets) as
#      buffers, shrunk to the band rate (empirical Bayes, m0 = 2, m1 = 4).
#      FedAvg averages the buffers like any tensor.
#   2. Drift correction: the band load of the current window shifts the
#      priors on the logit scale, delta = sigmoid(psi_b) (logit l_t - logit r_b).
#   3. Nine analytic experts (persistence, window mean, EWMA, 5-channel window
#      mean, drift-corrected prior, neighbourhood prior, load-scaled prior,
#      band load, h-step two-state Markov chain), mixed in probability space
#      by a gate MLP (17 -> 64 -> H*9, zero-initialised last layer) plus
#      per-band gate logits, then a per-band affine calibration ('loc_').
#   4. GEM-DCS (dcs_stream below): a per-cell logit offset that takes one
#      logistic step on each residual once the target sweep is sensed.
#  Structural parameters are stored as theta / GAIN (GAIN = 20), so they
#  converge within the few dozen optimiser steps the record allows.
# ==========================================================================
GAIN = 20.0


def _isp(v: float) -> float:
    """inverse softplus"""
    return float(math.log(math.expm1(v)))


def _lg(p: torch.Tensor) -> torch.Tensor:
    return torch.logit(p.clamp(1e-3, 1 - 1e-3))


def _smooth_c(a: torch.Tensor, k: int) -> torch.Tensor:
    """Average over k neighbouring sub-channels (last dim), replicate edges."""
    sh = a.shape
    x = a.reshape(-1, 1, sh[-1])
    x = F.pad(x, (k // 2, k // 2), mode="replicate")
    return F.avg_pool1d(x, k, stride=1).reshape(sh)


class PriorMemory(nn.Module):
    """Sufficient statistics of the TRAINING input windows per (scenario,
    sub-channel): occupancy counts and 0->1 / 1->0 transition counts.
    Updated in training mode only, from the inputs only."""

    def __init__(self, S: int, C: int):
        super().__init__()
        for nm in ("occ", "cnt", "t01", "n0", "t10", "n1"):
            self.register_buffer("mem_" + nm, torch.zeros(S, C))

    @torch.no_grad()
    def update(self, xb: torch.Tensor, sid: torch.Tensor) -> None:
        a, b = xb[:, :-1], xb[:, 1:]
        st = {"occ": xb.sum(1), "cnt": torch.full_like(xb[:, 0], float(xb.shape[1])),
              "t01": ((1 - a) * b).sum(1), "n0": (1 - a).sum(1),
              "t10": (a * (1 - b)).sum(1), "n1": a.sum(1)}
        for nm, v in st.items():
            getattr(self, "mem_" + nm).index_add_(0, sid, v.to(self.mem_occ.dtype))

    @staticmethod
    def _shrunk(num, den, m0):
        band = (num.sum(1, keepdim=True) + 0.5) / (den.sum(1, keepdim=True) + 1.0)
        return ((num + m0 * band) / (den + m0)).clamp(0.01, 0.99), band.clamp(0.01, 0.99)

    def rate(self):
        """-> per-channel rate [S,C], band rate [S,1] (0.5 before any update)."""
        return self._shrunk(self.mem_occ, self.mem_cnt, 2.0)

    def trans(self):
        """-> P(0->1) [S,C], P(1->0) [S,C], shrunk to the band's rates."""
        a, _ = self._shrunk(self.mem_t01, self.mem_n0, 4.0)
        b, _ = self._shrunk(self.mem_t10, self.mem_n1, 4.0)
        return a, b


class _PriorBase(nn.Module):
    """Prior memory + drift correction + context features shared by the three."""

    def __init__(self, W, H, C, S):
        super().__init__()
        self.W, self.H, self.C = W, H, C
        self.mem = PriorMemory(S, C)
        self.loc_psi = nn.Parameter(torch.zeros(S))          # drift-correction strength per band, sigmoid -> 0.5
        self.calib = ScenarioCalib(S, H, C)

    def prior(self, xb, sid):
        """-> drift-corrected channel prior mu [B,C], logit shift delta [B,1],
        raw band rate [B,1], band load of the window [B,1]."""
        if self.training:
            self.mem.update(xb, sid)
        mu, band = self.mem.rate()
        mu, band = mu[sid], band[sid]
        load = ((xb.sum((1, 2)) + 0.5) / (xb.shape[1] * xb.shape[2] + 1.0))[:, None]
        delta = torch.sigmoid(self.loc_psi * GAIN)[sid][:, None] * (_lg(load) - _lg(band))
        return torch.sigmoid(_lg(mu) + delta), delta, band, load

    def context(self, xb, xe, mu, load):
        """Per sub-channel context [B,C,F]: window statistics, prior, spectral
        neighbourhood of the last sweep, band load, last margin."""
        last = xb[:, -1]
        wm = xb.mean(1)
        f = [window_stats(xb, xe), mu[..., None], (mu - wm)[..., None],
             _smooth_c(last, 3)[..., None], _smooth_c(last, 7)[..., None],
             _smooth_c(wm, 5)[..., None], load[:, :, None].expand(-1, self.C, -1),
             last.mean(1, keepdim=True)[:, :, None].expand(-1, self.C, -1), xe[:, -1][..., None]]
        return torch.cat(f, -1)

    @staticmethod
    def n_ctx(W):
        return n_stats(W) + 8


class GEM(_PriorBase):
    """Gated Expert Mixture forecaster (novel).
    Nine analytic experts per cell and horizon -- persistence, window mean,
    fast EWMA, neighbourhood window mean, channel prior, neighbourhood prior,
    load-scaled prior, current band load and the h-step Markov chain of the
    memory -- mixed in
    PROBABILITY space by a softmax gate: per-band, per-horizon base weights
    ('loc_', fast) plus a zero-initialised MLP on the context. The forecast
    always lies inside the experts' range, so it cannot be worse than the
    worst expert and starts as their uniform average."""
    E = 9

    def __init__(self, W, H, C, S, d_model=64, dropout=0.1, **kw):
        super().__init__(W, H, C, S)
        self.loc_gate = nn.Parameter(torch.zeros(S, H, self.E))
        self.gate = nn.Sequential(nn.Linear(self.n_ctx(W), d_model), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, H * self.E))
        _zero_last(self.gate)
        w = 0.5 ** torch.arange(W - 1, -1, -1).float()
        self.register_buffer("ew", w / w.sum(), persistent=False)

    def forward(self, xb, xe, sid):
        B, W, C = xb.shape
        H = self.H
        mu, delta, band, load = self.prior(xb, sid)
        last, wm = xb[:, -1], (xb.sum(1) + 0.5) / (W + 1.0)
        a0, b0 = self.mem.trans()
        a0 = torch.sigmoid(_lg(a0[sid]) + delta); b0 = torch.sigmoid(_lg(b0[sid]) - delta)
        p, chain = last, []
        for _ in range(H):
            p = a0 * (1 - p) + (1 - b0) * p
            chain.append(p)
        rep = lambda v: v[:, None, :].expand(-1, H, -1)                               # noqa: E731
        scaled = (mu * (load / band).clamp(0.25, 4.0)).clamp(0.01, 0.99)
        ex = torch.stack([rep(last), rep(wm), rep(torch.einsum("w,bwc->bc", self.ew, xb)),
                          rep(_smooth_c(wm, 5)), rep(mu), rep(_smooth_c(mu, 5)), rep(scaled),
                          rep(load.expand(-1, C)), torch.stack(chain, 1)], -1).clamp(0.02, 0.98)   # [B,H,C,E]
        g = self.gate(self.context(xb, xe, mu, load)).reshape(B, C, H, self.E).permute(0, 2, 1, 3)
        g = g + (self.loc_gate * GAIN)[sid][:, :, None, :]
        return self.calib(_lg((torch.softmax(g, -1) * ex).sum(-1)), sid)


MODEL_REGISTRY = {"gem": GEM}


def build_model(name: str, cfg: Dict, C: int, S: int) -> nn.Module:
    return MODEL_REGISTRY[name](W=cfg["lookback"], H=cfg["horizon"], C=C, S=S,
                                d_model=cfg["d_model"], dropout=cfg["dropout"])


# ==========================================================================
#                     DCS: DRIFT-COMPENSATED STREAMING
# ==========================================================================
def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def dcs_stream(Z: np.ndarray, Y: np.ndarray, eta: float, beta0: Optional[np.ndarray] = None):
    """Z logits [N,H,C] of CONSECUTIVE windows (stride 1), Y labels [N,H,C].
    The forecast of window j for horizon h becomes observable at window j+h;
    before forecasting window i, the offset of horizon h takes one logistic
    step on the residual of window i-h. Returns (probabilities, final offset)."""
    N, H, C = Z.shape
    beta = np.zeros((H, C)) if beta0 is None else beta0.copy()
    P = np.empty_like(Z)
    for i in range(N):
        if eta > 0:
            for h in range(H):
                j = i - (h + 1)
                if j >= 0:
                    beta[h] += eta * (Y[j, h] - _sig(Z[j, h] + beta[h]))
        P[i] = _sig(Z[i] + beta)
    return P.astype(np.float32), beta


# ==========================================================================
#   SOTA FORECASTERS (Table III of the paper)
#     N-HiTS        C. Challu et al., AAAI 2023
#     PatchTST      Y. Nie et al., ICLR 2023
#     iTransformer  Y. Liu et al., ICLR 2024
#     Bi-ConvLSTM   L. Wang et al., Sensors 2024 (sequence-to-sequence)
#     SpectrumFM    F. Liu et al., IEEE GLOBECOM 2025 (masked + next-slot
#                   pre-training, then fine-tuning; federated pre-training in FL)
# ==========================================================================
class NHiTS(nn.Module):
    """Challu et al., 'N-HiTS: Neural Hierarchical Interpolation for Time
    Series Forecasting', AAAI 2023. Multi-rate pooling + hierarchical
    interpolation of backcast/forecast."""

    def __init__(self, W, H, C, d_model=64, dropout=0.1, pools=(4, 2, 1), **kw):
        super().__init__()
        self.W, self.H, self.C, self.pools = W, H, C, pools
        self.blocks = nn.ModuleList()
        for p in pools:
            wp = max(1, W // p)
            self.blocks.append(nn.Sequential(
                nn.Linear(wp * C, 4 * d_model), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(4 * d_model, 4 * d_model), nn.ReLU(),
                nn.Linear(4 * d_model, (wp + H) * C)))

    def forward(self, x):
        B = x.shape[0]
        res = x
        out = torch.zeros(B, self.H, self.C, device=x.device, dtype=x.dtype)
        for p, blk in zip(self.pools, self.blocks):
            xp = res.permute(0, 2, 1)
            xp = F.avg_pool1d(xp, p, stride=p) if p > 1 else xp
            wp = xp.shape[-1]
            h = blk(xp.permute(0, 2, 1).reshape(B, -1))
            back = h[:, :wp * self.C].reshape(B, wp, self.C)
            res = res - F.interpolate(back.permute(0, 2, 1), size=self.W, mode="linear",
                                      align_corners=False).permute(0, 2, 1)
            out = out + h[:, wp * self.C:].reshape(B, self.H, self.C)
        return out


class PatchTST(nn.Module):
    """Nie et al., 'A Time Series is Worth 64 Words', ICLR 2023.
    Channel-independent patches + Transformer encoder."""

    def __init__(self, W, H, C, d_model=64, dropout=0.1, patch=8, stride=4,
                 nhead=4, layers=2, **kw):
        super().__init__()
        self.H, self.patch, self.stride = H, patch, stride
        self.n_patch = max(1, (W - patch) // stride + 1)
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, 2 * d_model, dropout=dropout,
                                         batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(self.n_patch * d_model, H)

    def forward(self, x):
        B, W, C = x.shape
        z = x.permute(0, 2, 1).reshape(B * C, W)
        idx = (torch.arange(self.patch, device=x.device)[None, :]
               + self.stride * torch.arange(self.n_patch, device=x.device)[:, None]).clamp(max=W - 1)
        h = self.tr(self.embed(z[:, idx]) + self.pos)
        return self.head(h.reshape(B * C, -1)).reshape(B, C, self.H).permute(0, 2, 1)


class ITransformer(nn.Module):
    """Liu et al., 'iTransformer: Inverted Transformers Are Effective for Time
    Series Forecasting', ICLR 2024. Each sub-channel is one token."""

    def __init__(self, W, H, C, d_model=64, dropout=0.1, nhead=4, layers=2, **kw):
        super().__init__()
        self.embed = nn.Linear(W, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, 2 * d_model, dropout=dropout,
                                         batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, H)

    def forward(self, x):
        return self.head(self.tr(self.embed(x.permute(0, 2, 1)))).permute(0, 2, 1)


class _ConfBlock(nn.Module):
    """MHSA + depthwise-conv module + FFN (conv + attention encoder block)."""

    def __init__(self, d, nhead, dropout):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.dw = nn.Conv1d(d, d, 5, padding=2, groups=d)
        self.pw = nn.Conv1d(d, d, 1)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, z):
        h = self.n1(z)
        z = z + self.att(h, h, h, need_weights=False)[0]
        h = self.n2(z).transpose(1, 2)
        z = z + self.pw(F.gelu(self.dw(h))).transpose(1, 2)
        return z + self.ff(self.n3(z))


class SpectrumFMF(nn.Module):
    """Liu, Zhang, Wu, Zhou, Wu, Ng, Chae, 'SpectrumFM: Redefining Spectrum
    Cognition via Foundation Modeling', IEEE GLOBECOM 2025.
    Conv + multi-head self-attention encoder, self-supervised pre-training by
    masked reconstruction and next-slot prediction, then fine-tuned.
    ADAPTATION: tokens are the W time steps of the sub-channel vector (the
    paper tokenises I/Q); model size is reduced for this dataset; no external
    pre-training corpus -- pre-training uses only this run's TRAINING split."""

    def __init__(self, W, H, C, d_model=64, dropout=0.1, nhead=4, layers=3, **kw):
        super().__init__()
        self.W, self.H, self.C = W, H, C
        self.stem = nn.Sequential(nn.Conv1d(C, d_model, 3, padding=1), nn.GELU(),
                                  nn.Conv1d(d_model, d_model, 3, padding=1))
        self.pos = nn.Parameter(torch.zeros(1, W, d_model))
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList([_ConfBlock(d_model, nhead, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.recon = nn.Linear(d_model, C)          # pre-training head
        self.head = nn.Linear(2 * d_model, H * C)   # forecasting head
        self.mask_ratio = 0.3
        self.binary_input = True

    def encode(self, x, mask=None):
        z = self.stem(x.transpose(1, 2)).transpose(1, 2)
        if mask is not None:
            z = torch.where(mask[..., None], self.mask_tok.expand_as(z), z)
        z = z + self.pos
        for b in self.blocks:
            z = b(z)
        return self.norm(z)

    def forward(self, x):
        z = self.encode(x)
        return self.head(torch.cat([z.mean(1), z[:, -1]], -1)).reshape(-1, self.H, self.C)

    def pretrain_loss(self, x):
        B, W, _ = x.shape
        mask = torch.rand(B, W, device=x.device) < self.mask_ratio
        mask[:, -1] = True                          # next-slot prediction
        rec = self.recon(self.encode(x, mask))
        if self.binary_input:
            l = F.binary_cross_entropy_with_logits(rec, x, reduction="none")
        else:
            l = (rec - x) ** 2
        return (l.mean(-1) * mask).sum() / mask.sum().clamp(min=1)


class _ConvLSTMCell1d(nn.Module):
    """ConvLSTM cell whose convolution runs along the frequency axis."""

    def __init__(self, cin, hid, k=3):
        super().__init__()
        self.hid = hid
        self.conv = nn.Conv1d(cin + hid, 4 * hid, k, padding=k // 2)

    def forward(self, x, h, c):                       # x [B,cin,C], h,c [B,hid,C]
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class BiConvLSTMSeq2Seq(nn.Module):
    """Wang, Hu, Jiang, Chen, 'A Deep Long-Term Joint Temporal-Spectral Network
    for Spectrum Prediction', Sensors 24(5):1498, 2024. Bidirectional ConvLSTM
    encoder over time (convolution across frequency) + ConvLSTM seq2seq
    decoder that feeds back its own prediction for H steps."""

    def __init__(self, W, H, C, d_model=64, dropout=0.1, hid=16, **kw):
        super().__init__()
        self.H, self.hid = H, hid
        self.fw, self.bw = _ConvLSTMCell1d(1, hid), _ConvLSTMCell1d(1, hid)
        self.dec = _ConvLSTMCell1d(1, 2 * hid)
        self.out = nn.Conv1d(2 * hid, 1, 1)

    def forward(self, x):
        B, W, C = x.shape
        z = lambda n: torch.zeros(B, n, C, device=x.device, dtype=x.dtype)
        hf, cf, hb, cb = z(self.hid), z(self.hid), z(self.hid), z(self.hid)
        for t in range(W):
            hf, cf = self.fw(x[:, t:t + 1], hf, cf)
            hb, cb = self.bw(x[:, W - 1 - t:W - t], hb, cb)
        h, c = torch.cat([hf, hb], 1), torch.cat([cf, cb], 1)
        inp, outs = x[:, -1:], []
        for _ in range(self.H):
            h, c = self.dec(inp, h, c)
            o = self.out(h)                           # logits [B,1,C]
            outs.append(o)
            inp = torch.sigmoid(o)
        return torch.cat(outs, 1)


SOTA_REGISTRY: Dict[str, type] = {"nhits": NHiTS, "patchtst": PatchTST, "itransformer": ITransformer,
                                  "biconvlstm_s2s": BiConvLSTMSeq2Seq, "spectrumfm_f": SpectrumFMF}
SOTA_LABEL = {"persistence": "Persistence", "seasonal_naive": "Seasonal naive",
              "historical_mean": "Historical mean", "nhits": "N-HiTS", "patchtst": "PatchTST",
              "itransformer": "iTransformer", "biconvlstm_s2s": "Bi-ConvLSTM-S2S",
              "spectrumfm_f": "SpectrumFM-F"}


for _k, _v in SOTA_LABEL.items():
    MODEL_LABEL.setdefault(_k, _v)


class SotaWrap(nn.Module):
    """Gives a SOTA forecaster the (binary window, margin, scenario id)
    interface of the novel models. The SOTA model sees ONLY the binary window,
    exactly as in the SOTA run (input_repr = 'binary'); it has no 'loc_'
    parameters, so FedAvg / FedProx average all of it."""

    def __init__(self, inner):
        super().__init__()
        self.m = inner
        self.pretrain = False             # SpectrumFM-F: masked / next-slot pre-training phase

    def forward(self, xb, xe, sid):
        return self.m(xb)


def is_fm(model) -> bool:
    return isinstance(model, SotaWrap) and isinstance(model.m, SpectrumFMF)


_build_novel = build_model


def build_model(name: str, cfg: Dict, C: int, S: int) -> nn.Module:        # novel or SOTA
    if name in MODEL_REGISTRY:
        return _build_novel(name, cfg, C, S)
    m = SotaWrap(SOTA_REGISTRY[name](W=cfg["lookback"], H=cfg["horizon"], C=C,
                                     d_model=cfg["d_model"], dropout=cfg["dropout"]))
    if is_fm(m):
        m.m.mask_ratio = cfg["fm_mask_ratio"]
    return m


def family_of(name: str) -> str:
    base = name[:-4] if name.endswith("_dcs") else name
    if base in MODEL_REGISTRY:
        return "novel"
    if base in SOTA_REGISTRY:
        return "SOTA"
    return "baseline"


# ==========================================================================
#                          STAGE 1 - METRICS
# ==========================================================================
def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    _, first, cnt = np.unique(ps, return_index=True, return_counts=True)
    ranks = np.empty(len(p), np.float64)
    ranks[order] = np.repeat(first + (cnt - 1) / 2.0 + 1.0, cnt)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _auprc(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision (step-wise PR area), ties grouped."""
    npos = float(y.sum())
    if npos == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order].astype(np.float64), p[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]
    tp, fp = tp[last], fp[last]
    rec, prec = tp / npos, tp / np.maximum(tp + fp, 1e-12)
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def _ece(y, p, bins=15) -> float:
    e, n = 0.0, len(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def bin_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5, full: bool = True) -> Dict:
    y = y.reshape(-1).astype(np.float64)
    p = np.clip(p.reshape(-1).astype(np.float64), 1e-7, 1 - 1e-7)
    yb, pb = y > 0.5, p >= thr
    tp = int(np.sum(yb & pb)); tn = int(np.sum(~yb & ~pb))
    fp = int(np.sum(~yb & pb)); fn = int(np.sum(yb & ~pb))
    n = max(len(y), 1)
    rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    out = {"n_cells": int(len(y)), "accuracy": (tp + tn) / n, "precision": prec,
           "recall_pd": rec, "specificity": spec, "false_alarm_rate": 1 - spec,
           "miss_rate": 1 - rec,
           "f1": 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0,
           "balanced_accuracy": 0.5 * (rec + spec),
           "mcc": (tp * tn - fp * fn) / den,
           "brier": float(np.mean((p - y) ** 2)),
           "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
           "mae": float(np.mean(np.abs(p - y))),
           "rmse": float(math.sqrt(np.mean((p - y) ** 2))),
           "positive_rate": float(y.mean()), "pred_positive_rate": float(pb.mean())}
    if full:
        out.update({"auroc": _auroc(y, p), "auprc": _auprc(y, p), "ece": _ece(y, p)})
    return out


def f1_opt_threshold(y, p) -> float:
    y = y.reshape(-1) > 0.5
    p = p.reshape(-1)
    best, bt = -1.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        pb = p >= t
        tp = np.sum(y & pb); fp = np.sum(~y & pb); fn = np.sum(y & ~pb)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best:
            best, bt = f1, float(t)
    return bt


def snr_edges(emax: float, cfg: Dict) -> np.ndarray:
    lo, w = cfg["snr_bin_start_db"], cfg["snr_bin_width_db"]
    hi = lo + w * max(1, math.ceil((emax - lo) / w))
    return np.arange(lo, hi + 1e-9, w)


def _bin_mask(E, lo, hi, last, eta):
    lower = (E > lo) if abs(lo - eta) < 1e-9 else (E >= lo)
    return lower & ((E <= hi) if last else (E < hi))


def forecast_rows(model: str, mode: str, seed: int, scen: str, y, p, E, base: Dict,
                  thr: float, thr_val: float, edges: np.ndarray, cfg: Dict,
                  extra: Dict) -> Tuple[Dict, List[Dict], List[Dict], List[Dict]]:
    """-> overall row, per-horizon rows, per-SNR cell rows, per-SNR unit rows."""
    eta = cfg["eta_db"]
    key = {"model": MODEL_LABEL.get(model, model), "model_key": model, "mode": mode,
           "seed": seed, "scenario": scen}
    ov = {**key, **bin_metrics(y, p, thr)}
    vt = bin_metrics(y, p, thr_val, full=False)
    ov.update({"threshold_val_f1opt": thr_val, "f1_at_val_thr": vt["f1"],
               "accuracy_at_val_thr": vt["accuracy"], "recall_at_val_thr": vt["recall_pd"],
               "far_at_val_thr": vt["false_alarm_rate"]})
    bb = {k: float(np.mean((np.clip(v, 1e-7, 1 - 1e-7) - y) ** 2)) for k, v in base.items()}
    best_b = min(bb.values()) if bb else float("nan")
    ov["brier_skill_vs_persistence"] = 1 - ov["brier"] / bb["persistence"] if "persistence" in bb else float("nan")
    ov["brier_skill_vs_hist_mean"] = 1 - ov["brier"] / bb["historical_mean"] if "historical_mean" in bb else float("nan")
    ov["brier_skill_vs_best_baseline"] = 1 - ov["brier"] / best_b if bb else float("nan")
    ov["beats_hist_mean"] = bool(ov["brier"] < bb.get("historical_mean", -1))
    ov["beats_all_baselines"] = bool(ov["brier"] < best_b)
    ov.update(extra)

    hz = []
    for h in range(y.shape[1]):
        r = bin_metrics(y[:, h], p[:, h], thr)
        hz.append({**key, "horizon_step": h + 1, **{k: r[k] for k in
                   ("accuracy", "precision", "recall_pd", "false_alarm_rate", "f1",
                    "mcc", "auroc", "auprc", "brier", "logloss")}})

    cells = []
    nb = len(edges) - 1
    idle = E <= eta
    if idle.any():
        r = bin_metrics(y[idle], p[idle], thr, full=False)
        cells.append({**key, "snr_bin": f"<= {eta:g} (idle)", "snr_lo_db": float("-inf"),
                      "snr_hi_db": eta, "n_cells": int(idle.sum()),
                      "false_alarm_rate": r["false_alarm_rate"], "specificity": r["specificity"],
                      "recall_pd": float("nan"), "miss_rate": float("nan"),
                      "mean_prob": float(p[idle].mean()), "brier": r["brier"],
                      "logloss": r["logloss"],
                      "brier_best_baseline": min(float(np.mean((v[idle] - y[idle]) ** 2)) for v in base.values()) if base else float("nan")})
    for i in range(nb):
        lo, hi = float(edges[i]), float(edges[i + 1])
        m = _bin_mask(E, lo, hi, i == nb - 1, eta)
        if not m.any():
            cells.append({**key, "snr_bin": f"{lo:g}-{hi:g}", "snr_lo_db": lo, "snr_hi_db": hi,
                          "n_cells": 0})
            continue
        pm, ym = p[m], y[m]
        bbm = min(float(np.mean((v[m] - ym) ** 2)) for v in base.values()) if base else float("nan")
        br = float(np.mean((pm - ym) ** 2))
        cells.append({**key, "snr_bin": f"{lo:g}-{hi:g}", "snr_lo_db": lo, "snr_hi_db": hi,
                      "n_cells": int(m.sum()),
                      "recall_pd": float(np.mean(pm >= thr)), "miss_rate": float(np.mean(pm < thr)),
                      "recall_pd_at_val_thr": float(np.mean(pm >= thr_val)),
                      "mean_prob": float(pm.mean()), "brier": br,
                      "logloss": float(-np.mean(np.log(np.clip(pm, 1e-7, 1)))),
                      "brier_best_baseline": bbm,
                      "brier_skill_vs_best_baseline": 1 - br / bbm if bbm > 0 else float("nan")})

    units = []
    U = E.mean(1)                                     # [N, C]: mean target SNR
    for i in range(-1, nb):
        if i < 0:
            m, lab, lo, hi = U <= eta, f"<= {eta:g}", float("-inf"), eta
        else:
            lo, hi = float(edges[i]), float(edges[i + 1])
            m, lab = _bin_mask(U, lo, hi, i == nb - 1, eta), f"{lo:g}-{hi:g}"
        if not m.any():
            units.append({**key, "snr_bin": lab, "snr_lo_db": lo, "snr_hi_db": hi, "n_units": 0})
            continue
        ym = y.transpose(0, 2, 1)[m]                  # [n_units, H]
        pm = p.transpose(0, 2, 1)[m]
        r = bin_metrics(ym, pm, thr)
        bbm = min(float(np.mean((v.transpose(0, 2, 1)[m] - ym) ** 2)) for v in base.values()) if base else float("nan")
        units.append({**key, "snr_bin": lab, "snr_lo_db": lo, "snr_hi_db": hi,
                      "n_units": int(m.sum()), **r,
                      "brier_skill_vs_best_baseline": 1 - r["brier"] / bbm if bbm > 0 else float("nan")})
    return ov, hz, cells, units


# ==========================================================================
#                          COMPLEXITY PROFILING
# ==========================================================================
def count_flops(fn, *args) -> float:
    """FLOPs of one call, counted by torch's FlopCounterMode (matmul, conv,
    bmm, attention). Element-wise ops are not counted (standard practice)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                fn(*args)
        return float(fc.get_total_flops())
    except Exception:
        return float("nan")


def _time_call(fn, n_iter, n_warm, sync=None) -> float:
    for _ in range(n_warm):
        fn()
    if sync:
        sync()
    ts = []
    for _ in range(n_iter):
        t = time.perf_counter(); fn()
        if sync:
            sync()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def profile_forecaster(name: str, cfg: Dict, C: int, model: Optional[nn.Module] = None) -> Dict:
    W, H = cfg["lookback"], cfg["horizon"]
    row = {"model": MODEL_LABEL.get(name, name), "model_key": name}
    if model is None:                       # non-learned baseline
        X = np.random.default_rng(0).integers(0, 2, (cfg["prof_batch"], W, C)).astype(np.float32)
        x1 = X[:1]
        f1 = lambda: baseline_forecast(name, x1, H, 4)
        fb = lambda: baseline_forecast(name, X, H, 4)
        fl = {"persistence": 0, "seasonal_naive": 0, "historical_mean": W * C}[name]
        row.update({"params": 0, "params_M": 0.0, "size_MB": 0.0,
                    "flops_per_sample": float(fl), "mflops_per_sample": fl / 1e6,
                    "macs_per_sample": fl / 2.0,
                    "cpu_latency_ms_batch1": _time_call(f1, cfg["prof_iters"], cfg["prof_warmup"]) * 1e3,
                    "cpu_latency_ms_per_sample_b%d" % cfg["prof_batch"]:
                        _time_call(fb, cfg["prof_iters"], cfg["prof_warmup"]) * 1e3 / cfg["prof_batch"]})
        return row
    m = copy.deepcopy(model).cpu()
    n_par = sum(p.numel() for p in m.parameters())
    m.train()                                # disables the fused MHA fast path (uncountable)
    fl = count_flops(m, torch.zeros(2, W, C)) / 2.0
    m.eval()
    x1, xb = torch.zeros(1, W, C), torch.zeros(cfg["prof_batch"], W, C)
    with torch.inference_mode():
        t1 = _time_call(lambda: m(x1), cfg["prof_iters"], cfg["prof_warmup"])
        tb = _time_call(lambda: m(xb), max(3, cfg["prof_iters"] // 5), cfg["prof_warmup"])
    row.update({"params": n_par, "params_M": n_par / 1e6,
                "size_MB": sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6,
                "flops_per_sample": fl, "mflops_per_sample": fl / 1e6, "macs_per_sample": fl / 2.0,
                "cpu_latency_ms_batch1": t1 * 1e3,
                "cpu_latency_ms_per_sample_b%d" % cfg["prof_batch"]: tb * 1e3 / cfg["prof_batch"]})
    if torch.cuda.is_available():
        g = copy.deepcopy(model).cuda().eval()
        x1g, xbg = x1.cuda(), xb.cuda()
        with torch.inference_mode():
            row["gpu_latency_ms_batch1"] = _time_call(lambda: g(x1g), cfg["prof_iters"],
                                                      cfg["prof_warmup"], torch.cuda.synchronize) * 1e3
            row["gpu_latency_ms_per_sample_b%d" % cfg["prof_batch"]] = _time_call(
                lambda: g(xbg), cfg["prof_iters"], cfg["prof_warmup"],
                torch.cuda.synchronize) * 1e3 / cfg["prof_batch"]
        del g
    return row


# ==========================================================================
#                     TRAINING (centralised / local / FL)
# ==========================================================================
def _groups(model, wd):
    dec = [p for n, p in model.named_parameters() if "loc_" not in n]
    nod = [p for n, p in model.named_parameters() if "loc_" in n]
    return [{"params": dec, "weight_decay": wd}, {"params": nod, "weight_decay": 0.0}]


def _loss(model, b):
    xb, xe, sid, y = b
    if getattr(model, "pretrain", False):                       # SpectrumFM-F pre-training
        return model.m.pretrain_loss(xb)
    return F.binary_cross_entropy_with_logits(model(xb, xe, sid), y)


def fm_pretrain_central(model, Ttr, cfg, rng) -> None:
    """SpectrumFM-F: self-supervised pre-training on the TRAINING split only."""
    if not is_fm(model) or cfg["fm_pretrain_epochs"] <= 0:
        return
    model.pretrain = True
    train_epochs(model, Ttr, cfg, cfg["fm_pretrain_epochs"], rng)
    model.pretrain = False


def fm_pretrain_federated(model, clients, cfg, rng, weights=None) -> int:
    """SpectrumFM-F under FL: fm_pretrain_epochs FedAvg rounds of the
    pre-training objective (1 local epoch each). -> rounds run."""
    if not is_fm(model) or cfg["fm_pretrain_epochs"] <= 0:
        return 0
    model.pretrain = True
    g = copy.deepcopy(model.state_dict())
    for _ in range(cfg["fm_pretrain_epochs"]):
        states, ws = [], []
        for T in clients:
            model.load_state_dict(g)
            train_epochs(model, T, cfg, 1, rng)
            states.append(copy.deepcopy(model.state_dict()))
            ws.append(T["Xb"].shape[0])
        w = np.asarray(ws, np.float64); w = w / w.sum()
        g = {k: (sum(float(wi) * st[k] for wi, st in zip(w, states)) if v.dtype.is_floating_point
                 else states[0][k].clone()) for k, v in g.items()}
    model.load_state_dict(g)
    model.pretrain = False
    return cfg["fm_pretrain_epochs"]


def _batches(T: Dict, idx):
    return (T["Xb"][idx], T["Xe"][idx], T["sid"][idx], T["Y"][idx])


def train_epochs(model, T, cfg, epochs, rng, prox_ref=None, mu=0.0):
    opt = torch.optim.AdamW(_groups(model, cfg["weight_decay"]), lr=cfg["lr"])
    n, bs = T["Xb"].shape[0], cfg["batch_size"]
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for a in range(0, n, bs):
            j = torch.as_tensor(perm[a:a + bs], device=T["Xb"].device)
            if len(j) < 2:
                continue
            loss = _loss(model, _batches(T, j))
            if prox_ref is not None and mu > 0:
                loss = loss + 0.5 * mu * sum(((p - q) ** 2).sum() for (nm, p), q in
                                             zip(model.named_parameters(), prox_ref) if "loc_" not in nm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()


@torch.no_grad()
def eval_loss(model, T, bs=2048) -> float:
    model.eval()
    n, tot = T["Xb"].shape[0], 0.0
    for a in range(0, n, bs):
        j = torch.arange(a, min(n, a + bs), device=T["Xb"].device)
        tot += float(_loss(model, _batches(T, j))) * len(j)
    return tot / max(n, 1)


def cat_sets(sets: List[Dict]) -> Dict:
    return {k: torch.cat([s[k] for s in sets]) for k in sets[0]}


def fit_central(model, Ttr, Tva, cfg, seed) -> Dict:
    rng = np.random.default_rng(seed)
    best, state, bad, n, t0 = float("inf"), None, 0, 0, time.time()
    fm_pretrain_central(model, Ttr, cfg, rng)
    for ep in range(cfg["epochs"]):
        train_epochs(model, Ttr, cfg, 1, rng)
        va = eval_loss(model, Tva)
        n += 1
        if va < best - 1e-5:
            best, bad, state = va, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if state is not None:
        model.load_state_dict(state)
    return {"epochs_run": n, "best_val_loss": best, "train_s": time.time() - t0}


def model_bytes(model, shared_only=True) -> int:
    return sum(v.numel() * v.element_size() for k, v in model.state_dict().items()
               if not (shared_only and "loc_" in k))


def _aggregate(g_old: Dict, states: List[Dict], weights) -> Dict:
    """Shared parameters: sample-weighted average (FedAvg). 'loc_' parameters:
    every client contributes only ITS OWN change (they stay personal)."""
    w = np.asarray(weights, np.float64); w = w / w.sum()
    out = {}
    for k, v in g_old.items():
        if "loc_" in k:
            out[k] = v + sum(s[k] - v for s in states)
        elif v.dtype.is_floating_point:
            out[k] = sum(float(wi) * s[k] for wi, s in zip(w, states))
        else:
            out[k] = states[0][k].clone()
    return out


def fit_federated(model, clients: List[Dict], vals: List[Dict], cfg, algo, seed) -> Dict:
    mu = cfg["fedprox_mu"] if algo == "fedprox" else 0.0
    rng = np.random.default_rng(seed)
    g = copy.deepcopy(model.state_dict())
    best, state, bad, hist, t0 = float("inf"), None, 0, 0, time.time()
    nval = sum(v["Xb"].shape[0] for v in vals)
    pre = fm_pretrain_federated(model, clients, cfg, rng)
    for r in range(cfg["fl_rounds"]):
        states, ws = [], []
        for T in clients:
            model.load_state_dict(g)
            ref = [p.detach().clone() for p in model.parameters()] if mu > 0 else None
            train_epochs(model, T, cfg, cfg["fl_local_epochs"], rng, ref, mu)
            states.append(copy.deepcopy(model.state_dict()))
            ws.append(T["Xb"].shape[0])
        g = _aggregate(g, states, ws)
        model.load_state_dict(g)
        va = sum(eval_loss(model, V) * V["Xb"].shape[0] for V in vals) / max(nval, 1)
        hist += 1
        if va < best - 1e-5:
            best, bad, state = va, 0, copy.deepcopy(g)
        else:
            bad += 1
            if bad >= cfg["fl_patience"]:
                break
    if state is not None:
        model.load_state_dict(state)
    nb = model_bytes(model)
    return {"rounds_run": hist, "best_val_loss": best, "train_s": time.time() - t0,
            "comm_bytes": 2 * nb * len(clients) * (hist + pre), "model_bytes": nb,
            "n_clients": len(clients), "pretrain_rounds": pre}


@torch.no_grad()
def predict_logits(model, T, bs=2048) -> np.ndarray:
    model.eval()
    n, out = T["Xb"].shape[0], []
    for a in range(0, n, bs):
        j = torch.arange(a, min(n, a + bs), device=T["Xb"].device)
        xb, xe, sid, _ = _batches(T, j)
        out.append(model(xb, xe, sid).float().cpu().numpy())
    return np.concatenate(out, 0)


def train_one(mname, mode, seed, D, names, cfg, dev, C, S):
    """-> {scenario: {'zte','zva','ztr1'}} logits, info."""
    out, info = {}, {"model": mname, "mode": mode, "seed": seed}

    def new(tag):
        set_seed(run_seed(cfg, mname, mode, seed, tag))
        return build_model(mname, cfg, C, S).to(dev)

    def pred(m, scen):
        for s in scen:
            out[s] = {"zte": predict_logits(m, D[s]["te"]), "zva": predict_logits(m, D[s]["va"]),
                      "ztr1": predict_logits(m, D[s]["tr"])[:, 0, :]}

    if mode == "local":
        tot = 0.0
        for s in names:
            m = new(s)
            tot += fit_central(m, D[s]["tr"], D[s]["va"], cfg, seed)["train_s"]
            pred(m, [s])
        info.update({"train_s": tot, "comm_bytes": 0})
    elif mode == "centralized":
        m = new("all")
        r = fit_central(m, cat_sets([D[s]["tr"] for s in names]),
                        cat_sets([D[s]["va"] for s in names]), cfg, seed)
        info.update({"train_s": r["train_s"], "epochs_run": r["epochs_run"], "comm_bytes": 0})
        pred(m, names)
    else:
        if len(names) >= 2:
            clients, vals, ctype = [D[s]["tr"] for s in names], [D[s]["va"] for s in names], "scenario"
        else:
            T = D[names[0]]["tr"]
            cut = np.linspace(0, T["Xb"].shape[0], 5).astype(int)
            clients = [{k: v[a:b] for k, v in T.items()} for a, b in zip(cut[:-1], cut[1:])]
            vals, ctype = [D[names[0]]["va"]], "temporal (single scenario)"
        m = new("fl")
        r = fit_federated(m, clients, vals, cfg, mode, seed)
        info.update({"train_s": r["train_s"], "rounds_run": r["rounds_run"],
                     "comm_bytes": r["comm_bytes"], "model_bytes": r["model_bytes"],
                     "n_clients": r["n_clients"], "client_type": ctype})
        pred(m, names)
    return out, info


# ==========================================================================
#                      COMPLEXITY (params, FLOPs, latency)
# ==========================================================================
def _time_call(fn, n_iter, n_warm, sync=None) -> float:
    for _ in range(n_warm):
        fn()
    if sync:
        sync()
    ts = []
    for _ in range(n_iter):
        t = time.perf_counter(); fn()
        if sync:
            sync()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def profile_novel(name, cfg, C, S) -> Dict:
    W = cfg["lookback"]
    m = build_model(name, cfg, C, S).cpu()
    n_par = sum(p.numel() for p in m.parameters())
    mk = lambda B: (torch.zeros(B, W, C), torch.zeros(B, W, C), torch.zeros(B, dtype=torch.long))
    m.train()
    fl = count_flops(lambda: m(*mk(2))) / 2.0
    m.eval()
    x1, xb = mk(1), mk(cfg["prof_batch"])
    row = {"model": MODEL_LABEL[name], "model_key": name, "params": n_par, "params_M": n_par / 1e6,
           "params_shared_in_FL": sum(p.numel() for nm, p in m.named_parameters() if "loc_" not in nm),
           "size_MB": sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6,
           "flops_per_sample": fl, "mflops_per_sample": fl / 1e6, "macs_per_sample": fl / 2.0}
    with torch.inference_mode():
        row["cpu_latency_ms_batch1"] = _time_call(lambda: m(*x1), cfg["prof_iters"], cfg["prof_warmup"]) * 1e3
        row["cpu_latency_ms_per_sample_b%d" % cfg["prof_batch"]] = _time_call(
            lambda: m(*xb), max(3, cfg["prof_iters"] // 5), cfg["prof_warmup"]) * 1e3 / cfg["prof_batch"]
        if torch.cuda.is_available():
            g = copy.deepcopy(m).cuda().eval()
            g1 = tuple(t.cuda() for t in x1); gb = tuple(t.cuda() for t in xb)
            row["gpu_latency_ms_batch1"] = _time_call(lambda: g(*g1), cfg["prof_iters"], cfg["prof_warmup"],
                                                      torch.cuda.synchronize) * 1e3
            row["gpu_latency_ms_per_sample_b%d" % cfg["prof_batch"]] = _time_call(
                lambda: g(*gb), cfg["prof_iters"], cfg["prof_warmup"], torch.cuda.synchronize) * 1e3 / cfg["prof_batch"]
    row["dcs_extra_cost"] = "H x C additions + H x C sigmoids per window (streaming offset)"
    return row


# ==========================================================================
#                             CSV / CACHE
# ==========================================================================
def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    log(f"  -> {os.path.basename(path)} ({len(rows)} rows)")


def aggregate(rows: List[Dict], keys: List[str]) -> List[Dict]:
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in keys)].append(r)
    out = []
    for g, rs in groups.items():
        o = dict(zip(keys, g))
        o["n_seeds"] = len(rs)
        for k in rs[0]:
            if k in keys or k == "seed":
                continue
            v = [r[k] for r in rs if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
            if len(v) == len(rs):
                o[k] = float(np.mean(v)); o[k + "_std"] = float(np.std(v))
            elif k not in o:
                o[k] = rs[0][k]
        out.append(o)
    return out


def fingerprint(cfg: Dict, data_fp: str, *parts) -> str:
    keys = ["subchannel_mode", "subchannels_per_band", "subchannel_hz", "noise_quantile", "eta_db",
            "split_blocks", "max_gap_steps", "fl_norm", "min_per_class_per_client",
            "lookback", "horizon", "stride", "val_frac", "test_frac", "epochs", "batch_size", "lr",
            "weight_decay", "patience", "d_model", "dropout", "fl_rounds", "fl_local_epochs",
            "fedprox_mu", "fl_patience", "fm_pretrain_epochs", "fm_mask_ratio", "seed"]
    s = json.dumps({k: cfg[k] for k in keys}, sort_keys=True, default=str) + data_fp + "|".join(map(str, parts))
    return "nv" + hashlib.sha1(s.encode()).hexdigest()[:14]


# ==========================================================================
#                              ORCHESTRATION
# ==========================================================================
def run_forecasting(cfg, dev, D, names, C, S, edges, out_dir, data_fp):
    cache = os.path.join(out_dir, "cache"); os.makedirs(cache, exist_ok=True)
    thr, H = cfg["decision_threshold"], cfg["horizon"]
    OV, HZ, CELL, UNIT, COMM, REF = [], [], [], [], [], []
    store: Dict[Tuple, Dict] = {}
    base = {s: {b: {"te": baseline_forecast(b, D[s]["Xbin_te"], H, D[s]["period"]),
                    "va": baseline_forecast(b, D[s]["Xbin_va"], H, D[s]["period"])}
                for b in cfg["baselines"]} for s in names}

    def emit(model, mode, seed, P, extra, sink_ov):
        for s in names + (["Combined"] if len(names) > 1 else []):
            sc = names if s == "Combined" else [s]
            y = np.concatenate([D[k]["y_te"] for k in sc]); E = np.concatenate([D[k]["E_te"] for k in sc])
            p = np.concatenate([P[k]["te"] for k in sc])
            yv = np.concatenate([D[k]["y_va"] for k in sc]); pv = np.concatenate([P[k]["va"] for k in sc])
            bp = {b: np.concatenate([base[k][b]["te"] for k in sc]) for b in cfg["baselines"]}
            ov, hz, ce, un = forecast_rows(model, mode, seed, s, y, p, E, bp, thr,
                                           f1_opt_threshold(yv, pv), edges, cfg, extra)
            ov["val_brier"] = float(np.mean((pv - yv) ** 2))
            sink_ov.append(ov)
            if sink_ov is OV:
                HZ.extend(hz); CELL.extend(ce); UNIT.extend(un)

    for b in cfg["baselines"]:
        emit(b, "non-learned", 0, {s: base[s][b] for s in names}, {}, REF)
        store[(b, "non-learned", 0)] = {s: {"te": base[s][b]["te"]} for s in names}

    for seed in cfg["seeds"]:
        for mode in cfg["modes"]:
            for mname in cfg["forecast_models"]:
                fp = fingerprint(cfg, data_fp, mname, mode, seed)
                cp = os.path.join(cache, f"{mname}__{mode}__s{seed}__{fp}.npz")
                t0 = time.time()
                if cfg["resume"] and os.path.exists(cp):
                    z = np.load(cp, allow_pickle=True)
                    Z = {s: {k: z[f"{s}__{k}"] for k in ("zte", "zva", "ztr1")} for s in names}
                    info = json.loads(str(z["info"]))
                    log(f"  cached  {MODEL_LABEL[mname]:<6s} {mode:<12s} seed {seed}")
                else:
                    Z, info = train_one(mname, mode, seed, D, names, cfg, dev, C, S)
                    np.savez_compressed(cp, info=json.dumps(info),
                                        **{f"{s}__{k}": v for s, d in Z.items() for k, v in d.items()})
                    log(f"  trained {MODEL_LABEL[mname]:<6s} {mode:<12s} seed {seed} ({time.time() - t0:.0f} s)")
                extra = {"train_s": info.get("train_s", float("nan")),
                         "comm_MB": info.get("comm_bytes", 0) / 1e6}
                P = {s: {"te": _sig(Z[s]["zte"]).astype(np.float32), "va": _sig(Z[s]["zva"]).astype(np.float32),
                         "tr1": _sig(Z[s]["ztr1"]).astype(np.float32)} for s in names}
                store[(mname, mode, seed)] = P
                emit(mname, mode, seed, P, extra, OV)
                if cfg["dcs"]:
                    Pd, etas = {}, {}
                    for s in names:                       # eta chosen on validation (causal stream)
                        best = min(cfg["dcs_etas"], key=lambda e: float(np.mean(
                            (dcs_stream(Z[s]["zva"], D[s]["y_va"], e)[0] - D[s]["y_va"]) ** 2)))
                        pva, beta = dcs_stream(Z[s]["zva"], D[s]["y_va"], best)
                        pte, _ = dcs_stream(Z[s]["zte"], D[s]["y_te"], best, beta)
                        Pd[s] = {"te": pte, "va": pva, "tr1": P[s]["tr1"]}
                        etas[s] = best
                    store[(mname + "_dcs", mode, seed)] = Pd
                    emit(mname + "_dcs", mode, seed, Pd, {**extra, **{f"dcs_eta_{s}": e for s, e in etas.items()}}, OV)
                COMM.append({"model": MODEL_LABEL[mname], "mode": mode, "seed": seed,
                             "comm_MB_shared_params": info.get("comm_bytes", 0) / 1e6,
                             "model_MB_shared": info.get("model_bytes", 0) / 1e6,
                             "rounds_run": info.get("rounds_run", ""), "n_clients": info.get("n_clients", ""),
                             "client_type": info.get("client_type", ""), "train_s": info.get("train_s", float("nan"))})
            write_outputs(out_dir, OV, HZ, CELL, UNIT, COMM, REF, None)
    log("\n[complexity]")
    CX = []
    for mname in cfg["forecast_models"]:
        try:
            CX.append(profile_novel(mname, cfg, C, S))
            log(f"  {MODEL_LABEL[mname]:<6s} params {CX[-1]['params']:>9,d}  "
                f"{CX[-1]['mflops_per_sample']:8.3f} MFLOPs  CPU b1 {CX[-1]['cpu_latency_ms_batch1']:.2f} ms")
        except Exception as e:
            log(f"  profiling {mname} failed: {e}")
    write_outputs(out_dir, OV, HZ, CELL, UNIT, COMM, REF, CX)
    return store, OV


def write_outputs(out_dir, OV, HZ, CELL, UNIT, COMM, REF, CX):
    write_csv(os.path.join(out_dir, "forecast_overall.csv"), OV)
    write_csv(os.path.join(out_dir, "forecast_overall_agg.csv"), aggregate(OV, ["model", "model_key", "mode", "scenario"]))
    write_csv(os.path.join(out_dir, "forecast_per_horizon.csv"),
              aggregate(HZ, ["model", "model_key", "mode", "scenario", "horizon_step"]))
    write_csv(os.path.join(out_dir, "forecast_per_snr_cell.csv"),
              aggregate(CELL, ["model", "model_key", "mode", "scenario", "snr_bin"]))
    write_csv(os.path.join(out_dir, "forecast_per_snr_window.csv"),
              aggregate(UNIT, ["model", "model_key", "mode", "scenario", "snr_bin"]))
    write_csv(os.path.join(out_dir, "reference_baselines.csv"), REF)
    write_csv(os.path.join(out_dir, "fl_communication.csv"), COMM)
    if CX:
        write_csv(os.path.join(out_dir, "forecast_complexity.csv"), CX)


def write_stage2(cfg, out_dir, D, names, C, edges_snr, store, OV) -> None:
    """Same format as the SOTA script's stage2/: labels + forecasts of the
    best NOVEL model of each mode (validation Brier, Combined)."""
    d = os.path.join(out_dir, "stage2")
    os.makedirs(d, exist_ok=True)
    lab = {"names": json.dumps(names), "C": C, "edges_snr": edges_snr, "eta_db": cfg["eta_db"],
           "horizon": cfg["horizon"]}
    for s in names:
        for k in ("y_te", "E_te", "y_tr1", "E_tr1", "inr_busy", "inr_idle"):
            lab[f"{s}__{k}"] = D[s][k]
    np.savez_compressed(os.path.join(d, "labels.npz"), **lab)
    scope = "Combined" if len(names) > 1 else names[0]
    keys = sorted({k for k, m, s in store if m != "non-learned"})
    best = {}
    for mode in cfg["modes"]:
        cand = [(np.mean([r["val_brier"] for r in OV if r["model_key"] == k and r["mode"] == mode
                          and r["scenario"] == scope]), k) for k in keys
                if any(r["model_key"] == k and r["mode"] == mode for r in OV)]
        if cand:
            best[mode] = min(cand)[1]
    files = []
    for (m, mode, seed), P in store.items():
        if not (mode == "non-learned" or best.get(mode) == m):
            continue
        fn = f"preds__{m}__{mode}__s{seed}.npz"
        arr = {}
        for s in names:
            arr[f"{s}__te"] = P[s]["te"].astype(np.float16)
            if "tr1" in P[s]:
                arr[f"{s}__tr1"] = P[s]["tr1"].astype(np.float16)
        np.savez_compressed(os.path.join(d, fn), **arr)
        files.append(fn)
    man = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "scenarios": names, "C": C,
           "best_by_mode": best, "seeds": cfg["seeds"], "baselines": cfg["baselines"], "files": files,
           "labels": {k: cfg[k] for k in ("subchannel_hz", "noise_quantile", "eta_db", "lookback", "horizon")},
           "model_labels": MODEL_LABEL, "cache_dir": ""}
    with open(os.path.join(d, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    log(f"\n  stage-2 hand-off -> {d}")
    for mode, m in best.items():
        log(f"    best novel forecaster ({mode}, validation Brier): {MODEL_LABEL[m]}")


def _tt(a, dev):
    return torch.as_tensor(a, dtype=torch.float32, device=dev)


# ==========================================================================
#                 ABLATION: CLIENT PARTITIONS (IID / NON-IID)
# ==========================================================================
def snr_edges_ab(emax: float, cfg: Dict) -> np.ndarray:
    """2 dB bins from eta to snr_bin_max_db, then ONE overflow bin to the
    largest measured excess (nothing is dropped)."""
    lo, w, hi = cfg["snr_bin_start_db"], cfg["snr_bin_width_db"], cfg["snr_bin_max_db"]
    e = list(np.arange(lo, hi + 1e-9, w))
    if emax > hi + 1e-9:
        e.append(float(math.ceil(emax)))
    return np.asarray(e, np.float64)


def _subset(T: Dict, idx) -> Dict:
    j = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=T["Xb"].device)
    return {k: v[j] for k, v in T.items()}


def merge_until_balanced(groups: List[List[np.ndarray]], ypos: List[float], yneg: List[float],
                         min_per_class: int) -> Tuple[List[int], int]:
    """_merge_until_balanced() of 01_sensing.py: merge
    ADJACENT time chunks until every client holds at least min_per_class
    occupied and min_per_class vacant cells (a client with one class drags
    FedAvg towards 'always vacant'). -> chunk -> client map, merges done."""
    cl = [[i] for i in range(len(groups))]
    pos, neg = [ypos[i] for i in range(len(groups))], [yneg[i] for i in range(len(groups))]
    merges, changed = 0, True
    while changed and len(cl) > 1:
        changed = False
        for i in range(len(cl)):
            if pos[i] < min_per_class or neg[i] < min_per_class:
                j = i + 1 if i + 1 < len(cl) else i - 1
                a, b = min(i, j), max(i, j)
                cl[a] = cl[a] + cl[b]; pos[a] += pos[b]; neg[a] += neg[b]
                del cl[b], pos[b], neg[b]
                merges += 1; changed = True
                break
    out = np.empty(len(groups), int)
    for c, members in enumerate(cl):
        out[members] = c
    return out.tolist(), merges


def temporal_chunks(times: Dict[str, np.ndarray], pos: Dict[str, np.ndarray], ncell: float, scen: List[str],
                    K: int, min_per_class: int) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """Non-IID TEMPORAL clients of the sensing pipeline ('natural',
    single radio): K contiguous stretches of the evening with equal numbers
    of training windows (pooled over the scenarios), each client holding
    EVERY band of that stretch; adjacent stretches merged until balanced.
    -> [ {scenario: window indices} per client ], merges."""
    allt = np.sort(np.concatenate([times[s] for s in scen]))
    inner = np.quantile(allt, np.linspace(0, 1, K + 1)[1:-1]) if K > 1 else np.zeros(0)
    chunk = {s: np.searchsorted(inner, times[s], "right") for s in scen}
    ypos = [float(sum(pos[s][chunk[s] == k].sum() for s in scen)) for k in range(K)]
    ytot = [float(sum((chunk[s] == k).sum() * ncell for s in scen)) for k in range(K)]
    cmap, merges = merge_until_balanced(list(range(K)), ypos, [t - p for t, p in zip(ytot, ypos)], min_per_class)
    out = [dict() for _ in range(max(cmap) + 1)]
    for s in scen:
        for k in range(K):
            ix = np.flatnonzero(chunk[s] == k)
            if ix.size:
                c = cmap[k]
                out[c][s] = np.concatenate([out[c].get(s, np.zeros(0, int)), ix])
    return [o for o in out if o], merges


def partition_clients(D, names, partition, K, seed, cfg=None):
    """-> clients [dict of tensors], counts [K', S] windows per scenario.
      noniid  TEMPORAL clients (the sensing pipeline's 'natural' partition):
              a client = one receiver during one contiguous stretch of the
              evening, all bands of that stretch; with several radios the K
              clients are shared out among the radios. Adjacent stretches are
              merged until each client holds both classes (K' <= K).
      band    one band per client, contiguous time blocks of it (the
              (device, band) partition the sensing pipeline warns about:
              quiet DFS bands become single-class clients); K < S groups
              whole bands.
      iid     uniformly random equal shares of all training windows."""
    cfg = cfg or CONFIG
    S = len(names)
    clients, counts = [], []
    if partition == "noniid":
        radios = sorted({D[s]["radio"] for s in names})
        Kr = [K // len(radios) + (1 if i < K % len(radios) else 0) for i in range(len(radios))]
        for r, kr in zip(radios, Kr):
            scen = [s for s in names if D[s]["radio"] == r]
            if kr < 1:
                kr = 1
            chunks, merges = temporal_chunks({s: D[s]["t_tr"] for s in scen},
                                             {s: D[s]["ypos_tr"] for s in scen}, D[scen[0]]["ncell_tr"],
                                             scen, kr, cfg["min_per_class_per_client"])
            if merges:
                log(f"    [{r}] {merges} adjacent time stretch(es) merged so every client holds both classes")
            for ch in chunks:
                clients.append(cat_sets([_subset(D[s]["tr"], ix) for s, ix in ch.items()]))
                c = np.zeros(S)
                for s, ix in ch.items():
                    c[names.index(s)] = len(ix)
                counts.append(c)
    elif partition == "band" and K < S:
        for ci in range(K):
            mine = [si for si in range(S) if si % K == ci]
            clients.append(cat_sets([D[names[si]]["tr"] for si in mine]))
            c = np.zeros(S)
            for si in mine:
                c[si] = D[names[si]]["tr"]["Xb"].shape[0]
            counts.append(c)
    elif partition == "band":
        per = [K // S + (1 if i < K % S else 0) for i in range(S)]
        for si, s in enumerate(names):
            T = D[s]["tr"]
            n = T["Xb"].shape[0]
            cut = np.linspace(0, n, per[si] + 1).astype(int)
            for a, b in zip(cut[:-1], cut[1:]):
                clients.append(_subset(T, np.arange(a, b)))
                c = np.zeros(S); c[si] = b - a
                counts.append(c)
    elif partition == "iid":
        T = cat_sets([D[s]["tr"] for s in names])
        n = T["Xb"].shape[0]
        rng = np.random.default_rng(run_seed(cfg, "iid-partition", K, seed))
        for p in np.array_split(rng.permutation(n), K):
            p = np.sort(p)
            clients.append(_subset(T, p))
            sid = T["sid"][torch.as_tensor(p, device=T["sid"].device)].cpu().numpy()
            counts.append(np.bincount(sid, minlength=S).astype(float))
    else:
        raise ValueError(f"unknown partition {partition}")
    return clients, np.asarray(counts)


def convert_bn_to_gn(module: nn.Module, max_groups: int = 32) -> nn.Module:
    """convert_bn_to_gn() of 01_sensing.py: every
    BatchNorm -> GroupNorm for FEDERATED training (FedAvg of BatchNorm running
    statistics of heterogeneous clients gives a normaliser that fits no
    client; Hsieh et al., ICML 2020). Centralised / local keep BatchNorm."""
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            c = child.num_features
            g = max_groups
            while g > 1 and c % g != 0:
                g //= 2
            gn = nn.GroupNorm(max(1, g), c, affine=True)
            if child.affine:
                with torch.no_grad():
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)
            setattr(module, name, gn)
        else:
            convert_bn_to_gn(child, max_groups)
    return module


def np_threshold(y: np.ndarray, p: np.ndarray, target_pfa: float) -> float:
    """Neyman-Pearson operating point of the sensing pipeline (threshold_mode
    'target_pfa'): the smallest threshold whose VALIDATION false-alarm rate
    is <= target_pfa; applied unchanged to test."""
    y = y.reshape(-1) > 0.5
    neg = np.sort(p.reshape(-1)[~y])
    if neg.size == 0:
        return 0.5
    k = int(math.floor(target_pfa * neg.size))               # allowed false alarms
    thr = float(np.nextafter(neg[neg.size - k - 1], np.inf)) if k < neg.size else float(neg[0])
    return min(max(thr, 1e-6), 1.0)


def client_rows(clients, counts, names, partition, K, seed) -> List[Dict]:
    S = len(names)
    glob_mix = counts.sum(0) / counts.sum()
    tv = 0.5 * np.abs(counts / counts.sum(1, keepdims=True) - glob_mix).sum(1)
    occ = np.array([float(c["Y"].mean()) for c in clients])
    rows = []
    for i, (c, n) in enumerate(zip(clients, counts)):
        r = {"partition": partition, "n_clients": K, "n_clients_realized": len(clients), "seed": seed, "client": i,
             "n_windows": int(n.sum()), "occupancy": occ[i],
             "main_scenario": names[int(np.argmax(n))], "tv_distance_to_global_mix": float(tv[i])}
        for si, s in enumerate(names):
            r[f"frac_{s}"] = float(n[si] / max(n.sum(), 1))
        rows.append(r)
    summ = {"partition": partition, "n_clients": K, "n_clients_realized": len(clients), "seed": seed, "client": "ALL",
            "n_windows": int(counts.sum()), "occupancy": float(occ.mean()),
            "occupancy_std_across_clients": float(occ.std()),
            "windows_min": int(counts.sum(1).min()), "windows_max": int(counts.sum(1).max()),
            "tv_distance_to_global_mix": float(tv.mean()),
            "main_scenario": "mean over clients"}
    return rows + [summ]


def _aggregate_k(g_old: Dict, states: List[Dict], weights, counts: np.ndarray) -> Dict:
    """Shared parameters: sample-weighted average. 'loc_' parameters (first
    dimension = scenario): row s = average over the clients holding scenario
    s, weighted by their windows of s (untouched if nobody holds it)."""
    w = np.asarray(weights, np.float64); w = w / w.sum()
    out = {}
    for k, v in g_old.items():
        if "loc_" in k:
            new = v.clone()
            for si in range(v.shape[0]):
                cw = counts[:, si]
                if cw.sum() <= 0:
                    continue
                cw = cw / cw.sum()
                new[si] = sum(float(c) * s[k][si] for c, s in zip(cw, states) if c > 0)
            out[k] = new
        elif v.dtype.is_floating_point:
            out[k] = sum(float(wi) * s[k] for wi, s in zip(w, states))
        else:
            out[k] = states[0][k].clone()
    return out


def fit_federated_k(model, clients: List[Dict], counts: np.ndarray, vals: List[Dict], cfg, algo, seed) -> Dict:
    mu = cfg["fedprox_mu"] if algo == "fedprox" else 0.0
    rng = np.random.default_rng(seed)
    g = copy.deepcopy(model.state_dict())
    best, state, bad, rounds, t0 = float("inf"), None, 0, 0, time.time()
    nval = sum(v["Xb"].shape[0] for v in vals)
    pre = fm_pretrain_federated(model, clients, cfg, rng)
    for r in range(cfg["fl_rounds"]):
        states, ws = [], []
        for T in clients:
            model.load_state_dict(g)
            ref = [p.detach().clone() for p in model.parameters()] if mu > 0 else None
            train_epochs(model, T, cfg, cfg["fl_local_epochs"], rng, ref, mu)
            states.append(copy.deepcopy(model.state_dict()))
            ws.append(T["Xb"].shape[0])
        g = _aggregate_k(g, states, ws, counts)
        model.load_state_dict(g)
        va = sum(eval_loss(model, V) * V["Xb"].shape[0] for V in vals) / max(nval, 1)
        rounds += 1
        if va < best - 1e-5:
            best, bad, state = va, 0, copy.deepcopy(g)
        else:
            bad += 1
            if bad >= cfg["fl_patience"]:
                break
    if state is not None:
        model.load_state_dict(state)
    nb = model_bytes(model)                                       # shared parameters
    row_b = {k: v[0].numel() * v.element_size() for k, v in model.state_dict().items() if "loc_" in k}
    row_bytes = sum(row_b.values())
    shared_rows = (counts > 0).sum(0) >= 2                        # scenario held by >= 2 clients
    loc_per_round = 2 * row_bytes * float(((counts > 0) & shared_rows[None, :]).sum())
    comm = 2 * nb * len(clients) * (rounds + pre) + loc_per_round * rounds
    return {"rounds_run": rounds, "best_val_loss": best, "train_s": time.time() - t0,
            "comm_bytes": comm, "comm_bytes_per_client_per_round": comm / max(rounds * len(clients), 1),
            "model_bytes": nb, "n_clients": len(clients)}


def train_fl_k(mname, algo, partition, K, seed, D, names, cfg, dev, C, S, clients, counts):
    set_seed(run_seed(cfg, mname, algo, partition, K, seed, "fl"))
    m = build_model(mname, cfg, C, S)
    if cfg.get("fl_norm", "groupnorm") == "groupnorm":      # as the sensing pipeline's federated arm
        convert_bn_to_gn(m)
    m = m.to(dev)
    r = fit_federated_k(m, clients, counts, [D[s]["va"] for s in names], cfg, algo, seed)
    out = {s: {"zte": predict_logits(m, D[s]["te"]), "zva": predict_logits(m, D[s]["va"]),
               "ztr1": predict_logits(m, D[s]["tr"])[:, 0, :]} for s in names}
    info = {"model": mname, "mode": algo, "seed": seed, "partition": partition, "n_clients": K,
            "fl_norm": cfg.get("fl_norm", "groupnorm"), **r}
    return out, info


# ==========================================================================
#                          ABLATION ORCHESTRATION
# ==========================================================================
def ab_tasks(cfg) -> List[Tuple]:
    """(K, partition) pairs, the K-free references, then one sensing-source
    ablation part per other source (SLURM array index)."""
    return ([("K", k, p) for k in cfg["ab_clients"] for p in cfg["ab_partitions"]] + [("ref", None, None)]
            + [("src", None, x) for x in cfg.get("_source_ablation_list", [])])


def task_tag(t) -> str:
    return {"ref": "references", "src": f"source-{t[2]}"}.get(t[0], f"K{t[1]:02d}-{t[2]}" if t[1] else "")


def source_ablation_list(cfg) -> List[str]:
    """Other sensing sources of the hand-off ('auto') or the listed ones."""
    if cfg["input"] not in ("sensing", "sensing_grid"):
        return []
    man = json.load(open(os.path.join(cfg["_handoff_dir"], "manifest.json")))
    z = np.load(os.path.join(cfg["_handoff_dir"], "frames.npz"))
    have = [k for k in man.get("sources", {}) if k == "energy" or f"p__{k}" in z.files]
    want = have if cfg["source_ablation"] == "auto" else list(cfg["source_ablation"] or [])
    return [x for x in want if x in have and x != cfg["_sensing_source"]]


def all_models(cfg, main: bool = True) -> List[str]:
    """Novel models always; every SOTA model in the MAIN comparison
    (references and non-IID K = S), sota_models_ablation elsewhere."""
    sota = cfg["sota_models"] if main else cfg["sota_models_ablation"]
    return list(cfg["forecast_models"]) + list(sota)


def is_main(part, K, cfg) -> bool:
    """MAIN federated run = non-IID temporal clients, K = main_clients (8),
    the sensing pipeline's C."""
    return part == "noniid" and K == int(cfg["main_clients"])


def mode_key(algo, partition, K) -> str:
    """Name of one configuration in the stage-2 hand-off."""
    return algo if partition in ("pooled", "isolated") else f"{algo}-{partition}-K{K}"


class Sink:
    def __init__(self):
        self.OV, self.HZ, self.CELL, self.CELLH, self.UNIT = [], [], [], [], []
        self.COMM, self.REF, self.CLI, self.CX = [], [], [], []


def run_task(task, cfg, dev, D, names, C, S, edges, out_dir, data_fp, base):
    sink = Sink()
    cache = os.path.join(out_dir, "cache"); os.makedirs(cache, exist_ok=True)
    thr, SC = cfg["decision_threshold"], build_scopes(D, names)
    scopes = list(SC)
    src_tag = cfg.get("_sensing_source", "psd")
    s2 = os.path.join(out_dir, "stage2"); os.makedirs(s2, exist_ok=True)

    def emit(model, mode, seed, P, tags, extra, to_ref=False):
        tags = {**tags, "sensing_source": src_tag}
        for s in scopes:
            spec = SC[s]                                  # valid channels only (padding masked)
            y = scope_cat(spec, lambda k: D[k]["y_te"]); E = scope_cat(spec, lambda k: D[k]["E_te"])
            p = scope_cat(spec, lambda k: P[k]["te"])
            yv = scope_cat(spec, lambda k: D[k]["y_va"]); pv = scope_cat(spec, lambda k: P[k]["va"])
            bp = {b: scope_cat(spec, lambda k, b=b: base[k][b]["te"]) for b in cfg["baselines"]}
            tv = f1_opt_threshold(yv, pv)
            ov, hz, ce, un = forecast_rows(model, mode, seed, s, y, p, E, bp, thr, tv, edges, cfg, extra)
            ov["val_brier"] = float(np.mean((pv - yv) ** 2))
            tn = np_threshold(yv, pv, cfg["target_pfa"])          # sensing pipeline's operating point
            nm = bin_metrics(y, p, tn, full=False)
            ov.update({"threshold_np_val": tn, "target_pfa": cfg["target_pfa"],
                       "accuracy_at_np": nm["accuracy"], "balanced_accuracy_at_np": nm["balanced_accuracy"],
                       "f1_at_np": nm["f1"], "pd_at_np": nm["recall_pd"],
                       "fdr_pfa_at_np": nm["false_alarm_rate"], "mdr_at_np": nm["miss_rate"]})
            for r in [ov] + hz + ce + un:
                r.update(tags); r["family"] = family_of(model)
            if to_ref:
                sink.REF.append(ov)
                continue
            sink.OV.append(ov); sink.HZ.extend(hz); sink.CELL.extend(ce); sink.UNIT.extend(un)
            for h in cfg["snr_horizons"]:
                if h > y.shape[1]:
                    continue
                sl = slice(h - 1, h)
                _, _, ch, _ = forecast_rows(model, mode, seed, s, y[:, sl], p[:, sl], E[:, sl],
                                            {b: v[:, sl] for b, v in bp.items()}, thr, tv, edges, cfg, {})
                for r in ch:
                    r.update(tags); r["horizon_step"] = h; r["family"] = family_of(model)
                sink.CELLH.extend(ch)

    def handle(mname, mode, seed, Z, info, tags, keep):
        P = {s: {"te": _sig(Z[s]["zte"]).astype(np.float32), "va": _sig(Z[s]["zva"]).astype(np.float32),
                 "tr1": _sig(Z[s]["ztr1"]).astype(np.float32)} for s in names}
        extra = {"train_s": info.get("train_s", float("nan")), "comm_MB": info.get("comm_bytes", 0) / 1e6,
                 "rounds_run": info.get("rounds_run", float("nan"))}
        emit(mname, mode, seed, P, tags, extra)
        if keep is not None:
            keep[(mname, seed)] = P
        if cfg["dcs"] and mname in MODEL_REGISTRY:          # DCS is part of the novel design only
            Pd, etas = {}, {}
            for s in names:
                best = min(cfg["dcs_etas"], key=lambda e: float(np.mean(
                    (dcs_stream(Z[s]["zva"], D[s]["y_va"], e)[0] - D[s]["y_va"]) ** 2)))
                pva, beta = dcs_stream(Z[s]["zva"], D[s]["y_va"], best)
                pte, _ = dcs_stream(Z[s]["zte"], D[s]["y_te"], best, beta)
                Pd[s] = {"te": pte, "va": pva, "tr1": P[s]["tr1"]}
                etas[s] = best
            emit(mname + "_dcs", mode, seed, Pd, tags, {**extra, **{f"dcs_eta_{s}": e for s, e in etas.items()}})
            if keep is not None:
                keep[(mname + "_dcs", seed)] = Pd
        sink.COMM.append({"model": MODEL_LABEL[mname], "model_key": mname, "family": family_of(mname),
                          "mode": mode, **tags, "sensing_source": src_tag, "seed": seed,
                          "rounds_run": info.get("rounds_run", ""),
                          "comm_MB_total": info.get("comm_bytes", 0) / 1e6,
                          "comm_MB_per_client_per_round": info.get("comm_bytes_per_client_per_round", 0) / 1e6,
                          "model_MB_shared": info.get("model_bytes", 0) / 1e6,
                          "train_s": info.get("train_s", float("nan"))})

    def cached(fn, key, trainer):
        fp = fingerprint(cfg, data_fp, *key)
        cp = os.path.join(cache, f"{fn}__{fp}.npz")
        if cfg["resume"] and os.path.exists(cp):
            z = np.load(cp, allow_pickle=True)
            return {s: {k: z[f"{s}__{k}"] for k in ("zte", "zva", "ztr1")} for s in names}, json.loads(str(z["info"])), True
        Z, info = trainer()
        np.savez_compressed(cp, info=json.dumps(info), **{f"{s}__{k}": v for s, d in Z.items() for k, v in d.items()})
        return Z, info, False

    def handoff(mkey, keep):
        """Best NOVEL and best SOTA forecaster of this configuration (validation
        Brier, Combined) -> preds__<model>__<mkey>__s<seed>.npz +
        best_novel__<mkey>.json / best_sota__<mkey>.json."""
        scope = "Combined" if len(names) > 1 else names[0]
        rows = [r for r in sink.OV if r.get("stage2_key") == mkey and r["scenario"] == scope]
        for fam in ("novel", "SOTA"):
            fr = [r for r in rows if r["family"] == fam]
            if not fr:
                continue
            vb = {k: float(np.mean([r["val_brier"] for r in fr if r["model_key"] == k]))
                  for k in {r["model_key"] for r in fr}}
            best = sorted(vb, key=lambda k: (vb[k], k))[0]          # deterministic tie-break
            old = os.path.join(s2, f"best__{mkey}.json")             # pick of the first ablation run
            if fam == "novel" and os.path.isfile(old):
                prev = json.load(open(old)).get("model")
                if prev in vb and vb[prev] <= vb[best] + 1e-12:      # equally good -> keep it, so its
                    best = prev                                       # allocation / sharing results are re-used
            for (mk, seed), P in keep.items():
                if mk != best:
                    continue
                arr = {}
                for s in names:
                    arr[f"{s}__te"] = P[s]["te"].astype(np.float16)
                    arr[f"{s}__tr1"] = P[s]["tr1"].astype(np.float16)
                _atomic_npz(os.path.join(s2, f"preds__{best}__{mkey}__s{seed}.npz"), arr)
            _atomic_json(os.path.join(s2, f"best_{fam.lower()}__{mkey}.json"),
                         {"model": best, "label": MODEL_LABEL.get(best, best), "family": fam,
                          "seeds": sorted({sd for _, sd in keep})})
            log(f"    hand-off {mkey} ({fam}): {MODEL_LABEL.get(best, best)}")

    kind, K, part0 = task
    if kind == "src":                       # sensing-source ablation: the MAIN comparison on another source
        for b in cfg["baselines"]:
            emit(b, "non-learned", 0, {s: base[s][b] for s in names},
                 {"partition": "none", "n_clients": 0, "comparison": "source_ablation"}, {}, to_ref=True)
        tags = {"partition": "pooled", "n_clients": 1, "comparison": "source_ablation", "stage2_key": ""}
        for seed in cfg["seeds"]:
            for mname in all_models(cfg):
                t0 = time.time()
                Z, info, hit = cached(f"{mname}__centralized__s{seed}", (mname, "centralized", seed, "ab-ref"),
                                      lambda: train_one(mname, "centralized", seed, D, names, cfg, dev, C, S))
                log(f"  {'cached ' if hit else 'trained'} {MODEL_LABEL[mname]:<15s} centralized  [{part0}] "
                    f"seed {seed}" + ("" if hit else f" ({time.time() - t0:.0f} s)"))
                handle(mname, "centralized", seed, Z, info, tags, None)
        Km, algo = int(cfg["main_clients"]), cfg["ab_algos"][0]
        mkey = mode_key(algo, "noniid", Km)
        tags = {"partition": "noniid", "n_clients": Km, "comparison": "source_ablation", "stage2_key": ""}
        for seed in cfg["seeds"]:
            clients, counts = partition_clients(D, names, "noniid", Km, seed)
            for mname in all_models(cfg):
                t0 = time.time()
                Z, info, hit = cached(f"{mname}__{mkey}__s{seed}", (mname, algo, "noniid", Km, seed, "ab"),
                                      lambda: train_fl_k(mname, algo, "noniid", Km, seed, D, names, cfg, dev, C, S,
                                                         clients, counts))
                log(f"  {'cached ' if hit else 'trained'} {MODEL_LABEL[mname]:<15s} {algo:<8s} noniid  "
                    f"K={Km:<3d} [{part0}] seed {seed}" + ("" if hit else f" ({time.time() - t0:.0f} s)"))
                handle(mname, algo, seed, Z, info, tags, None)
            del clients
        return sink
    if kind == "ref":
        for b in cfg["baselines"]:
            emit(b, "non-learned", 0, {s: base[s][b] for s in names}, {"partition": "none", "n_clients": 0}, {},
                 to_ref=True)
        for mode in cfg["ab_reference_modes"]:
            tags = {"partition": "pooled" if mode == "centralized" else "isolated",
                    "comparison": "main",
                    "n_clients": 1 if mode == "centralized" else len(names),
                    "stage2_key": mode_key(mode, "pooled" if mode == "centralized" else "isolated", 0)}
            keep = {} if mode == "centralized" else None
            for seed in cfg["seeds"]:
                for mname in all_models(cfg):
                    t0 = time.time()
                    Z, info, hit = cached(f"{mname}__{mode}__s{seed}", (mname, mode, seed, "ab-ref"),
                                          lambda: train_one(mname, mode, seed, D, names, cfg, dev, C, S))
                    log(f"  {'cached ' if hit else 'trained'} {MODEL_LABEL[mname]:<15s} {mode:<12s} seed {seed}"
                        + ("" if hit else f" ({time.time() - t0:.0f} s)"))
                    handle(mname, mode, seed, Z, info, tags, keep)
            if keep is not None:
                handoff(tags["stage2_key"], keep)
        for mname in all_models(cfg):
            try:
                sink.CX.append({**profile_novel(mname, cfg, C, S), "family": family_of(mname)})
            except Exception as e:
                log(f"  profiling {mname} failed: {e}")
        return sink
    for part in [part0]:
        for algo in cfg["ab_algos"]:
            mkey = mode_key(algo, part, K)
            tags = {"partition": part, "n_clients": K, "stage2_key": mkey,
                    "comparison": "main" if is_main(part, K, cfg) else "ablation"}
            keep = {} if algo in cfg["ab_stage2_algos"] else None
            for seed in cfg["seeds"]:
                clients, counts = partition_clients(D, names, part, K, seed)
                if algo == cfg["ab_algos"][0]:
                    sink.CLI.extend(client_rows(clients, counts, names, part, K, seed))
                for mname in all_models(cfg, is_main(part, K, cfg)):
                    t0 = time.time()
                    Z, info, hit = cached(f"{mname}__{mkey}__s{seed}", (mname, algo, part, K, seed, "ab"),
                                          lambda: train_fl_k(mname, algo, part, K, seed, D, names, cfg, dev, C, S,
                                                             clients, counts))
                    log(f"  {'cached ' if hit else 'trained'} {MODEL_LABEL[mname]:<15s} {algo:<8s} {part:<7s} "
                        f"K={K:<3d} seed {seed}" + ("" if hit else f" ({time.time() - t0:.0f} s, "
                                                                     f"{info.get('rounds_run')} rounds)"))
                    handle(mname, algo, seed, Z, info, tags, keep)
                del clients
            if keep is not None:
                handoff(mkey, keep)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    return sink


# ==========================================================================
#                        PARTS -> MERGED CSV OUTPUT
# ==========================================================================
def _atomic_npz(path, arr):
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, path)


def _atomic_json(path, obj):
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def save_part(out_dir, tag, sink: Sink):
    import pickle
    d = os.path.join(out_dir, "parts_v2"); os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f"{tag}.pkl.tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        pickle.dump(vars(sink), fh)
    os.replace(tmp, os.path.join(d, f"{tag}.pkl"))


def merge(out_dir):
    import pickle
    files = sorted(glob.glob(os.path.join(out_dir, "parts_v2", "*.pkl")))
    if not files:
        log("  no parts to merge yet"); return
    A = Sink()
    for f in files:
        with open(f, "rb") as fh:
            p = pickle.load(fh)
        for k, v in p.items():
            getattr(A, k).extend(v)
    k0 = ["model", "model_key", "family", "mode", "partition", "n_clients", "scenario", "sensing_source"]
    W = lambda n, rows: write_csv(os.path.join(out_dir, n), rows)
    W("forecast_overall.csv", A.OV)
    W("forecast_overall_agg.csv", aggregate(A.OV, k0))
    W("forecast_per_horizon.csv", aggregate(A.HZ, k0 + ["horizon_step"]))
    W("forecast_per_snr_cell.csv", aggregate(A.CELL, k0 + ["snr_bin"]))
    W("forecast_per_snr_window.csv", aggregate(A.UNIT, k0 + ["snr_bin"]))
    W("forecast_per_snr_cell_by_horizon.csv", aggregate(A.CELLH, k0 + ["horizon_step", "snr_bin"]))
    W("fl_communication.csv", A.COMM)
    W("fl_communication_agg.csv", aggregate(A.COMM, ["model", "model_key", "family", "mode", "partition", "n_clients",
                                                      "sensing_source"]))
    W("client_partitions.csv", A.CLI)
    W("reference_baselines.csv", A.REF)
    if A.CX:
        W("forecast_complexity.csv", A.CX)
    log(f"  merged {len(files)} part(s): {', '.join(os.path.basename(f)[:-4] for f in files)}")


# ==========================================================================
#                                   MAIN
# ==========================================================================
def load_all(cfg, dev):
    recs = None
    if cfg["input"] == "sensing":
        log("\n[data] sensing hand-off (01_sensing.py) -> per-band occupancy threshold")
        recs = sensing_recordings(cfg)
    log("\n[data] discovery")
    paths = discover_sources(cfg)
    if not paths:
        log(f"\nNO DATASET FOUND under {cfg['data_root']} (set CONFIG['data_root'] / CONFIG['source'] "
            "or --root / --source). Nothing was trained and no data was invented.")
        sys.exit(1)
    pool = pool_sources(paths, cfg)
    if not pool:
        log("\nNO USABLE BAND in the data. Nothing was trained and no data was invented.")
        sys.exit(1)
    radios = sorted({r for r, _ in pool})
    multi = len(radios) > 1
    if multi:
        log(f"  {len(radios)} radios {radios}: scenario = radio:band; non-IID clients are "
            f"(radio, time stretch) as in the sensing pipeline")
    spans = {}
    for (r, b), parts in pool.items():
        sp = [freq_span(s, b) for s in parts]
        spans[(r, b)] = (max(a for a, _ in sp), min(z for _, z in sp))       # common coverage
    keys = sorted(pool, key=lambda k: (k[0], spans[k][0]))
    nm = {k: (f"{k[0]}:{k[1]}" if multi else k[1]) for k in keys}
    edges_b, C = band_edges({nm[k]: spans[k] for k in keys}, cfg)
    PT, radio_of, src_of, band_of = {}, {}, {}, {}
    for k in keys:
        n = nm[k]
        band_of[n] = k[1]
        P, t, ni = merge_parts([to_subchannels(s, k[1], edges_b[n]) for s in pool[k]])
        if P.shape[0] < 8:
            log(f"  [{n}] only {P.shape[0]} steps -> skipped")
            continue
        PT[n], radio_of[n] = (P, t, ni), k[0]
        src_of[n] = " + ".join(s["src"] for s in pool[k])
    names = [nm[k] for k in keys if nm[k] in PT]
    if not names:
        log("\nNO SCENARIO WITH ENOUGH STEPS. Nothing was trained and no data was invented.")
        sys.exit(1)
    for n in names:
        tt = PT[n][1]
        log(f"    {n:<16s} {len(tt):6,d} sweeps, median {float(np.median(np.diff(tt))) if len(tt) > 1 else 0:.1f} s/sweep")
    W, H, B, tedges, dropped = plan_split({n: PT[n][1] for n in names}, cfg)
    names = [n for n in names if n not in dropped]
    for n in dropped:
        PT.pop(n, None)
    cfg["lookback"], cfg["horizon"], cfg["split_blocks_used"] = W, H, B
    cfg["snr_horizons"] = sorted({min(h, H) for h in cfg["snr_horizons"]})
    log(f"  {len(names)} scenarios: " + ", ".join(
        f"{n} {edges_b[n][0] / 1e6:.0f}-{edges_b[n][-1] / 1e6:.0f} MHz "
        f"({(edges_b[n][1] - edges_b[n][0]) / 1e6:.2f} MHz x {C})" for n in names))
    log(f"  split: chronological {1 - cfg['val_frac'] - cfg['test_frac']:.2f}/{cfg['val_frac']:.2f}/"
        f"{cfg['test_frac']:.2f} within each of {B} time block(s) and each band, W+H gaps; W={W}, H={H}")
    D, cards, fp_parts = {}, [], [cfg.get("_sensing_source", "psd")]
    for si, n in enumerate(names):
        P, t, ni = PT.pop(n)
        calib = None
        if recs is not None:
            calib = recs.get((radio_of[n], band_of[n]))
            if calib is None:                                  # radio named differently: match the band
                hit = [v for (r, b), v in recs.items() if b == band_of[n]]
                calib = hit[0] if len(hit) == 1 else None
        sc, w = build_scenario(n, P, t, edges_b[n], cfg, ni, src_of[n], radio_of[n], tedges, calib)
        eta = sc["eta"]
        cards.append(sc["card"])
        idx_in = w["start"][:, None] + np.arange(W)[None, :]
        Xe = np.clip((sc["E"][idx_in] - eta) / 10.0, -2.0, 4.0).astype(np.float32)

        def pack(ix):
            return {"Xb": _tt(w["Xbin"][ix], dev), "Xe": _tt(Xe[ix], dev),
                    "sid": torch.full((len(ix),), si, dtype=torch.long, device=dev),
                    "Y": _tt(w["Y"][ix], dev)}
        D[n] = {"tr": pack(w["tr"]), "va": pack(w["va"]), "te": pack(w["te"]),
                "Xbin_te": w["Xbin"][w["te"]], "Xbin_va": w["Xbin"][w["va"]],
                "y_te": w["Y"][w["te"]], "E_te": w["E"][w["te"]], "y_va": w["Y"][w["va"]],
                "y_tr1": w["Y"][w["tr"]][:, 0, :], "E_tr1": w["E"][w["tr"]][:, 0, :],
                "ypos_tr": w["Y"][w["tr"]].sum((1, 2)), "ncell_tr": float(H * C),
                "t_tr": w["t0"][w["tr"]], "t_te": w["t0"][w["te"]],
                "period": w["period"], "inr_busy": w["inr_busy"], "inr_idle": w["inr_idle"],
                "edges_hz": edges_b[n], "radio": radio_of[n]}
        fp_parts.append(f"{n}:{sc['T']}:{float(sc['Y'].sum()):.0f}:{float(sc['E'].sum()):.3f}:{B}:{eta:g}")
    audit_classes(D, names, cfg)
    data_fp = hashlib.sha1("|".join(fp_parts).encode()).hexdigest()[:12]
    return D, names, C, cards, data_fp


def audit_classes(D, names, cfg) -> None:
    """Class-balance audit of the sensing pipeline: per band, and both
    classes in every pooled split (else the metrics are undefined)."""
    log("\n  per-scenario occupancy (a band that is ~one class is the 'frequency confound' of the sensing")
    log("  pipeline: per-band rows of such a band measure little):")
    for n in names:
        nv = int(D[n].get("nvalid", D[n]["y_te"].shape[-1]))
        r = float(np.concatenate([D[n]["y_tr1"][..., :nv].ravel(), D[n]["y_te"][..., :nv].ravel()]).mean())
        flag = "  <-- almost single-class" if r < 0.005 or r > 0.995 else ""
        log(f"    {n:<22s} occupied {r * 100:5.1f}%{flag}")
    for k, nm in (("y_tr1", "train"), ("y_va", "val"), ("y_te", "test")):
        y = np.concatenate([D[n][k].ravel() for n in names])
        if y.size == 0 or y.min() == y.max():
            sys.exit(f"{nm} split holds a single class (or nothing): metrics would be undefined. "
                     f"Check eta_db / noise_quantile or the band selection.")


# ==========================================================================
#   SENSING-DRIVEN INPUT (default): the hand-off of 01_sensing.py
# --------------------------------------------------------------------------
#   frames.npz holds, for EVERY I/Q frame of the evening, its time, radio,
#   band, tile centre frequency, SNR proxy and the detectors' P(H1). A
#   sweeping receiver revisits every 20 MHz tile once per sweep, so:
#     scenario   one RECEIVER (its whole sweep); channels = its tiles in
#                frequency order (each tile carries its band name -> per-band
#                rows); several radios: tiles padded to the largest count,
#                padded channels are masked out of every metric
#     visit      the frames of one dwell on one tile (consecutive frames
#                closer than the dwell gap, found from the frame spacing)
#     time step  one sweep = the median revisit period of the tiles; a step
#                with < min_tile_coverage of the tiles observed is a gap
#                (windows across gaps are dropped, as for PSD sweeps); a
#                missing tile inside a kept step holds its last value
#     label      occupied when >= visit_busy_frac of the visit's frames are
#                sensed H1 by the source detector at ITS validation
#                (Neyman-Pearson) threshold
#     excess E   mean SNR proxy of the visit (dB) -> per-SNR rows, INR
# ==========================================================================
def discover_handoff(cfg: Dict) -> Tuple[Optional[str], List[str]]:
    """Folder holding frames.npz + manifest.json of the sensing stage."""
    cands = []
    for p in [cfg.get("sensing_dir", "")] + list(cfg.get("sensing_dir_fallbacks", [])):
        if p:
            p = os.path.expanduser(p)
            cands += [p, os.path.join(p, "sensing_handoff")]
    for c in cands:
        if os.path.isfile(os.path.join(c, "frames.npz")) and os.path.isfile(os.path.join(c, "manifest.json")):
            return c, cands
    return None, cands


def _otsu_log_gap(d: np.ndarray) -> float:
    """Dwell gap: frames of one visit are milliseconds apart, visits a sweep
    apart. Otsu split of log10(frame spacing); 0 when the spacing is unimodal
    (one frame per visit)."""
    d = d[d > 0]
    if d.size < 4:
        return 0.0
    ld = np.log10(d)
    if ld.max() - ld.min() < 1.0:                       # < 10x spread: one frame per visit
        return 0.0
    h, e = np.histogram(ld, 64)
    c = 0.5 * (e[1:] + e[:-1])
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    k = int(np.argmax(w0 * w1 * (m0 - m1) ** 2))
    return float(10 ** c[k])


def sensed_grid(z, sel: np.ndarray, dec: np.ndarray, prob: np.ndarray, cfg: Dict) -> Dict:
    """Frames of ONE receiver -> [T, C_tiles] occupancy / SNR grid."""
    t_all, f_all = z["utc"][sel], np.round(z["center_hz"][sel] / 1e3) * 1e3
    tiles = np.unique(f_all)
    gaps = np.concatenate([np.diff(np.sort(t_all[f_all == f])) for f in tiles])
    dwell_gap = _otsu_log_gap(gaps)
    V = []                                                # (tile, t, occ, prob, snr, test_frac)
    tband = []
    for ci, f in enumerate(tiles):
        ix = sel[f_all == f]
        ix = ix[np.argsort(z["utc"][ix], kind="stable")]
        t = z["utc"][ix]
        vid = np.concatenate([[0], np.cumsum(np.diff(t) > dwell_gap)]) if len(t) > 1 else np.zeros(1, int)
        nv = int(vid[-1]) + 1
        cnt = np.bincount(vid, minlength=nv).astype(np.float64)
        m = lambda a: np.bincount(vid, weights=np.asarray(a, np.float64), minlength=nv) / cnt  # noqa: E731
        V.append((np.full(nv, ci), m(t), m(dec[ix]), m(prob[ix]), m(z["snr"][ix]),
                  m(z["split"][ix] == 2)))
        b, n = np.unique(z["band"][ix], return_counts=True)
        tband.append(str(b[np.argmax(n)]))
    per_tile_period = [float(np.median(np.diff(v[1]))) for v in V if len(v[1]) > 2]
    step = float(np.median(per_tile_period)) if per_tile_period else 1.0
    ci, tv, occ, pr, snr, tst = (np.concatenate(a) for a in zip(*V))
    t0 = float(tv.min())
    b = np.round((tv - t0) / step).astype(int)
    T, C = int(b.max()) + 1, len(tiles)
    acc = {k: np.zeros((T, C)) for k in ("occ", "prob", "snr", "tst", "n")}
    np.add.at(acc["n"], (b, ci), 1.0)
    for k, v in (("occ", occ), ("prob", pr), ("snr", snr), ("tst", tst)):
        np.add.at(acc[k], (b, ci), v)
    seen = acc["n"] > 0
    for k in ("occ", "prob", "snr", "tst"):
        acc[k] = np.where(seen, acc[k] / np.maximum(acc["n"], 1), np.nan)
    keep = seen.mean(1) >= cfg["min_tile_coverage"]
    n_interp = 0
    for k in ("occ", "prob", "snr", "tst"):
        a = acc[k][keep]
        for c in range(C):                                 # hold the last visit (first: next visit)
            col = a[:, c]
            bad = np.isnan(col)
            if bad.all():
                col[:] = 0.0
                continue
            idx = np.where(~bad, np.arange(len(col)), 0)
            np.maximum.accumulate(idx, out=idx)
            first = int(np.flatnonzero(~bad)[0])
            col[:first] = col[first]
            col[first:] = col[idx[first:]]
            if k == "occ":
                n_interp += int(bad.sum())
        acc[k] = a
    t = t0 + np.flatnonzero(keep) * step
    Y = (acc["occ"] >= cfg["visit_busy_frac"]).astype(np.float32)
    return {"Y": Y, "E": acc["snr"].astype(np.float32), "prob": acc["prob"].astype(np.float32),
            "tst": acc["tst"].astype(np.float32), "t": t, "tiles_hz": tiles, "tile_band": tband,
            "n_interp": n_interp, "step": step, "dwell_gap": dwell_gap,
            "steps_dropped": int((~keep).sum()), "frames": int(sel.size)}


def build_scenario_sensed(name: str, G: Dict, C: int, cfg: Dict, radio: str,
                          tedges: np.ndarray, src_desc: str) -> Tuple[Dict, Dict]:
    """Windows + split of one receiver's sensed grid (same windowing, split,
    gap rule and statistics as build_scenario). Channels beyond the
    receiver's own tiles (several radios) are padding: Y = 0, E = idle."""
    Y0, E0, t = G["Y"], G["E"], G["t"]
    T, nv = Y0.shape
    Y = np.zeros((T, C), np.float32); Y[:, :nv] = Y0
    E = np.full((T, C), -10.0, np.float32); E[:, :nv] = E0
    TST = np.zeros((T, C), np.float32); TST[:, :nv] = G["tst"]
    W, H = cfg["lookback"], cfg["horizon"]
    starts = window_starts(t, W, H, cfg)
    tr, va, te, blk = split_windows(starts, t, tedges, W, H, cfg)
    trmask = np.zeros(T, bool)
    for s0 in starts[tr]:
        trmask[s0:s0 + W + H] = True
    s = Y[:, :nv].mean(1) - Y[:, :nv].mean()
    ac1 = float(np.corrcoef(s[:-1], s[1:])[0, 1]) if s.std() > 1e-9 else float("nan")
    idx_in = starts[:, None] + np.arange(W)[None, :]
    idx_out = starts[:, None] + W + np.arange(H)[None, :]
    Xsrc = (np.clip((E - cfg["eta_db"]) / 10.0, -2.0, 4.0).astype(np.float32)
            if cfg["input_repr"] == "excess" else Y)
    w = {"X": Xsrc[idx_in], "Xbin": Y[idx_in], "Y": Y[idx_out], "E": E[idx_out],
         "t0": t[starts + W], "start": starts, "tr": tr, "va": va, "te": te}
    w["period"] = estimate_period(Y[trmask][:, :nv], W)
    inr = np.maximum(np.power(10.0, E[trmask] / 10.0) - 1.0, 0.0)
    yb = Y[trmask] > 0.5
    w["inr_busy"] = np.array([inr[yb[:, c], c].mean() if yb[:, c].any() else 10.0 for c in range(C)], np.float32)
    w["inr_idle"] = np.array([inr[~yb[:, c], c].mean() if (~yb[:, c]).any() else 0.0 for c in range(C)], np.float32)
    tst_te = TST[idx_out][te][..., :nv]
    w["sens_test_frac"] = float(tst_te.mean()) if tst_te.size else float("nan")
    ysplit = {k: w["Y"][w[k]][..., :nv] for k in ("tr", "va", "te")}
    occ_runs = [r for c in range(nv) for r in _runs(Y[:, c].astype(int), 1)]
    idle_runs = [r for c in range(nv) for r in _runs(Y[:, c].astype(int), 0)]
    sc = {"name": name, "E": E, "Y": Y, "T": T, "C": C}
    bands = sorted(set(G["tile_band"]))
    sc["card"] = {
        "scenario": name, "radio": radio, "input": "sensing", "sensing_source": src_desc,
        "frames": G["frames"], "tiles": nv, "channels_padded": C - nv,
        "bands": " ".join(f"{b}:{G['tile_band'].count(b)}" for b in bands),
        "f_lo_MHz": float(G["tiles_hz"][0] / 1e6), "f_hi_MHz": float(G["tiles_hz"][-1] / 1e6),
        "steps": T, "sweep_period_s": G["step"], "dwell_gap_s": G["dwell_gap"],
        "duration_min": float((t[-1] - t[0]) / 60.0), "steps_dropped_low_coverage": G["steps_dropped"],
        "cells_held_from_last_visit": G["n_interp"],
        "windows_dropped_gap": int(T - W - H + 1 - len(starts)),
        "split_blocks": len(tedges) - 1, "windows_train": len(tr), "windows_val": len(va), "windows_test": len(te),
        "visit_busy_frac": cfg["visit_busy_frac"], "min_tile_coverage": cfg["min_tile_coverage"],
        "positive_rate": float(Y[:, :nv].mean()),
        "positive_rate_train": float(ysplit["tr"].mean()) if len(tr) else float("nan"),
        "positive_rate_val": float(ysplit["va"].mean()) if len(va) else float("nan"),
        "positive_rate_test": float(ysplit["te"].mean()) if len(te) else float("nan"),
        "test_targets_from_sensing_test_frames": w["sens_test_frac"],
        "on_run_mean_steps": float(np.mean(occ_runs)) if occ_runs else 0.0,
        "off_run_mean_steps": float(np.mean(idle_runs)) if idle_runs else 0.0,
        "lag1_acf": ac1, "snr_proxy_max_db": float(E0.max()),
    }
    log(f"  [{name}] {nv} tiles ({sc['card']['bands']}), {T} sweeps of {G['step']:.2f} s "
        f"({sc['card']['duration_min']:.1f} min), sensed occupancy {Y[:, :nv].mean() * 100:.1f}% "
        f"(train/val/test {sc['card']['positive_rate_train'] * 100:.1f}/"
        f"{sc['card']['positive_rate_val'] * 100:.1f}/{sc['card']['positive_rate_test'] * 100:.1f}%), "
        f"windows {len(tr)}/{len(va)}/{len(te)}, period {w['period']}")
    return sc, w


def load_all_sensed(cfg, dev):
    """Sensing hand-off -> the same D / names / C structure as load_all()."""
    log("\n[data] sensing hand-off (01_sensing.py)")
    hd, tried = discover_handoff(cfg)
    if hd is None:
        log("\nNO SENSING HAND-OFF FOUND (frames.npz + manifest.json). Looked in:\n  " + "\n  ".join(tried)
            + "\nRun 01_sensing.py first (stage 'handoff'), set CONFIG['sensing_dir'] / "
              "--sensing-dir, or use --input psd for the PSD-sweep forecasting.\n"
              "Nothing was trained and no data was invented.")
        sys.exit(1)
    man = json.load(open(os.path.join(hd, "manifest.json")))
    z = dict(np.load(os.path.join(hd, "frames.npz")))
    src = cfg["sensing_source"]
    if src == "main":
        src = man.get("main_source", "novel")
    if src != "energy" and f"p__{src}" not in z:
        sys.exit(f"sensing source '{src}' not in the hand-off (have: energy"
                 f"{''.join(', ' + k[3:] for k in z if k.startswith('p__'))})")
    if src == "energy":
        prob = z["y_energy"].astype(np.float32); dec = prob.copy()
        desc = "energy (margin labels)"
    else:
        info = man["sources"][src]
        prob = z[f"p__{src}"].astype(np.float32)
        dec = (prob >= float(info["threshold"])).astype(np.float32)
        desc = f"{src}: {info['model']} ({info['mode']}, thr {float(info['threshold']):.4f})"
    cfg["_sensing_source"], cfg["_sensing_desc"], cfg["_handoff_dir"] = src, desc, hd
    log(f"  {hd}\n  source {desc}; {len(prob):,} frames")
    keep = np.ones(len(prob), bool)
    if cfg.get("devices"):
        keep &= np.isin(z["device"], cfg["devices"])
    if cfg.get("bands"):
        keep &= np.isin(z["band"], cfg["bands"])
    if cfg.get("exclude_bands"):
        keep &= ~np.isin(z["band"], cfg["exclude_bands"])
    radios = sorted(set(z["device"][keep].tolist()))
    grids = {}
    for r in radios:
        sel = np.flatnonzero(keep & (z["device"] == r))
        G = sensed_grid(z, sel, dec, prob, cfg)
        if G["Y"].shape[0] < 8:
            log(f"  [{r}] only {G['Y'].shape[0]} sweeps -> skipped")
            continue
        grids[r] = G
    if not grids:
        log("\nNO RECEIVER WITH ENOUGH SWEEPS. Nothing was trained and no data was invented.")
        sys.exit(1)
    names = sorted(grids)
    C = max(G["Y"].shape[1] for G in grids.values())
    W, H, B, tedges, dropped = plan_split({n: grids[n]["t"] for n in names}, cfg)
    names = [n for n in names if n not in dropped]
    C = max(grids[n]["Y"].shape[1] for n in names)
    cfg["lookback"], cfg["horizon"], cfg["split_blocks_used"] = W, H, B
    cfg["snr_horizons"] = sorted({min(h, H) for h in cfg["snr_horizons"]})
    if len(names) > 1 and len({G["Y"].shape[1] for G in grids.values()}) > 1:
        log(f"  receivers have different tile counts -> C = {C}, the smaller ones padded "
            f"(padding masked out of every metric; unusable in allocation / sharing)")
    log(f"  split: chronological {1 - cfg['val_frac'] - cfg['test_frac']:.2f}/{cfg['val_frac']:.2f}/"
        f"{cfg['test_frac']:.2f} within each of {B} time block(s), W+H gaps; W={W}, H={H} sweeps")
    D, cards, fp_parts = {}, [], [src]
    eta = cfg["eta_db"]
    for si, n in enumerate(names):
        G = grids[n]
        nv = G["Y"].shape[1]
        sc, w = build_scenario_sensed(n, G, C, cfg, n, tedges, desc)
        cards.append(sc["card"])
        idx_in = w["start"][:, None] + np.arange(W)[None, :]
        Xe = np.clip((sc["E"][idx_in] - eta) / 10.0, -2.0, 4.0).astype(np.float32)

        def pack(ix):
            return {"Xb": _tt(w["Xbin"][ix], dev), "Xe": _tt(Xe[ix], dev),
                    "sid": torch.full((len(ix),), si, dtype=torch.long, device=dev),
                    "Y": _tt(w["Y"][ix], dev)}
        half = 10e6 if nv < 2 else float(np.median(np.diff(G["tiles_hz"]))) / 2
        cen = np.concatenate([G["tiles_hz"], G["tiles_hz"][-1] + 2 * half * np.arange(1, C - nv + 1)])
        D[n] = {"tr": pack(w["tr"]), "va": pack(w["va"]), "te": pack(w["te"]),
                "Xbin_te": w["Xbin"][w["te"]], "Xbin_va": w["Xbin"][w["va"]],
                "y_te": w["Y"][w["te"]], "E_te": w["E"][w["te"]], "y_va": w["Y"][w["va"]],
                "y_tr1": w["Y"][w["tr"]][:, 0, :], "E_tr1": w["E"][w["tr"]][:, 0, :],
                "ypos_tr": w["Y"][w["tr"]][..., :nv].sum((1, 2)), "ncell_tr": float(H * nv),
                "t_tr": w["t0"][w["tr"]], "t_te": w["t0"][w["te"]],
                "period": w["period"], "inr_busy": w["inr_busy"], "inr_idle": w["inr_idle"],
                "edges_hz": np.concatenate([cen - half, [cen[-1] + half]]), "radio": n,
                "nvalid": nv, "ch_band": np.array(G["tile_band"] + [""] * (C - nv)),
                "tiles_hz": G["tiles_hz"]}
        fp_parts.append(f"{n}:{sc['T']}:{float(sc['Y'].sum()):.0f}:{float(sc['E'].sum()):.3f}:{B}:{W}:{H}")
    audit_classes(D, names, cfg)
    log(f"  test targets taken from sensing-TEST frames: "
        + ", ".join(f"{c['scenario']} {c['test_targets_from_sensing_test_frames'] * 100:.0f}%" for c in cards)
        + " (the rest were scored by the detector on frames it trained / validated on)")
    data_fp = hashlib.sha1("|".join(fp_parts).encode()).hexdigest()[:12]
    return D, names, C, cards, data_fp


def build_scopes(D, names) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    """Reporting scopes -> [(scenario, valid channel indices)]: every
    scenario, 'Combined' (several scenarios) and, when the channels carry
    band names (sensing input), every BAND of every receiver."""
    sc = {s: [(s, np.arange(D[s].get("nvalid", D[s]["y_te"].shape[-1])))] for s in names}
    if len(names) > 1:
        sc["Combined"] = [v[0] for v in sc.values()]
    for s in names:
        cb = D[s].get("ch_band")
        if cb is None:
            continue
        for b in sorted({x for x in cb.tolist() if x}):
            sc[f"{s}|{b}" if len(names) > 1 else f"band:{b}"] = [(s, np.flatnonzero(cb == b))]
    return sc


def scope_cat(spec, get) -> np.ndarray:
    """Concatenate the valid channels of a scope as (window x channel) units:
    [N, H, C] -> [sum N*|ix|, H, 1]. Cell-level metrics are unchanged, padded
    channels drop out."""
    out = []
    for s, ix in spec:
        a = get(s)[..., ix]
        out.append(np.ascontiguousarray(a.transpose(0, 2, 1)).reshape(-1, a.shape[1])[..., None])
    return np.concatenate(out)


# ==========================================================================
#   SENSING-CALIBRATED PSD LABELS (default input "sensing")
# --------------------------------------------------------------------------
#   The raw I/Q of the Fireworks capture exists only in short recordings
#   (~1.4 s per tile, a few minutes apart), so the sensed frames alone give
#   only ~10 time steps per tile: far too few to forecast. The PSD sweeps
#   cover every tile in every sweep. The chain is therefore:
#     1. every I/Q recording (frames of one tile closer than
#        recording_gap_s) gets the SENSED occupancy of the stage-1 detector:
#        busy when >= visit_busy_frac of its frames are H1 at the detector's
#        own validation (Neyman-Pearson) threshold;
#     2. the recording is matched to the PSD sweep of its band nearest in
#        time (<= calib_max_lag_s) and to the sub-channels inside its tile;
#        PSD evidence = the largest excess E (dB over the training noise
#        floor) of those sub-channels;
#     3. per band, the occupancy threshold eta is the value on calib_eta_grid
#        that best AGREES with the detector (balanced accuracy of
#        E_max > eta vs the sensed label), fitted on recordings inside the
#        forecasting TRAINING span only; ties -> closest to eta_db;
#     4. every sweep / sub-channel of the band is labelled E > eta_band.
#   The forecasting target is thus the PSD occupancy calibrated to what the
#   sensing detector decides, at full sweep resolution.
# ==========================================================================
def resolve_sensing_source(cfg: Dict) -> Tuple[str, Dict, Dict, str]:
    """-> (handoff dir, manifest, frames, source key) for cfg['sensing_source']."""
    hd, tried = discover_handoff(cfg)
    if hd is None:
        log("\nNO SENSING HAND-OFF FOUND (frames.npz + manifest.json). Looked in:\n  " + "\n  ".join(tried)
            + "\nRun 01_sensing.py first (stage 'handoff'), set CONFIG['sensing_dir'] / "
              "--sensing-dir, or use --input psd (fixed eta_db labels).\n"
              "Nothing was trained and no data was invented.")
        sys.exit(1)
    man = json.load(open(os.path.join(hd, "manifest.json")))
    z = dict(np.load(os.path.join(hd, "frames.npz")))
    src = cfg["sensing_source"]
    if src == "main":
        src = man.get("main_source", "novel")
    if src != "energy" and f"p__{src}" not in z:
        sys.exit(f"sensing source '{src}' not in the hand-off (have: energy"
                 f"{''.join(', ' + k[3:] for k in z if k.startswith('p__'))})")
    return hd, man, z, src


def sensing_recordings(cfg: Dict) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
    """Hand-off frames -> one row per I/Q recording, grouped by (radio, band):
    t (mid time), f_lo / f_hi (tile), busy (sensed share of H1 frames),
    split (share of sensing-test frames)."""
    hd, man, z, src = resolve_sensing_source(cfg)
    if src == "energy":
        dec = z["y_energy"].astype(np.float32)
        desc = "energy (margin labels)"
    else:
        info = man["sources"][src]
        dec = (z[f"p__{src}"] >= float(info["threshold"])).astype(np.float32)
        desc = f"{src}: {info['model']} ({info['mode']}, thr {float(info['threshold']):.4f})"
    cfg["_sensing_source"], cfg["_sensing_desc"], cfg["_handoff_dir"] = src, desc, hd
    gap = float(cfg.get("recording_gap_s", 1.0))
    out: Dict[Tuple[str, str], Dict[str, list]] = {}
    key = np.array([f"{a}\x00{b}\x00{c:.0f}" for a, b, c in zip(z["device"], z["band"], z["center_hz"])])
    for k in np.unique(key):
        ix = np.flatnonzero(key == k)
        ix = ix[np.argsort(z["utc"][ix], kind="stable")]
        t = z["utc"][ix]
        rid = np.concatenate([[0], np.cumsum(np.diff(t) > gap)])
        radio, band, _ = k.split("\x00")
        o = out.setdefault((radio, band), {"t": [], "fc": [], "busy": [], "tst": [], "n": []})
        for r in range(int(rid[-1]) + 1):
            m = rid == r
            o["t"].append(float(t[m].mean())); o["fc"].append(float(z["center_hz"][ix[0]]))
            o["busy"].append(float(dec[ix[m]].mean())); o["tst"].append(float((z["split"][ix[m]] == 2).mean()))
            o["n"].append(int(m.sum()))
    res = {}
    for kk, o in out.items():
        a = {k: np.asarray(v) for k, v in o.items()}
        fcs = np.unique(a["fc"])
        half = (float(np.median(np.diff(fcs))) / 2 if fcs.size > 1 else 10e6) \
            if not cfg.get("tile_half_hz") else float(cfg["tile_half_hz"])
        a["f_lo"], a["f_hi"] = a["fc"] - half, a["fc"] + half
        res[kk] = a
    n_rec = sum(len(a["t"]) for a in res.values())
    log(f"  sensing source {desc}: {n_rec:,} I/Q recordings "
        f"(median {int(np.median(np.concatenate([a['n'] for a in res.values()])))} frames each) "
        f"in {len(res)} (radio, band) groups -> calibrate the PSD occupancy threshold per band")
    return res


def calibrate_eta(E: np.ndarray, t: np.ndarray, edges: np.ndarray, trmask: np.ndarray,
                  rec: Optional[Dict[str, np.ndarray]], cfg: Dict) -> Tuple[float, Dict]:
    """Per-band eta that best agrees with the sensed recordings (see above)."""
    eta0 = float(cfg["eta_db"])
    info = {"eta_db": eta0, "calib_points": 0, "calib_points_train": 0, "calib_ba": float("nan"),
            "calib_ba_at_default_eta": float("nan"), "calib_sensed_busy_rate": float("nan"),
            "calib_test_frame_share": float("nan"), "calib_status": "no recordings in this band"}
    if rec is None or len(rec["t"]) == 0:
        return eta0, info
    step = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    lag = float(cfg.get("calib_max_lag_s") or max(2.0 * step, 5.0))
    k = np.clip(np.searchsorted(t, rec["t"]), 1, len(t) - 1)
    k = np.where(np.abs(t[k - 1] - rec["t"]) <= np.abs(t[k] - rec["t"]), k - 1, k)
    ok = np.abs(t[k] - rec["t"]) <= lag
    cen = 0.5 * (edges[:-1] + edges[1:])
    ev = np.full(len(rec["t"]), np.nan)
    for i in np.flatnonzero(ok):
        ch = np.flatnonzero((cen >= rec["f_lo"][i]) & (cen <= rec["f_hi"][i]))
        if ch.size:
            ev[i] = float(E[k[i], ch].max())
    m = np.isfinite(ev)
    if m.sum() == 0:
        info["calib_status"] = "no recording overlaps a PSD sweep / sub-channel"
        return eta0, info
    lab = rec["busy"] >= cfg["visit_busy_frac"]
    use = m & trmask[k]                                   # forecasting TRAINING span only
    status = "fitted on training-span recordings"
    if use.sum() < cfg["calib_min_points"] or lab[use].min() == lab[use].max():
        use = m
        status = "fitted on all recordings (too few / one class in the training span)"
    if use.sum() < cfg["calib_min_points"] or lab[use].min() == lab[use].max():
        info.update({"calib_points": int(m.sum()), "calib_status": "too few points or one sensed class -> eta_db"})
        return eta0, info

    def ba(th, ix):
        pr, yy = ev[ix] > th, lab[ix]
        return 0.5 * ((pr & yy).sum() / max(yy.sum(), 1) + (~pr & ~yy).sum() / max((~yy).sum(), 1))
    lo, hi, st = cfg["calib_eta_grid"]
    grid = np.arange(lo, hi + 1e-9, st)
    scores = np.array([ba(g, use) for g in grid])
    best = grid[np.flatnonzero(scores >= scores.max() - 1e-12)]
    eta = float(best[np.argmin(np.abs(best - eta0))])
    info.update({"eta_db": eta, "calib_points": int(m.sum()), "calib_points_train": int((m & trmask[k]).sum()),
                 "calib_ba": float(ba(eta, m)), "calib_ba_at_default_eta": float(ba(eta0, m)),
                 "calib_sensed_busy_rate": float(lab[m].mean()),
                 "calib_test_frame_share": float(rec["tst"][m].mean()), "calib_status": status})
    return eta, info


def write_stage2_static(cfg, out_dir, D, names, C, edges_snr, base):
    d = os.path.join(out_dir, "stage2"); os.makedirs(d, exist_ok=True)
    lab = {"names": json.dumps(names), "C": C, "edges_snr": edges_snr, "eta_db": cfg["eta_db"],
           "horizon": cfg["horizon"]}
    for s in names:
        for k in ("y_te", "E_te", "y_tr1", "E_tr1", "inr_busy", "inr_idle"):
            lab[f"{s}__{k}"] = D[s][k]
        lab[f"{s}__edges_hz"] = D[s]["edges_hz"]
        lab[f"{s}__t_tr1"] = D[s]["t_tr"]                     # time of every training slot (temporal clients)
        lab[f"{s}__radio"] = np.array(D[s]["radio"])
        lab[f"{s}__nvalid"] = np.array(int(D[s].get("nvalid", C)))     # channels beyond = padding
        if "ch_band" in D[s]:
            lab[f"{s}__ch_band"] = D[s]["ch_band"]
    _atomic_npz(os.path.join(d, "labels.npz"), lab)
    for b in ("persistence", "historical_mean"):
        if b in cfg["baselines"]:
            _atomic_npz(os.path.join(d, f"preds__{b}__non-learned__s0.npz"),
                        {f"{s}__te": base[s][b]["te"].astype(np.float16) for s in names})
    _atomic_json(os.path.join(d, "manifest.json"),
                 {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "scenarios": names, "C": C,
                  "kind": "ablation_clients", "best_by_mode": {}, "seeds": cfg["seeds"],
                  "ab_clients": cfg["ab_clients"], "ab_partitions": cfg["ab_partitions"],
                  "main_clients": cfg["main_clients"], "min_per_class_per_client": cfg["min_per_class_per_client"],
                  "split_blocks_used": cfg.get("split_blocks_used"), "target_pfa": cfg["target_pfa"],
                  "ab_stage2_algos": cfg["ab_stage2_algos"], "baselines": cfg["baselines"],
                  "labels": {k: cfg[k] for k in ("subchannel_mode", "subchannels_per_band", "subchannel_hz",
                                                 "noise_quantile", "eta_db", "lookback", "horizon")},
                  "dataset": ("fireworks multi-band, sensed tile occupancy (scenario = receiver sweep, "
                              "channel = 20 MHz tile)" if cfg["input"] == "sensing_grid" else
                              "fireworks multi-band PSD, occupancy threshold calibrated per band to the "
                              "stage-1 detector (scenario = band)" if cfg["input"] == "sensing" else
                              "fireworks multi-band PSD (scenario = band, or radio:band with several radios)"),
                  "input": cfg["input"], "sensing_source": cfg.get("_sensing_source", ""),
                  "sensing_desc": cfg.get("_sensing_desc", ""), "sensing_handoff": cfg.get("_handoff_dir", ""),
                  "nvalid": {s: int(D[s].get("nvalid", C)) for s in names},
                  "band_edges_MHz": {s: [float(D[s]["edges_hz"][0] / 1e6), float(D[s]["edges_hz"][-1] / 1e6)]
                                     for s in names},
                  "model_labels": MODEL_LABEL, "cache_dir": ""})


def resolve_clients(cfg, S: int) -> None:
    """'auto' -> the sensing pipeline's client counts [4, 8, 12, 16, 24, 32];
    main_clients is always part of the grid."""
    if cfg["ab_clients"] == "auto" or not cfg["ab_clients"]:
        cfg["ab_clients"] = [4, 8, 12, 16, 24, 32]
    cfg["ab_clients"] = sorted({int(k) for k in cfg["ab_clients"]} | {int(cfg["main_clients"])})


def main() -> None:
    global _LOG_FH
    ap = argparse.ArgumentParser(description="Stage 2: sub-channel occupancy forecasting (GEM / GEM-DCS and "
                                             "five SOTA forecasters), centralized and federated")
    ap.add_argument("--family", choices=["all", "sota", "gem"], default=CONFIG["family"],
                    help="gem = GEM + GEM-DCS, sota = the five SOTA forecasters, all = both (default)")
    ap.add_argument("--input", choices=["sensing", "psd", "sensing_grid"],
                    help="sensing (PSD labelled by the 01 detector, default) | psd (eta_db) | sensing_grid")
    ap.add_argument("--results", help="pipeline results root (default $SFS_RESULTS or ./results): "
                                      "reads <results>/01_sensing, writes <results>/02_forecasting and "
                                      "<results>/02b_forecasting_gem")
    ap.add_argument("--sensing-dir", help="01_sensing.py results folder or its sensing_handoff/")
    ap.add_argument("--sensing-source", help="main (OSCAR-Net) | sota | energy")
    ap.add_argument("--source-ablation", help="auto | comma list | '' (none)")
    ap.add_argument("--root", help="dataset root for --input psd (session folder / export .h5)")
    ap.add_argument("--source", help="explicit session folder or .h5")
    ap.add_argument("--out"); ap.add_argument("--seeds")
    ap.add_argument("--models", help="proposed models of the gem family (default gem)")
    ap.add_argument("--sota-models", help="SOTA models of the main comparison ('' = none)")
    ap.add_argument("--sota-models-ablation", help="SOTA models of the off-main ablation parts")
    ap.add_argument("--bands", help="subset of bands, e.g. ISM_2400,UNII_1")
    ap.add_argument("--devices", help="subset of radios (the .h5 'radio' attribute)")
    ap.add_argument("--clients", help="e.g. 4,7,14 (default: CONFIG['ab_clients'])")
    ap.add_argument("--partitions", help="iid,noniid"); ap.add_argument("--algos", help="fedavg,fedprox")
    ap.add_argument("--task", type=int, help="run one part only (SLURM array index)")
    ap.add_argument("--list-tasks", action="store_true", help="print the part indices and exit")
    ap.add_argument("--merge", action="store_true", help="only rebuild the merged CSVs from parts_v2/")
    ap.add_argument("--device"); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.family == "all":                       # run both families, each in its own process
        import subprocess
        argv = [x for x in sys.argv[1:]]
        if "--family" in argv:
            k = argv.index("--family"); del argv[k:k + 2]
        argv = [x for x in argv if not x.startswith("--family=")]
        for fam in ("sota", "gem"):
            log(f"\n######## family {fam} ########")
            subprocess.run([sys.executable, os.path.abspath(__file__), "--family", fam] + argv, check=True)
        return
    cfg = CONFIG
    cfg["family"] = a.family
    cfg.update(copy.deepcopy(FAMILIES[a.family]))
    if a.results:
        cfg["out_dir"] = os.path.join(a.results, os.path.basename(FAMILIES[a.family]["out_dir"]))
        cfg["sensing_dir"] = os.path.join(a.results, "01_sensing", "sensing_handoff")
        cfg["sensing_dir_fallbacks"] = [os.path.join(a.results, "01_sensing")]
    if a.quick:
        cfg.update(QUICK)
    if a.input:
        cfg["input"] = a.input
    if a.sensing_dir:
        cfg["sensing_dir"] = a.sensing_dir
    if a.sensing_source:
        cfg["sensing_source"] = a.sensing_source
    if a.source_ablation is not None:
        cfg["source_ablation"] = ("auto" if a.source_ablation == "auto"
                                  else [x.strip() for x in a.source_ablation.split(",") if x.strip()])
    if a.root:
        cfg["data_root"] = a.root
    if a.source:
        cfg["source"] = a.source
    if a.out:
        cfg["out_dir"] = a.out
    if a.seeds:
        cfg["seeds"] = [int(x) for x in a.seeds.split(",")]
    if a.models:
        cfg["forecast_models"] = [x.strip() for x in a.models.split(",") if x.strip()]
    if a.sota_models is not None:
        cfg["sota_models"] = [x.strip() for x in a.sota_models.split(",") if x.strip()]
    if a.sota_models_ablation is not None:
        cfg["sota_models_ablation"] = [x.strip() for x in a.sota_models_ablation.split(",") if x.strip()]
    if a.bands:
        cfg["bands"] = [x.strip() for x in a.bands.split(",") if x.strip()]
    if a.devices:
        cfg["devices"] = [x.strip() for x in a.devices.split(",") if x.strip()]
    if a.clients:
        cfg["ab_clients"] = [int(x) for x in a.clients.split(",")]
    if a.partitions:
        cfg["ab_partitions"] = [x.strip() for x in a.partitions.split(",")]
    if a.algos:
        cfg["ab_algos"] = [x.strip() for x in a.algos.split(",")]
    if a.device:
        cfg["device"] = a.device
    bad = [m for m in cfg["forecast_models"] if m not in MODEL_REGISTRY]
    if bad:
        sys.exit(f"unknown novel model(s) {bad}; choose from {list(MODEL_REGISTRY)}")
    bad = [m for m in cfg["sota_models"] + cfg["sota_models_ablation"] if m not in SOTA_REGISTRY]
    if bad:
        sys.exit(f"unknown SOTA model(s) {bad}; choose from {list(SOTA_REGISTRY)}")
    out_dir = os.path.abspath(os.path.expanduser(cfg["out_dir"]))
    os.makedirs(out_dir, exist_ok=True)
    tag0 = "merge" if a.merge else ("all" if a.task is None else f"task{a.task}")
    _LOG_FH = open(os.path.join(out_dir, f"run_log_{tag0}.txt"), "a", encoding="utf-8")
    log("=" * 78)
    log(f"02_forecasting [{cfg['family']}]  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if a.merge:
        merge(out_dir); return
    dev = get_device(cfg)
    loader = load_all_sensed if cfg["input"] == "sensing_grid" else load_all
    D, names, C, cards, data_fp = loader(cfg, dev)
    S, H = len(names), cfg["horizon"]
    resolve_clients(cfg, S)
    main_src = cfg.get("_sensing_source", "")
    cfg["_source_ablation_list"] = source_ablation_list(cfg)
    tasks = ab_tasks(cfg)
    if a.list_tasks or (a.task is not None and not 0 <= a.task < len(tasks)):
        msg = ", ".join(f"{i}={task_tag(t)}" for i, t in enumerate(tasks))
        if a.list_tasks:
            log(msg); return
        sys.exit(f"--task must be 0..{len(tasks) - 1}: {msg}")
    sel = tasks if a.task is None else [tasks[a.task]]
    log(f"device {dev} | parts: {', '.join(task_tag(t) for t in sel)}")
    if cfg["input"] == "sensing":
        log(f"input: PSD sweeps, {C} sub-channels per band, eta calibrated per band to "
            f"{cfg['_sensing_desc']}; W={cfg['lookback']} H={H} | source ablation: "
            f"{cfg['_source_ablation_list'] or 'none'}")
    elif cfg["input"] == "sensing_grid":
        log(f"input: sensed occupancy ({cfg['_sensing_desc']}), {C} tile channels per receiver; "
            f"W={cfg['lookback']} H={H} sweeps | source ablation: {cfg['_source_ablation_list'] or 'none'}")
    else:
        log(f"labels: Q{cfg['noise_quantile'] * 100:g} noise floor (train span), eta = {cfg['eta_db']:g} dB, "
            f"{C} sub-channels per band ({cfg['subchannel_mode']}); W={cfg['lookback']} H={H}")
    log(f"grid: clients {cfg['ab_clients']} (main: non-IID temporal K = {cfg['main_clients']}) x partitions "
        f"{cfg['ab_partitions']} x "
        f"{cfg['ab_algos']} | novel {cfg['forecast_models']} (+DCS) | SOTA main {cfg['sota_models']} | "
        f"SOTA ablation {cfg['sota_models_ablation']} | seeds {cfg['seeds']}")
    log("=" * 78)
    _atomic_json(os.path.join(out_dir, "config_used.json"), cfg)
    emax = max(float(D[n]["E_te"].max()) for n in names)
    edges_snr = snr_edges_ab(emax, cfg)
    log(f"  SNR bins (dB): {', '.join(f'{edges_snr[i]:g}-{edges_snr[i + 1]:g}' for i in range(len(edges_snr) - 1))}")
    write_csv(os.path.join(out_dir, "dataset_card.csv"), cards)
    base = {s: {b: {"te": baseline_forecast(b, D[s]["Xbin_te"], H, D[s]["period"]),
                    "va": baseline_forecast(b, D[s]["Xbin_va"], H, D[s]["period"])}
                for b in cfg["baselines"]} for s in names}
    write_stage2_static(cfg, out_dir, D, names, C, edges_snr, base)
    for t in sel:
        log(f"\n[part {task_tag(t)}]")
        if t[0] == "src":                                     # same grid / split, other sensing source
            cfg["sensing_source"] = t[2]
            D2, n2, C2, cards2, fp2 = loader(cfg, dev)
            base2 = {s: {b: {"te": baseline_forecast(b, D2[s]["Xbin_te"], H, D2[s]["period"]),
                             "va": baseline_forecast(b, D2[s]["Xbin_va"], H, D2[s]["period"])}
                         for b in cfg["baselines"]} for s in n2}
            write_csv(os.path.join(out_dir, f"dataset_card_source-{t[2]}.csv"), cards2)
            sink = run_task(t, cfg, dev, D2, n2, C2, len(n2), edges_snr, out_dir, fp2, base2)
            cfg["sensing_source"] = "main"
            cfg["_sensing_source"] = main_src
            del D2
            save_part(out_dir, task_tag(t), sink)
            merge(out_dir)
            continue
        sink = run_task(t, cfg, dev, D, names, C, S, edges_snr, out_dir, data_fp, base)
        save_part(out_dir, task_tag(t), sink)
        merge(out_dir)
    log(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
