#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_sensing.py -- Stage 1 of the sense-forecast-share pipeline
=============================================================
Spectrum sensing on the raw-I/Q tier of the Fireworks mass-gathering dataset
(IEEE DataPort, DOI 10.21227/f2rv-ms26).

  * Labels  (paper Sec. III-B): every frame of 32,768 samples (1.31 ms at
    25 MS/s) is cut into 20-us windows; noise floor N0 = 10th percentile of
    the window powers, margin eta = 6 dB, occupied iff >= 10 % of the windows
    exceed N0 + eta. SNR proxy = Q0.99 - N0. Gain invariant.
  * Models: OSCAR-Net (proposed) and the five SOTA detectors of Table II:
        oscar_net_p1             OSCAR-Net (this work)
        deepsense_s4             DeepSense   (IEEE INFOCOM 2021)
        tprime_s9                T-PRIME     (IEEE INFOCOM 2024)
        spectrum_transformer_s5  SpecFormer  (IEEE TWC 2024)
        cnn_transformer_s6       CNN-Former  (IEEE WCL 2025)
        denoise_transformer_s7   DnFormer    (IEEE COMML 2025)
  * Training: centralized and FedAvg (8 non-IID temporal clients, half of
    them per round, 2 local epochs, GroupNorm); operating point = validation
    threshold with false-alarm rate closest below 10 %.
  * Metrics: accuracy, F1, recall (Pd), FDR = FP/(FP+TN), MDR = 1 - Pd,
    NMSE, MAE, per-SNR curves, parameters, FLOPs, CPU latency.

STAGES (--stages, comma separated; default main,clients,handoff)
  main     Table II: every model x {centralized, federated}
  clients  Fig. 8(a): client count K in {4, 8, 12, 16, 32} x {non-IID, IID}
  handoff  runs OSCAR-Net (FedAvg) over every frame and writes
           <out>/sensing_handoff/frames.npz + manifest.json for stage 2
  lobo     optional: leave-one-band-out audit (not in the paper)
  labels   optional: label-scheme ablation (not in the paper)

USAGE
  python 01_sensing.py --root DATA_DIR --out RESULTS/01_sensing
  python 01_sensing.py --stages main --models oscar_net_p1,deepsense_s4
  python 01_sensing.py --quick              # 5 epochs / 5 rounds smoke test

Every finished (model, mode) unit is checkpointed under <out>/_completed/;
rerunning the same command after an interruption resumes where it stopped.
"""

from __future__ import annotations

import collections
import copy
import csv
import glob
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import h5py
except ImportError:
    sys.exit("h5py required:  pip install h5py")

# deterministic cuBLAS kernels need this BEFORE torch initialises CUDA
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, Subset
except ImportError:
    sys.exit("PyTorch required:  pip install torch")

from sklearn.metrics import (auc, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_curve)


# ==========================================================================
#                                 CONFIG
# ==========================================================================
# Paths can be given on the command line (--root / --out) or via the
# environment variables SFS_DATA (export .h5 folder) and SFS_RESULTS.
ROOT = os.environ.get("SFS_DATA", os.path.join(".", "data", "fireworks_export"))
PIPELINE_ROOT = os.environ.get("SFS_RESULTS", os.path.join(".", "results"))
OUTPUT_ROOT = PIPELINE_ROOT

CONFIG: Dict = {
    # ---- data -----------------------------------------------------------
    "h5_files": [],              # [] = every *.h5 under root/ and root/*/export/
    "root": ROOT,
    "dataset_arms": [{"name": "combined", "devices": []}],   # one pooled arm
    "frame_len": 32768,          # 1.31 ms at 25 MS/s
    "use_spectrogram": False,    # no 2-D detector in the paper
    "spec_size": (128, 128),
    "max_frames_per_file": 0,    # 0 = all frames
    "cache_dir": None,           # None = <out>/_frame_cache (memmap, ~17 GB)
    "store_dtype": "float16",
    "load_block": 256,
    "reuse_frame_cache": True,
    "normalise": "per_frame_rms",
    # ---- labels (paper eq. (1)-(2)) -------------------------------------
    "label_segment": "model_input",
    "crop_mode": "head",
    "min_label_windows": 32,
    "label_scheme": "margin",
    "noise_percentile": 10.0,    # N0 = Q_0.10 of the window powers
    "occ_margin_db": 6.0,        # eta = 6 dB
    "margin_busy_frac": 0.10,    # beta_0 = 0.10
    "label_win_us": 20.0,        # 20-us windows (500 samples at 25 MS/s)
    # legacy two-threshold scheme, only used by the optional 'labels' stage
    "label_hi_db": 10.0,
    "label_lo_db": 3.0,
    "label_busy_frac_hi": 0.15,
    "label_busy_frac_lo": 0.02,
    "drop_ambiguous": True,
    "snr_bins_db": list(range(-10, 32, 4)),
    # ---- models ---------------------------------------------------------
    "models": [
        "deepsense_s4", "tprime_s9", "spectrum_transformer_s5",
        "cnn_transformer_s6", "denoise_transformer_s7",
        "oscar_net_p1",
    ],
    # ---- training -------------------------------------------------------
    "batch_size": 128,
    "epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 5e-4,
    "early_stopping_patience": 8,
    "min_epochs": 12,
    "lr_scheduler_patience": 4,
    "lr_scheduler_factor": 0.5,
    "min_lr": 1e-6,
    "grad_clip": 5.0,
    "augment_train": True,
    "aug_time_shift": True,
    "aug_phase_rotate": True,
    "aug_time_reverse": False,
    "label_smoothing": 0.05,
    "val_frac": 0.15,
    "test_frac": 0.20,
    "split_by": "time",          # split inside 32 time blocks: no leakage
    "split_blocks": 32,
    # ---- federated learning ---------------------------------------------
    "training_modes": ["centralized", "federated"],
    "client_partition": "natural",   # natural = contiguous time blocks (non-IID)
    "n_time_blocks": 3,
    "n_time_blocks_single_device": 8,   # -> 8 clients for one radio
    "min_per_class_per_client": 8,
    "n_clients": 6,              # only for the 'iid' / 'dirichlet' partitions
    "client_fraction": 0.5,
    "dirichlet_alpha": 0.5,
    "fl_rounds": 100,
    "fl_local_epochs": 2,
    "fl_early_stop_rounds": 10,
    "fl_norm": "groupnorm",
    "fl_algorithm": "fedavg",    # the paper uses FedAvg ("mira" = optional)
    "mira_lambda": 0.30,
    "mira_server_lr": 1.0,
    "mira_adjacency": "temporal",
    "mira_tau": 1.5,
    "mira_eval": "consensus",
    "fl_min_rounds": 12,
    "report_per_client": True,
    # ---- stage 'clients' (Fig. 8a) --------------------------------------
    "client_count_ablation": [4, 8, 12, 16, 32],
    "ablation_models": [],       # [] = every model in "models"
    "ablation_partitions": ["natural", "iid"],
    # ---- optional stages ------------------------------------------------
    "lobo_models": [],
    "lobo_modes": ["centralized"],
    "label_scheme_ablation": ["dead_zone"],
    "label_ablation_models": [],
    "eval_leave_one_band_out": True,
    # ---- stage 'handoff' (sensing -> forecasting) -----------------------
    "handoff_detector": "oscar_net_p1",
    "handoff_mode": "federated",
    "handoff_sota": "auto",      # best SOTA detector by validation accuracy
    "handoff_dir": None,         # None = <out>/sensing_handoff
    "handoff_batch": 256,
    # ---- complexity profile (CPU only) ----------------------------------
    "profile_complexity": True,
    "prof_batch": 1024,
    "prof_warmup": 5,
    "prof_iters": 20,
    "prof_measure_b1": True,
    "prof_b1_iters": 20,
    "prof_threads": 0,
    # ---- operating point --------------------------------------------------
    "threshold_mode": "target_pfa",
    "target_pfa": 0.10,
    # ---- bookkeeping ------------------------------------------------------
    "results_dir": os.path.join(PIPELINE_ROOT, "01_sensing"),
    "resume": True,
    "force_retrain": [],         # e.g. ["combined__tprime_s9__federated"]
    "ignore_fingerprint": False,
    "seed": 20260807,
    "train_seeds": [20260807],   # add seeds to repeat the main table
    "deterministic": True,
    "num_workers": 0,
    "verbose": True,
}

SEED = CONFIG["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if CONFIG.get("deterministic", True):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_seed(cfg: Dict) -> int:
    """Seed of the TRAINING randomness (init, shuffling, augmentation, FL
    client sampling). Defaults to cfg['seed']."""
    t = cfg.get("train_seed")
    return int(t) if t is not None else int(cfg.get("seed", 0))
if CONFIG.get("prof_threads"):
    torch.set_num_threads(int(CONFIG["prof_threads"]))


def _stable_seed(*parts) -> int:
    """Process-independent hash. Python's hash() of a str is salted per
    process (PYTHONHASHSEED), so seeds built from it differ between runs."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode())
               .hexdigest()[:8], 16)


def seed_unit(base: int, *parts) -> None:
    """Reseed every RNG at the start of one training unit, so a unit's
    result does not depend on which units ran before it (or were resumed
    from checkpoints) -- each (arm, model, mode) is reproducible alone."""
    s_ = (int(base) + _stable_seed(*parts)) % (2 ** 31 - 1)
    random.seed(s_)
    np.random.seed(s_)
    torch.manual_seed(s_)
    torch.cuda.manual_seed_all(s_)


def log(m: str = "") -> None:
    if CONFIG["verbose"]:
        print(m, flush=True)


# ==========================================================================
#                          DATA: LOAD + LABEL
# ==========================================================================
def energy_labels(iq: np.ndarray, samp_rate: float, cfg: Dict
                  ) -> Tuple[int, float, float]:
    """
    Energy detection on ONE frame. Two schemes, selected by
    cfg["label_scheme"]:

    "margin" (default) -- floor = Q_{noise_percentile}{dBFS power}, single
        margin eta=occ_margin_db. H1 iff the busy fraction above floor+eta
        reaches margin_busy_frac; H0 otherwise. No dead zone: label is NEVER
        -1 under this scheme, so every frame that survives the
        minimum-window check is kept. Looser than "dead_zone" by
        construction (lower floor percentile, smaller margin, one threshold
        instead of two), so it admits marginal/low-SNR frames that
        "dead_zone" would discard.

    "dead_zone" (legacy) -- the original two-threshold rule. Frames between
        the thresholds are AMBIGUOUS (returned as -1; caller discards them),
        so only high-confidence, easily-separable frames survive.

    Returns (label, snr_proxy_db, busy_frac). snr_proxy_db is always
    (Q_0.99 - floor) so the two schemes stay plottable on the same axis.
    """
    nwin = max(8, int(round(cfg["label_win_us"] * 1e-6 * samp_rate)))
    need = int(cfg.get("min_label_windows", 32))
    # The percentiles that define the label are computed over iq.size/nwin
    # windows. With too few windows they are dominated by sampling noise and
    # the labels become close to random, which no model can recover from.
    # Shrink the window rather than emit a noisy label.
    if iq.size // nwin < need:
        nwin = max(8, iq.size // need)
    n = (iq.size // nwin) * nwin
    if n < nwin * 8:
        return -1, float("nan"), float("nan")
    pw = (np.abs(iq[:n].reshape(-1, nwin)) ** 2).mean(axis=1)
    pdb = 10.0 * np.log10(pw + 1e-20)
    peak = float(np.percentile(pdb, 99.0))

    scheme = cfg.get("label_scheme", "margin")
    if scheme == "margin":
        floor_pct = float(cfg.get("noise_percentile", 10.0))
        floor = float(np.percentile(pdb, floor_pct))
        snr_proxy = peak - floor
        eta = float(cfg.get("occ_margin_db", 6.0))
        busy = float(np.mean(pdb > floor + eta))
        beta = float(cfg.get("margin_busy_frac", 0.10))
        label = 1 if busy >= beta else 0
        return label, snr_proxy, busy

    # ---- legacy "dead_zone" scheme ----
    floor = float(np.percentile(pdb, 25.0))
    snr_proxy = peak - floor
    busy_hi = float(np.mean(pdb > floor + cfg["label_hi_db"]))
    busy_lo = float(np.mean(pdb > floor + cfg["label_lo_db"]))

    if snr_proxy >= cfg["label_hi_db"] and busy_hi >= cfg["label_busy_frac_hi"]:
        return 1, snr_proxy, busy_hi
    if snr_proxy <= cfg["label_lo_db"] and busy_lo <= cfg["label_busy_frac_lo"]:
        return 0, snr_proxy, busy_lo
    return -1, snr_proxy, busy_hi


def _frames_group(f):
    """I/Q frames of an .h5 file: the /iq_frames group of an export file, or
    the root of derived/frames.h5 written by the dataset's converter
    (code/sigmf_to_dataset.py of the IEEE DataPort release)."""
    if "iq_frames" in f:
        return f["iq_frames"]
    if "iq" in f and "sample_rate" in f and "center_hz" in f and "utc" in f:
        return f
    return None


def discover_h5(cfg: Dict) -> List[str]:
    """Entries may be .h5 files OR directories to search for .h5."""
    if cfg["h5_files"]:
        out: List[str] = []
        for e in cfg["h5_files"]:
            if os.path.isdir(e):
                out.extend(sorted(glob.glob(os.path.join(e, "*.h5"))))
            elif os.path.exists(e):
                out.append(e)
            else:
                log(f"  WARNING: path not found, skipped: {e}")
        return sorted(set(out))
    pats = [os.path.join(cfg["root"], "*", "export", "*.h5"),
            os.path.join(cfg["root"], "derived", "*.h5"),        # DataPort converter output
            os.path.join(cfg["root"], "*.h5")]
    out: List[str] = []
    for p in pats:
        out.extend(sorted(glob.glob(p)))
    return sorted(set(out))


_CACHE_KEYS = [
    "frame_len", "use_spectrogram", "normalise", "label_segment", "crop_mode",
    "min_label_windows", "label_scheme", "noise_percentile", "occ_margin_db",
    "margin_busy_frac", "label_hi_db", "label_lo_db", "label_busy_frac_hi",
    "label_busy_frac_lo", "label_win_us", "drop_ambiguous",
    "max_frames_per_file", "seed", "store_dtype",
]


def _frame_cache_key(cfg: Dict, files: List[str]) -> str:
    """Hash of everything that changes the labelled, normalised frame set.
    Same key -> the on-disk cache can be reused without re-reading the .h5."""
    payload = {k: cfg.get(k) for k in _CACHE_KEYS}
    payload["files"] = [(os.path.abspath(p), os.path.getsize(p),
                         int(os.path.getmtime(p))) for p in files]
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _normalise_iq(a: np.ndarray, mode: str) -> np.ndarray:
    """Per-frame amplitude normalisation, applied BEFORE the frame is stored,
    so the full array never has to be held in RAM to normalise it."""
    if mode == "per_frame_rms":
        r = float(np.sqrt((a * a).sum(axis=1).mean())) + 1e-12
        return a / r
    if mode == "per_frame_peak":
        return a / (float(np.abs(a).max()) + 1e-12)
    return a


def load_dataset(cfg: Dict) -> Dict[str, np.ndarray]:
    """
    Read every exported .h5 and build the labelled frame set.

    MEMORY. The previous version appended every frame to a Python list and
    np.stack-ed it at the end, which needs the whole corpus in RAM twice:
    128,778 frames x 32,768 samples x 2 x float32 is ~34 GB for the I/Q alone,
    ~8 GB more for spectrograms. Under the old dead-zone labels most frames
    were discarded, so it fitted; under the "margin" labels nothing is
    discarded and it does not.

    Frames are now normalised one at a time and written straight into
    disk-backed .npy arrays (numpy memmap) under cfg["cache_dir"]. RAM use is
    one read block plus the small metadata vectors, regardless of corpus
    size. The cache is keyed on the files and every labelling/normalisation
    setting, so a rerun with the same configuration skips the .h5 pass
    entirely; change any of those settings and a fresh cache is built.
    """
    files = discover_h5(cfg)
    if not files:
        sys.exit(f"No .h5 found. Download the export .h5 files of the dataset first, or set "
                 f"CONFIG['h5_files'].\nLooked under: {cfg['root']}")

    L = cfg["frame_len"]
    store_dt = np.dtype(cfg.get("store_dtype", "float16"))
    key = _frame_cache_key(cfg, files)
    cdir = os.path.join(cfg.get("cache_dir") or
                        os.path.join(cfg["results_dir"], "_frame_cache"), key)
    os.makedirs(cdir, exist_ok=True)
    p_iq = os.path.join(cdir, "X_iq.npy")
    p_sp = os.path.join(cdir, "X_spec.npy")
    p_meta = os.path.join(cdir, "meta.npz")
    p_done = os.path.join(cdir, "done.json")

    if cfg.get("reuse_frame_cache", True) and os.path.exists(p_done):
        with open(p_done) as f:
            info = json.load(f)
        n = int(info["n"])
        log(f"  frame cache HIT  {cdir}")
        log(f"  ({n:,} labelled frames; delete this folder to rebuild)")
        native_lens = set(info["native_lens"])
        n_amb = int(info["n_amb"])
        meta = np.load(p_meta, allow_pickle=False)
        d = {k: meta[k] for k in meta.files}
        d["X_iq"] = np.load(p_iq, mmap_mode="r")[:n]
        if info.get("has_spec"):
            d["X_spec"] = np.load(p_sp, mmap_mode="r")[:n]
    else:
        # ---- pass 1: sizes only (no sample data read) ----
        plan = []
        cap, all_spec, spec_shape = 0, True, None
        for path in files:
            with h5py.File(path, "r") as f:
                if _frames_group(f) is None:
                    log(f"  {os.path.basename(path)}: no I/Q frames (/iq_frames or /iq), skipped")
                    continue
                g = _frames_group(f)
                n_total = g["iq"].shape[0]
                n_use = (min(n_total, cfg["max_frames_per_file"])
                         if cfg["max_frames_per_file"] else n_total)
                plan.append((path, n_use, n_total))
                cap += n_use
                if "spectrogram" in g:
                    spec_shape = tuple(g["spectrogram"].shape[1:])
                else:
                    all_spec = False
        if not plan:
            sys.exit("No .h5 file contained iq_frames.")
        use_spec = bool(cfg["use_spectrogram"] and all_spec and spec_shape)
        if cfg["use_spectrogram"] and not use_spec:
            log("  NOTE: not every .h5 has spectrograms -> 2-D models disabled")

        need = cap * L * 2 * store_dt.itemsize
        if use_spec:
            need += cap * int(np.prod(spec_shape)) * store_dt.itemsize
        free = shutil.disk_usage(cdir).free
        log(f"  frame cache MISS -> building {cdir}")
        log(f"  up to {cap:,} frames, {need/1e9:.1f} GB on disk "
            f"({store_dt.name}); {free/1e9:.1f} GB free")
        if need > free * 0.95:
            sys.exit(f"Not enough disk for the frame cache ({need/1e9:.1f} GB "
                     f"needed, {free/1e9:.1f} GB free). Point cache_dir at a "
                     f"larger drive, keep store_dtype='float16', or set "
                     f"max_frames_per_file to subsample.")

        mm_iq = np.lib.format.open_memmap(p_iq, mode="w+", dtype=store_dt,
                                          shape=(cap, L, 2))
        mm_sp = (np.lib.format.open_memmap(p_sp, mode="w+", dtype=store_dt,
                                           shape=(cap,) + spec_shape)
                 if use_spec else None)

        rng_crop = np.random.default_rng(cfg["seed"])
        y, snr, dev, band, tstamp, cfreq = [], [], [], [], [], []
        n_amb, k = 0, 0
        native_lens: set = set()
        step = int(cfg.get("load_block", 256))
        norm = cfg["normalise"]

        for path, n_use, n_total in plan:
            with h5py.File(path, "r") as f:
                radio = f.attrs.get("radio", "unknown")
                if isinstance(radio, bytes):
                    radio = radio.decode()
                g = _frames_group(f)
                fl = int(g.attrs.get("frame_len", g["iq"].shape[1]))
                native_lens.add(fl)
                fs_col = (g["full_scale"][:n_use] if "full_scale" in g
                          else np.full(n_use, 32767.0))
                sr_col = g["sample_rate"][:n_use]
                bn_col = g["band_name"][:n_use]
                cf_col = g["center_hz"][:n_use]
                ut_col = g["utc"][:n_use]
                log(f"  {os.path.basename(path)}  [{radio}]  "
                    f"{n_use:,}/{n_total:,} frames of {fl:,}")

                t0 = time.time()
                for s in range(0, n_use, step):
                    e = min(s + step, n_use)
                    blk = g["iq"][s:e]                    # native dtype (int16)
                    spb = g["spectrogram"][s:e] if use_spec else None
                    for j in range(blk.shape[0]):
                        fs = float(fs_col[s + j]) or 32767.0
                        fr = blk[j].astype(np.float32)
                        iq_full = (fr[:, 0] + 1j * fr[:, 1]) / fs

                        # ---- pick the segment the MODEL will actually see ----
                        if iq_full.size >= L:
                            if cfg.get("crop_mode", "head") == "random":
                                off = int(rng_crop.integers(0, iq_full.size - L + 1))
                            else:
                                off = 0
                            seg = iq_full[off:off + L]
                        else:
                            seg = np.zeros(L, dtype=np.complex64)
                            seg[:iq_full.size] = iq_full

                        # ---- label THAT segment, not the whole stored frame ----
                        to_label = (seg if cfg.get("label_segment", "model_input")
                                    == "model_input" else iq_full)
                        lab, sp, _bf = energy_labels(to_label,
                                                     float(sr_col[s + j]), cfg)
                        if lab < 0:
                            n_amb += 1
                            if cfg["drop_ambiguous"]:
                                continue
                            lab = 1 if sp >= 0.5 * (cfg["label_hi_db"]
                                                    + cfg["label_lo_db"]) else 0
                        a = np.stack([seg.real, seg.imag],
                                     axis=-1).astype(np.float32)
                        mm_iq[k] = _normalise_iq(a, norm)
                        if mm_sp is not None:
                            sg = spb[j].astype(np.float32)
                            mm_sp[k] = (sg - sg.mean()) / (sg.std() + 1e-6)
                        y.append(lab)
                        snr.append(sp)
                        dev.append(str(radio))
                        b = bn_col[s + j]
                        band.append(b.decode() if isinstance(b, bytes) else str(b))
                        cfreq.append(float(cf_col[s + j]))
                        tstamp.append(float(ut_col[s + j]))
                        k += 1
                    if (s // step) % 40 == 0:
                        rate = e / max(time.time() - t0, 1e-6)
                        log(f"    {e:,}/{n_use:,} read  ({rate:,.0f} frames/s, "
                            f"{k:,} kept so far)")

        n = k
        if n == 0:
            sys.exit("No frames survived labelling. For label_scheme='margin', "
                     "lower noise_percentile / occ_margin_db / margin_busy_frac; "
                     "for 'dead_zone', loosen label_hi_db / label_lo_db; or "
                     "check that the .h5 contains iq_frames.")
        mm_iq.flush()
        del mm_iq                      # release the write handle (Windows)
        if mm_sp is not None:
            mm_sp.flush()
            del mm_sp
        np.savez(p_meta, y=np.array(y, dtype=np.int64),
                 snr=np.array(snr, dtype=np.float32), device=np.array(dev),
                 band=np.array(band), center_hz=np.array(cfreq, dtype=np.float64),
                 utc=np.array(tstamp, dtype=np.float64))
        # commit marker LAST: a crash before this line just rebuilds next time
        _atomic_write_json(p_done, {"n": n, "capacity": cap, "n_amb": n_amb,
                                    "native_lens": sorted(native_lens),
                                    "has_spec": use_spec, "key": key})
        meta = np.load(p_meta, allow_pickle=False)
        d = {kk: meta[kk] for kk in meta.files}
        d["X_iq"] = np.load(p_iq, mmap_mode="r")[:n]
        if use_spec:
            d["X_spec"] = np.load(p_sp, mmap_mode="r")[:n]

    # ---------------- label / input consistency report ----------------
    nat = max(native_lens) if native_lens else L
    log()
    log(f"  stored frame length   {nat:,} samples")
    log(f"  model input length    {L:,} samples "
        f"({L/nat*100:.0f}% of the stored frame)")
    if L < nat:
        if cfg.get("label_segment", "model_input") == "model_input":
            log(f"  labels are computed on the SAME {L:,}-sample segment the "
                f"network receives - consistent.")
            if cfg.get("crop_mode") == "head":
                log(f"  NOTE: crop_mode='head' always takes the first {L:,} "
                    f"samples, so {100-L/nat*100:.0f}% of every recording is "
                    f"never used. Set crop_mode='random' to sample across the "
                    f"whole frame, or raise frame_len to {nat:,}.")
        else:
            log(f"  *** WARNING: label_segment='full_frame'. Labels come from "
                f"all {nat:,} samples but the network sees only the first "
                f"{L:,}.")
            log(f"  *** A burst anywhere in the other {nat-L:,} samples makes "
                f"the frame H1 while the network is shown pure noise. This "
                f"puts a hard ceiling on every 1-D model and CANNOT be fixed "
                f"by architecture. Set label_segment='model_input'.")
    # envelope windows available for the percentile statistics
    if "X_iq" in dir():
        pass
    n1 = int((d["y"] == 1).sum())
    n0 = int((d["y"] == 0).sum())
    log()
    log(f"  frames kept   {len(d['y']):,}   (H1 {n1:,} / H0 {n0:,})")
    if cfg.get("label_scheme", "margin") == "margin":
        log(f"  labelling     'margin' scheme: floor=Q{cfg.get('noise_percentile',10.0):.0f}, "
            f"eta={cfg.get('occ_margin_db',6.0):.0f} dB, "
            f"busy_frac>={cfg.get('margin_busy_frac',0.10):.2f} -> H1  "
            f"({n_amb:,} frames too short for the min-window check, discarded)")
    else:
        log(f"  ambiguous     {n_amb:,} discarded (dead zone between "
            f"{cfg['label_lo_db']:.0f} and {cfg['label_hi_db']:.0f} dB)")
    devs = sorted(set(d["device"].tolist()))
    log(f"  devices       {devs}")
    if len(devs) < 2:
        log(f"  single-radio setup ({devs[0]}). Federated clients will be "
            f"TIME blocks of one receiver, which is the single-sensor "
            f"temporal-federation setting - see partition_clients().")
    log(f"  bands         {len(set(d['band'].tolist()))}")
    log(f"  est SNR       {np.nanmin(d['snr']):.1f} .. "
        f"{np.nanmax(d['snr']):.1f} dB")
    # ---- per-band class balance: the confound audit, printed every run ----
    log()
    log("  per-band class balance (a band that is 100% one class lets a model")
    log("  score by identifying the BAND instead of detecting a signal):")
    degenerate = []
    for b in sorted(set(d["band"].tolist())):
        m = d["band"] == b
        yy = d["y"][m]
        f1r = float((yy == 1).mean())
        flag = ""
        if f1r in (0.0, 1.0):
            flag = "  <-- single class"
            degenerate.append(b)
        log(f"    {b:<20s} n={int(m.sum()):6,d}  H1={f1r*100:5.1f}%{flag}")
    if degenerate:
        log(f"  WARNING: {len(degenerate)}/{len(set(d['band'].tolist()))} bands "
            f"are single-class.")
        log(f"  Band identity then predicts the label almost perfectly, and any")
        log(f"  model that can see spectral shape (the 2-D spectrogram models")
        log(f"  especially) can reach ~1.0 AUC without learning detection at all.")
        log(f"  Treat near-perfect scores as evidence of this confound, not of")
        log(f"  performance, and run the leave-one-band-out audit.")

    if n0 == 0 or n1 == 0:
        sys.exit("Only one class survived labelling - the detector thresholds "
                 "do not fit this data. Inspect the sweeps CSV duty_cycle "
                 "distribution and adjust occ_margin_db/noise_percentile "
                 "(label_scheme='margin') or label_hi_db/label_lo_db "
                 "(label_scheme='dead_zone').")
    imb = max(n0, n1) / max(1, min(n0, n1))
    if imb > 3:
        log(f"  NOTE: class imbalance {imb:.1f}:1 - class weights applied")
    return d


def filter_arm(d: Dict[str, np.ndarray], devices: List[str]
               ) -> Optional[Dict[str, np.ndarray]]:
    if not devices:
        return d
    m = np.isin(d["device"], devices)
    if m.sum() == 0:
        return None
    if m.all():
        return d
    # NOTE: boolean indexing a memmap materialises the selected rows in RAM.
    # Fine for a small per-device arm; for a large one, prefer one combined
    # arm (devices=[]), which never copies.
    log(f"  arm filter copies {int(m.sum()):,} frames into RAM "
        f"(~{int(m.sum()) * d['X_iq'][0].nbytes / 1e9:.1f} GB of I/Q)")
    return {k: (v[m] if isinstance(v, np.ndarray) and v.shape[0] == m.size
                else v) for k, v in d.items()}


def arm_stats(d: Dict[str, np.ndarray]) -> Dict:
    y, snr = d["y"], d["snr"]
    fin = np.isfinite(snr)
    q = (lambda p: float(np.percentile(snr[fin], p)) if fin.any()
         else float("nan"))
    return {"n_frames": int(y.size), "n_H1": int((y == 1).sum()),
            "n_H0": int((y == 0).sum()), "h1_fraction": float((y == 1).mean()),
            "majority_baseline": float(max((y == 0).mean(), (y == 1).mean())),
            "n_bands": int(len(set(d["band"].tolist()))),
            "n_devices": int(len(set(d["device"].tolist()))),
            "snr_p10_db": q(10), "snr_p50_db": q(50), "snr_p90_db": q(90)}


class FrameDS(Dataset):
    """
    Frame dataset with optional TRAIN-TIME augmentation.

    IMPORTANT: these augmentations add NO synthetic signal or noise. They are
    exact symmetries of the detection problem, so an augmented frame is a
    physically valid observation of the same hypothesis:

      circular time shift  - whether a channel is occupied does not depend on
                             where in the window you start looking
      global phase rotation- the receiver LO phase is arbitrary; multiplying
                             the whole frame by e^{j phi} is what would have
                             happened had the dwell begun a fraction of a
                             carrier cycle earlier
      time reversal        - optional, off by default; it conjugates the
                             spectrum, which is a symmetry for occupancy but
                             NOT for asymmetric modulations, so it is only
                             safe for pure detection

    Applying these is not the same as adding noise to inflate a dataset. Each
    output is a measurement the radio could genuinely have produced.
    """

    def __init__(self, d: Dict[str, np.ndarray], idx: np.ndarray,
                 use_spec: bool, augment: bool = False,
                 cfg: Optional[Dict] = None):
        self.d, self.idx, self.use_spec = d, idx, use_spec
        self.augment = augment
        self.cfg = cfg or {}
        self.rng = np.random.default_rng(train_seed(self.cfg) + len(idx))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        if self.use_spec:
            a = self.d["X_spec"][j]
            x = torch.from_numpy(
                np.ascontiguousarray(a, dtype=np.float32)).unsqueeze(0)
            if self.augment and self.cfg.get("aug_time_shift", True):
                sh = int(self.rng.integers(0, x.shape[-1]))
                x = torch.roll(x, shifts=sh, dims=-1)   # time axis only
        else:
            # memmap row (float16 on disk) -> float32 copy in RAM before any
            # arithmetic, so augmentation is not done at half precision
            a = np.asarray(self.d["X_iq"][j], dtype=np.float32)
            if self.augment:
                if self.cfg.get("aug_time_shift", True):
                    a = np.roll(a, int(self.rng.integers(0, a.shape[0])), axis=0)
                if self.cfg.get("aug_phase_rotate", True):
                    ph = float(self.rng.uniform(0, 2 * np.pi))
                    c, s_ = math.cos(ph), math.sin(ph)
                    i_, q_ = a[:, 0].copy(), a[:, 1].copy()
                    a = np.stack([i_ * c - q_ * s_, i_ * s_ + q_ * c], axis=-1)
                if self.cfg.get("aug_time_reverse", False) and \
                        self.rng.random() < 0.5:
                    a = a[::-1].copy()
            x = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        return x, int(self.d["y"][j]), float(self.d["snr"][j])


# ==========================================================================
#                                 MODELS
# ==========================================================================
# ==========================================================================
#            SOTA SPECTRUM-SENSING BASELINES  (Table II of the paper)
# ==========================================================================
#  [S4] D. Uvaydov, S. D'Oro, F. Restuccia, T. Melodia, "DeepSense: Fast
#       Wideband Spectrum Sensing Through Real-Time In-the-Loop Deep
#       Learning," IEEE INFOCOM 2021.
#  [S5] W. Zhang, Y. Wang, X. Chen, Z. Cai, Z. Tian, "Spectrum Transformer:
#       An Attention-Based Wideband Spectrum Detector," IEEE Trans. Wireless
#       Commun., vol. 23, no. 9, pp. 12343-12353, Sep. 2024.   (SpecFormer)
#  [S6] X. Fang, M. Jin, Q. Guo, T. Jiang, "CNN-Transformer-Based Cooperative
#       Spectrum Sensing in Cognitive Radio Networks," IEEE Wireless Commun.
#       Lett., vol. 14, no. 5, pp. 1576-1580, May 2025.        (CNN-Former)
#  [S7] Y. Chen, Y. Wang, S. Peng, J. Li, R. Xiao, "Deep Denoising and
#       Transformer-Based 3D Spectrum Sensing," IEEE Commun. Lett., vol. 29,
#       no. 12, pp. 2894-2898, Dec. 2025.                      (DnFormer)
#  [S9] M. Belgiovine et al., "T-PRIME: Transformer-Based Protocol
#       Identification for Machine-Learning at the Edge," IEEE INFOCOM 2024.
#
#  Each detector is re-implemented for one sensor and this input shape; the
#  cooperative fusion of [S6] is not reproduced (one antenna, no spatial
#  diversity). BatchNorm is replaced by GroupNorm in the federated runs.
# ==========================================================================


class DeepSense_S4(nn.Module):
    """[S4] Uvaydov et al., IEEE INFOCOM 2021. Small 1-D CNN over raw I/Q,
    designed to be FPGA-deployable in the receive loop."""

    def __init__(self, num_classes=2, in_ch=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 16, 7, stride=2, padding=3), nn.BatchNorm1d(16),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.BatchNorm1d(32),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8, 128),
                                nn.ReLU(), nn.Dropout(0.3),
                                nn.Linear(128, num_classes))

    def forward(self, x):
        return self.fc(self.net(x.permute(0, 2, 1)))


class _PosEnc(nn.Module):
    def __init__(self, d, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SpectrumTransformer_S5(nn.Module):
    """[S5] Zhang et al., IEEE TWC 2024. Patch-tokenised attention detector."""

    def __init__(self, num_classes=2, patch=64, d_model=128, nhead=4,
                 layers=3, in_ch=2):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(patch * in_ch, d_model)
        self.pos = _PosEnc(d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4,
                                         dropout=0.1, batch_first=True,
                                         norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, num_classes))

    def forward(self, x):
        B, N, C = x.shape
        n_p = N // self.patch
        t = x[:, :n_p * self.patch].reshape(B, n_p, self.patch * C)
        t = self.pos(self.proj(t))
        t = torch.cat([self.cls.expand(B, -1, -1), t], dim=1)
        return self.head(self.tr(t)[:, 0])


class CNNTransformer_S6(nn.Module):
    """[S6] Fang et al., IEEE WCL 2025. Conv front end + attention fusion.
    Published as cooperative; here the attention fuses SUB-BAND tokens from a
    single sensor rather than reports from several sensors."""

    def __init__(self, num_classes=2, d_model=128, nhead=4, layers=2, in_ch=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, d_model, 5, stride=4, padding=2),
            nn.BatchNorm1d(d_model), nn.ReLU(),
            nn.Conv1d(d_model, d_model, 3, stride=2, padding=1),
            nn.BatchNorm1d(d_model), nn.ReLU())
        self.pos = _PosEnc(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2,
                                         dropout=0.1, batch_first=True,
                                         norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, num_classes))

    def forward(self, x):
        h = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        if h.shape[1] > 512:
            h = torch.nn.functional.adaptive_avg_pool1d(
                h.permute(0, 2, 1), 512).permute(0, 2, 1)
        return self.head(self.tr(self.pos(h)).mean(dim=1))


class DenoiseTransformer_S7(nn.Module):
    """[S7] Chen et al., IEEE Commun. Lett. 2025. Denoising autoencoder front
    end feeding a transformer. The denoiser is trained jointly through the
    classification loss (the paper pre-trains it; joint training is used here
    because this dataset has no clean reference to pre-train against)."""

    def __init__(self, num_classes=2, d_model=128, nhead=4, layers=2,
                 patch=64, in_ch=2):
        super().__init__()
        self.patch = patch
        self.den = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, in_ch, 7, padding=3))
        self.proj = nn.Linear(patch * in_ch, d_model)
        self.pos = _PosEnc(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2,
                                         dropout=0.1, batch_first=True,
                                         norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, num_classes))

    def forward(self, x):
        z = x.permute(0, 2, 1)
        z = (z + self.den(z)).permute(0, 2, 1)         # residual denoising
        B, N, C = z.shape
        n_p = N // self.patch
        t = z[:, :n_p * self.patch].reshape(B, n_p, self.patch * C)
        return self.head(self.tr(self.pos(self.proj(t))).mean(dim=1))


class TPRIME_S9(nn.Module):
    """[S9] Belgiovine et al., IEEE INFOCOM 2024. Compact transformer over
    short I/Q slices, sized for edge deployment."""

    def __init__(self, num_classes=2, seg=24, d_model=64, nhead=8, layers=2,
                 in_ch=2):
        super().__init__()
        self.seg = seg
        self.proj = nn.Linear(seg * in_ch, d_model)
        self.pos = _PosEnc(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4,
                                         dropout=0.1, batch_first=True,
                                         norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, N, C = x.shape
        n = N // self.seg
        t = x[:, :n * self.seg].reshape(B, n, self.seg * C)
        if n > 512:                                    # keep attention tractable
            t = torch.nn.functional.adaptive_avg_pool1d(
                t.permute(0, 2, 1), 512).permute(0, 2, 1)
        return self.head(self.tr(self.pos(self.proj(t))).mean(dim=1))


# ==========================================================================
#            PROPOSED DETECTOR: OSCAR-Net  (paper Sec. V, Algorithm 1)
# ==========================================================================
#  OSCAR-Net = Order-Statistic CFAR-Aware Residual network.
#   1. multi-scale log-power envelope e_w, w in {128, 512, 2048} samples
#   2. learned noise floor f_w = softmax-weighted quantiles
#      {Q0.05, Q0.10, Q0.25, Q0.50} of e_w (initialised on Q0.10)
#   3. soft duty-cycle spectrum d_{w,k} = mean_n sigmoid((r_w - tau_k)/T_k),
#      margins tau initialised at 3/6/9/12 dB; descriptor s_w in R^8
#   4. shared dilated conv encoder on the residual r_w = e_w - f_w, pooled
#      by mean and max -> g_w in R^64; z in R^216 -> MLP 216 -> 64 -> 2
#  20,646 parameters, 4.7 MFLOPs per frame, GroupNorm only (the same network
#  centrally and federated), invariant to the receiver gain (Prop. 1).
# ==========================================================================
def _log_envelope(x: torch.Tensor, win: int) -> torch.Tensor:
    """(B, L, 2) I/Q -> (B, L // win) log-power envelope in dB."""
    B, L, _ = x.shape
    n = L // win
    p = (x[:, :n * win] ** 2).sum(-1).reshape(B, n, win).mean(-1)
    return 10.0 * torch.log10(p.clamp_min(1e-10))


class FloorRelativeEnvelope(nn.Module):
    """
    Differentiable order-statistic CFAR front end of OSCAR-Net.

    For each window length w it forms the log-power envelope e_w[n] and
    estimates the frame's own noise floor as a LEARNED convex combination of
    fixed order statistics {Q_0.05, Q_0.10, Q_0.25, Q_0.50} (initialised to
    sit on Q_0.10, the labelling floor). From the floor-relative excess
    x_w[n] = e_w[n] - floor_w it returns
      - the excess sequence itself (for a shape encoder downstream), and
      - per scale: a SOFT DUTY-CYCLE SPECTRUM, mean_n sigmoid((x_w[n]-tau_k)/T_k)
        for K learnable margins tau_k (initialised 3, 6, 9, 12 dB around
        eta = 6 dB) with learnable temperatures; the 50/90/99th percentiles of
        the excess; and the mean-to-floor gap. dB quantities are divided by 10.
    Everything is differentiable (torch.quantile back-propagates through the
    sort), so the floor estimator and the margins are trained end to end.
    Per-frame RMS normalisation shifts e_w by a constant, which the floor
    subtraction removes exactly: the features are scale-free by construction.
    """
    QS = (0.05, 0.10, 0.25, 0.50)
    QX = (0.50, 0.90, 0.99)

    def __init__(self, scales=(128, 512, 2048), taus=(3.0, 6.0, 9.0, 12.0)):
        super().__init__()
        self.scales = tuple(scales)
        S, K = len(self.scales), len(taus)
        init = torch.full((S, len(self.QS)), -2.0)
        init[:, 1] = 2.0                                  # start near Q_0.10
        self.floor_logits = nn.Parameter(init)
        self.tau = nn.Parameter(torch.tensor(taus).repeat(S, 1))
        self.log_temp = nn.Parameter(torch.zeros(S, K))    # T = 1 dB at init
        self.register_buffer("q_floor", torch.tensor(self.QS))
        self.register_buffer("q_ex", torch.tensor(self.QX))
        self.n_stats_per_scale = K + len(self.QX) + 1
        self._xflops = 0.0

    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape
        excess, stats, xf = [], [], 0.0
        for s, w0 in enumerate(self.scales):
            w = max(8, min(w0, L // 16))                  # >= 16 windows
            e = _log_envelope(x, w)                        # (B, N)
            n = e.shape[1]
            qs = torch.quantile(e, self.q_floor, dim=1).T  # (B, 4)
            floor = (qs * torch.softmax(self.floor_logits[s], 0)).sum(
                1, keepdim=True)
            ex = e - floor                                  # (B, N) dB
            T = self.log_temp[s].exp().clamp(0.1, 10.0)
            duty = torch.sigmoid((ex.unsqueeze(-1) - self.tau[s]) / T).mean(1)
            exq = torch.quantile(ex, self.q_ex, dim=1).T / 10.0
            gap = (e.mean(1, keepdim=True) - floor) / 10.0
            stats.append(torch.cat([duty, exq, gap], 1))
            excess.append(ex)
            K = self.tau.shape[1]
            xf += 3 * L + 3 * n + 2 * n * max(1.0, math.log2(n)) + 4 * n * K
        self._xflops = xf                                  # per frame
        return excess, torch.cat(stats, 1)


class _GNAct(nn.Sequential):
    """Conv1d -> GroupNorm -> GELU."""

    def __init__(self, cin, cout, k, stride=1, dilation=1, groups=1):
        pad = dilation * (k - 1) // 2
        g = 8 if cout % 8 == 0 else (4 if cout % 4 == 0 else 1)
        super().__init__(
            nn.Conv1d(cin, cout, k, stride=stride, padding=pad,
                      dilation=dilation, groups=groups),
            nn.GroupNorm(g, cout), nn.GELU())


class OSCARNet_P1(nn.Module):
    """
    [P1] OSCAR-Net: Order-Statistic CFAR-Aware Residual network (proposed).

    Learnable OS-CFAR front end (FloorRelativeEnvelope at 5/20/80 us) feeding
    (a) the soft duty-cycle spectra and excess quantiles directly to the head
    and (b) one weight-shared dilated conv encoder over each scale's
    floor-relative excess sequence, pooled by mean AND max so a sparse burst
    is not averaged away. 20,646 parameters and 4.7 MFLOPs per frame; every
    feature is floor-relative, so the output is invariant to the receiver
    gain (paper Proposition 1).
    """

    def __init__(self, num_classes=2, scales=(128, 512, 2048), ch=32):
        super().__init__()
        self.fre = FloorRelativeEnvelope(scales)
        self.enc = nn.Sequential(_GNAct(1, ch, 5), _GNAct(ch, ch, 3, dilation=2),
                                 _GNAct(ch, ch, 3, dilation=4))
        d_in = len(scales) * (self.fre.n_stats_per_scale + 2 * ch)
        self.head = nn.Sequential(nn.Linear(d_in, 64), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(64, num_classes))

    def forward(self, x):
        excess, stats = self.fre(x)
        feats = [stats]
        for ex in excess:
            h = self.enc(ex.unsqueeze(1) / 10.0)
            feats += [h.mean(-1), h.amax(-1)]
        return self.head(torch.cat(feats, 1))


PROPOSED_MODELS = {"oscar_net_p1"}


def build_model(name: str, cfg: Dict) -> nn.Module:
    """OSCAR-Net (proposed) and the five SOTA detectors compared in the paper."""
    if name == "oscar_net_p1":
        return OSCARNet_P1()
    if name == "deepsense_s4":
        return DeepSense_S4()
    if name == "spectrum_transformer_s5":
        return SpectrumTransformer_S5()
    if name == "cnn_transformer_s6":
        return CNNTransformer_S6()
    if name == "denoise_transformer_s7":
        return DenoiseTransformer_S7()
    if name == "tprime_s9":
        return TPRIME_S9()
    raise ValueError(f"unknown model {name}")


IS_2D: set = set()          # no spectrogram (2-D) detector in the paper
# the detector proposed in the paper; every other model is a baseline
NOVEL_DETECTORS = ("oscar_net_p1",)
ALL_STAGES = ("main", "clients", "lobo", "labels", "handoff")
DEFAULT_STAGES = ("main", "clients", "handoff")   # what the paper needs
STAGES = set(DEFAULT_STAGES)
ARM_DATA: Dict[str, Dict] = {}


def convert_bn_to_gn(module: nn.Module, max_groups: int = 32) -> nn.Module:
    """
    Replace every BatchNorm with GroupNorm, in place.

    WHY THIS EXISTS
      FedAvg averages BatchNorm's running_mean / running_var across clients.
      Those statistics describe each client's own input distribution, and here
      the clients genuinely differ - different phases of the evening see
      different noise floors and different occupancy. Averaging their BN statistics produces a normaliser that fits
      no client, and the global model sits at chance: measured at val loss
      0.69 (= ln 2) and AUC 0.18 with BatchNorm, versus AUC 0.92 for the same
      architecture trained centrally.

      GroupNorm normalises within each sample, so it carries no cross-client
      running statistics and nothing incompatible gets averaged. This is the
      standard remedy in the non-IID federated literature (Hsieh et al., "The
      Non-IID Data Quagmire of Decentralized Machine Learning", ICML 2020).

      The centralized arm keeps BatchNorm, so if you compare the two arms
      state that the normalisation layer differs - or set fl_norm="batchnorm"
      to compare like with like and report the collapse as a finding.
    """
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


# ==========================================================================
#     COMPLEXITY PROFILING  (methodology of EfficiencyReport.py, verbatim)
# ==========================================================================
#  BEST / AVERAGE / WORST MEANS DEPLOYMENT SCENARIO, NOT TIMING JITTER.
#
#  EfficiencyReport.py defines the three cases by the routing fraction p of a
#  cascade: Best p->0, Average p=p_G, Worst p->1. It then states, and this is
#  the part that governs everything below:
#
#      "STANDALONE model -> Best = Average = Worst for ALL SIX metrics.
#       A single network runs the same ops on every frame [...] The three
#       columns are emitted (you asked for them) and they are identical BY
#       CONSTRUCTION. Do not present this as if it were measured variation."
#
#  Every model here is standalone: no gate, no routing, no
#  data-dependent control flow. So for every one of them B = A = W, exactly,
#  and each row carries a `baw_identical` flag so it cannot ship by accident.
#
#  My earlier version reported min / mean / max of the timing samples in the
#  Best / Average / Worst columns. Under your own methodology that is wrong -
#  bench_latency() calls that spread "MEASUREMENT DISPERSION on this machine",
#  and says it "is NOT the Best/Average/Worst routing scenario, and the two
#  must never be conflated." The dispersion is still reported, in separate
#  columns named meas_lat_*, where it belongs.
#
#  Other points taken from EfficiencyReport.py rather than reinvented:
#    * CPU-only by design, so numbers are comparable across machines
#    * latency is AMORTISED: one forward over PROF_BATCH, divided by the batch
#    * batch-1 latency measured separately (deployment per-frame cost)
#    * MACs and FLOPs are distinct: FLOPs = 2*MACs + elementwise
#    * BatchNorm, biases and RNN elementwise ops are counted
#    * activation bytes summed over every leaf module, not just the peak
#    * torch.inference_mode(), not no_grad()
#    * random tensors of the right shape are timing-equivalent to real IQ
# ==========================================================================
MB = 1e6


def param_stats(model: nn.Module) -> Tuple[int, int, int]:
    n_params = sum(p.numel() for p in model.parameters())
    pbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    bbytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return n_params, pbytes, bbytes


def count_macs(model: nn.Module, sample: torch.Tensor
               ) -> Tuple[float, float, float]:
    """Exact per-frame MACs / FLOPs / activation bytes, hook based, batch = 1."""
    store = {"macs": 0.0, "ew": 0.0, "act_bytes": 0.0}
    handles = []

    def lin(m, i, o):
        store["macs"] += o.numel() * m.in_features
        if m.bias is not None:
            store["ew"] += o.numel()

    def conv(m, i, o):
        k = 1
        for kk in m.kernel_size:
            k *= kk
        store["macs"] += o.numel() * (m.in_channels // m.groups) * k
        if m.bias is not None:
            store["ew"] += o.numel()

    def norm(m, i, o):
        n = i[0].numel()
        store["macs"] += n
        store["ew"] += n

    def rnn(gates):
        def hook(m, i, o):
            x = i[0]
            B, T = x.shape[0], x.shape[1]
            H = m.hidden_size
            nd = 2 if m.bidirectional else 1
            for layer in range(m.num_layers):
                in_dim = m.input_size if layer == 0 else H * nd
                store["macs"] += B * T * nd * gates * (H * in_dim + H * H)
                store["ew"] += B * T * nd * (gates + 3) * H
        return hook

    def mha(m, i, o):
        # nn.MultiheadAttention applies its Q/K/V/out projections through
        # F.multi_head_attention_forward on raw weight tensors, so NO Linear
        # hook ever fires for them, and the QK^T / AV products are not
        # modules at all. Without this hook every transformer (SpecFormer,
        # CNN-Former, DnFormer, T-PRIME) was under-counted.
        q, k = i[0], i[1]
        if m.batch_first:
            B, Tq, Tk = q.shape[0], q.shape[1], k.shape[1]
        else:
            B, Tq, Tk = q.shape[1], q.shape[0], k.shape[0]
        E = m.embed_dim
        store["macs"] += B * (Tq * E * E + 2 * Tk * E * E    # Q, K, V proj
                              + 2 * Tq * Tk * E              # QK^T and AV
                              + Tq * E * E)                  # out proj
        store["ew"] += B * m.num_heads * Tq * Tk * 3         # scale+softmax

    def act(m, i, o):
        def walk(t):
            if torch.is_tensor(t):
                store["act_bytes"] += t.numel() * t.element_size()
            elif isinstance(t, (list, tuple)):
                for e in t:
                    walk(e)
        walk(o)

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(lin))
        elif isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            handles.append(mod.register_forward_hook(conv))
        elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            handles.append(mod.register_forward_hook(norm))
        elif isinstance(mod, nn.LSTM):
            handles.append(mod.register_forward_hook(rnn(4)))
        elif isinstance(mod, nn.GRU):
            handles.append(mod.register_forward_hook(rnn(3)))
        elif isinstance(mod, nn.MultiheadAttention):
            handles.append(mod.register_forward_hook(mha))
        if len(list(mod.children())) == 0:
            handles.append(mod.register_forward_hook(act))

    # The fused inference fast path of nn.TransformerEncoderLayer bypasses
    # submodule calls, so hooks would silently see nothing. Disable it while
    # counting (no effect on training or on latency measurement).
    fp = None
    try:
        fp = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
    except AttributeError:
        pass
    model.eval()
    with torch.no_grad():
        model(sample[:1])
    if fp is not None:
        torch.backends.mha.set_fastpath_enabled(fp)
    for h in handles:
        h.remove()
    # Non-module tensor ops (FFT, sort/quantile, envelope) that a model
    # reports for itself in a `_xflops` attribute (proposed models only).
    store["ew"] += sum(float(getattr(m, "_xflops", 0.0)) for m in model.modules())
    return store["macs"], 2.0 * store["macs"] + store["ew"], store["act_bytes"]


def bench_latency(model: nn.Module, shape: Tuple[int, ...], batch: int,
                  warmup: int, iters: int) -> Dict[str, float]:
    """
    One forward over the whole batch; per-frame latency is the amortised cost.

    The min/mean/max over timed iterations is MEASUREMENT DISPERSION on this
    machine. It is NOT the Best/Average/Worst deployment scenario, and the two
    must never be conflated - hence the meas_ prefix on those fields.
    """
    import statistics
    model.eval()
    inp = torch.randn(batch, *shape)
    with torch.inference_mode():
        for _ in range(warmup):
            model(inp)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(inp)
            times.append(time.perf_counter() - t0)
    mean_b = statistics.mean(times)
    return {
        "mean_batch_ms": mean_b * 1e3,
        "std_batch_ms": (statistics.pstdev(times) if len(times) > 1 else 0.0) * 1e3,
        "med_batch_ms": statistics.median(times) * 1e3,
        "per_sample_ms": mean_b / batch * 1e3,
        "meas_lat_min_ms": min(times) / batch * 1e3,
        "meas_lat_max_ms": max(times) / batch * 1e3,
        "throughput_sps": batch / mean_b,
    }


def profile_model(model: nn.Module, sample_shape: Tuple[int, ...],
                  name: str, cfg: Dict) -> Dict:
    """Profile one standalone network. Best = Average = Worst, by construction."""
    dev = torch.device("cpu")          # CPU-only by design, as in EfficiencyReport
    model = model.to(dev).eval()

    n_par, pbytes, bbytes = param_stats(model)
    macs, flops, act_bytes = count_macs(model, torch.randn(1, *sample_shape))
    lat = bench_latency(model, sample_shape, cfg["prof_batch"],
                        cfg["prof_warmup"], cfg["prof_iters"])
    b1 = (bench_latency(model, sample_shape, 1, 3, cfg["prof_b1_iters"])
          if cfg["prof_measure_b1"] else None)

    per_sample = lat["per_sample_ms"]
    thr = lat["throughput_sps"]
    row = {
        "model": name,
        "role": "standalone",
        "params": n_par,
        "params_M": n_par / 1e6,
        "model_mem_MB": (pbytes + bbytes) / MB,
        "weight_MB": pbytes / MB,
        "buffer_MB": bbytes / MB,
        "activation_MB": act_bytes / MB,
        "macs_per_frame": macs,
        "macs_M": macs / 1e6,
        "flops_per_frame": flops,
        "flops_M": flops / 1e6,

        # ---- Best / Average / Worst: DEPLOYMENT SCENARIO ----
        # Standalone network -> no routing -> identical by construction.
        "params_best": n_par, "params_avg": n_par, "params_worst": n_par,
        "mem_MB_best": (pbytes + bbytes) / MB,
        "mem_MB_avg": (pbytes + bbytes) / MB,
        "mem_MB_worst": (pbytes + bbytes) / MB,
        "macs_M_best": macs / 1e6, "macs_M_avg": macs / 1e6,
        "macs_M_worst": macs / 1e6,
        "flops_M_best": flops / 1e6, "flops_M_avg": flops / 1e6,
        "flops_M_worst": flops / 1e6,
        "latency_ms_best": per_sample, "latency_ms_avg": per_sample,
        "latency_ms_worst": per_sample,
        "throughput_best": thr, "throughput_avg": thr, "throughput_worst": thr,
        "baw_identical": True,
        "baw_note": "standalone: no routing, B=A=W by construction",

        # ---- measurement dispersion, NOT the B/A/W scenario ----
        "meas_lat_min_ms": lat["meas_lat_min_ms"],
        "meas_lat_mean_ms": per_sample,
        "meas_lat_max_ms": lat["meas_lat_max_ms"],
        "meas_batch_mean_ms": lat["mean_batch_ms"],
        "meas_batch_med_ms": lat["med_batch_ms"],
        "meas_batch_std_ms": lat["std_batch_ms"],
        "batch1_latency_ms": b1["mean_batch_ms"] if b1 else float("nan"),
        "batch1_min_ms": b1["meas_lat_min_ms"] if b1 else float("nan"),
        "batch1_max_ms": b1["meas_lat_max_ms"] if b1 else float("nan"),

        "prof_batch": cfg["prof_batch"],
        "prof_iters": cfg["prof_iters"],
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
    }
    return row


# ==========================================================================
#                                METRICS
# ==========================================================================
def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   y_prob1: np.ndarray) -> Dict[str, float]:
    """All eight requested metrics for one subset."""
    if y_true.size == 0:
        return {}
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = (tp + tn) / y_true.size
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)          # Pd
    f1 = f1_score(y_true, y_pred, zero_division=0)

    mdr = fn / (fn + tp) if (fn + tp) > 0 else 0.0               # missed det.
    fdr = fp / (fp + tn) if (fp + tn) > 0 else 0.0               # false alarm

    mae = float(np.mean(np.abs(y_true - y_prob1)))
    # NMSE normalises by the energy of the target vector. On an all-H0 subset
    # that energy is zero and the ratio is meaningless (it explodes to ~1e12
    # rather than failing), so report nan instead of a fabricated number.
    denom = float(np.sum(y_true.astype(np.float64) ** 2))
    nmse = (float(np.sum((y_true - y_prob1) ** 2) / denom)
            if denom > 0 else float("nan"))

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob1)
        a = float(auc(fpr, tpr))
    else:
        a = float("nan")     # AUC undefined on a single-class subset

    return {"n": int(y_true.size), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": acc, "precision": prec, "recall_pd": rec, "f1": f1,
            "mdr": mdr, "fdr": fdr, "nmse": nmse, "mae": mae, "auc": a}


def _scores(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, ...]:
    model.eval()
    P, Y, S = [], [], []
    with torch.no_grad():
        for x, yb, sb in loader:
            P.append(torch.softmax(model(x.to(DEVICE)), 1)[:, 1].cpu().numpy())
            Y.append(yb.numpy())
            S.append(sb.numpy())
    return np.concatenate(P), np.concatenate(Y), np.concatenate(S)


def pick_threshold(model: nn.Module, val_loader: DataLoader, cfg: Dict) -> float:
    """
    Choose the decision threshold on VALIDATION only, never on test.

    target_pfa: the Neyman-Pearson operating point - the smallest threshold
    whose validation false-alarm rate does not exceed target_pfa. This is how
    spectrum sensing is actually specified (IEEE 802.22 fixes Pfa at 0.1), and
    it makes Pd across models directly comparable because every model is held
    to the same false-alarm budget.
    """
    mode = cfg.get("threshold_mode", "target_pfa")
    if mode == "fixed":
        return 0.5
    p1, y, _ = _scores(model, val_loader)
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p1)
    if mode == "youden":
        return float(thr[int(np.argmax(tpr - fpr))])
    # roc_curve prepends an infinite sentinel threshold; drop non-finite
    # entries or the "no false alarms" point returns inf and classifies
    # everything as H0.
    fin = np.isfinite(thr)
    fpr, tpr, thr = fpr[fin], tpr[fin], thr[fin]
    if thr.size == 0:
        return 0.5
    ok = np.flatnonzero(fpr <= cfg.get("target_pfa", 0.10))
    return float(thr[ok[-1]]) if ok.size else float(thr[0])


def evaluate(model: nn.Module, loader: DataLoader, cfg: Dict,
             threshold: float = 0.5) -> Dict:
    p1, y, s = _scores(model, loader)
    pred = (p1 >= threshold).astype(np.int64)

    overall = binary_metrics(y, pred, p1)
    overall["threshold"] = float(threshold)
    # the naive cut, reported alongside so the calibration effect is visible
    naive = binary_metrics(y, (p1 >= 0.5).astype(np.int64), p1)
    overall["accuracy_at_0.5"] = naive["accuracy"]
    overall["pd_at_0.5"] = naive["recall_pd"]
    # A detector that always answers H0 scores this. Any accuracy at or below
    # it is worthless no matter how it looks in isolation, and on an 85/15
    # split that bar is 0.85 - higher than several models scored.
    maj = float(max((y == 0).mean(), (y == 1).mean()))
    overall["majority_baseline_acc"] = maj
    overall["acc_minus_baseline"] = overall["accuracy"] - maj
    overall["beats_baseline"] = bool(overall["accuracy"] > maj)
    overall["balanced_accuracy"] = 0.5 * (overall["recall_pd"] +
                                          (1.0 - overall["fdr"]))

    # ---------------- per-SNR curves, done correctly ----------------
    # THE BUG THIS REPLACES. The previous version binned ALL frames by
    # estimated SNR and computed accuracy/AUC inside each bin. Because the
    # labels come from the same energy statistic as the axis (H1 needs
    # snr >= label_hi_db, H0 needs snr <= label_lo_db), every bin held exactly
    # ONE class: AUC was nan in every bin, and "accuracy" was just Pd on the
    # high bins and 1-Pfa on the low bins wearing an SNR label.
    #
    # The classical formulation has no such problem, because Pd and Pfa live
    # on different hypotheses:
    #     Pd  is a property of H1 and IS a function of SNR
    #     Pfa is a property of H0 (noise only) and has NO SNR axis
    # So each bin takes the H1 frames at that SNR and scores them against the
    # ENTIRE H0 pool. AUC is then well defined, Pfa is constant across bins by
    # construction, and the curve is the Pd-vs-SNR curve the sensing
    # literature actually plots.
    per_snr = {}
    h0_mask = (y == 0)
    p1_h0 = p1[h0_mask]
    pred_h0 = pred[h0_mask]
    n_h0 = int(h0_mask.sum())
    global_fdr = float(pred_h0.mean()) if n_h0 else float("nan")

    edges = cfg["snr_bins_db"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi) & (y == 1)
        n_h1 = int(m.sum())
        if n_h1 < 5:
            continue
        yy = np.concatenate([np.ones(n_h1, dtype=np.int64),
                             np.zeros(n_h0, dtype=np.int64)])
        pp = np.concatenate([p1[m], p1_h0])
        dd = np.concatenate([pred[m], pred_h0])
        r = binary_metrics(yy, dd, pp)
        r.update({"snr_lo_db": lo, "snr_hi_db": hi,
                  "snr_mid_db": 0.5 * (lo + hi),
                  "n_h1_in_bin": n_h1, "n_h0_pool": n_h0,
                  "pd": float(dd[:n_h1].mean()),
                  "mdr_at_snr": 1.0 - float(dd[:n_h1].mean()),
                  "fdr_global": global_fdr})
        per_snr[f"{lo}_{hi}"] = r

    return {"overall": overall, "per_snr": per_snr, "threshold": threshold,
            "y": y, "p1": p1, "snr": s}


# ==========================================================================
#                       OVERFITTING DIAGNOSTICS
# ==========================================================================
#  A common federated-tuning rule scores a configuration as
#      score = val + FL_TUNE_OVERFIT_LAMBDA * max(0, val - train)
#  i.e. it penalises the generalisation gap explicitly rather than trusting
#  validation loss alone, and a typical training script logs train and
#  val loss side by side every epoch for the same reason. Both are reproduced
#  here, plus a train/val/test accuracy comparison, because on THIS dataset
#  overfitting has a specific shape worth catching: only ~600 labelled frames
#  survive the dead-zone filter, so a 0.5 M-parameter network can memorise the
#  training split outright. Training loss reaching ~0.001 while validation sits
#  at 0.14 (observed here) is exactly that.
# ==========================================================================
OVERFIT_LAMBDA = 0.5        # weight of the generalisation-gap penalty


@torch.no_grad()
def _loss_acc(model: nn.Module, loader: DataLoader, crit
              ) -> Tuple[float, float, float, float]:
    """Returns (loss, raw accuracy, balanced accuracy, majority baseline)."""
    model.eval()
    tot, ls, correct = 0, 0.0, 0
    tp = tn = fp = fn = 0
    for x, yb, _ in loader:
        x, yb = x.to(DEVICE), yb.to(DEVICE)
        out = model(x)
        ls += crit(out, yb).item() * x.size(0)
        pr = out.argmax(1)
        correct += (pr == yb).sum().item()
        tp += int(((pr == 1) & (yb == 1)).sum())
        tn += int(((pr == 0) & (yb == 0)).sum())
        fp += int(((pr == 1) & (yb == 0)).sum())
        fn += int(((pr == 0) & (yb == 1)).sum())
        tot += x.size(0)
    n = max(tot, 1)
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    base = max(tp + fn, tn + fp) / n
    return ls / n, correct / n, 0.5 * (tpr + tnr), base


def overfit_report(model: nn.Module, tr_loader, va_loader, te_loader,
                   crit, history: List[Dict]) -> Dict:
    """
    Generalisation diagnostics, measured on BALANCED accuracy.

    WHY NOT RAW ACCURACY. The chronological split leaves the three subsets
    with different class priors - measured on this dataset: train+val is
    10.2% H1 (majority baseline 0.8976) while test is 15.0% H1 (baseline
    0.8502). Raw accuracy is not comparable across sets with different
    priors, so a naive train-minus-test gap already contains 0.0475 of pure
    prior shift before any model has done anything. On a reported gap of
    0.19 that is a quarter of the number, and it was being called
    overfitting.

    Balanced accuracy = (TPR + TNR) / 2 is invariant to the class prior, so
    the gap it produces is attributable to the model. Raw accuracy and the
    per-split baseline are still reported, because seeing accuracy sit BELOW
    its own split's baseline is the fastest way to spot a model that has
    learned nothing.
    """
    tr_l, tr_a, tr_b, tr_base = _loss_acc(model, tr_loader, crit)
    va_l, va_a, va_b, va_base = _loss_acc(model, va_loader, crit)
    te_l, te_a, te_b, te_base = _loss_acc(model, te_loader, crit)

    loss_gap = va_l - tr_l
    bal_gap = tr_b - va_b                 # the honest generalisation gap
    raw_gap = tr_a - va_a                 # contaminated by the prior shift
    prior_artifact = tr_base - va_base
    score = va_l + OVERFIT_LAMBDA * max(0.0, loss_gap)

    diverged = False
    if len(history) >= 6:
        h = history[-6:]
        tr_t = h[-1].get("train_loss", float("nan")) - h[0].get("train_loss",
                                                               float("nan"))
        va_t = h[-1]["val_loss"] - h[0]["val_loss"]
        diverged = bool(np.isfinite(tr_t) and tr_t < -1e-4 and va_t > 1e-4)

    # ---- verdict, on balanced accuracy ----
    if te_b < 0.55:
        verdict = ("NOT LEARNING (balanced acc ~ chance) - the model is not "
                   "solving the task at all; overfitting is not the issue")
    elif te_b > 0.95:
        verdict = ("SUSPICIOUSLY PERFECT - check the band confound before "
                   "believing this (see the per-band table)")
    elif te_a < te_base:
        verdict = (f"BELOW BASELINE (acc {te_a:.3f} < majority {te_base:.3f}) "
                   f"- balanced acc {te_b:.3f} is the metric to quote")
    elif bal_gap > 0.20:
        verdict = "SEVERE overfitting (balanced-accuracy gap)"
    elif bal_gap > 0.10:
        verdict = "moderate overfitting (balanced-accuracy gap)"
    elif diverged:
        verdict = "diverging (val loss rising while train falls)"
    else:
        verdict = "acceptable"

    best_ep = (int(np.argmin([h["val_loss"] for h in history]))
               if history else -1)
    return {
        "train_loss": tr_l, "val_loss": va_l, "test_loss": te_l,
        "train_acc": tr_a, "val_acc": va_a, "test_acc": te_a,
        "train_bal_acc": tr_b, "val_bal_acc": va_b, "test_bal_acc": te_b,
        "train_baseline": tr_base, "val_baseline": va_base,
        "test_baseline": te_base,
        "bal_acc_gap_train_minus_val": bal_gap,
        "raw_acc_gap_train_minus_val": raw_gap,
        "prior_shift_artifact": prior_artifact,
        "acc_gap_train_minus_val": bal_gap,   # kept for CSV compatibility
        "loss_gap_val_minus_train": loss_gap,
        "test_acc_minus_baseline": te_a - te_base,
        "overfit_score": score,
        "val_loss_diverging": diverged,
        "best_epoch_or_round": best_ep,
        "verdict": verdict,
    }


# ==========================================================================
#                          TRAINING (CENTRALIZED)
# ==========================================================================
def class_weights(y: np.ndarray) -> torch.Tensor:
    n0 = max(1, int((y == 0).sum()))
    n1 = max(1, int((y == 1).sum()))
    t = n0 + n1
    return torch.tensor([t / (2 * n0), t / (2 * n1)], dtype=torch.float32)


def train_centralized(model, tr_loader, va_loader, cfg, w) -> Tuple[nn.Module, Dict]:
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                           weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min", factor=cfg["lr_scheduler_factor"],
        patience=cfg["lr_scheduler_patience"], min_lr=cfg["min_lr"])
    crit = nn.CrossEntropyLoss(weight=w.to(DEVICE),
                               label_smoothing=cfg.get("label_smoothing", 0.0))

    best, best_state, patience = float("inf"), None, 0
    hist = []
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        tot, run = 0, 0.0
        for x, yb, _ in tr_loader:
            x, yb = x.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            run += loss.item() * x.size(0)
            tot += x.size(0)
        tr_loss = run / max(tot, 1)

        model.eval()
        vs, vt = 0.0, 0
        with torch.no_grad():
            for x, yb, _ in va_loader:
                x, yb = x.to(DEVICE), yb.to(DEVICE)
                vs += crit(model(x), yb).item() * x.size(0)
                vt += x.size(0)
        va_loss = vs / max(vt, 1)
        sched.step(va_loss)
        hist.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})

        if va_loss < best - 1e-5:
            best, patience = va_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if (ep + 1) >= cfg["min_epochs"] \
                    and patience >= cfg["early_stopping_patience"]:
                log(f"      early stop @ epoch {ep+1} (val {best:.4f})")
                break
        if (ep + 1) % 5 == 0:
            log(f"      ep {ep+1:3d}  train {tr_loss:.4f}  val {va_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": best, "epochs_run": len(hist),
                   "train_time_s": time.time() - t0, "history": hist}


# ==========================================================================
#     FEDERATED  (FedAvg)
# ==========================================================================
def _merge_until_balanced(blocks: List[np.ndarray], y: np.ndarray,
                          min_per_class: int = 8
                          ) -> Tuple[List[np.ndarray], int]:
    """
    Merge adjacent time blocks until every client holds both classes.

    Needed because occupancy is NON-STATIONARY across an event: the H1 rate
    climbs as the crowd arrives, so the earliest contiguous blocks can contain
    almost no positives. FedAvg over a client with zero positives is
    destructive - the local model learns "always H0" and drags the global
    average with it (measured earlier at AUC 0.28, worse than chance).

    Merging ADJACENT blocks preserves temporal locality, so a client is still
    "this receiver during this stretch of the evening" rather than a random
    shuffle. That keeps the federation physically meaningful while making
    every client trainable.
    """
    merged = [b for b in blocks if len(b)]
    n_merges = 0
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, b in enumerate(merged):
            yy = y[b]
            if int((yy == 1).sum()) < min_per_class or \
                    int((yy == 0).sum()) < min_per_class:
                j = i + 1 if i + 1 < len(merged) else i - 1
                merged[min(i, j)] = np.concatenate([merged[i], merged[j]])
                del merged[max(i, j)]
                n_merges += 1
                changed = True
                break
    return merged, n_merges


def partition_clients(d: Dict, idx: np.ndarray, cfg: Dict
                      ) -> Tuple[List[np.ndarray], List[str]]:
    """
    Split TRAIN indices into DISJOINT client shards.

    MODES
      "natural"  adapts to the data:
                   * 2+ receivers -> one client per (device, time-block).
                     Heterogeneity is real hardware plus real time.
                   * 1 receiver   -> contiguous TIME blocks of that receiver,
                     then adjacent blocks are merged until each holds both
                     classes. This is the single-sensor temporal federation:
                     each client is the same radio over a different stretch
                     of the evening, with genuinely different occupancy
                     statistics because the crowd builds and disperses.
      "device"       one client per receiver (degenerate with one radio).
      "device_band"  one client per (device, band). AVOID on this dataset:
                     a DFS block is quiet all evening and 2.4 GHz ch 1 is
                     busy all evening, so most clients end up single-class.
      "dirichlet"    synthetic Dirichlet label skew (control).
      "iid"          control.
    """
    rng = np.random.default_rng(cfg["seed"])
    mode = cfg["client_partition"]
    y = d["y"]
    shards: List[np.ndarray] = []
    names: List[str] = []

    n_dev = len(set(d["device"][idx].tolist()))

    if mode in ("natural", "device", "device_band"):
        if mode == "device":
            key = d["device"][idx].astype(object)
        elif mode == "device_band":
            key = np.array([f"{a}|{b}" for a, b in
                            zip(d["device"][idx], d["band"][idx])], dtype=object)
        elif n_dev == 1:
            # ---- SINGLE RECEIVER: contiguous time blocks, then balance ----
            nb = int(cfg.get("n_time_blocks_single_device", 8))
            if "_split_block" in d:
                # v3: a client = a contiguous group of whole split blocks, so
                # every client owns train, val AND test frames (make_splits)
                sb = d["_split_block"][idx]
                ub = np.unique(sb)
                groups = [g for g in np.array_split(ub, min(nb, len(ub)))
                          if len(g)]
                blocks = [idx[np.isin(sb, g)] for g in groups]
                blocks = [b[np.argsort(d["utc"][b])] for b in blocks if len(b)]
            else:
                order = idx[np.argsort(d["utc"][idx])]
                blocks = [b for b in np.array_split(order, nb) if len(b)]
            merged, n_m = _merge_until_balanced(
                blocks, y, int(cfg.get("min_per_class_per_client", 8)))
            if n_m:
                log(f"      merged {n_m} adjacent time block(s) to remove "
                    f"single-class clients ({nb} -> {len(merged)})")
            shards = merged
            names = [f"{d['device'][idx][0]}|T{i}" for i in range(len(merged))]
            key = None
        else:
            # several receivers: (device, time block). In the client-COUNT
            # sweep the C clients are shared out among the radios
            # (_ablation_clients); the main run keeps n_time_blocks per radio.
            devs = sorted(set(d["device"][idx].tolist()))
            c_tot = int(cfg.get("_ablation_clients") or 0)
            key = np.empty(len(idx), dtype=object)
            for di, dev in enumerate(devs):
                nb = (max(1, c_tot // len(devs) + (1 if di < c_tot % len(devs) else 0))
                      if c_tot else max(1, int(cfg.get("n_time_blocks", 3))))
                m = np.flatnonzero(d["device"][idx] == dev)
                order = m[np.argsort(d["utc"][idx][m])]
                for bi, part in enumerate(np.array_split(order, nb)):
                    key[part] = f"{dev}|T{bi:02d}"

        if key is not None:
            for k in sorted(set(key.tolist())):
                sel = idx[key == k]
                if len(sel) >= 8:
                    shards.append(sel)
                    names.append(str(k))

        if len(shards) < 2:
            log("      partition gave <2 usable clients; using dirichlet")
            return partition_clients(
                d, idx, dict(cfg, client_partition="dirichlet"))

    else:
        n_cl = cfg["n_clients"]
        buckets: List[List[int]] = [[] for _ in range(n_cl)]
        if mode == "dirichlet":
            for lbl in (0, 1):
                sub = idx[y[idx] == lbl]
                rng.shuffle(sub)
                props = rng.dirichlet([cfg["dirichlet_alpha"]] * n_cl)
                cuts = (np.cumsum(props) * len(sub)).astype(int)[:-1]
                for c, part in enumerate(np.split(sub, cuts)):
                    buckets[c].extend(part.tolist())
        else:
            sub = idx.copy()
            rng.shuffle(sub)
            for c, part in enumerate(np.array_split(sub, n_cl)):
                buckets[c].extend(part.tolist())
        for c in range(n_cl):
            if not buckets[c]:
                donor = max(range(n_cl), key=lambda k: len(buckets[k]))
                if len(buckets[donor]) > 1:
                    buckets[c].append(buckets[donor].pop())
        shards = [np.array(bb) for bb in buckets]
        names = [f"client_{i}" for i in range(n_cl)]

    single = [n for n, sh in zip(names, shards)
              if len(set(y[sh].tolist())) < 2]
    if single:
        log(f"      WARNING: {len(single)}/{len(shards)} clients hold only ONE "
            f"class: {single[:5]}")
        log(f"      FedAvg over single-class clients diverges. Raise "
            f"min_per_class_per_client or lower "
            f"n_time_blocks_single_device.")
    return shards, names


def fedavg(global_state, client_states, sizes):
    """Weighted FedAvg: w = sum_k (n_k/n) w_k. Float tensors (including BN
    running statistics) averaged; integer buffers taken from the server."""
    total = float(sum(sizes))
    new = {}
    for k, gv in global_state.items():
        if torch.is_floating_point(gv):
            acc = torch.zeros_like(gv, dtype=torch.float64)
            for cs, sz in zip(client_states, sizes):
                acc += cs[k].to(torch.float64) * (sz / total)
            new[k] = acc.to(gv.dtype)
        else:
            new[k] = gv.clone()
    return new


def local_train(model, loader, lr, epochs, crit, cfg):
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=cfg["weight_decay"])
    model.train()
    for _ in range(epochs):
        for x, yb, _ in loader:
            x, yb = x.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(x), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ==========================================================================
#   MIRA -- Federated Multi-Task Learning with Laplacian Regularisation
# ==========================================================================
#  A. Elbakary, C. Ben Issaid, T. ElBatt, K. Seddik, M. Bennis, "MIRA: A
#  Method of Federated MultI-Task Learning for LaRge LAnguage Models".
#
#  WHAT MIRA CHANGES. FedAvg collapses every client into one global model by
#  averaging (Eq. fedavg). When clients are genuinely heterogeneous, that
#  single model is a compromise that can be worse than purely local training
#  on some clients -- exactly the effect the MIRA authors report in their
#  Fig. 2. MIRA instead keeps ONE MODEL PER CLIENT and couples them through a
#  graph Laplacian, so similar clients pull on one another while dissimilar
#  clients are left free to specialise.
#
#  Objective (paper Eq. 1-3):
#      min_W  J(W) = F(W) + lambda * R(W)
#      F(W)  = sum_k f_k(w_k)
#      R(W)  = W^T L W = (1/2) sum_k sum_{l in N_k} a_kl ||w_k - w_l||^2
#  with L = D - M the Laplacian of the client-similarity graph.
#
#  Server regularisation step (paper Eq. 4 / Eq. 8):
#      w_k^{t+1} = w_{k,R}^{t} - eta*lambda * sum_{l in N_k} a_kl
#                                  (w_{k,R}^{t} - w_{l,R}^{t})
#  and non-sampled clients are left untouched (paper Eq. 9):
#      w_k^{t+1} = w_k^{t},  for k not in S^{t}.
#
#  WHY IT SUITS THIS DATASET. Our clients are contiguous time blocks of one
#  evening, and their H1 rates differ substantially as the crowd builds and
#  disperses. That is label-distribution heterogeneity, which is precisely
#  the regime MIRA targets. It also lets the similarity graph carry real
#  structure -- adjacent intervals of the evening genuinely resemble each
#  other -- rather than the random adjacency used in the original paper.
#
#  ADAPTATION NOTE. MIRA was published for LoRA fine-tuning of LLMs, where
#  w_k is the low-rank update Delta_W. Here the detectors are small enough to
#  federate in full, so w_k is the entire parameter vector. The optimisation
#  is identical; only the parameterisation differs. There is no LoRA in this
#  work and none is claimed.
# ==========================================================================
def build_adjacency(shards: List[np.ndarray], d: Dict, cfg: Dict) -> np.ndarray:
    """
    Similarity matrix a_{k,l}, zero diagonal, rows normalised to sum to 1.

    Row normalisation is not cosmetic. The server step contracts client k
    toward its neighbours by eta*lambda*sum_l a_kl; if that product exceeds 1
    the update overshoots and the client models oscillate. With rows summing
    to 1 the condition for a contraction is simply eta*lambda <= 1, which is
    checked explicitly below.
    """
    K = len(shards)
    mode = cfg.get("mira_adjacency", "temporal")
    A = np.zeros((K, K), dtype=np.float64)

    if mode == "temporal":
        tau = float(cfg.get("mira_tau", 1.5))
        for k in range(K):
            for l in range(K):
                if k != l:
                    A[k, l] = math.exp(-abs(k - l) / max(tau, 1e-6))
    elif mode == "label_sim":
        rates = np.array([float((d["y"][sh] == 1).mean()) for sh in shards])
        for k in range(K):
            for l in range(K):
                if k != l:
                    A[k, l] = math.exp(-abs(rates[k] - rates[l]) /
                                       max(float(rates.std()), 1e-3))
    elif mode == "random":
        rng = np.random.default_rng(cfg["seed"])
        A = rng.uniform(0.0, 1.0, size=(K, K))
        A = 0.5 * (A + A.T)
        np.fill_diagonal(A, 0.0)
    else:                                   # uniform
        A = np.ones((K, K), dtype=np.float64)
        np.fill_diagonal(A, 0.0)

    rs = A.sum(axis=1, keepdims=True)
    rs[rs <= 0] = 1.0
    return A / rs


def mira_server_step(states: List[Dict], A: np.ndarray, participated: List[bool],
                     eta: float, lam: float) -> List[Dict]:
    """
    Apply paper Eq. 8 to every participating client, Eq. 9 to the rest.

    Only floating-point tensors are regularised; integer buffers (e.g. batch
    counters) are left alone, since a convex pull between integer counters is
    meaningless.
    """
    K = len(states)
    new = [dict(sd) for sd in states]
    coef = eta * lam
    for k in range(K):
        if not participated[k]:
            continue                        # Eq. 9: untouched
        for key, vk in states[k].items():
            if not torch.is_floating_point(vk):
                continue
            acc = torch.zeros_like(vk, dtype=torch.float64)
            for l in range(K):
                if l == k or A[k, l] == 0.0:
                    continue
                acc += A[k, l] * (vk.to(torch.float64)
                                  - states[l][key].to(torch.float64))
            new[k][key] = (vk.to(torch.float64) - coef * acc).to(vk.dtype)
    return new


def _weighted_average(states: List[Dict], weights: List[float]) -> Dict:
    tot = float(sum(weights)) or 1.0
    out = {}
    for key, v0 in states[0].items():
        if torch.is_floating_point(v0):
            acc = torch.zeros_like(v0, dtype=torch.float64)
            for sd, wt in zip(states, weights):
                acc += sd[key].to(torch.float64) * (wt / tot)
            out[key] = acc.to(v0.dtype)
        else:
            out[key] = v0.clone()
    return out


@torch.no_grad()
def _eval_loss(model: nn.Module, loader: DataLoader, crit) -> float:
    model.eval()
    tot, ls = 0, 0.0
    for x, yb, _ in loader:
        x, yb = x.to(DEVICE), yb.to(DEVICE)
        ls += crit(model(x), yb).item() * x.size(0)
        tot += x.size(0)
    return ls / max(tot, 1)


def train_mira(model_name, d, tr_idx, va_loader, cfg, w,
               client_shards: Optional[Tuple[List[np.ndarray], List[str]]] = None
               ) -> Tuple[nn.Module, Dict, List[Dict]]:
    """
    Returns (deployed_model, info, best_client_states).

    best_client_states is the list of K per-client state dicts AT THE SAME
    ROUND as the deployed artefact (i.e. whichever round minimised the
    mira_eval validation criterion). It is returned separately from `info`
    (not JSON-serialised) so a caller can additionally score each client's
    OWN personalised model on its OWN local data -- the true per-client
    evaluation MIRA's design calls for, as opposed to the single
    consensus/mean artefact in `deployed_model`.

    `client_shards`, if given, overrides the internal partition_clients(d,
    tr_idx, cfg) call. Pass the TRAIN-only intersection of a client
    partition computed over the FULL session (train+val+test) so that
    "client k" denotes the same contiguous interval of the evening in
    training as in the later per-client test evaluation; leaving it None
    reproduces the original behaviour (partition derived from tr_idx alone).
    """
    if client_shards is not None:
        shards, names = client_shards
    else:
        shards, names = partition_clients(d, tr_idx, cfg)
    K = len(shards)
    use_spec = model_name in IS_2D
    loaders = [DataLoader(FrameDS(d, sh, use_spec,
                                  augment=cfg.get("augment_train", False),
                                  cfg=cfg),
                          batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=cfg["num_workers"])
               for sh in shards]
    sizes_all = [len(sh) for sh in shards]

    lam = float(cfg.get("mira_lambda", 0.3))
    eta = float(cfg.get("mira_server_lr", 1.0))
    A = build_adjacency(shards, d, cfg)

    log(f"      MIRA: {K} clients ({cfg['client_partition']}), "
        f"lambda={lam}, eta={eta}, adjacency={cfg.get('mira_adjacency')}")
    for n, sh in zip(names, shards):
        y = d["y"][sh]
        log(f"        {n[:40]:<40s} n={len(sh):5,d}  "
            f"H1={int((y==1).sum()):5,d}  H0={int((y==0).sum()):5,d}")
    if eta * lam > 1.0:
        log(f"      WARNING: eta*lambda = {eta*lam:.2f} > 1. The server step "
            f"overshoots and client models may oscillate. Reduce mira_lambda.")

    def _mk():
        m = build_model(model_name, cfg)
        if cfg.get("fl_norm", "groupnorm") == "groupnorm":
            m = convert_bn_to_gn(m)
        return m.to(DEVICE)

    work = _mk()                                   # single reusable module
    init = {k: v.detach().cpu().clone() for k, v in work.state_dict().items()}
    # every client starts from the SAME initialisation, as in the paper
    client_states = [{k: v.clone() for k, v in init.items()} for _ in range(K)]

    crit = nn.CrossEntropyLoss(weight=w.to(DEVICE),
                               label_smoothing=cfg.get("label_smoothing", 0.0))
    rng = random.Random(train_seed(cfg) + _stable_seed(model_name) % 10000)

    def select():
        k = max(1, int(round(cfg["client_fraction"] * K)))
        return rng.sample(range(K), min(k, K))

    best, best_state, best_client_states, patience = float("inf"), None, None, 0
    hist = []
    t0 = time.time()

    for rnd in range(cfg["fl_rounds"]):
        sel = select()
        participated = [False] * K

        # ---- local instruction/optimisation rounds (paper Eq. 6) ----
        for ci in sel:
            work.load_state_dict(client_states[ci])
            client_states[ci] = local_train(
                work, loaders[ci], cfg["learning_rate"],
                cfg["fl_local_epochs"], crit, cfg)
            participated[ci] = True

        # ---- server regularisation (paper Eq. 8 / Eq. 9) ----
        client_states = mira_server_step(client_states, A, participated,
                                         eta, lam)

        # ---- evaluate the deployable artefact on the shared validation set --
        mode = cfg.get("mira_eval", "consensus")
        if mode == "consensus":
            cand = _weighted_average(client_states,
                                     [float(s) for s in sizes_all])
            work.load_state_dict(cand)
            val = _eval_loss(work, va_loader, crit)
        else:
            per = []
            for ci in range(K):
                work.load_state_dict(client_states[ci])
                per.append(_eval_loss(work, va_loader, crit))
            if mode == "best_client":
                bi = int(np.argmin(per))
                cand = {k: v.clone() for k, v in client_states[bi].items()}
                val = per[bi]
            else:                                   # mean_clients
                cand = _weighted_average(client_states,
                                         [float(s) for s in sizes_all])
                val = float(np.mean(per))

        # per-client spread quantifies how much personalisation MIRA retained
        div = float(np.mean([
            float(torch.norm(
                torch.cat([(client_states[k][kk].to(torch.float64)
                            - client_states[(k + 1) % K][kk].to(torch.float64)
                            ).flatten()
                           for kk in client_states[k]
                           if torch.is_floating_point(client_states[k][kk])]))
                  ) for k in range(K)])) if K > 1 else 0.0

        hist.append({"round": rnd, "val_loss": val, "train_loss": float("nan"),
                     "clients": sel, "client_divergence": div})

        if val < best - 1e-5:
            best, patience = val, 0
            best_state = {k: v.clone() for k, v in cand.items()}
            # snapshot every client's OWN state at this same round, so the
            # per-client evaluation and the deployed artefact share one
            # training horizon rather than two independently-chosen ones.
            best_client_states = [{k: v.clone() for k, v in cs.items()}
                                   for cs in client_states]
        else:
            patience += 1
            if (rnd + 1) >= cfg["fl_min_rounds"] \
                    and patience >= cfg["fl_early_stop_rounds"]:
                log(f"      MIRA early stop @ round {rnd+1} (val {best:.4f})")
                break
        if (rnd + 1) % 5 == 0:
            log(f"      round {rnd+1:3d}  val {val:.4f}  "
                f"clients {len(sel)}/{K}  inter-client dist {div:.3f}")

    if best_state:
        work.load_state_dict(best_state)
    if best_client_states is None:              # fl_rounds==0 edge case
        best_client_states = client_states
    return work, {"best_val_loss": best, "rounds_run": len(hist),
                  "train_time_s": time.time() - t0, "n_clients": K,
                  "client_names": names, "client_sizes": sizes_all,
                  "fl_algorithm": "mira", "mira_lambda": lam,
                  "mira_adjacency": cfg.get("mira_adjacency"),
                  "history": hist}, best_client_states


# ==========================================================================
#   TRUE PER-CLIENT MIRA EVALUATION
#   (each client is scored on its own local test frames)
# ==========================================================================
def full_session_client_shards(d: Dict, tr: np.ndarray, va: np.ndarray,
                                te: np.ndarray, cfg: Dict
                               ) -> Tuple[List[np.ndarray], List[np.ndarray],
                                          List[np.ndarray], List[str]]:
    """
    One client partition computed over the WHOLE session (train+val+test
    together), then intersected with each split.

    Calling partition_clients on tr_idx alone (as the original code did)
    means "client k" is defined only by the train-time boundaries and has no
    well-defined counterpart in the test split. Partitioning the full
    session first and intersecting afterwards makes "client k" the same
    contiguous interval of the evening in every split, so client k's final
    personalised model can be scored on client k's OWN held-out test frames
    rather than on the pooled test set or a consensus/averaged model.

    v3: with make_splits splitting inside every time block, each client
    owns local train/val/test frames. Adjacent clients are then merged until
    every client's TRAIN shard holds >= min_per_class_per_client frames of
    BOTH classes (an under-populated client would otherwise sit at its
    initialisation, or crash the loader if empty). A client whose local val
    or test shard is single-class is still trained, but skipped -- and
    logged -- by evaluate_per_client.

    Returns (shards_tr, shards_va, shards_te, names), aligned lists.
    """
    full_idx = np.concatenate([tr, va, te])
    shards_full, names = partition_clients(d, full_idx, cfg)
    tr_set, va_set, te_set = set(tr.tolist()), set(va.tolist()), set(te.tolist())
    min_pc = int(cfg.get("min_per_class_per_client", 8))
    y = d["y"]
    is_tr = np.zeros(len(y), dtype=bool)
    is_tr[tr] = True

    def tr_ok(sh):
        t = sh[is_tr[sh]]
        return (int((y[t] == 1).sum()) >= min_pc and
                int((y[t] == 0).sum()) >= min_pc)

    merged, n_merge = list(shards_full), 0
    while len(merged) > 1:
        bad = [i for i, sh in enumerate(merged) if not tr_ok(sh)]
        if not bad:
            break
        i = bad[0]
        j = i + 1 if i + 1 < len(merged) else i - 1
        lo, hi = min(i, j), max(i, j)
        merged[lo] = np.concatenate([merged[lo], merged[hi]])
        del merged[hi]
        n_merge += 1
    if n_merge:
        log(f"      merged {n_merge} adjacent client(s) so every client's "
            f"TRAIN shard holds both classes ({len(shards_full)} -> "
            f"{len(merged)} clients)")
    dev0 = str(d["device"][full_idx[0]])
    names = [f"{dev0}|T{k}" for k in range(len(merged))] \
        if len(merged) != len(shards_full) else names

    shards_tr, shards_va, shards_te = [], [], []
    for sh in merged:
        s_ = set(sh.tolist())
        shards_tr.append(np.array(sorted(s_ & tr_set), dtype=np.int64))
        shards_va.append(np.array(sorted(s_ & va_set), dtype=np.int64))
        shards_te.append(np.array(sorted(s_ & te_set), dtype=np.int64))
    return shards_tr, shards_va, shards_te, names


def evaluate_per_client(model_name: str, cfg: Dict, d: Dict,
                             client_states: List[Dict], names: List[str],
                             shards_va: List[np.ndarray],
                             shards_te: List[np.ndarray],
                             sizes_tr: List[int],
                             fixed_threshold: Optional[float] = None
                             ) -> Tuple[List[Dict], Dict]:
    """
    Score EACH client's own final personalised state on its OWN local test
    shard, with a threshold picked on its OWN local validation shard. This
    is the primary MIRA metric: it measures what the personalisation
    objective in (paper Eq. miraobj) actually optimises for, rather than the
    size-weighted consensus, which MIRA's own authors motivate against using
    on heterogeneous clients.

    A client with too few local val or test frames to evaluate (an interval
    of the evening with almost no held-out data) is skipped and logged, not
    silently zero-filled.
    """
    use_spec = model_name in IS_2D
    rows: List[Dict] = []
    for k, (name, cs) in enumerate(zip(names, client_states)):
        va_idx, te_idx = shards_va[k], shards_te[k]
        if len(va_idx) < 5 or len(te_idx) < 5:
            log(f"      per-client eval SKIPPED for {name} "
                f"(n_val={len(va_idx)}, n_test={len(te_idx)} too few)")
            continue
        if len(set(d["y"][te_idx].tolist())) < 2:
            log(f"      per-client eval SKIPPED for {name} "
                f"(local test shard is single-class)")
            continue
        m = build_model(model_name, cfg)
        if cfg.get("fl_norm", "groupnorm") == "groupnorm":
            m = convert_bn_to_gn(m)
        m.load_state_dict(cs)
        m = m.to(DEVICE)
        va_l = DataLoader(FrameDS(d, va_idx, use_spec, augment=False, cfg=cfg),
                          batch_size=cfg["batch_size"], shuffle=False,
                          num_workers=cfg["num_workers"])
        te_l = DataLoader(FrameDS(d, te_idx, use_spec, augment=False, cfg=cfg),
                          batch_size=cfg["batch_size"], shuffle=False,
                          num_workers=cfg["num_workers"])
        # MIRA: each client deploys its own model, so its own threshold.
        # FedAvg: ONE global model with ONE threshold (fixed_threshold) --
        # per-client numbers then show how evenly that single detector
        # serves each interval of the evening, with no personalisation.
        thr = (fixed_threshold if fixed_threshold is not None
               else pick_threshold(m, va_l, cfg))
        res = evaluate(m, te_l, cfg, threshold=thr)
        o = res["overall"]
        rows.append({"client": name, "n_train": sizes_tr[k] if k < len(sizes_tr) else -1,
                     "n_val": len(va_idx), "n_test": len(te_idx), **o})

    if not rows:
        log("      WARNING: no client had enough local val+test data for "
            "per-client MIRA evaluation; perclient_* columns will be nan")
        return rows, {"perclient_n_clients_evaluated": 0,
                      "perclient_bal_acc_mean": float("nan"),
                      "perclient_bal_acc_spread": float("nan"),
                      "perclient_f1_mean": float("nan"),
                      "perclient_recall_pd_mean": float("nan"),
                      "perclient_fdr_mean": float("nan")}

    bal = np.array([r["balanced_accuracy"] for r in rows])
    agg = {
        "perclient_n_clients_evaluated": len(rows),
        "perclient_bal_acc_mean": float(np.mean(bal)),
        "perclient_bal_acc_median": float(np.median(bal)),
        "perclient_bal_acc_spread": float(np.max(bal) - np.min(bal)),
        "perclient_f1_mean": float(np.mean([r["f1"] for r in rows])),
        "perclient_recall_pd_mean": float(np.mean([r["recall_pd"] for r in rows])),
        "perclient_fdr_mean": float(np.mean([r["fdr"] for r in rows])),
    }
    return rows, agg


def train_federated(model_name, d, tr_idx, va_loader, cfg, w,
                    client_shards: Optional[Tuple[List[np.ndarray],
                                                  List[str]]] = None
                    ) -> Tuple[nn.Module, Dict]:
    """FedAvg. `client_shards` (train-only intersection of the full-session
    partition, see full_session_client_shards) makes the clients identical to
    the ones used for per-client evaluation; None = partition tr_idx here."""
    if client_shards is not None:
        shards, names = client_shards
    else:
        shards, names = partition_clients(d, tr_idx, cfg)
    use_spec = model_name in IS_2D
    loaders = [DataLoader(FrameDS(d, sh, use_spec,
                                  augment=cfg.get("augment_train", False),
                                  cfg=cfg),
                          batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=cfg["num_workers"])
               for sh in shards]
    sizes_all = [len(s) for s in shards]
    log(f"      {len(shards)} clients ({cfg['client_partition']})")
    for n, sh in zip(names, shards):
        y = d["y"][sh]
        log(f"        {n[:40]:<40s} n={len(sh):5,d}  "
            f"H1={int((y==1).sum()):5,d}  H0={int((y==0).sum()):5,d}")

    def _mk():
        m = build_model(model_name, cfg)
        if cfg.get("fl_norm", "groupnorm") == "groupnorm":
            m = convert_bn_to_gn(m)
        return m.to(DEVICE)

    gm, lm = _mk(), _mk()
    if cfg.get("fl_norm", "groupnorm") == "groupnorm":
        log("      norm: GroupNorm (BatchNorm running stats are not "
            "FedAvg-safe across heterogeneous clients)")
    g_state = {k: v.detach().cpu().clone() for k, v in gm.state_dict().items()}
    crit = nn.CrossEntropyLoss(weight=w.to(DEVICE),
                               label_smoothing=cfg.get("label_smoothing", 0.0))
    rng = random.Random(train_seed(cfg) + _stable_seed(model_name) % 10000)

    def select():
        k = max(1, int(round(cfg["client_fraction"] * len(shards))))
        return rng.sample(range(len(shards)), min(k, len(shards)))

    best, best_state, patience = float("inf"), None, 0
    hist = []
    t0 = time.time()
    for rnd in range(cfg["fl_rounds"]):
        sel = select()
        states, sizes = [], []
        for ci in sel:
            lm.load_state_dict(g_state)
            states.append(local_train(lm, loaders[ci], cfg["learning_rate"],
                                      cfg["fl_local_epochs"], crit, cfg))
            sizes.append(len(shards[ci]))
        g_state = fedavg(g_state, states, sizes)
        gm.load_state_dict(g_state)

        gm.eval()
        vs, vt = 0.0, 0
        with torch.no_grad():
            for x, yb, _ in va_loader:
                x, yb = x.to(DEVICE), yb.to(DEVICE)
                vs += crit(gm(x), yb).item() * x.size(0)
                vt += x.size(0)
        val = vs / max(vt, 1)
        hist.append({"round": rnd, "val_loss": val, "train_loss": float("nan"),
                     "clients": sel})

        if val < best - 1e-5:
            best, patience = val, 0
            best_state = {k: v.clone() for k, v in g_state.items()}
        else:
            patience += 1
            if (rnd + 1) >= cfg["fl_min_rounds"] \
                    and patience >= cfg["fl_early_stop_rounds"]:
                log(f"      FL early stop @ round {rnd+1} (val {best:.4f})")
                break
        if (rnd + 1) % 5 == 0:
            log(f"      round {rnd+1:3d}  val {val:.4f}  "
                f"clients {len(sel)}/{len(shards)}")

    if best_state:
        gm.load_state_dict(best_state)
    return gm, {"best_val_loss": best, "rounds_run": len(hist),
                "train_time_s": time.time() - t0, "n_clients": len(shards),
                "client_names": names, "client_sizes": sizes_all,
                "fl_algorithm": "fedavg", "history": hist}


# ==========================================================================
#                              SPLITS + IO
# ==========================================================================
def make_splits(d: Dict, cfg: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological split STRATIFIED BY BAND.

    A naive global time split fails badly on this data: frames are recorded
    band by band, so sorting by timestamp sorts by band, and the test tail
    ends up containing a single band and therefore a single class - accuracy
    0.0 and AUC undefined. That is not a modelling failure, it is a split
    failure, and it is easy to miss because the code runs without error.

    So the split is applied WITHIN each band: earliest frames of every band to
    train, latest to test. Temporal ordering is preserved inside each band
    (no leakage from future to past), and every band and both classes appear
    in all three splits.

    v3: ALSO WITHIN EACH OF `split_blocks` CONTIGUOUS TIME BLOCKS. With one
    block (the v1/v2 behaviour) "test" is the END of the event and "train"
    its beginning, so federated clients -- which are themselves time
    intervals -- had either no test frames (early clients) or no training
    frames (late clients), and true per-client evaluation was impossible.
    Splitting chronologically inside each (time block, band) gives every
    stretch of the evening its own train/val/test, every client (a group of
    whole blocks, see partition_clients) local held-out data, and one test
    set shared by the centralised arm, the federated arm and every client
    count in the C-ablation. It also removes most of the train/test prior
    shift the v1/v2 split induced. split_blocks=1 reproduces v1/v2.
    The block id is stored in d["_split_block"] for partition_clients.
    """
    n = len(d["y"])
    rng = np.random.default_rng(cfg["seed"])
    tr, va, te = [], [], []

    nb = max(1, int(cfg.get("split_blocks", 32)))
    nb = max(nb, int(cfg.get("n_time_blocks_single_device", 8)),
             *[int(c) for c in cfg.get("client_count_ablation", []) or [1]])
    order = np.argsort(d["utc"], kind="stable")
    blk = np.empty(n, dtype=np.int64)
    for bi, part in enumerate(np.array_split(order, nb)):
        blk[part] = bi
    d["_split_block"] = blk

    if cfg["split_by"] == "time":
        groups = np.array([f"{b}|{k}" for b, k in zip(d["band"], blk)])
    else:
        groups = np.array(["_all"] * n)
    for gname in sorted(set(groups.tolist())):
        gi = np.flatnonzero(groups == gname)
        gi = gi[np.argsort(d["utc"][gi])] if cfg["split_by"] == "time" \
            else gi[rng.permutation(len(gi))]
        m = len(gi)
        n_te = max(1, int(round(cfg["test_frac"] * m)))
        n_va = max(1, int(round(cfg["val_frac"] * m)))
        if m - n_te - n_va < 1:                 # tiny band: everything trains
            tr.extend(gi.tolist())
            continue
        tr.extend(gi[:m - n_va - n_te].tolist())
        va.extend(gi[m - n_va - n_te:m - n_te].tolist())
        te.extend(gi[m - n_te:].tolist())

    tr, va, te = np.array(tr), np.array(va), np.array(te)
    for nm, idx in (("train", tr), ("val", va), ("test", te)):
        if idx.size == 0:
            sys.exit(f"{nm} split is empty - too few frames after labelling.")
        cls = set(d["y"][idx].tolist())
        if len(cls) < 2:
            sys.exit(
                f"{nm} split contains only class {cls}. Metrics such as AUC "
                f"are undefined on a single-class subset, so training would "
                f"report meaningless numbers.\n"
                f"  Cause: not enough of one class survives labelling in some "
                f"band.\n  Fix: loosen occ_margin_db/noise_percentile "
                f"(label_scheme='margin') or label_hi_db/label_lo_db "
                f"(label_scheme='dead_zone'), or add sessions covering both "
                f"quiet and busy spectrum.")
    return tr, va, te


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


# ==========================================================================
def _env(cfg: Dict) -> Dict:
    m = os.environ.get("SENSING_MODES", "").strip()
    if m:
        cfg["training_modes"] = [x for x in m.split(",") if x]
    if os.environ.get("SENSING_AUDIT"):
        cfg["eval_leave_one_band_out"] = True
    if os.environ.get("SENSING_QUICK"):
        cfg.update({"epochs": 5, "min_epochs": 2, "fl_rounds": 5,
                    "fl_min_rounds": 2, "latency_runs": 20})
    o = os.environ.get("SENSING_OUT", "").strip()
    if o:
        cfg["results_dir"] = o
    return cfg


# ==========================================================================
#                    HARDWARE / PROVENANCE AUDIT
# ==========================================================================
#  This is a TRAINING script. It cannot set the antenna port, the sample rate
#  or the tuning range - those are decided in the capture software
#  at capture time and are already baked into the files by the time we get
#  here. What it CAN do is check that the data loaded is consistent with the
#  hardware you say produced it, and state plainly which claims it cannot
#  check. Silent inconsistency is expensive: an ADC scale wrong by 16x shifts
#  every dB figure by 24 dB without raising an error.
# ==========================================================================
DECLARED_HW = {
    "radios": {
        "USRP B205mini-i": {"freq_min_hz": 50e6, "freq_max_hz": 6000e6,
                            "max_bw_hz": 56e6, "adc_full_scale": 32767.0,
                            "expected_antenna": "RX2"},
        "ADALM-PLUTO": {"freq_min_hz": 325e6, "freq_max_hz": 3800e6,
                        "max_bw_hz": 20e6, "adc_full_scale": 2048.0,
                        "expected_antenna": "RX"},
    },
    "antenna": "Superbat dual-band 2.4/5/5.8 GHz 8 dBi omni, SMA male",
    # The Superbat is DUAL-band: it has two design passbands, not one. A
    # single-resonance check wrongly flags the whole 5 GHz sweep as
    # mismatched, which is the opposite of true - 5/5.8 GHz is exactly what
    # the antenna's upper band is for.
    "antenna_bands_hz": [(2.400e9, 2.500e9), (5.150e9, 5.925e9)],
    "antenna_tolerance_octaves": 0.35,
}


def hardware_audit(h5_files: List[str]) -> Dict:
    log("\n" + "=" * 74)
    log("HARDWARE / PROVENANCE AUDIT")
    log("=" * 74)
    findings: List[str] = []
    seen: Dict[str, Dict] = {}
    for path in h5_files:
        try:
            fh = h5py.File(path, "r")
        except Exception as e:
            findings.append(f"unreadable {os.path.basename(path)}: {e}")
            continue
        with fh:
            radio = str(fh.attrs.get("radio", "unknown"))
            g = _frames_group(fh)
            if g is None:
                findings.append(f"{os.path.basename(path)}: no iq_frames")
                continue
            ant = ""
            if "iq_raw" in fh and len(fh["iq_raw"]):
                k0 = list(fh["iq_raw"].keys())[0]
                ant = str(fh["iq_raw"][k0].attrs.get("antenna", ""))
            d = seen.setdefault(radio, {"n": 0, "sr": set(), "f_lo": np.inf,
                                        "f_hi": -np.inf, "scale": set(),
                                        "ant": set(), "files": 0})
            d["n"] += int(g["iq"].shape[0]); d["files"] += 1
            if "sample_rate" in g:
                d["sr"].update(np.unique(np.round(g["sample_rate"][:])).tolist())
            if "center_hz" in g:
                c = g["center_hz"][:]
                d["f_lo"] = min(d["f_lo"], float(c.min()))
                d["f_hi"] = max(d["f_hi"], float(c.max()))
            if "full_scale" in g:
                d["scale"].update(np.unique(g["full_scale"][:]).tolist())
            if ant:
                d["ant"].add(ant)

    for radio, d in seen.items():
        spec = DECLARED_HW["radios"].get(radio)
        log(f"\n  {radio}   ({d['n']:,} frames, {d['files']} file(s))")
        if spec is None:
            findings.append(f"{radio}: not a declared radio, cannot verify")
            log("    NOT in DECLARED_HW; nothing to check against.")
            continue
        if d["sr"]:
            hi = max(d["sr"])
            log(f"    sample rate   up to {hi/1e6:.3f} MS/s "
                f"(device max {spec['max_bw_hz']/1e6:.0f} MHz)")
            if hi > spec["max_bw_hz"] + 1e3:
                findings.append(f"{radio}: {hi/1e6:.1f} MS/s EXCEEDS the "
                                f"{spec['max_bw_hz']/1e6:.0f} MHz device limit")
            elif hi < spec["max_bw_hz"] * 0.5:
                log(f"    note: {hi/spec['max_bw_hz']*100:.0f}% of the device "
                    f"maximum. Deliberate (USB throughput vs ADC dynamic "
                    f"range), not a fault.")
        if np.isfinite(d["f_lo"]):
            log(f"    tuned range   {d['f_lo']/1e6:.1f} - {d['f_hi']/1e6:.1f} "
                f"MHz (device {spec['freq_min_hz']/1e6:.0f} - "
                f"{spec['freq_max_hz']/1e6:.0f})")
            if d["f_hi"] > spec["freq_max_hz"]:
                findings.append(f"{radio}: tuned to {d['f_hi']/1e6:.0f} MHz, "
                                f"ABOVE the {spec['freq_max_hz']/1e6:.0f} MHz "
                                f"ceiling - out of specification")
            if d["f_lo"] < spec["freq_min_hz"]:
                findings.append(f"{radio}: tuned below the device floor")
            # distance to the NEAREST antenna passband, not to one resonance
            def _off(f):
                best = min(0.0 if lo <= f <= hi
                           else min(abs(math.log2(f / lo)),
                                    abs(math.log2(f / hi)))
                           for lo, hi in DECLARED_HW["antenna_bands_hz"])
                return best
            tol = DECLARED_HW["antenna_tolerance_octaves"]
            worst_f = max([d["f_lo"], d["f_hi"]], key=_off)
            if _off(worst_f) > tol:
                findings.append(
                    f"{radio}: tuning reaches {worst_f/1e6:.0f} MHz, "
                    f"{_off(worst_f):.2f} octaves outside the antenna's "
                    f"passbands "
                    f"{[f'{a/1e9:.2f}-{b/1e9:.2f} GHz' for a, b in DECLARED_HW['antenna_bands_hz']]}"
                    f" - mismatched there, levels untrustworthy")
        if d["scale"]:
            got = sorted(d["scale"])
            log(f"    ADC scale     {', '.join(f'{v:.0f}' for v in got)} "
                f"(expected {spec['adc_full_scale']:.0f})")
            if any(abs(v - spec["adc_full_scale"]) > 1 for v in got):
                findings.append(
                    f"{radio}: ADC full scale {got} != expected "
                    f"{spec['adc_full_scale']:.0f}; every dBFS figure is off "
                    f"by {20*math.log10(max(got)/spec['adc_full_scale']):+.1f} dB")
        if d["ant"]:
            log(f"    antenna port  {', '.join(sorted(d['ant']))} "
                f"(expected {spec['expected_antenna']})")
            if spec["expected_antenna"] not in d["ant"]:
                findings.append(f"{radio}: antenna port {sorted(d['ant'])} is "
                                f"not {spec['expected_antenna']}")
        else:
            log("    antenna port  not recorded in this export (no iq_raw)")

    log("\n  Cannot be verified from exported data - check the capture log:")
    log("    which antenna was physically attached, and its real gain")
    log("    whether the USB link negotiated USB 3")
    log("    whether your own WiFi/Bluetooth were off")
    log("    antenna orientation and ground plane")
    if findings:
        log("\n  AUDIT FINDINGS:")
        for x in findings:
            log(f"    ! {x}")
    else:
        log("\n  No inconsistencies against the declared setup.")
    log("=" * 74)
    return {"radios": list(seen), "findings": findings}


MODEL_REFS = {
    "deepsense_s4": "DeepSense, IEEE INFOCOM 2021 - compact 1-D CNN",
    "tprime_s9": "T-PRIME, IEEE INFOCOM 2024 - edge transformer",
    "spectrum_transformer_s5": "SpecFormer, IEEE TWC 2024 - patch transformer",
    "cnn_transformer_s6": "CNN-Former, IEEE WCL 2025 - conv + attention",
    "denoise_transformer_s7": "DnFormer, IEEE COMML 2025 - denoise + transformer",
    "oscar_net_p1": "OSCAR-Net (proposed) - learnable OS-CFAR + excess encoder",
}


def print_run_matrix(cfg: Dict, have_spec: bool) -> None:
    models = [m for m in cfg["models"] if m not in IS_2D or have_spec]
    modes = list(cfg["training_modes"])
    arms = [a["name"] for a in cfg["dataset_arms"]]
    total = len(models) * len(modes) * len(arms)
    log("\n" + "=" * 74)
    log(f"RUN MATRIX: {len(models)} models x {len(modes)} modes x "
        f"{len(arms)} arms = {total} runs")
    log("=" * 74)
    log(f"  arms  : {', '.join(arms)}")
    log(f"  modes : {', '.join(modes)}")
    log("  models:")
    for i, m in enumerate(models, 1):
        tag = "  [2-D]" if m in IS_2D else ""
        log(f"    {i:2d}. {m:<24s} {MODEL_REFS.get(m,''):<50s}{tag}")
    per_c = cfg["epochs"] * 1.2
    per_f = cfg["fl_rounds"] * cfg["fl_local_epochs"] * 2.0
    est = len(arms) * len(models) * ((per_c if "centralized" in modes else 0)
                                     + (per_f if "federated" in modes else 0))
    log(f"\n  Rough estimate {est/60:.0f} min ({est/3600:.1f} h) plus "
        f"profiling. Order-of-magnitude only.")
    if DEVICE.type == "cpu":
        log("  Training on CPU. A CUDA GPU cuts this a lot; complexity "
            "profiling stays on CPU so those numbers stay comparable.")
    log("=" * 74)


# ==========================================================================
#                    CRASH-SAFE CHECKPOINT / RESUME
# ==========================================================================
def _atomic_write_json(path: str, obj: Dict) -> None:
    """
    Write a JSON file such that it is either fully present or absent.

    A direct open()/write() can be interrupted by a power loss or forced
    restart, leaving a truncated file that json.load() may or may not reject.
    Writing to a temporary file in the same directory and then renaming is
    atomic on both NTFS and POSIX, so a reader never observes a partial file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    return o


# Configuration keys that change what a trained model IS. Cosmetic keys
# (verbose, results_dir, force_retrain) are excluded so that changing them
# does not needlessly invalidate hours of completed work.
FINGERPRINT_KEYS = [
    "frame_len", "use_spectrogram", "spec_size", "normalise", "label_segment",
    "crop_mode", "min_label_windows", "label_scheme", "noise_percentile",
    "occ_margin_db", "margin_busy_frac", "label_hi_db", "label_lo_db",
    "label_busy_frac_hi", "label_busy_frac_lo", "label_win_us",
    "drop_ambiguous", "batch_size", "epochs", "learning_rate", "weight_decay",
    "early_stopping_patience", "min_epochs", "lr_scheduler_patience",
    "lr_scheduler_factor", "grad_clip", "augment_train", "aug_time_shift",
    "aug_phase_rotate", "aug_time_reverse", "label_smoothing", "val_frac",
    "test_frac", "split_by", "split_blocks", "client_partition", "n_time_blocks",
    "n_time_blocks_single_device", "min_per_class_per_client", "n_clients",
    "client_fraction", "dirichlet_alpha", "fl_rounds", "fl_local_epochs",
    "fl_early_stop_rounds", "fl_norm", "fl_min_rounds", "mira_eval",
    "report_per_client", "threshold_mode",
    "target_pfa", "snr_bins_db", "seed",
]


def config_fingerprint(cfg: Dict) -> str:
    payload = {k: cfg.get(k) for k in FINGERPRINT_KEYS}
    # only added when it differs from "seed", so the primary run keeps its
    # fingerprint and its checkpoints stay valid
    if cfg.get("train_seed") is not None and int(cfg["train_seed"]) != int(cfg["seed"]):
        payload["train_seed"] = int(cfg["train_seed"])
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def dataset_fingerprint(d: Dict) -> str:
    """Identify the loaded corpus, so a changed dataset invalidates markers."""
    payload = {
        "n": int(len(d["y"])),
        "n_h1": int((d["y"] == 1).sum()),
        "bands": sorted(set(d["band"].tolist())),
        "devices": sorted(set(d["device"].tolist())),
        "t0": round(float(d["utc"].min()), 3),
        "t1": round(float(d["utc"].max()), 3),
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _done_dir(cfg: Dict) -> str:
    return os.path.join(cfg["results_dir"], "_completed")


def unit_key(arm: str, model: str, mode: str) -> str:
    return f"{arm}__{model}__{mode}"


def _marker_path(cfg: Dict, key: str) -> str:
    return os.path.join(_done_dir(cfg), f"{key}.json")


def load_marker(cfg: Dict, key: str, cfp: str, dfp: str) -> Optional[Dict]:
    """Return a completed unit's payload, or None if absent/stale/corrupt."""
    if not cfg.get("resume", True):
        return None
    for pat in (key, key.split("__")[1] if "__" in key else key):
        if pat in cfg.get("force_retrain", []):
            return None
    if key in cfg.get("force_retrain", []):
        return None
    path = _marker_path(cfg, key)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            m = json.load(f)
    except Exception as e:
        log(f"      marker for {key} is unreadable ({e}); retraining")
        return None
    if not m.get("complete"):
        return None
    if not cfg.get("ignore_fingerprint", False):
        if m.get("config_fingerprint") != cfp:
            if "main" in STAGES:            # only the training stage would retrain it
                log(f"      marker for {key} was produced under a DIFFERENT "
                    f"configuration; retraining")
            return None
        if m.get("dataset_fingerprint") != dfp:
            log(f"      marker for {key} was produced on a DIFFERENT dataset; "
                f"retraining")
            return None
    return m


def save_marker(cfg: Dict, key: str, cfp: str, dfp: str,
                summary_row: Dict, per_snr: List[Dict]) -> None:
    _atomic_write_json(_marker_path(cfg, key), {
        "complete": True,
        "key": key,
        "config_fingerprint": cfp,
        "dataset_fingerprint": dfp,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary_row": _jsonable(summary_row),
        "per_snr_rows": _jsonable(per_snr),
    })


def load_profile_marker(cfg: Dict, model: str, cfp: str) -> Optional[Dict]:
    if not cfg.get("resume", True) or model in cfg.get("force_retrain", []):
        return None
    path = os.path.join(_done_dir(cfg), f"_profile__{model}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            m = json.load(f)
    except Exception:
        return None
    if not cfg.get("ignore_fingerprint", False) and \
            m.get("config_fingerprint") != cfp:
        return None
    return m.get("profile")


def save_profile_marker(cfg: Dict, model: str, cfp: str, prof: Dict) -> None:
    _atomic_write_json(os.path.join(_done_dir(cfg), f"_profile__{model}.json"),
                       {"config_fingerprint": cfp, "profile": _jsonable(prof)})


def collect_markers(cfg: Dict, cfp: str, dfp: str
                    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Rebuild the aggregate tables from every valid marker on disk."""
    sm, ps, cx = [], [], []
    dd = _done_dir(cfg)
    if not os.path.isdir(dd):
        return sm, ps, cx
    for path in sorted(glob.glob(os.path.join(dd, "*.json"))):
        base = os.path.basename(path)[:-5]
        try:
            with open(path) as f:
                m = json.load(f)
        except Exception:
            continue
        stale = (not cfg.get("ignore_fingerprint", False)
                 and m.get("config_fingerprint") != cfp)
        if base.startswith("_profile__"):
            if not stale and m.get("profile"):
                cx.append(m["profile"])
            continue
        if base.startswith(("ablation__", "lobo__")):
            continue        # client-count sweep / LOBO: own CSV, not the main table
        if stale or m.get("dataset_fingerprint") != dfp or not m.get("complete"):
            continue
        if m.get("summary_row"):
            sm.append(m["summary_row"])
        ps.extend(m.get("per_snr_rows", []))
    return sm, ps, cx


def plan_status(cfg: Dict, arms: List[str], models: List[str],
                cfp: str, dfp: str) -> Tuple[int, int, List[str]]:
    """Report how many units are already complete before any training starts."""
    todo, done = [], 0
    for a in arms:
        for m in models:
            for md in cfg["training_modes"]:
                k = unit_key(a, m, md)
                if load_marker(cfg, k, cfp, dfp) is not None:
                    done += 1
                else:
                    todo.append(k)
    return done, done + len(todo), todo


def run_arm(arm_name: str, d: Dict, cfg: Dict, out_dir: str,
            cfp: str = "", dfp: str = ""):
    """Full model sweep on one dataset arm."""
    os.makedirs(out_dir, exist_ok=True)
    have_spec = "X_spec" in d
    models = [m for m in cfg["models"] if m not in IS_2D or have_spec]
    if len(models) < len(cfg["models"]):
        log("    NOTE: no spectrograms available -> 2-D models skipped")

    st = arm_stats(d)
    log(f"    frames {st['n_frames']:,}  H1 {st['n_H1']:,} / H0 {st['n_H0']:,}"
        f"  ({st['h1_fraction']*100:.1f}% H1)")
    log(f"    majority-class baseline accuracy = {st['majority_baseline']:.4f}"
        f"   <- any model at or below this is worthless")
    log(f"    bands {st['n_bands']}  devices {st['n_devices']}  |  est SNR "
        f"p10/p50/p90 = {st['snr_p10_db']:.1f}/{st['snr_p50_db']:.1f}/"
        f"{st['snr_p90_db']:.1f} dB")

    tr, va, te = make_splits(d, cfg)
    log(f"    split: train {len(tr):,} / val {len(va):,} / test {len(te):,}")
    w = class_weights(d["y"][tr])

    arm_cfg = dict(cfg)
    if st["n_devices"] == 1 and cfg["client_partition"] == "natural":
        log(f"    single-radio arm -> federated clients will be "
            f"{cfg.get('n_time_blocks_single_device', 8)} contiguous time "
            f"blocks, auto-merged to keep both classes in every client")

    # ---- client partition over the FULL session, for true per-client MIRA
    # evaluation (see full_session_client_shards docstring). Computed once
    # per arm since it does not depend on the model being trained.
    shards_tr_full = shards_va_full = shards_te_full = client_names_full = None
    if "federated" in cfg["training_modes"]:
        shards_tr_full, shards_va_full, shards_te_full, client_names_full = \
            full_session_client_shards(d, tr, va, te, cfg)
        empty_te = sum(1 for sh in shards_te_full if len(sh) == 0)
        log(f"    full-session client partition: {len(client_names_full)} "
            f"clients  ({empty_te} with no local test frames -> excluded "
            f"from per-client evaluation)")

    summary, per_snr_rows, cx_rows, perclient_rows = [], [], [], []
    for mi, mname in enumerate(models, 1):
        use_spec = mname in IS_2D
        mk = lambda idx, sh: DataLoader(          # noqa: E731
            FrameDS(d, idx, use_spec,
                    augment=(sh and cfg.get("augment_train", False)), cfg=cfg),
            batch_size=cfg["batch_size"], shuffle=sh,
            num_workers=cfg["num_workers"])
        tr_l, va_l, te_l = mk(tr, True), mk(va, False), mk(te, False)

        if cfg["profile_complexity"]:
            cached = load_profile_marker(cfg, mname, cfp)
            if cached is not None:
                log(f"\n  [{arm_name}] profile {mname} "
                    f"({mi}/{len(models)}) loaded from checkpoint")
                prof = dict(cached)
                prof["arm"] = arm_name
                cx_rows.append(prof)
                cx_rows[-1]["_cached"] = True
            else:
                log(f"\n  [{arm_name}] profiling {mname} "
                    f"({mi}/{len(models)}, CPU, batch {cfg['prof_batch']})")
                xb, _, _ = next(iter(te_l))
                prof = profile_model(build_model(mname, cfg),
                                     tuple(xb.shape[1:]), mname, cfg)
                prof["arm"] = arm_name
                save_profile_marker(cfg, mname, cfp, prof)
                cx_rows.append(prof)
            log(f"    params {prof['params_M']:.4f} M | mem "
                f"{prof['model_mem_MB']:.2f} MB | MACs {prof['macs_M']:.1f} M "
                f"| FLOPs {prof['flops_M']:.1f} M")
            log(f"    latency B/A/W {prof['latency_ms_avg']:.4f} ms/frame "
                f"(identical, standalone) | batch-1 "
                f"{prof['batch1_latency_ms']:.3f} ms | thr "
                f"{prof['throughput_avg']:,.0f} f/s")

        for mode in cfg["training_modes"]:
            ukey = unit_key(arm_name, mname, mode)
            done = load_marker(cfg, ukey, cfp, dfp)
            if done is not None and mode == "federated":
                prev = done["summary_row"].get("fl_algorithm", "mira")
                if prev != cfg.get("fl_algorithm", "fedavg"):
                    log(f"\n  [{arm_name}] {mname} - federated: checkpoint was "
                        f"trained with {prev}, config says "
                        f"{cfg.get('fl_algorithm')} -> retraining")
                    done = None
            if done is not None:
                r = done["summary_row"]
                log(f"\n  [{arm_name}] {mname} - {mode}  "
                    f"ALREADY TRAINED (completed {done.get('completed_utc','')})"
                    f" -> skipping")
                log(f"    ACC {r.get('accuracy', float('nan')):.4f} | "
                    f"AUC {r.get('auc', float('nan')):.4f} | "
                    f"balAcc {r.get('balanced_accuracy', float('nan')):.4f}")
                summary.append(r)
                per_snr_rows.extend(done.get("per_snr_rows", []))
                continue

            log(f"\n  [{arm_name}] {mname} - {mode}")
            seed_unit(train_seed(cfg), arm_name, mname, mode)
            client_states = None
            algo = cfg.get("fl_algorithm", "fedavg")
            if mode == "centralized":
                model, info = train_centralized(
                    build_model(mname, cfg), tr_l, va_l, cfg, w)
            else:
                client_shards = ((shards_tr_full, client_names_full)
                                 if shards_tr_full is not None else None)
                if algo == "mira":
                    model, info, client_states = train_mira(
                        mname, d, tr, va_l, arm_cfg, w,
                        client_shards=client_shards)
                else:
                    model, info = train_federated(
                        mname, d, tr, va_l, arm_cfg, w,
                        client_shards=client_shards)
                    gstate = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                    client_states = [gstate] * len(info["client_names"])

            crit_eval = nn.CrossEntropyLoss(weight=w.to(DEVICE))
            ofit = overfit_report(model, tr_l, va_l, te_l, crit_eval,
                                  info.get("history", []))
            thr = pick_threshold(model, va_l, cfg)
            res = evaluate(model, te_l, cfg, threshold=thr)
            o = res["overall"]
            log(f"    ACC {o['accuracy']:.4f} (baseline "
                f"{o['majority_baseline_acc']:.4f}, "
                f"{'BEATS' if o['beats_baseline'] else 'FAILS'}) | "
                f"F1 {o['f1']:.4f} | AUC {o['auc']:.4f}")
            log(f"    Pd {o['recall_pd']:.4f} | MDR {o['mdr']:.4f} | FDR "
                f"{o['fdr']:.4f} | NMSE {o['nmse']:.4f} | MAE {o['mae']:.4f}")
            log(f"    balanced acc {o['balanced_accuracy']:.4f} "
                f"(train {ofit['train_bal_acc']:.4f} / "
                f"val {ofit['val_bal_acc']:.4f}) | "
                f"bal gap {ofit['bal_acc_gap_train_minus_val']:+.4f}")
            log(f"    verdict: {ofit['verdict']}")

            row = {"arm": arm_name, "model": mname, "mode": mode, **o,
                   **{f"ofit_{k}": v for k, v in ofit.items()},
                   "train_time_s": round(info["train_time_s"], 1),
                   "best_val_loss": round(info["best_val_loss"], 5)}
            row.update({k: info[k] for k in
                        ("epochs_run", "rounds_run", "n_clients") if k in info})
            row.update({f"arm_{k}": v for k, v in st.items()})
            if mode == "federated":
                row["fl_algorithm"] = algo      # read by the resume guard

            # ---- per-client evaluation on each client's own test frames ----
            if client_states is not None and \
                    cfg.get("report_per_client", True) and \
                    shards_va_full is not None:
                pc_rows, pc_agg = evaluate_per_client(
                    mname, cfg, d, client_states, client_names_full,
                    shards_va_full, shards_te_full, info.get("client_sizes", []),
                    fixed_threshold=(thr if algo == "fedavg" else None))
                row.update(pc_agg)
                row["fl_algorithm"] = algo
                log(f"    {algo} per-client: mean balanced acc "
                    f"{pc_agg['perclient_bal_acc_mean']:.4f}  spread "
                    f"{pc_agg['perclient_bal_acc_spread']:.4f}  over "
                    f"{pc_agg['perclient_n_clients_evaluated']} clients  "
                    f"(vs. pooled test {o['balanced_accuracy']:.4f})")
                for r in pc_rows:
                    perclient_rows.append({"arm": arm_name, "model": mname,
                                                "mode": mode, **r})
            summary.append(row)
            for _k, r in sorted(res["per_snr"].items(),
                                key=lambda kv: kv[1]["snr_mid_db"]):
                per_snr_rows.append({"arm": arm_name, "model": mname,
                                     "mode": mode, **r})

            tag = f"{mname}_{mode}"
            torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}.pt"))
            np.savez_compressed(os.path.join(out_dir, f"roc_{tag}.npz"),
                                y=res["y"], p1=res["p1"], snr=res["snr"])
            with open(os.path.join(out_dir, f"history_{tag}.json"), "w") as f:
                json.dump(info, f, indent=2, default=str)

            # Marker LAST: it is the commit point. If the machine dies before
            # this line, the unit is simply retrained on the next run; if it
            # dies after, the unit is never repeated.
            unit_snr = [{"arm": arm_name, "model": mname, "mode": mode, **r}
                        for _k, r in sorted(res["per_snr"].items(),
                                            key=lambda kv: kv[1]["snr_mid_db"])]
            save_marker(cfg, ukey, cfp, dfp, row, unit_snr)
            log(f"    checkpoint saved -> {ukey}.json")

    write_csv(os.path.join(out_dir, "summary_overall.csv"), summary)
    write_csv(os.path.join(out_dir, "metrics_per_snr.csv"), per_snr_rows)
    write_csv(os.path.join(out_dir, "complexity.csv"), cx_rows)
    # Per-client breakdown. NOTE: unlike the other
    # three CSVs, this one is populated only from units trained THIS run --
    # a unit loaded from a resume marker contributes its aggregate
    # perclient_* columns to summary_overall.csv (saved in the marker)
    # but not its row-level breakdown here, since markers do not store it.
    # Delete the checkpoint folder (or force_retrain the federated units) to
    # regenerate this file in full.
    write_csv(os.path.join(out_dir, "summary_mira_per_client.csv"),
              perclient_rows)
    return summary, per_snr_rows, cx_rows, perclient_rows


# ==========================================================================
#   CLIENT-COUNT SENSITIVITY SWEEP  (paper Fig. 8a)
# ==========================================================================
PART_NAME = {"natural": "noniid", "iid": "iid", "dirichlet": "dirichlet"}


def run_client_count_ablation(arm_name: str, d: Dict, cfg: Dict, out_dir: str,
                              cfp: str = "", dfp: str = "") -> List[Dict]:
    """
    Client COUNT x PARTITION sweep (paper Fig. 8a):
    FedAvg at every C in cfg["client_count_ablation"] ({4, 8, 12, 16, 24, 32})
    under every partition in cfg["ablation_partitions"]
        natural  non-IID: a client = one stretch of the evening (whole split
                 blocks), adjacent blocks merged until both classes are held;
                 several radios: the C clients are shared out among them
        iid      control: uniformly random equal shares
    for every model in cfg["ablation_models"] ([] = all 13). Same test frames
    for every unit; IID and non-IID at the same (C, model) start from the
    same initial weights; C = n_time_blocks_single_device (8) of the natural
    partition is seeded like the main federated unit, so it reproduces the
    Table I FedAvg row. Writes ablation_num_clients.csv (partition column)
    and ablation_num_clients_per_client.csv. Every unit is checkpointed.

    The realised client count can be LOWER than requested: adjacent blocks
    are merged until every client holds both classes (C_realized column).
    """
    have_spec = "X_spec" in d
    models = cfg.get("ablation_models") or cfg["models"]
    models = [m for m in models if m not in IS_2D or have_spec]
    c_list = [int(c) for c in cfg.get("client_count_ablation", [])]
    parts = list(cfg.get("ablation_partitions") or ["natural"])
    if not models or not c_list:
        return []

    tr, va, te = make_splits(d, cfg)
    w = class_weights(d["y"][tr])
    n_dev = len(set(d["device"].tolist()))
    rows: List[Dict] = []
    pc_all: List[Dict] = []
    algo = cfg.get("fl_algorithm", "fedavg")
    base_c = int(cfg.get("n_time_blocks_single_device", 8))

    log("\n" + "=" * 74)
    log(f"CLIENT-COUNT x PARTITION ABLATION ({algo})  arm={arm_name}  C in {c_list}  "
        f"partitions {[PART_NAME.get(p, p) for p in parts]}")
    log(f"  models: {models}")
    log(f"  ({len(c_list) * len(models) * len(parts)} federated trainings; finished "
        f"units are skipped)")
    log("=" * 74)

    for part in parts:
        for C in c_list:
            c_cfg = dict(cfg)
            c_cfg["client_partition"] = part
            c_cfg["n_time_blocks_single_device"] = C          # non-IID count, one radio
            c_cfg["n_clients"] = C                            # IID count
            if n_dev > 1:
                c_cfg["_ablation_clients"] = C                # non-IID count, several radios
            log(f"\n  [{PART_NAME.get(part, part)}] C = {C}")
            shards_tr, shards_va, shards_te, names = \
                full_session_client_shards(d, tr, va, te, c_cfg)
            realized_k = len(names)
            cfp_c = config_fingerprint(c_cfg)

            for mname in models:
                use_spec = mname in IS_2D
                # natural keeps the v3 key, so finished v3 sweeps are re-used
                key = (f"ablation__{arm_name}__{algo}__C{C}__{mname}" if part == "natural"
                       else f"ablation__{arm_name}__{algo}__{part}__C{C}__{mname}")
                cached = load_marker(c_cfg, key, cfp_c, dfp)
                if cached is not None:
                    r = dict(cached["summary_row"]); r.setdefault("partition", PART_NAME.get(part, part))
                    rows.append(r)
                    pc_all.extend(cached.get("per_snr_rows", []))
                    log(f"  C={C:<3d} {mname:<24s} ALREADY RUN -> skipping")
                    continue

                va_l = DataLoader(FrameDS(d, va, use_spec, augment=False, cfg=c_cfg),
                                  batch_size=c_cfg["batch_size"], shuffle=False,
                                  num_workers=c_cfg["num_workers"])
                te_l = DataLoader(FrameDS(d, te, use_spec, augment=False, cfg=c_cfg),
                                  batch_size=c_cfg["batch_size"], shuffle=False,
                                  num_workers=c_cfg["num_workers"])
                log(f"  C={C:<3d} (realised {realized_k})  {mname}")
                # seed independent of the partition: IID and non-IID start from
                # the same weights; C = base_c reproduces the main FedAvg unit
                if C == base_c:
                    seed_unit(cfg["seed"], arm_name, mname, "federated")
                else:
                    seed_unit(cfg["seed"], arm_name, mname, "federated", f"C{C}")
                if algo == "mira":
                    model, info, client_states = train_mira(
                        mname, d, tr, va_l, c_cfg, w,
                        client_shards=(shards_tr, names))
                else:
                    model, info = train_federated(
                        mname, d, tr, va_l, c_cfg, w,
                        client_shards=(shards_tr, names))
                    g = {k: v.detach().cpu().clone()
                         for k, v in model.state_dict().items()}
                    client_states = [g] * len(names)
                thr = pick_threshold(model, va_l, c_cfg)
                o = evaluate(model, te_l, c_cfg, threshold=thr)["overall"]
                pc_rows, pc_agg = evaluate_per_client(
                    mname, c_cfg, d, client_states, names, shards_va, shards_te,
                    info.get("client_sizes", []),
                    fixed_threshold=(thr if algo == "fedavg" else None))
                pc_rows = [{"arm": arm_name, "model": mname, "fl_algorithm": algo,
                            "partition": PART_NAME.get(part, part),
                            "C_requested": C, "C_realized": realized_k, **r}
                           for r in pc_rows]
                pc_all.extend(pc_rows)
                row = {"arm": arm_name, "model": mname, "fl_algorithm": algo,
                       "partition": PART_NAME.get(part, part),
                       "family": "proposed" if mname in NOVEL_DETECTORS else "baseline",
                       "C_requested": C,
                       "C_realized": realized_k,
                       "consensus_balanced_accuracy": o["balanced_accuracy"],
                       "consensus_f1": o["f1"], "consensus_fdr": o["fdr"],
                       "consensus_recall_pd": o["recall_pd"],
                       "consensus_auc": o["auc"], "consensus_accuracy": o["accuracy"],
                       "threshold": thr,
                       **pc_agg, "rounds_run": info.get("rounds_run"),
                       "train_time_s": round(info["train_time_s"], 1)}
                log(f"    pooled-test balAcc {o['balanced_accuracy']:.4f}  |  "
                    f"per-client mean balAcc "
                    f"{pc_agg['perclient_bal_acc_mean']:.4f} "
                    f"(spread {pc_agg['perclient_bal_acc_spread']:.4f}, "
                    f"{pc_agg['perclient_n_clients_evaluated']} clients)")
                rows.append(row)
                save_marker(c_cfg, key, cfp_c, dfp, row, pc_rows)

    rows.sort(key=lambda r: (str(r.get("partition")), str(r.get("model")),
                             int(r.get("C_requested", 0))))
    write_csv(os.path.join(out_dir, "ablation_num_clients.csv"), rows)
    write_csv(os.path.join(out_dir, "ablation_num_clients_per_client.csv"), pc_all)
    # cost of heterogeneity: BA(IID) - BA(non-IID) at every (model, C)
    gap = []
    by = {(r["model"], int(r["C_requested"]), r.get("partition")): r for r in rows}
    for (m, C, p), r in sorted(by.items()):
        if p == "noniid" and (m, C, "iid") in by:
            gap.append({"arm": arm_name, "model": m, "C_requested": C,
                        "bal_acc_noniid": r["consensus_balanced_accuracy"],
                        "bal_acc_iid": by[(m, C, "iid")]["consensus_balanced_accuracy"],
                        "cost_of_heterogeneity": by[(m, C, "iid")]["consensus_balanced_accuracy"]
                        - r["consensus_balanced_accuracy"]})
    write_csv(os.path.join(out_dir, "ablation_heterogeneity_gap.csv"), gap)
    return rows


# ==========================================================================
#   LEAVE-ONE-BAND-OUT CONFOUND AUDIT  (point 3 of the header: the frequency
#   confound). Previously only a flag; implemented here.
# ==========================================================================
def _load_trained(arm_dir: str, mname: str, mode: str, cfg: Dict) -> Optional[nn.Module]:
    """Model saved by run_arm (<arm>/<model>_<mode>.pt) or None."""
    p = os.path.join(arm_dir, f"{mname}_{mode}.pt")
    if not os.path.isfile(p):
        return None
    m = build_model(mname, cfg)
    if mode == "federated" and cfg.get("fl_norm", "groupnorm") == "groupnorm":
        m = convert_bn_to_gn(m)
    m.load_state_dict(torch.load(p, map_location="cpu"))
    return m.to(DEVICE).eval()


def _subset_metrics(model: nn.Module, d: Dict, idx: np.ndarray, use_spec: bool,
                    cfg: Dict, thr: float) -> Dict:
    if len(idx) == 0:
        return {}
    ld = DataLoader(FrameDS(d, idx, use_spec, augment=False, cfg=cfg),
                    batch_size=cfg["batch_size"], shuffle=False,
                    num_workers=cfg["num_workers"])
    p1, y, _ = _scores(model, ld)
    o = binary_metrics(y, (p1 >= thr).astype(np.int64), p1)
    two = len(np.unique(y)) > 1
    o["balanced_accuracy"] = (0.5 * (o["recall_pd"] + 1.0 - o["fdr"]) if two
                              else float("nan"))
    o["h1_fraction"] = float((y == 1).mean())
    return o


def run_leave_one_band_out(arm_name: str, d: Dict, cfg: Dict, out_dir: str,
                           cfp: str = "", dfp: str = "") -> List[Dict]:
    """
    For every band b: retrain each model WITHOUT any frame of b (train and
    validation of the other bands, threshold on their validation), then
    score it on the TEST frames of b. The same test frames are scored with
    the main model of the run matrix (trained with b) -> 'inband_*' columns.
    A large drop (delta_*) means the model rode the band identity
    (frequency confound) instead of detecting energy structure.
    On a single-class band only Pd (all-H1) or Pfa (all-H0) is defined; the
    other columns are NaN, not fabricated. Writes leave_one_band_out.csv.
    """
    have_spec = "X_spec" in d
    models = [m for m in (cfg.get("lobo_models") or cfg["models"])
              if m not in IS_2D or have_spec]
    modes = list(cfg.get("lobo_modes") or ["centralized"])
    tr, va, te = make_splits(d, cfg)
    bands = sorted(set(d["band"].tolist()))
    log("\n" + "=" * 74)
    log(f"LEAVE-ONE-BAND-OUT  arm={arm_name}  {len(bands)} bands x {len(models)} models x "
        f"{modes}")
    log("=" * 74)
    arm_cfg = dict(cfg)
    rows: List[Dict] = []
    main_rows = collect_markers(cfg, cfp, dfp)[0]          # thresholds of the in-band models
    for b in bands:
        inb = d["band"] == b
        tr_b, va_b, te_b = tr[~inb[tr]], va[~inb[va]], te[inb[te]]
        if len(te_b) == 0 or len(set(d["y"][tr_b].tolist())) < 2 or \
                len(set(d["y"][va_b].tolist())) < 2:
            log(f"  {b}: skipped (no test frames, or the other bands are single-class)")
            continue
        w = class_weights(d["y"][tr_b])
        log(f"\n  held-out band {b}: train {len(tr_b):,} / val {len(va_b):,} (other bands), "
            f"test {len(te_b):,} (H1 {float((d['y'][te_b] == 1).mean()) * 100:.1f}%)")
        for mname in models:
            use_spec = mname in IS_2D
            for mode in modes:
                key = f"lobo__{arm_name}__{b}__{mname}__{mode}"
                cached = load_marker(cfg, key, cfp, dfp)
                if cached is not None:
                    rows.append(cached["summary_row"])
                    log(f"    {mname:<24s} {mode:<12s} ALREADY RUN -> skipping")
                    continue
                seed_unit(cfg["seed"], arm_name, mname, mode, "lobo", b)
                mk = lambda idx, sh: DataLoader(            # noqa: E731
                    FrameDS(d, idx, use_spec, augment=(sh and cfg.get("augment_train", False)),
                            cfg=cfg),
                    batch_size=cfg["batch_size"], shuffle=sh, num_workers=cfg["num_workers"])
                va_l = mk(va_b, False)
                if mode == "centralized":
                    model, info = train_centralized(build_model(mname, cfg), mk(tr_b, True),
                                                    va_l, cfg, w)
                else:
                    model, info = train_federated(mname, d, tr_b, va_l, arm_cfg, w)
                thr = pick_threshold(model, va_l, cfg)
                o = _subset_metrics(model, d, te_b, use_spec, cfg, thr)
                row = {"arm": arm_name, "held_out_band": b, "model": mname, "mode": mode,
                       "family": "proposed" if mname in NOVEL_DETECTORS else "baseline",
                       "n_test": o.get("n"), "h1_fraction": o.get("h1_fraction"),
                       "threshold": thr,
                       **{k: o.get(k) for k in ("balanced_accuracy", "accuracy", "recall_pd",
                                                "fdr", "mdr", "f1", "auc")},
                       "train_time_s": round(info["train_time_s"], 1)}
                ref = _load_trained(out_dir, mname, mode, cfg)
                if ref is not None:
                    mrow = next((r for r in main_rows
                                 if r.get("arm") == arm_name and r.get("model") == mname
                                 and r.get("mode") == mode), None)
                    thr_ref = float(mrow["threshold"]) if mrow else 0.5
                    oi = _subset_metrics(ref, d, te_b, use_spec, cfg, thr_ref)
                    for k in ("balanced_accuracy", "recall_pd", "fdr", "auc"):
                        row[f"inband_{k}"] = oi.get(k)
                        a_, b_ = o.get(k), oi.get(k)
                        row[f"delta_{k}"] = (a_ - b_) if (a_ is not None and b_ is not None) \
                            else float("nan")
                log(f"    {mname:<24s} {mode:<12s} held-out balAcc "
                    f"{row.get('balanced_accuracy', float('nan')):.4f}  Pd "
                    f"{row.get('recall_pd', float('nan')):.4f}  FDR "
                    f"{row.get('fdr', float('nan')):.4f}"
                    + (f"  (in-band balAcc {row['inband_balanced_accuracy']:.4f})"
                       if row.get("inband_balanced_accuracy") is not None else ""))
                rows.append(row)
                save_marker(cfg, key, cfp, dfp, row, [])
    write_csv(os.path.join(out_dir, "leave_one_band_out.csv"), rows)
    return rows


# ==========================================================================
#   LABEL-SCHEME ABLATION  (point 6: dead-zone vs margin labels)
# ==========================================================================
def run_label_scheme_ablation(cfg: Dict, main_rows: List[Dict]) -> List[Dict]:
    """
    Re-label the corpus with every scheme in cfg["label_scheme_ablation"]
    (own frame cache, keyed on the label settings) and re-run the run matrix
    (centralised + federated) for cfg["label_ablation_models"] ([] = all).
    ablation_label_scheme.csv holds these rows next to the main 'margin'
    rows, so the before/after effect of the labels is one table. Both
    schemes are scored against their OWN labels: the comparison is of how
    hard each labelled problem is, not of a common ground truth.
    """
    out = [dict(r, label_scheme=cfg["label_scheme"]) for r in main_rows]
    for scheme in cfg.get("label_scheme_ablation") or []:
        if scheme == cfg["label_scheme"]:
            continue
        c2 = dict(cfg)
        c2["label_scheme"] = scheme
        c2["models"] = list(cfg.get("label_ablation_models") or cfg["models"])
        c2["profile_complexity"] = False
        log("\n" + "=" * 74)
        log(f"LABEL-SCHEME ABLATION: '{scheme}'  models {c2['models']}")
        log("=" * 74)
        try:
            d2 = load_dataset(c2)
        except SystemExit as e:
            log(f"  scheme '{scheme}' aborted: {e}")
            continue
        cfp2, dfp2 = config_fingerprint(c2), dataset_fingerprint(d2)
        for arm in cfg["dataset_arms"]:
            dd = filter_arm(d2, arm["devices"])
            if dd is None or len(dd["y"]) < 40:
                continue
            name = f"{arm['name']}__labels_{scheme}"
            try:
                sm, _, _, _ = run_arm(name, dd, c2, os.path.join(
                    cfg["results_dir"], "ablation_label_scheme", scheme, arm["name"]),
                    cfp=cfp2, dfp=dfp2)
            except SystemExit as e:
                log(f"  arm '{name}' aborted: {e}")
                continue
            st = arm_stats(dd)
            out.extend(dict(r, label_scheme=scheme, n_frames_labelled=st["n_frames"])
                       for r in sm)
    write_csv(os.path.join(cfg["results_dir"], "ablation_label_scheme.csv"), out)
    return out


# ==========================================================================
#   SENSING -> FORECASTING HAND-OFF
# ==========================================================================
def export_handoff(cfg: Dict, summary: List[Dict]) -> Optional[str]:
    """
    Run the chosen detectors over EVERY frame of the pooled arm and write
    <handoff_dir>/frames.npz + manifest.json for 02_forecasting.py:
        utc, device, band, center_hz, snr (estimated SNR proxy, dB),
        y_energy  margin energy-detector label,
        split     0 / 1 / 2 = train / val / test of make_splits (the
                  forecasting stage reports how many of its test targets
                  come from sensing-TEST frames),
        block     split block,
        p__<src>  P(H1) of each source detector; its validation (Neyman-
                  Pearson, Pfa = target_pfa) threshold is in the manifest.
    Sources (all of the same mode, FedAvg by default):
             oscar_net_p1
                    = OSCAR-Net, the main source of stage 2 (handoff_detector),
             sota   = best non-proposed detector by VALIDATION balanced
                      accuracy (never by test),
             energy = y_energy itself.
    """
    arms = [a for a in cfg["dataset_arms"] if not a["devices"]] or cfg["dataset_arms"]
    arm = arms[0]["name"]
    d = ARM_DATA.get(arm)
    if d is None:
        log(f"\n  hand-off: arm '{arm}' not loaded -> nothing exported")
        return None
    arm_dir = os.path.join(cfg["results_dir"], arm)
    hd = cfg.get("handoff_dir") or os.path.join(cfg["results_dir"], "sensing_handoff")
    os.makedirs(hd, exist_ok=True)
    mode = cfg.get("handoff_mode", "federated")
    have_spec = "X_spec" in d
    rows = [r for r in summary if r.get("arm") == arm and r.get("mode") == mode]
    # Trained under another configuration (other epochs, seeds, version of
    # this file ...)? The hand-off only needs each model's saved weights
    # (<arm>/<model>_<mode>.pt) and its validation threshold, so it also
    # reads markers whose CONFIG fingerprint differs (same dataset required).
    have = {r["model"] for r in rows}
    dfp_now = dataset_fingerprint(d)
    other_cfg = []
    for m in list(dict.fromkeys(list(cfg["models"]) + list(NOVEL_DETECTORS))):
        if m in have or not os.path.isfile(os.path.join(arm_dir, f"{m}_{mode}.pt")):
            continue
        try:
            with open(_marker_path(cfg, unit_key(arm, m, mode))) as fh:
                mk = json.load(fh)
        except Exception:
            continue
        if not mk.get("complete") or not mk.get("summary_row"):
            continue
        if mk.get("dataset_fingerprint") != dfp_now:
            log(f"  {m}: marker is from a DIFFERENT dataset -> not used")
            continue
        rows.append(dict(mk["summary_row"], arm=arm, mode=mode, model=m))
        other_cfg.append(m)
    if other_cfg:
        log(f"  using trained weights + validation thresholds of {len(other_cfg)} model(s) whose "
            f"markers carry another config fingerprint: {', '.join(other_cfg)}")
    row_of = {r["model"]: r for r in rows}
    log("\n" + "=" * 74)
    log(f"SENSING -> FORECASTING HAND-OFF  (arm {arm}, mode {mode})  -> {hd}")
    log("=" * 74)

    def vba(r):
        v = r.get("ofit_val_bal_acc")
        return float(v) if v is not None and np.isfinite(float(v)) else -1.0

    src: Dict[str, str] = {m: m for m in NOVEL_DETECTORS if m in row_of}   # all proposed detectors
    nov = cfg.get("handoff_detector", "oscar_net_p1")
    if nov not in src:
        cand = [r for r in rows if r["model"] in NOVEL_DETECTORS]
        if cand:
            nov = max(cand, key=lambda r: (vba(r), r["model"]))["model"]
            log(f"  {cfg.get('handoff_detector')} ({mode}) not trained -> main source = {nov}")
    main_key = nov if nov in src else None
    sota = cfg.get("handoff_sota", "auto")
    if sota == "auto":
        cand = [r for r in rows if r["model"] not in NOVEL_DETECTORS
                and (r["model"] not in IS_2D or have_spec)]
        if cand:
            src["sota"] = max(cand, key=lambda r: (vba(r), r["model"]))["model"]
    elif sota in row_of:
        src["sota"] = sota

    n = len(d["y"])
    tr, va, te = make_splits(d, cfg)
    split = np.full(n, -1, dtype=np.int8)
    split[tr], split[va], split[te] = 0, 1, 2
    arr = {"utc": np.asarray(d["utc"], np.float64),
           "device": np.asarray(d["device"]).astype(str),
           "band": np.asarray(d["band"]).astype(str),
           "center_hz": np.asarray(d["center_hz"], np.float64),
           "snr": np.asarray(d["snr"], np.float32),
           "y_energy": np.asarray(d["y"], np.int8),
           "split": split, "block": np.asarray(d["_split_block"], np.int32)}
    man_src = {"energy": {"model": "energy_detector", "mode": "rule", "threshold": 0.5,
                          "description": f"{cfg['label_scheme']} labels (floor Q{cfg['noise_percentile']:g}, "
                                         f"eta {cfg['occ_margin_db']:g} dB, busy >= "
                                         f"{cfg['margin_busy_frac']:g})"}}
    for key, m in src.items():
        model = _load_trained(arm_dir, m, mode, cfg)
        if model is None:
            log(f"  {key}: {m}_{mode}.pt not found in {arm_dir} -> source skipped "
                f"(run --stages main first)")
            continue
        r = row_of[m]
        ld = DataLoader(FrameDS(d, np.arange(n), m in IS_2D, augment=False, cfg=cfg),
                        batch_size=int(cfg.get("handoff_batch", 256)), shuffle=False,
                        num_workers=cfg["num_workers"])
        t0 = time.time()
        p1, _, _ = _scores(model, ld)
        thr = float(r.get("threshold", 0.5))
        arr[f"p__{key}"] = p1.astype(np.float32)
        yh = (p1 >= thr).astype(np.int8)
        man_src[key] = {
            "model": m, "mode": mode, "threshold": thr,
            "family": "proposed" if m in NOVEL_DETECTORS else "baseline",
            "selected_by": ("proposed detector" if key in NOVEL_DETECTORS else
                            ("validation balanced accuracy" if sota == "auto" else "handoff_sota")),
            "val_balanced_accuracy": r.get("ofit_val_bal_acc"),
            "test_balanced_accuracy": r.get("balanced_accuracy"), "test_auc": r.get("auc"),
            "test_pd": r.get("recall_pd"), "test_fdr": r.get("fdr"),
            "agreement_with_energy_labels": {nm: float((yh[ix] == d["y"][ix]).mean())
                                             for nm, ix in (("train", tr), ("val", va), ("test", te))},
            "sensed_occupancy_rate": float(yh.mean())}
        log(f"  {key:<15s} {m:<24s} thr {thr:.4f}  test balAcc "
            f"{float(r.get('balanced_accuracy', float('nan'))):.4f}  sensed occupancy "
            f"{yh.mean() * 100:.1f}%  ({time.time() - t0:.0f} s, {n:,} frames)")
    if not any(k.startswith("p__") for k in arr):
        log(f"  NO trained detector found in {arm_dir} (no <model>_{mode}.pt with a marker): "
            f"nothing exported, the existing hand-off is left untouched. Run --stages main first.")
        return None
    tmp = os.path.join(hd, f"frames.tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, os.path.join(hd, "frames.npz"))
    nb = max(1, int(cfg.get("split_blocks", 32)),
             int(cfg.get("n_time_blocks_single_device", 8)),
             *[int(c) for c in cfg.get("client_count_ablation", []) or [1]])
    _atomic_write_json(os.path.join(hd, "manifest.json"), {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "01_sensing", "arm": arm, "n_frames": n,
        "devices": sorted(set(arr["device"].tolist())),
        "bands": sorted(set(arr["band"].tolist())),
        "main_source": (main_key if main_key and f"p__{main_key}" in arr else
                        next((k for k in src if f"p__{k}" in arr), "energy")),
        "sources": {k: v for k, v in man_src.items() if k == "energy" or f"p__{k}" in arr},
        "seed": cfg["seed"], "split_blocks": nb, "val_frac": cfg["val_frac"],
        "test_frac": cfg["test_frac"], "target_pfa": cfg.get("target_pfa", 0.10),
        "threshold_mode": cfg.get("threshold_mode"),
        "label_scheme": cfg["label_scheme"], "noise_percentile": cfg["noise_percentile"],
        "occ_margin_db": cfg["occ_margin_db"], "margin_busy_frac": cfg["margin_busy_frac"],
        "n_time_blocks_single_device": cfg["n_time_blocks_single_device"],
        "client_count_ablation": cfg.get("client_count_ablation"),
        "min_per_class_per_client": cfg["min_per_class_per_client"],
        "fl_norm": cfg.get("fl_norm"), "frame_len": cfg["frame_len"]})
    log(f"  -> {os.path.join(hd, 'frames.npz')}  +  manifest.json")
    return hd

# ==========================================================================
#   REPEATED TRAINING SEEDS  (main table as mean +- std)
# ==========================================================================
SEED_METRICS = ("balanced_accuracy", "f1", "auc", "recall_pd", "fdr", "accuracy",
                "accuracy_at_0.5", "pd_at_0.5", "threshold", "rounds_run", "epochs_run")


def run_seed_repeats(cfg: Dict, primary_rows: List[Dict], dfp: str) -> List[Dict]:
    """
    Re-run the main run matrix for every extra seed in cfg["train_seeds"]
    (same data, split, labels and clients; only the training randomness
    changes) and write
        ALL_arms_summary_seeds.csv     one row per (seed, arm, model, mode)
        ALL_arms_summary_mean_std.csv  mean, std (ddof=1) and 'mean +- std'
    The primary seed (== cfg["seed"]) is the run already in results_dir.
    Extra seeds live in results_dir/seeds/seed_<s>/ with their own markers,
    so an interrupted repeat resumes like the main sweep.
    """
    base = int(cfg["seed"])
    seeds = [int(x) for x in (cfg.get("train_seeds") or [base])]
    seeds = list(dict.fromkeys(seeds))                    # unique, keep order
    long: List[Dict] = ([dict(r, train_seed=base) for r in primary_rows]
                        if base in seeds else [])
    for sd in [x for x in seeds if x != base]:
        c = dict(cfg)
        c["train_seed"] = sd
        c["results_dir"] = os.path.join(cfg["results_dir"], "seeds", f"seed_{sd}")
        c["profile_complexity"] = False                   # identical across seeds
        os.makedirs(c["results_dir"], exist_ok=True)
        cfp_s = config_fingerprint(c)
        log("\n" + "=" * 74)
        log(f"REPEAT SEED {sd}  ({seeds.index(sd) + 1}/{len(seeds)})  -> {c['results_dir']}")
        log("=" * 74)
        for arm in cfg["dataset_arms"]:
            d = ARM_DATA.get(arm["name"])
            if d is None:
                continue
            try:
                sm, _, _, _ = run_arm(arm["name"], d, c,
                                      os.path.join(c["results_dir"], arm["name"]),
                                      cfp=cfp_s, dfp=dfp)
            except SystemExit as e:
                log(f"  seed {sd}, arm '{arm['name']}' aborted: {e}")
                continue
            long.extend(dict(r, train_seed=sd) for r in sm)
    if len(seeds) < 2 or not long:
        return long

    rd = cfg["results_dir"]
    write_csv(os.path.join(rd, "ALL_arms_summary_seeds.csv"), long)
    groups: Dict[Tuple, List[Dict]] = collections.OrderedDict()
    for r in long:
        groups.setdefault((r.get("arm"), r.get("model"), r.get("mode")), []).append(r)
    agg: List[Dict] = []
    for (a, m, md), rs in groups.items():
        row = {"arm": a, "model": m, "mode": md,
               "family": "proposed" if m in NOVEL_DETECTORS else "baseline",
               "n_seeds": len(rs),
               "seeds": " ".join(str(r["train_seed"]) for r in rs)}
        for k in SEED_METRICS:
            v = np.array([float(r[k]) for r in rs
                          if r.get(k) not in (None, "") and np.isfinite(float(r[k]))])
            mu = float(v.mean()) if v.size else float("nan")
            sdv = float(v.std(ddof=1)) if v.size > 1 else float("nan")
            row[f"{k}_mean"], row[f"{k}_std"] = mu, sdv
            row[f"{k}_pm"] = f"{mu:.4f} +- {sdv:.4f}" if v.size > 1 else f"{mu:.4f}"
        agg.append(row)
    # rank inside each (arm, mode) by mean balanced accuracy
    for key in sorted({(r["arm"], r["mode"]) for r in agg}):
        sub = sorted([r for r in agg if (r["arm"], r["mode"]) == key],
                     key=lambda r: -r["balanced_accuracy_mean"])
        for i, r in enumerate(sub, 1):
            r["rank_bal_acc"] = i
    agg.sort(key=lambda r: (str(r["arm"]), str(r["mode"]), r["rank_bal_acc"]))
    write_csv(os.path.join(rd, "ALL_arms_summary_mean_std.csv"), agg)

    log("\n" + "=" * 74)
    log(f"MAIN TABLE OVER {len(seeds)} TRAINING SEEDS (mean +- std, test set)")
    log("=" * 74)
    for key in sorted({(r["arm"], r["mode"]) for r in agg}):
        log(f"\n  [{key[0]}] {key[1]}")
        log(f"  {'#':>2} {'model':<24}{'n':>3}  {'balAcc':<17}{'F1':<17}{'AUC':<17}"
            f"{'Acc@0.5':<17}")
        for r in [r for r in agg if (r["arm"], r["mode"]) == key]:
            log(f"  {r['rank_bal_acc']:>2} {r['model']:<24}{r['n_seeds']:>3}  "
                f"{r['balanced_accuracy_pm']:<17}{r['f1_pm']:<17}{r['auc_pm']:<17}"
                f"{r['accuracy_at_0.5_pm']:<17}")
    log(f"\n  -> ALL_arms_summary_mean_std.csv, ALL_arms_summary_seeds.csv in {rd}")
    return long


def _parse_cli(cfg: Dict) -> set:
    """Command line (all optional; no arguments = the paper's stages)."""
    import argparse
    ap = argparse.ArgumentParser(description="Stage 1: spectrum sensing (OSCAR-Net + five SOTA "
                                             "detectors), centralized and FedAvg, + hand-off")
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                    help=f"subset of {','.join(ALL_STAGES)} (default {','.join(DEFAULT_STAGES)})")
    ap.add_argument("--root", help="dataset folder with the export .h5 files (or $SFS_DATA)")
    ap.add_argument("--out", help="results folder (default $SFS_RESULTS/01_sensing)")
    ap.add_argument("--cache", help="frame-cache folder (default <out>/_frame_cache)")
    ap.add_argument("--max-frames", type=int, help="frames per file (0 = all)")
    ap.add_argument("--models", help="models of the run matrix, e.g. oscar_net_p1,deepsense_s4")
    ap.add_argument("--ablation-models", help="client-count sweep models ('' = all)")
    ap.add_argument("--lobo-models", help="leave-one-band-out models ('' = all)")
    ap.add_argument("--label-models", help="label-scheme ablation models ('' = all)")
    ap.add_argument("--counts", help="client counts, e.g. 4,8,12,16,24,32")
    ap.add_argument("--partitions", help="natural,iid")
    ap.add_argument("--frame-len", type=int, help="model input length (default 32768)")
    ap.add_argument("--quick", action="store_true", help="5 epochs / 5 rounds smoke test")
    ap.add_argument("--seeds", help="training seeds of the main table, e.g. "
                                    "20260807,20260808,20260809 (first = primary run)")
    a = ap.parse_args()
    lst = lambda v: [x.strip() for x in v.split(",") if x.strip()]      # noqa: E731
    st = set(lst(a.stages))
    bad = st - set(ALL_STAGES)
    if bad:
        sys.exit(f"unknown stage(s) {sorted(bad)}; choose from {ALL_STAGES}")
    if a.root:
        cfg["root"] = a.root
    if a.out:
        cfg["results_dir"] = a.out
    if a.cache:
        cfg["cache_dir"] = a.cache
    if a.max_frames is not None:
        cfg["max_frames_per_file"] = int(a.max_frames)
    if a.models:
        cfg["models"] = lst(a.models)
    if a.ablation_models is not None:
        cfg["ablation_models"] = lst(a.ablation_models)
    if a.lobo_models is not None:
        cfg["lobo_models"] = lst(a.lobo_models)
    if a.label_models is not None:
        cfg["label_ablation_models"] = lst(a.label_models)
    if a.counts:
        cfg["client_count_ablation"] = [int(x) for x in lst(a.counts)]
    if a.partitions:
        cfg["ablation_partitions"] = lst(a.partitions)
    if a.frame_len:
        cfg["frame_len"] = int(a.frame_len)
    if a.seeds:
        cfg["train_seeds"] = [int(x) for x in lst(a.seeds)]
    if a.quick:
        cfg.update({"epochs": 5, "min_epochs": 2, "fl_rounds": 5, "fl_min_rounds": 2,
                    "prof_iters": 2, "prof_warmup": 1, "prof_b1_iters": 2})
    unknown = [m for m in cfg["models"] + list(cfg.get("ablation_models") or [])
               + list(cfg.get("lobo_models") or []) + list(cfg.get("label_ablation_models") or [])
               if m not in MODEL_REFS]
    if unknown:
        sys.exit(f"unknown model(s) {sorted(set(unknown))}; choose from {sorted(MODEL_REFS)}")
    return st


def main() -> None:
    global STAGES
    cfg = _env(CONFIG)
    STAGES = _parse_cli(cfg)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    log("=" * 74)
    log(f"Spectrum sensing on real captures | train {DEVICE} | profiling CPU")
    log("=" * 74)

    files = discover_h5(cfg)
    hardware_audit(files)

    log(f"  stages: {', '.join(s for s in ALL_STAGES if s in STAGES)}")
    log("\n[1] Loading exported sessions")
    d_all = load_dataset(cfg)
    print_run_matrix(cfg, "X_spec" in d_all)

    # ---------------- resume bookkeeping ----------------
    cfp = config_fingerprint(cfg)
    dfp = dataset_fingerprint(d_all)
    have_spec = "X_spec" in d_all
    _models = [m for m in cfg["models"] if m not in IS_2D or have_spec]
    _arms = [a["name"] for a in cfg["dataset_arms"]]
    n_done, n_total, todo = plan_status(cfg, _arms, _models, cfp, dfp)
    log("\n" + "=" * 74)
    log("RESUME STATUS")
    log("=" * 74)
    log(f"  output root        {cfg['results_dir']}")
    log(f"  checkpoint folder  {_done_dir(cfg)}")
    log(f"  config  fingerprint {cfp}")
    log(f"  dataset fingerprint {dfp}")
    log(f"  completed          {n_done}/{n_total} units")
    log(f"  remaining          {len(todo)}")
    if cfg.get("force_retrain"):
        log(f"  force_retrain      {cfg['force_retrain']}")
    if n_done and not todo:
        log("  Every unit is already trained. Aggregates will be rebuilt from")
        log("  the checkpoints and no training will run. Delete the checkpoint")
        log("  folder, or set force_retrain, to train again.")
    elif todo:
        log(f"  next: {todo[0]}")
    log("=" * 74)

    all_sm, all_ps, all_cx, all_mpc, arm_rows = [], [], [], [], []
    t0 = time.time()
    for arm in cfg["dataset_arms"]:
        name, devs = arm["name"], arm["devices"]
        log("\n" + "=" * 74)
        log(f"ARM: {name}   devices: {devs if devs else 'ALL'}")
        log("=" * 74)
        d = filter_arm(d_all, devs)
        if d is None or len(d["y"]) < 40:
            log(f"  skipped: too few frames for arm '{name}'")
            continue
        st = arm_stats(d); st["arm"] = name
        arm_rows.append(st)
        arm_dir = os.path.join(cfg["results_dir"], name)
        ARM_DATA[name] = d
        if "main" in STAGES:
            try:
                sm, ps, cx, mpc = run_arm(name, d, cfg, arm_dir, cfp=cfp, dfp=dfp)
            except SystemExit as e:
                log(f"  arm '{name}' aborted: {e}")
                continue
            all_sm += sm; all_ps += ps; all_cx += cx; all_mpc += mpc

        # ---- client-count x partition sweep (every model, every radio count) ----
        if "clients" in STAGES and cfg.get("client_count_ablation") and \
                "federated" in cfg["training_modes"]:
            try:
                run_client_count_ablation(name, d, cfg, arm_dir, cfp=cfp, dfp=dfp)
            except SystemExit as e:
                log(f"  client ablation of arm '{name}' aborted: {e}")

        # ---- leave-one-band-out confound audit ----
        if "lobo" in STAGES and cfg.get("eval_leave_one_band_out"):
            try:
                run_leave_one_band_out(name, d, cfg, arm_dir, cfp=cfp, dfp=dfp)
            except SystemExit as e:
                log(f"  LOBO of arm '{name}' aborted: {e}")

    rd = cfg["results_dir"]
    # Rebuild aggregates from every valid marker on disk, not only from what
    # this process trained. A run that completed four units yesterday and two
    # today still produces a six-unit CSV.
    m_sm, m_ps, m_cx = collect_markers(cfg, cfp, dfp)
    if len(m_sm) >= len(all_sm):
        _seen = set()
        merged = []
        for r in m_sm + all_sm:
            k = (r.get("arm"), r.get("model"), r.get("mode"))
            if k in _seen:
                continue
            _seen.add(k)
            merged.append(r)
        order = {unit_key(a["name"], m, md): i for i, (a, m, md) in enumerate(
            [(a, m, md) for a in cfg["dataset_arms"] for m in cfg["models"]
             for md in cfg["training_modes"]])}
        merged.sort(key=lambda r: order.get(
            unit_key(str(r.get("arm")), str(r.get("model")),
                     str(r.get("mode"))), 10**6))
        all_sm = merged
        all_ps = m_ps if len(m_ps) >= len(all_ps) else all_ps
        if len(m_cx) > len(all_cx):
            all_cx = m_cx
        log(f"\nAggregates rebuilt from {len(all_sm)} checkpointed units.")

    write_csv(os.path.join(rd, "ALL_arms_summary.csv"), all_sm)
    write_csv(os.path.join(rd, "ALL_arms_per_snr.csv"), all_ps)
    write_csv(os.path.join(rd, "ALL_arms_complexity.csv"), all_cx)
    write_csv(os.path.join(rd, "ALL_arms_mira_per_client.csv"), all_mpc)
    write_csv(os.path.join(rd, "arm_dataset_stats.csv"), arm_rows)
    with open(os.path.join(rd, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    # ---- repeated training seeds of the main table (mean +- std) ----
    if "main" in STAGES and len(cfg.get("train_seeds") or []) > 1:
        run_seed_repeats(cfg, all_sm, dfp)

    # ---- label-scheme ablation (re-labels the corpus, own frame cache) ----
    if "labels" in STAGES and cfg.get("label_scheme_ablation"):
        run_label_scheme_ablation(cfg, all_sm)

    # ---- SENSING -> FORECASTING hand-off ----
    if "handoff" in STAGES:
        export_handoff(cfg, all_sm)

    if not all_sm:
        log("\nNo main-table units on disk yet (run --stages main).")
        log(f"\nTotal wall time {(time.time()-t0)/60:.1f} min")
        return

    log("\n" + "=" * 74)
    log("DATASET COMPOSITION PER ARM (read before comparing arms)")
    log("=" * 74)
    log(f"{'arm':<12}{'frames':>8}{'H1':>7}{'H0':>7}{'%H1':>7}{'baseline':>10}"
        f"{'bands':>7}{'SNRp50':>8}")
    for a in arm_rows:
        log(f"{a['arm']:<12}{a['n_frames']:8,}{a['n_H1']:7,}{a['n_H0']:7,}"
            f"{a['h1_fraction']*100:6.1f}%{a['majority_baseline']:10.4f}"
            f"{a['n_bands']:7}{a['snr_p50_db']:8.1f}")

    log("\n" + "=" * 74)
    log("RESULTS (test set).  'vs base' = accuracy minus majority baseline")
    log("=" * 74)
    log(f"{'arm':<10}{'model':<24}{'mode':<12}{'ACC':>7}{'vs base':>9}"
        f"{'F1':>7}{'AUC':>7}{'Pd':>7}{'MDR':>7}{'FDR':>7}")
    for r in all_sm:
        log(f"{r['arm']:<10}{r['model']:<24}{r['mode']:<12}"
            f"{r['accuracy']:7.4f}{r['acc_minus_baseline']:+9.4f}"
            f"{r['f1']:7.4f}{r['auc']:7.4f}{r['recall_pd']:7.4f}"
            f"{r['mdr']:7.4f}{r['fdr']:7.4f}")

    if all_cx:
        log("\n" + "=" * 74)
        log("COMPLEXITY (CPU, standalone -> Best = Average = Worst)")
        log("=" * 74)
        log(f"{'model':<24}{'Params':>10}{'Mem MB':>9}{'MACs M':>9}"
            f"{'FLOPs M':>9}{'Lat ms':>9}{'Thr f/s':>10}")
        seen = set()
        for c in all_cx:
            if c["model"] in seen:
                continue
            seen.add(c["model"])
            log(f"{c['model']:<24}{c['params']:10,}{c['model_mem_MB']:9.2f}"
                f"{c['macs_M']:9.1f}{c['flops_M']:9.1f}"
                f"{c['latency_ms_avg']:9.4f}{c['throughput_avg']:10,.0f}")

    log("\n" + "=" * 74)
    log("OVERFITTING")
    log("=" * 74)
    log("Gaps are on BALANCED accuracy (prior-invariant). Raw accuracy is")
    log("not comparable across splits here: train baseline "
        f"{all_sm[0]['ofit_train_baseline']:.4f} vs test baseline "
        f"{all_sm[0]['ofit_test_baseline']:.4f}, a "
        f"{all_sm[0]['ofit_prior_shift_artifact']:+.4f} artifact before any "
        "model acts.")
    log("")
    log(f"{'model':<24}{'mode':<12}{'trBal':>7}{'vaBal':>7}{'teBal':>7}"
        f"{'balGap':>8}{'rawGap':>8}  verdict")
    for r in all_sm:
        log(f"{r['model']:<24}{r['mode']:<12}"
            f"{r['ofit_train_bal_acc']:7.4f}{r['ofit_val_bal_acc']:7.4f}"
            f"{r['ofit_test_bal_acc']:7.4f}"
            f"{r['ofit_bal_acc_gap_train_minus_val']:+8.4f}"
            f"{r['ofit_raw_acc_gap_train_minus_val']:+8.4f}  "
            f"{r['ofit_verdict']}")

    log(f"\nTotal wall time {(time.time()-t0)/60:.1f} min")
    log(f"Results -> {rd}/  (ALL_arms_*.csv, arm_dataset_stats.csv, <arm>/)")


if __name__ == "__main__":
    main()
