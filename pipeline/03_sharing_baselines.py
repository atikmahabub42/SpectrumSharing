#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
03_sharing_baselines.py -- Stage 3 (baselines) of the sense-forecast-share pipeline
===================================================================================
Trace-driven primary / secondary spectrum sharing on the Fireworks dataset
(IEEE DataPort, DOI 10.21227/f2rv-ms26) with the five SOTA sharing methods of
Table IV. TIDE, the proposed method, is in 03b_sharing_tide.py; both scripts
use the same secondary network, PU protection rule, test slots and metrics.

INPUT   <results>/02_forecasting/stage2/ (02_forecasting.py --family sota):
        measured PU occupancy and PU power of every test slot + forecasts of
        the best SOTA forecaster of each configuration (N-HiTS centrally,
        SpectrumFM federated in the paper).
MEASURED  PU occupancy y and PU power over noise S_pu = 10^(excess/10) - 1.
SIMULATED the secondary network only: K = 48 SU links of 2-8 m in a
          15 x 15 m area, path-loss exponent 3, 4 dB shadowing, 40 dB
          reference SNR, and the gain of each SU towards the PU receiver.

PU PROTECTION (per busy sub-channel c)
  I_pu[c] = sum_{SU k on c} p_k G_k ;  violation iff S_pu[c] / (1 + I_pu[c]) < 3 dB
SU UTILITY = sum_k log2(1 + SINR_k) - mu * #violated sub-channels, mu = 10

METHODS
  lbt_sense      listen-before-talk on the last sensed slot
  auction        BLAST second-price auction (Abognah & Basir, arXiv:2604.12127;
                 mechanism only, no LLM agents, no ledger)
  dqn_dss        DQN dynamic spectrum sharing (Challita & Sandberg)
  adapshare_ppo  AdapShare, PPO (Gopal et al.)
  ma_idqn        multi-agent independent DQN (Shin et al., IEEE COMML 2024)
  oracle         true future occupancy and PU strength (regret reference)

RUN GRID  references (centralized) and K in {4, 8, 12, 16, 24, 32} x
  {non-IID temporal, IID}; learned policies are trained with FedAvg over the
  same clients as the forecaster. Main federated setting: non-IID, K = 8.

OUTPUT (CSV)  share_overall(_agg).csv (utility, sum rate, PU violation, PU
  hit, idle capture, regret, latency ...), share_per_snr.csv (PU hit and
  violation rate per PU-SNR bin), share_complexity.csv,
  fl_communication(_agg).csv, client_partitions.csv

USAGE
  python 03_sharing_baselines.py                       # every part + merge
  python 03_sharing_baselines.py --quick               # smoke test
  python 03_sharing_baselines.py --list-tasks / --task i / --merge
Finished parts are stored in <out>/parts_v2/; reruns resume.
"""
from __future__ import annotations

import argparse
import hashlib
import copy
import csv
import glob
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
# Paths: --forecast-dir / --out on the command line, or $SFS_RESULTS.
_RES = os.environ.get("SFS_RESULTS", os.path.join(".", "results"))

CONFIG: Dict = {
    "forecast_dir": os.path.join(_RES, "02_forecasting"),          # 02_forecasting.py --family sota
    "out_dir": os.path.join(_RES, "03_sharing_baselines"),
    "methods": [],                    # the proposed method (TIDE) is in 03b_sharing_tide.py
    # SOTA methods of the main comparison (references + non-IID K = 8) ...
    "sota_methods": ["lbt_sense", "auction", "dqn_dss", "adapshare_ppo", "ma_idqn"],
    # ... and of the other (K, partition) parts of the client sweep
    "sota_methods_ablation": ["lbt_sense", "auction", "dqn_dss", "adapshare_ppo", "ma_idqn"],
    "forecasters": ["best"],          # the best SOTA forecaster of each configuration
    # training settings of the SOTA sharing run, used for the SOTA methods
    "sota_overrides": {"train_steps": 1500, "alloc_batch": 128, "alloc_lr": 1e-3,
                       "fl_rounds": 30, "fl_local_steps": 50, "fl_steps_per_round": 200,
                       "power_levels": (0.1, 0.3, 1.0),
                       "interweave_target_collision": 0.05, "underlay_busy_threshold": 0.5,
                       "dqn_gamma": 0.9, "dqn_eps": (1.0, 0.05), "dqn_replay": 200000,
                       "dqn_target_every": 200, "idqn_gamma": 0.5, "idqn_parallel": 8,
                       "ppo_clip": 0.2, "ppo_epochs": 4, "adapshare_levels": 9,
                       "ddpg_noise": 0.2, "auction_rounds": 48, "wmmse_iters": 30},
    # ---- ABLATION grid (must match the forecasting ablation) -----------------
    "ab_clients": "from_forecast",        # the K list of the forecasting run (its stage2 manifest)
    "ab_partitions": ["noniid", "iid"],  # default: the forecasting run's partitions (manifest)
    "main_clients": 8,                # main federated run = non-IID temporal K = 8 (manifest)
    "min_per_class_per_client": 8,
    "seeds": [0, 1, 2],
    "seed": 20260807,                 # same base seed as the sensing / forecasting scripts
    "horizons": [1, 2],               # forecast horizons evaluated (paper: h = 1)
    # ---- secondary network + PU protection: IDENTICAL to the SOTA run -----------
    "su_links": 48, "su_area_m": 15.0, "su_link_dist_m": (2.0, 8.0),
    "su_pathloss_exp": 3.0, "su_shadow_db": 4.0, "su_snr_ref_db": 40.0,
    "tx_power_min": 1e-3, "topo_pool_test": 64, "topo_block": 50,
    "pu_sinr_min_db": 3.0, "violation_penalty": 10.0, "collision_penalty": 2.0,
    "eta_db": 6.0, "wmmse_iters": 30,
    # ---- base training settings (each method gets them overridden by sota_overrides) ----
    "power_levels": (0.03, 0.1, 0.3, 1.0),
    "train_steps": 800, "alloc_batch": 32, "alloc_lr": 1e-3,
    "fl_rounds": 16, "fl_local_steps": 50,
    "fl_steps_per_round": 200, "fl_min_local_steps": 4,      # local steps per client = max(4, 200 / K)

    # ---- profiling -----------------------------------------------------------------
    "prof_iters": 30, "prof_warmup": 3,
    "device": "auto",
    "verbose": True,
}

QUICK = {"train_steps": 4, "fl_rounds": 1, "fl_local_steps": 2, "fl_steps_per_round": 8, "fl_min_local_steps": 1, "ab_partitions": ["iid", "noniid"], "seeds": [0], "horizons": [1],
         "prof_iters": 3, "prof_warmup": 1}

QUICK["sota_overrides"] = {**CONFIG["sota_overrides"], "train_steps": 4, "fl_rounds": 1, "fl_steps_per_round": 8,
                           "alloc_batch": 16, "dqn_replay": 5000}

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
#          INPUT: the stage2/ hand-off of 02_forecasting.py
# ==========================================================================
def _search_bases() -> List[str]:
    b = [os.getcwd(), _RES, os.path.dirname(os.path.abspath(__file__))]
    return list(dict.fromkeys(b))


def find_stage2(forecast_dir: str):
    """The forecasting out_dir (or its stage2/ folder itself), given as an
    absolute path or relative to: the current folder, Results,
    Fireworks, the code folder. -> (stage2 dir or None, places tried)."""
    p = os.path.expanduser(forecast_dir)
    roots = [p] if os.path.isabs(p) else [os.path.join(b, p) for b in _search_bases()]
    tried = []
    for r in roots:
        for d in (os.path.join(r, "stage2"), r):
            d = os.path.normpath(d)
            if d in tried:
                continue
            tried.append(d)
            if os.path.isfile(os.path.join(d, "manifest.json")):
                return d, tried
    return None, tried


def resolve_out(out_dir: str) -> str:
    p = os.path.expanduser(out_dir)
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(os.getcwd(), p))


class Bundle:
    """Measured labels + forecasts written by script 1. Nothing is generated."""

    def __init__(self, cfg: Dict):
        d, tried = find_stage2(cfg["forecast_dir"])
        if d is None:
            log("\nNO FORECAST HAND-OFF FOUND. Looked for stage2/manifest.json in:\n  "
                + "\n  ".join(tried) +
                "\nRun 02_forecasting.py first (it writes stage2/),\n"
                "or set CONFIG['forecast_dir'] / --forecast-dir to its out_dir.\n"
                "Nothing was run and no data was invented.")
            sys.exit(1)
        mp = os.path.join(d, "manifest.json")
        self.dir = d
        self.man = json.load(open(mp))
        z = np.load(os.path.join(d, "labels.npz"))
        self.names = json.loads(str(z["names"]))
        self.C = int(z["C"])
        self.edges = np.asarray(z["edges_snr"], dtype=np.float64)
        self.eta = float(z["eta_db"])
        self.H = int(z["horizon"])
        self.D = {s: {k: np.asarray(z[f"{s}__{k}"], dtype=np.float32)
                      for k in ("y_te", "E_te", "y_tr1", "E_tr1", "inr_busy", "inr_idle")}
                  for s in self.names}
        for s in self.names:                       # time of every training slot + radio (temporal clients)
            self.D[s]["t_tr1"] = (np.asarray(z[f"{s}__t_tr1"], dtype=np.float64) if f"{s}__t_tr1" in z
                                  else np.arange(len(self.D[s]["y_tr1"]), dtype=np.float64))
            self.D[s]["radio"] = str(z[f"{s}__radio"]) if f"{s}__radio" in z else "radio"
        self.nv = {s: int(z[f"{s}__nvalid"]) if f"{s}__nvalid" in z else self.C for s in self.names}
        for s in self.names:                       # padded channels: permanent strong incumbent
            nv = self.nv[s]
            if nv < self.C:
                for k in ("y_te", "y_tr1"):
                    self.D[s][k][..., nv:] = 1.0
                for k in ("E_te", "E_tr1"):
                    self.D[s][k][..., nv:] = PAD_EXCESS_DB
                for k in ("inr_busy", "inr_idle"):
                    self.D[s][k][nv:] = 10 ** (PAD_EXCESS_DB / 10) - 1
        self.input = self.man.get("input", "psd")
        if self.input == "sensing_grid":
            log(f"  input: sensed tile occupancy ({self.man.get('sensing_desc', '')}); "
                f"tiles per receiver {self.nv}")
        elif self.input == "sensing":
            log(f"  input: PSD occupancy, threshold calibrated per band to "
                f"{self.man.get('sensing_desc', '')}")
        n_te = {s: int(self.D[s]["y_te"].shape[0]) for s in self.names}
        log(f"  test slots per scenario {n_te}")
        self.labels = self.man.get("model_labels", {})
        log(f"  hand-off: {d}\n  scenarios {self.names}, C = {self.C}, H = {self.H}, "
            f"best forecaster by mode {self.man['best_by_mode']}")

    def label(self, key):
        return self.labels.get(key, key)

    def preds(self, model: str, mode: str, seed: int) -> Optional[Dict]:
        """{scenario: {'te': [N,H,C], 'tr1': [N,C]}} or None."""
        fn = os.path.join(self.dir, f"preds__{model}__{mode}__s{seed}.npz")
        if not os.path.isfile(fn):
            c = sorted(glob.glob(os.path.join(self.man.get("cache_dir", ""),
                                              f"{model}__{mode}__s{seed}__*.npz")),
                       key=os.path.getmtime)
            if not c:
                return None
            fn = c[-1]
        z = np.load(fn, allow_pickle=True)
        out = {}
        for s in self.names:
            out[s] = {"te": np.asarray(z[f"{s}__te"], dtype=np.float32)}
            if f"{s}__tr1" in z:
                out[s]["tr1"] = np.asarray(z[f"{s}__tr1"], dtype=np.float32)
            nv = self.nv.get(s, self.C)
            if nv < self.C:                        # padded channels: certainly occupied
                for k in out[s]:
                    out[s][k][..., nv:] = 1.0
        return out


# ==========================================================================
#                               SMALL HELPERS
# ==========================================================================
def _bin_mask(E, lo, hi, last, eta):
    lower = (E > lo) if abs(lo - eta) < 1e-9 else (E >= lo)
    return lower & ((E <= hi) if last else (E < hi))


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
                o[k] = float(np.nanmean(v)) if not all(np.isnan(v)) else float("nan")
                o[k + "_std"] = float(np.nanstd(v)) if not all(np.isnan(v)) else float("nan")
            elif k not in o:
                o[k] = rs[0][k]
        out.append(o)
    return out


def model_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.state_dict().values())


def _avg_states(states, weights):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = {}
    for k in states[0]:
        if states[0][k].dtype.is_floating_point:
            out[k] = sum(float(wi) * s[k] for wi, s in zip(w, states))
        else:
            out[k] = states[0][k].clone()
    return out


def count_flops(fn, *args) -> float:
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


# ==========================================================================
#              SECONDARY NETWORK (simulated) + MEASURED PU CHANNEL
# ==========================================================================
#  K SU links (tx -> rx) in an indoor area; the spectrum sensor that recorded
#  the data sits at the centre and stands for the PU receiver.
#    G[j, k]  gain tx j -> rx k,  Gsp[k]  gain SU tx k -> PU receiver
#  normalised so the noise power of one sub-channel is 1. This network is
#  SIMULATED (seeded); everything about the primary users is MEASURED:
#    y[c]    true occupancy of sub-channel c in the slot
#    inr[c]  10^(excess/10) - 1 >= 0 : PU power over noise at the receivers
#  A decision may use only the forecast phat[c] and the expected INR
#    inr_hat[c] = phat*E[inr | busy] + (1 - phat)*E[inr | idle]
#  (conditional means from the TRAINING span).
# ==========================================================================
def gen_topologies(rng: np.random.Generator, n: int, cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    K, L = cfg["su_links"], cfg["su_area_m"]
    dmin, dmax = cfg["su_link_dist_m"]
    G = np.empty((n, K, K), np.float32)
    Gsp = np.empty((n, K), np.float32)
    pu = np.array([L / 2, L / 2])
    pl = lambda D, shape: (cfg["su_snr_ref_db"] - 10 * cfg["su_pathloss_exp"] * np.log10(D)
                           + rng.normal(0, cfg["su_shadow_db"], shape))
    for i in range(n):
        tx = rng.uniform(0, L, (K, 2))
        d = rng.uniform(dmin, dmax, K)
        th = rng.uniform(0, 2 * np.pi, K)
        rx = np.clip(tx + d[:, None] * np.stack([np.cos(th), np.sin(th)], 1), 0, L)
        D = np.maximum(np.linalg.norm(tx[:, None, :] - rx[None, :, :], axis=-1), 0.5)
        G[i] = np.power(10.0, pl(D, (K, K)) / 10.0)
        Dp = np.maximum(np.linalg.norm(tx - pu[None], axis=-1), 0.5)
        Gsp[i] = np.power(10.0, pl(Dp, (K,)) / 10.0)
    return G, Gsp


def inr_from_excess(e):
    return torch.clamp(torch.pow(10.0, e / 10.0) - 1.0, min=0.0)


class SlotBatch:
    """B slots: forecast, truth, measured excess, SU topology, on device."""

    def __init__(self, phat, y, e, G, Gsp, inr_busy, inr_idle, dev, ppers=None, nv=None):
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
        self.nv = None if nv is None else torch.as_tensor(np.asarray(nv), dtype=torch.long, device=dev)
        self.phat, self.y, self.e, self.G, self.Gsp = t(phat), t(y), t(e), t(G), t(Gsp)
        self.ppers = t(ppers) if ppers is not None else None
        self.inr = inr_from_excess(self.e)
        ib, ii = t(inr_busy), t(inr_idle)
        if ib.dim() == 1:
            ib, ii = ib[None].expand_as(self.phat), ii[None].expand_as(self.phat)
        self.inr_busy, self.inr_idle = ib, ii
        self.inr_hat = self.phat * ib + (1 - self.phat) * ii
        self.B, self.C = self.phat.shape
        self.K = self.G.shape[1]
        self.dev = dev

    def oracle(self):
        o = copy.copy(self)
        o.phat, o.inr_hat = self.y, self.inr
        return o

    def gdirect_db(self):
        return 10 * torch.log10(torch.diagonal(self.G, dim1=1, dim2=2) + 1e-12)

    def node_feats(self):
        B, K, C = self.B, self.K, self.C
        return torch.cat([self.phat[:, None, :].expand(B, K, C),
                          (torch.log1p(self.inr_hat) / 3.0)[:, None, :].expand(B, K, C),
                          (self.gdirect_db() / 40.0)[..., None]], -1)       # [B, K, 2C+1]

    def edge_feats(self):
        return 10 * torch.log10(self.G + 1e-12) / 40.0                      # [B, K, K]


def su_side(sb: SlotBatch, ch, pw, cfg):
    """SU rates under the TRUE measured PU interference. ch in [0, C], C = idle."""
    C, K = sb.C, sb.K
    tx = (ch < C) & (pw > cfg["tx_power_min"])
    chc = ch.clamp(max=C - 1)
    A = F.one_hot(chc, C).float() * tx[..., None].float()
    same = A @ A.transpose(1, 2) * (1 - torch.eye(K, device=sb.dev))
    p = pw * tx.float()
    I = (same * p[:, :, None] * sb.G).sum(1)
    sinr = p * torch.diagonal(sb.G, dim1=1, dim2=2) / (1.0 + torch.gather(sb.inr, 1, chc) + I)
    rate = torch.log2(1 + sinr) * tx.float()
    return tx, chc, A, p, sinr, rate


def _rates(S, N, I):
    return torch.log2(1 + S / (N + I))


@torch.no_grad()
def greedy_assign(sb: SlotBatch, cfg: Dict, oracle: bool = False, allowed=None, lam=None):
    """Links in decreasing direct gain; each takes the sub-channel with the
    largest positive MARGINAL expected utility (own expected rate, minus the
    rate it removes from links already there, minus the expected collision
    charge), or stays idle. Reuse allowed. Forecast only, unless oracle=True.
    allowed: optional [B, C] bool mask of sub-channels that may be used."""
    s = sb.oracle() if oracle else sb
    B, K, C, dev = s.B, s.K, s.C, s.dev
    lam = cfg["collision_penalty"] if lam is None else lam
    Gd = torch.diagonal(s.G, dim1=1, dim2=2)
    order = torch.argsort(-Gd, dim=1)
    ch = torch.full((B, K), C, dtype=torch.long, device=dev)
    A = torch.zeros(B, K, C, device=dev)
    Icur = torch.zeros(B, K, device=dev); Ncur = torch.ones(B, K, device=dev)
    qcur = torch.zeros(B, K, device=dev)
    ar = torch.arange(B, device=dev)
    for i in range(K):
        k = order[:, i]
        Sk = Gd[ar, k]
        Ikc = torch.einsum("bjc,bj->bc", A, s.G[ar, :, k])
        own = (1 - s.phat) * _rates(Sk[:, None], 1 + s.inr_hat, Ikc) - lam * s.phat
        g_out = s.G[ar, k, :]
        assigned = A.sum(-1) > 0
        loss_j = qcur * (_rates(Gd, Ncur, Icur) - _rates(Gd, Ncur, Icur + g_out)) * assigned
        val = own - torch.einsum("bjc,bj->bc", A, loss_j)
        if allowed is not None:
            val = val.masked_fill(~allowed, -1e9)
        best, c_star = val.max(1)
        ok = best > 0
        if ok.any():
            b = ar[ok]; kk = k[ok]; cc = c_star[ok]
            ch[b, kk] = cc
            Icur[b] = Icur[b] + A[b, :, cc] * g_out[ok]
            A[b, kk, cc] = 1.0
            Icur[b, kk] = Ikc[ok, cc]
            Ncur[b, kk] = 1 + s.inr_hat[b, cc]
            qcur[b, kk] = 1 - s.phat[b, cc]
    return ch, (ch < C).float()


@torch.no_grad()
def wmmse_power(sb: SlotBatch, ch, cfg: Dict, noise=None, iters=None):
    """WMMSE power control (Shi, Razaviyayn, Luo, He, IEEE TSP 2011) on the
    co-channel interference graph of a fixed sub-channel assignment."""
    iters = iters or cfg["wmmse_iters"]
    C = sb.C
    tx = (ch < C).float()
    chc = ch.clamp(max=C - 1)
    A = F.one_hot(chc, C).float() * tx[..., None]
    h2 = sb.G * (A @ A.transpose(1, 2))
    hkk = torch.sqrt(torch.diagonal(sb.G, dim1=1, dim2=2))
    if noise is None:
        noise = 1.0 + torch.gather(sb.inr_hat, 1, chc)
    v = tx.clone()
    for _ in range(iters):
        u = hkk * v / (noise + torch.einsum("bjk,bj->bk", h2, v ** 2))
        w = 1.0 / (1.0 - u * hkk * v).clamp(min=1e-6)
        den = torch.einsum("bkj,bj->bk", h2, w * u ** 2)
        v = (w * u * hkk / den.clamp(min=1e-12)).clamp(0, 1) * tx
    return v ** 2


# ==========================================================================
#                   TRAINING SLOTS + TEST BLOCKS
# ==========================================================================
class SlotSource:
    """Training slots of one scenario: forecast (h=1) + measured truth.
    Topologies are drawn fresh from the training RNG (never the test pool)."""

    def __init__(self, name, phat, y, e, inr_busy, inr_idle, cfg, dev):
        self.name, self.phat, self.y, self.e = name, phat, y, e
        self.inr_busy, self.inr_idle, self.cfg, self.dev = inr_busy, inr_idle, cfg, dev
        self.N = len(phat)

    def _sb(self, idx, G, Gsp):
        return SlotBatch(self.phat[idx], self.y[idx], self.e[idx], G, Gsp,
                         self.inr_busy, self.inr_idle, self.dev)

    def batch(self, rng, B):
        G, Gsp = gen_topologies(rng, B, self.cfg)
        return self._sb(rng.integers(0, self.N, B), G, Gsp)

    def sequence(self, rng, M, L):
        if self.N < 2:                                   # one slot: one-step sequence
            G, Gsp = gen_topologies(rng, M, self.cfg)
            return [self._sb(np.zeros(M, dtype=int), G, Gsp)]
        L = min(L, self.N - 1)
        st = rng.integers(0, self.N - L, M)
        G, Gsp = gen_topologies(rng, M, self.cfg)
        return [self._sb(st + t, G, Gsp) for t in range(L)]


class PooledSource:
    def __init__(self, sources):
        self.sources = sources
        self.N = sum(s.N for s in sources)
        self.w = np.array([s.N for s in sources], np.float64) / self.N

    def pick(self, rng):
        return self.sources[int(rng.choice(len(self.sources), p=self.w))]

    def batch(self, rng, B):
        return self.pick(rng).batch(rng, B)

    def sequence(self, rng, M, L):
        return self.pick(rng).sequence(rng, M, L)


def topo_len(bd, cfg) -> int:
    """Block length: topo_block, or the shortest test set when a band has
    fewer test slots (exports with few sweeps per band)."""
    L = min(int(cfg["topo_block"]), min(int(bd.D[s]["y_te"].shape[0]) for s in bd.names))
    if L < 1:
        sys.exit("a scenario has no test slots: nothing to evaluate")
    return L


def test_blocks(bd: Bundle, phat_of, ppers_of, h: int, pool, cfg):
    """Consecutive test slots of every scenario, cut into blocks of
    topo_block slots; each block uses one SU topology of the fixed TEST pool."""
    L = topo_len(bd, cfg)
    G_pool, Gsp_pool = pool
    out = defaultdict(list)
    for si, s in enumerate(bd.names):
        D = bd.D[s]
        ph = phat_of(s)[:, h - 1, :]
        pp = ppers_of(s)[:, h - 1, :] if ppers_of else ph
        y, e = D["y_te"][:, h - 1, :], D["E_te"][:, h - 1, :]
        for b in range(ph.shape[0] // L):
            sl = slice(b * L, (b + 1) * L)
            j = (zlib.crc32(s.encode()) + b) % len(G_pool)
            out["PH"].append(ph[sl]); out["PP"].append(pp[sl])
            out["Y"].append(y[sl]); out["E"].append(e[sl])
            out["G"].append(G_pool[j]); out["Gsp"].append(Gsp_pool[j])
            out["SID"].append(si); out["IB"].append(D["inr_busy"]); out["II"].append(D["inr_idle"])
            out["NV"].append(bd.nv.get(s, bd.C))
    return {k: np.stack(v) if k != "SID" else np.array(v) for k, v in out.items()}


def block_slot(blk, t, dev):
    return SlotBatch(blk["PH"][:, t], blk["Y"][:, t], blk["E"][:, t], blk["G"], blk["Gsp"],
                     blk["IB"], blk["II"], dev, ppers=blk["PP"][:, t], nv=blk.get("NV"))


# ==========================================================================
#                          COMMON POLICY PIECES
# ==========================================================================
class Allocator:
    learned = False
    sequential = False
    label = ""
    ref = ""

    def __init__(self, cfg, dev, C):
        self.cfg, self.dev, self.C = cfg, dev, C
        self.nets: Dict[str, nn.Module] = {}

    def reset(self, B):
        pass

    def act(self, sb: SlotBatch):
        raise NotImplementedError

    def feedback(self, sb, ch, out):
        pass

    def train_steps(self, src, n, rng, client=0):
        pass

    def opt_for(self, key, net):
        if not hasattr(self, "_opts"):
            self._opts = {}
        return torch.optim.Adam(net.parameters(), lr=self.cfg["alloc_lr"])


class QNet(nn.Module):
    def __init__(self, fin, nout, h=256):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(fin, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                               nn.Linear(h, nout))

    def forward(self, x):
        return self.f(x)


class _Replay:
    def __init__(self, cap):
        self.cap, self.data, self.pos, self.n = cap, None, 0, 0

    def add(self, *ts):
        ts = [t.detach() for t in ts]
        if self.data is None:
            self.data = [torch.zeros((self.cap,) + t.shape[1:], dtype=t.dtype, device=t.device) for t in ts]
        m = ts[0].shape[0]
        idx = (torch.arange(m, device=ts[0].device) + self.pos) % self.cap
        for d, t in zip(self.data, ts):
            d[idx] = t
        self.pos = (self.pos + m) % self.cap
        self.n = min(self.n + m, self.cap)

    def sample(self, rng, bs):
        j = torch.as_tensor(rng.integers(0, self.n, bs), device=self.data[0].device)
        return [d[j] for d in self.data]


def train_policy(alloc: Allocator, sources: List[SlotSource], mode: str, cfg, seed) -> Dict:
    """Centralised (pooled slots) or FedAvg (one client per scenario)."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    if not alloc.learned:
        return {"train_s": 0.0, "comm_bytes": 0}
    if mode == "fedavg" and len(sources) > 1:
        rounds = cfg["fl_rounds"]
        for _ in range(rounds):
            g = {k: copy.deepcopy(n.state_dict()) for k, n in alloc.nets.items()}
            states, ws = [], []
            for ci, src in enumerate(sources):
                for k, n in alloc.nets.items():
                    n.load_state_dict(g[k])
                alloc.train_steps(src, cfg["fl_local_steps"], rng, client=ci)
                states.append({k: copy.deepcopy(n.state_dict()) for k, n in alloc.nets.items()})
                ws.append(src.N)
            for k, n in alloc.nets.items():
                n.load_state_dict(_avg_states([s[k] for s in states], ws))
        nb = sum(model_bytes(n) for n in alloc.nets.values())
        return {"train_s": time.time() - t0, "comm_bytes": 2 * nb * len(sources) * rounds,
                "rounds": rounds}
    alloc.train_steps(PooledSource(sources), cfg["train_steps"], rng)
    return {"train_s": time.time() - t0, "comm_bytes": 0}


def profile_policy(a: Allocator, C: int, cfg: Dict) -> Dict:
    """Never stops the run: a profiling failure is logged and reported as NaN."""
    try:
        return _profile_policy(a, C, cfg)
    except Exception as e:
        log(f"  profiling {a.label} failed ({type(e).__name__}: {e}) -> NaN in the complexity CSV")
        return {"method": a.label, "method_key": getattr(a, "key", type(a).__name__),
                "reference": a.ref, "profiling_error": f"{type(e).__name__}: {e}"}


def _profile_policy(a: Allocator, C: int, cfg: Dict) -> Dict:
    """Params, torch-counted FLOPs and latency of ONE slot decision
    (K links, C sub-channels): CPU batch 1, and GPU batch 1 if available."""
    rng = np.random.default_rng(0)
    G, Gsp = gen_topologies(rng, 1, cfg)
    ph = rng.uniform(0, 1, (1, C)).astype(np.float32)
    ib, ii = np.full(C, 10.0, np.float32), np.zeros(C, np.float32)
    row = {"method": a.label, "method_key": getattr(a, "key", type(a).__name__),
           "reference": a.ref, "su_links": cfg["su_links"], "subchannels": C,
           "params": int(sum(p.numel() for n in a.nets.values() for p in n.parameters()))}
    row["params_M"] = row["params"] / 1e6
    for devname in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
        dv = torch.device(devname)
        b = copy.copy(a)
        b.dev = dv
        b.nets = {k: copy.deepcopy(n).to(dv) for k, n in a.nets.items()}
        for attr, val in vars(a).items():           # e.g. power levels, diffusion schedule
            if attr == "nets":
                continue
            if torch.is_tensor(val):
                setattr(b, attr, val.to(dv))
            elif isinstance(val, nn.Module):
                setattr(b, attr, copy.deepcopy(val).to(dv))
        b.prev = None
        sb = SlotBatch(ph, (ph > 0.5).astype(np.float32), np.zeros((1, C), np.float32),
                       G, Gsp, ib, ii, dv, ppers=ph)
        b.reset(1)
        sync = torch.cuda.synchronize if devname == "cuda" else None
        if devname == "cpu":
            row["flops_per_slot"] = count_flops(lambda: b.act(sb))
            row["mflops_per_slot"] = row["flops_per_slot"] / 1e6
        row[f"{devname}_latency_ms_per_slot"] = _time_call(
            lambda: b.act(sb), max(3, cfg["prof_iters"] // 5), cfg["prof_warmup"], sync) * 1e3
    return row


# ==========================================================================
#         NOVEL STAGE-2 ENGINE: sensing memory, fusion, exact expectation
# ==========================================================================
#  Facts from the SOTA results that the novel methods act on:
#   * Greedy with the TRUE future reaches 168 of the oracle's 169, but only
#     138 with the forecast: the loss is in how uncertainty is used, not in
#     the assignment search.
#   * The classic greedy scores a sub-channel with ONE rate at the MEAN
#     interference. Busy-channel PU power spans 6-50 dB, so E[log(1+SINR)]
#     differs strongly from log(1+SINR at mean INR) (Jensen).
#   * The sensor keeps measuring: at forecast time the occupancy and PU
#     power of the last slot are known (sensing memory). No SOTA method
#     used them; a persistent PU keeps a similar strength.
#   * Weak PUs (6-16 dB) cause most violations; strong PUs tolerate sharing.
# ==========================================================================
Q_ATOMS = 8


def scenario_stats(y1: np.ndarray, e1: np.ndarray, eta: float, C: int) -> Dict:
    """From TRAINING slots (h = 1) of one scenario: busy-INR quantile atoms per
    sub-channel, idle INR mean, and how often a busy PU keeps its strength."""
    inr = np.maximum(np.power(10.0, e1 / 10.0) - 1.0, 0.0)
    busy = y1 > 0.5
    lv = (np.arange(Q_ATOMS) + 0.5) / Q_ATOMS
    allb = inr[busy] if busy.any() else np.array([10.0])
    atoms = np.empty((C, Q_ATOMS), np.float32)
    idle = np.empty(C, np.float32)
    for c in range(C):
        v = inr[busy[:, c], c]
        atoms[c] = np.quantile(v if v.size >= 8 else allb, lv)
        w = inr[~busy[:, c], c]
        idle[c] = w.mean() if w.size else 0.0
    both = busy[1:] & busy[:-1]
    keep = float((np.abs(e1[1:] - e1[:-1]) <= 3.0)[both].mean()) if both.any() else 0.0
    return {"atoms": atoms, "inr_idle": idle, "persist": keep}


def memory_of(y: np.ndarray, e: np.ndarray):
    """Sensing memory of CONSECUTIVE windows: what was measured at the forecast
    time of window i is the h=1 target of window i-1 (observed, causal)."""
    ym = np.zeros_like(y); em = np.zeros_like(e); ok = np.zeros(len(y), np.float32)
    ym[1:], em[1:], ok[1:] = y[:-1], e[:-1], 1.0
    return ym, em, ok


class FSF(nn.Module):
    """Forecast-Sensing Fusion (novel): calibrated occupancy probability from
    the forecast logit, the last sensed state and the last margin of the same
    sub-channel. Fitted per scenario on TRAINING slots (local, never shared)."""

    def __init__(self, C):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        self.b = nn.Parameter(torch.zeros(C))

    def forward(self, ph, ym, em, ok, eta):
        lg = torch.logit(ph.clamp(1e-4, 1 - 1e-4))
        x = torch.stack([lg, (2 * ym - 1) * ok[:, None], ((em - eta) / 10).clamp(-2, 4) * ok[:, None],
                         ok[:, None].expand_as(lg)], -1)
        return torch.sigmoid((x * self.w).sum(-1) + self.b)


def fit_fsf(ph, y, ym, em, ok, eta, C, dev, steps=300) -> FSF:
    m = FSF(C).to(dev)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    ph, y, ym, em, ok = t(ph), t(y), t(ym), t(em), t(ok)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(steps):
        p = m(ph, ym, em, ok, eta)
        loss = F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


# ---------------------------------------------------------------- memory-aware data
class MemSource(SlotSource):
    """Training slots with sensing memory (consecutive training windows)."""

    def __init__(self, name, sid, phat, y, e, inr_busy, inr_idle, cfg, dev):
        super().__init__(name, phat, y, e, inr_busy, inr_idle, cfg, dev)
        self.sid = sid
        self.ym, self.em, self.ok = memory_of(y, e)

    def _sb(self, idx, G, Gsp):
        sb = SlotBatch(self.phat[idx], self.y[idx], self.e[idx], G, Gsp,
                       self.inr_busy, self.inr_idle, self.dev)
        return attach_memory(sb, self.ym[idx], self.em[idx], self.ok[idx], np.full(len(idx), self.sid))


def attach_memory(sb, ym, em, ok, sid):
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=sb.dev)
    sb.ym, sb.em, sb.mok, sb.sid = t(ym), t(em), t(ok), np.asarray(sid)
    return sb


def test_blocks_mem(bd, phat_of, ppers_of, h, pool, cfg):
    blk = test_blocks(bd, phat_of, ppers_of, h, pool, cfg)
    L = topo_len(bd, cfg)
    YM, EM, OK = [], [], []
    for s in bd.names:
        D = bd.D[s]
        ym, em, ok = memory_of(D["y_te"][:, 0, :], D["E_te"][:, 0, :])
        for b in range(phat_of(s).shape[0] // L):
            sl = slice(b * L, (b + 1) * L)
            YM.append(ym[sl]); EM.append(em[sl]); OK.append(ok[sl])
    blk["YM"], blk["EM"], blk["OK"] = np.stack(YM), np.stack(EM), np.stack(OK)
    return blk


def block_slot_mem(blk, t, dev):
    sb = block_slot(blk, t, dev)
    return attach_memory(sb, blk["YM"][:, t], blk["EM"][:, t], blk["OK"][:, t], blk["SID"])


# ==========================================================================
#             SHARING OUTCOME (measured future, PU protection)
# ==========================================================================
def gamma_pu(cfg):
    return 10 ** (cfg["pu_sinr_min_db"] / 10.0)


# ---- padded channels (several receivers with different tile counts) -------
# A receiver with fewer tiles than C has its extra channels marked as a
# permanently occupied, strong incumbent (y = 1, excess 40 dB, forecast 1), so
# no method sees them as an opportunity; at scoring, a decision on a padded
# channel is folded onto a real tile of that receiver (c mod n_valid) and
# padded channels are excluded from every busy / idle count.
PAD_EXCESS_DB = 40.0


def fold_padding(sb, ch):
    nv = getattr(sb, "nv", None)
    if nv is None:
        return ch, torch.ones(sb.B, sb.C, dtype=torch.bool, device=sb.dev)
    nvb = nv[:, None]
    ch = torch.where((ch < sb.C) & (ch >= nvb), ch % nvb, ch)
    valid = torch.arange(sb.C, device=sb.dev)[None, :] < nvb
    return ch, valid


def evaluate_share(sb: SlotBatch, ch, pw, cfg: Dict) -> Dict:
    ch, valid = fold_padding(sb, ch)
    tx, chc, A, p, sinr, rate = su_side(sb, ch, pw, cfg)
    Ipu = torch.einsum("bkc,bk,bk->bc", A, p, sb.Gsp)
    busy = (sb.y > 0.5) & valid
    hit = A.sum(1) > 0
    viol = busy & hit & (sb.inr / (1.0 + Ipu) < gamma_pu(cfg))
    viol_link = tx & torch.gather(viol, 1, chc)
    col_link = tx & (torch.gather(sb.y, 1, chc) > 0.5)
    sr = rate.sum(1)
    return {"rate": rate, "tx": tx, "sinr": sinr, "pw": p, "busy": busy, "hit": hit,
            "viol": viol, "viol_link": viol_link, "col_link": col_link, "Ipu": Ipu,
            "sum_rate": sr, "utility": sr - cfg["violation_penalty"] * viol.float().sum(1), "valid": valid}


# ==========================================================================
#                             NOVEL SHARING METHODS
# ==========================================================================


METHODS: Dict[str, type] = {}        # proposed method: see 03b_sharing_tide.py


# ==========================================================================
#                              EVALUATION
# ==========================================================================
def eval_method(a, blk, cfg, dev, edges, eta):
    NB, L, C = blk["PH"].shape
    K = blk["G"].shape[1]
    acc = defaultdict(lambda: np.zeros(NB))
    rate_acc = np.zeros((NB, K))
    nb = len(edges) - 1
    sh, sv, sn = np.zeros((NB, nb)), np.zeros((NB, nb)), np.zeros((NB, nb))
    a.reset(NB)
    t_act = 0.0
    cpu = lambda v: v.float().detach().cpu().numpy()
    for t in range(L):
        sb = block_slot_mem(blk, t, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ch, pw = a.act(sb)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_act += time.perf_counter() - t0
        o = evaluate_share(sb, ch, pw, cfg)
        a.feedback(sb, ch, o)
        tx, hit, busy, viol = o["tx"], o["hit"], o["busy"], o["viol"]
        acc["sum_rate"] += cpu(o["sum_rate"]); acc["utility"] += cpu(o["utility"])
        acc["n_tx"] += cpu(tx.sum(1)); acc["n_col"] += cpu(o["col_link"].sum(1))
        acc["n_viol"] += cpu(viol.sum(1))
        acc["hit_busy"] += cpu((hit & busy).sum(1)); acc["n_busy"] += cpu(busy.sum(1))
        acc["hit_idle"] += cpu((hit & ~busy).sum(1)); acc["n_idle"] += cpu((~busy & o["valid"]).sum(1))
        acc["n_hit"] += cpu(hit.sum(1))
        acc["sinr_loss_db"] += cpu((10 * torch.log10(1 + o["Ipu"]) * (hit & busy).float()).sum(1))
        acc["outage"] += cpu(((o["sinr"] < 1) & tx).sum(1))
        acc["sum_pw"] += cpu(o["pw"].sum(1))
        rate_acc += cpu(o["rate"])
        e, bz, hz, vz = blk["E"][:, t], cpu(busy) > 0.5, cpu(hit) > 0.5, cpu(viol) > 0.5
        for i in range(nb):
            m = bz & _bin_mask(e, edges[i], edges[i + 1], i == nb - 1, eta)
            sn[:, i] += m.sum(1); sh[:, i] += (m & hz).sum(1); sv[:, i] += (m & vz).sum(1)
    s1, s2 = rate_acc.sum(1), (rate_acc ** 2).sum(1)
    acc["jain"] = np.where(s2 > 0, s1 ** 2 / (K * np.maximum(s2, 1e-12)), 0.0)
    return acc, sh, sv, sn, t_act / (NB * L)


def share_row(acc, mask, L, K):
    sl = float(mask.sum() * L)
    S = lambda k: float(acc[k][mask].sum())
    ntx = max(S("n_tx"), 1.0)
    return {"n_slots": int(sl), "su_sum_rate_per_slot": S("sum_rate") / sl,
            "su_rate_per_link": S("sum_rate") / sl / K, "su_rate_per_active_link": S("sum_rate") / ntx,
            "utility_per_slot": S("utility") / sl,
            "pu_violation_rate": S("n_viol") / max(S("n_busy"), 1.0),
            "pu_violations_per_slot": S("n_viol") / sl,
            "pu_collision_rate": S("hit_busy") / max(S("n_busy"), 1.0),
            "su_tx_on_busy_frac": S("n_col") / ntx,
            "pu_sinr_loss_db_when_shared": S("sinr_loss_db") / max(S("hit_busy"), 1.0),
            "idle_capture": S("hit_idle") / max(S("n_idle"), 1.0),
            "spectrum_reuse_su_per_used_channel": S("n_tx") / max(S("n_hit"), 1.0),
            "access_ratio": S("n_tx") / (sl * K), "outage_rate": S("outage") / ntx,
            "jain_fairness": float(acc["jain"][mask].mean()),
            "energy_eff_bit_per_unit_power": S("sum_rate") / max(S("sum_pw"), 1e-9),
            "su_bits_per_pu_violation": S("sum_rate") / max(S("n_viol"), 1.0)}


def oracle_utility(blk, cfg, dev):
    """Same oracle as the SOTA sharing script (greedy + WMMSE + exact PU cap
    with the true future), so regret_vs_oracle is directly comparable."""
    NB, L, C = blk["PH"].shape
    U = np.zeros(NB)
    gam = gamma_pu(cfg)
    for t in range(L):
        sb = block_slot_mem(blk, t, dev)
        ch, _ = greedy_assign(sb, cfg, oracle=True, lam=0.0)
        chc = ch.clamp(max=C - 1)
        pw = wmmse_power(sb, ch, cfg, noise=1.0 + torch.gather(sb.inr, 1, chc))
        tx = (ch < C).float()
        n_on = torch.einsum("bkc->bc", F.one_hot(chc, C).float() * tx[..., None])
        allow = (sb.inr / gam - 1.0).clamp(min=0)
        cap = (torch.gather(allow, 1, chc) / (torch.gather(n_on, 1, chc).clamp(min=1) * sb.Gsp + 1e-12)).clamp(0, 1)
        busy = torch.gather(sb.y, 1, chc) > 0.5
        pw = torch.where(busy, torch.minimum(pw, cap), pw)
        U += evaluate_share(sb, ch, pw, cfg)["utility"].cpu().numpy()
    return U


def fit_fsf_all(bd, P, names, cfg, dev):
    out = []
    for s in names:
        ym, em, ok = memory_of(bd.D[s]["y_tr1"], bd.D[s]["E_tr1"])
        out.append(fit_fsf(P[s]["tr1"], bd.D[s]["y_tr1"], ym, em, ok, cfg["eta_db"], bd.C, dev))
    return out


# ==========================================================================
#   SOTA SHARING METHODS (identical classes to the SOTA sharing script)
# ==========================================================================


@torch.no_grad()
def auction_assign(sb: SlotBatch, cfg: Dict):
    """Repeated sealed-bid second-price auctions, one item per sub-channel per
    round. Each unassigned link bids its best marginal expected utility; the
    highest bidder of each sub-channel wins and pays the second-highest bid.
    Market mechanism of BLAST (Abognah & Basir 2026), WITHOUT its LLM agents."""
    B, K, C, dev = sb.B, sb.K, sb.C, sb.dev
    lam = cfg["collision_penalty"]
    Gd = torch.diagonal(sb.G, dim1=1, dim2=2)
    ch = torch.full((B, K), C, dtype=torch.long, device=dev)
    A = torch.zeros(B, K, C, device=dev)
    Icur = torch.zeros(B, K, device=dev); Ncur = torch.ones(B, K, device=dev)
    qcur = torch.zeros(B, K, device=dev)
    active = torch.ones(B, K, dtype=torch.bool, device=dev)
    revenue = torch.zeros(B, device=dev)
    for _ in range(cfg["auction_rounds"]):
        if not active.any():
            break
        Ikc = torch.einsum("bjc,bjk->bkc", A, sb.G)
        own = (1 - sb.phat[:, None, :]) * _rates(Gd[..., None], 1 + sb.inr_hat[:, None, :], Ikc) \
            - lam * sb.phat[:, None, :]
        asg = (A.sum(-1) > 0).float()
        dl = qcur[:, None, :] * (_rates(Gd[:, None, :], Ncur[:, None, :], Icur[:, None, :])
                                 - _rates(Gd[:, None, :], Ncur[:, None, :], Icur[:, None, :] + sb.G)) \
            * asg[:, None, :]
        val = own - torch.einsum("bkj,bjc->bkc", dl, A)
        bid, bch = val.max(-1)
        active = active & (bid > 0)
        if not active.any():
            break
        bm = torch.where(active[..., None] & (F.one_hot(bch, C) > 0), bid[..., None],
                         torch.full_like(val, -1e9))
        top2 = bm.topk(min(2, K), dim=1)
        win_val, win_k = top2.values[:, 0], top2.indices[:, 0]
        second = top2.values[:, 1].clamp(min=0) if K > 1 else torch.zeros_like(win_val)
        has = win_val > -1e8
        if not has.any():
            break
        bi, ci = torch.nonzero(has, as_tuple=True)
        ki = win_k[bi, ci]
        revenue.index_add_(0, bi, second[bi, ci])
        newA = torch.zeros_like(A)
        newA[bi, ki, ci] = 1.0
        Icur = Icur + torch.einsum("bkc,bjc,bkj->bj", newA, A, sb.G)
        Icur[bi, ki] = Ikc[bi, ki, ci]
        Ncur[bi, ki] = 1 + sb.inr_hat[bi, ci]
        qcur[bi, ki] = 1 - sb.phat[bi, ci]
        ch[bi, ki] = ci
        active[bi, ki] = False
        A = A + newA
    return ch, (ch < C).float(), revenue


# ==========================================================================
#                                 METHODS
# ==========================================================================


class LBTSense(Allocator):
    label, ref = "LBT-sense", "listen-before-talk on the last sensed slot (interweave)"

    def act(self, sb):
        pp = sb.ppers if sb.ppers is not None else sb.phat
        s2 = copy.copy(sb)
        s2.phat = pp
        s2.inr_hat = pp * sb.inr_busy + (1 - pp) * sb.inr_idle
        return greedy_assign(s2, self.cfg, allowed=pp < 0.5)


class AuctionShare(Allocator):
    label, ref = "Auction (BLAST mechanism)", "Abognah & Basir, arXiv:2604.12127; LLM agents not reproduced"

    def act(self, sb):
        ch, pw, rev = auction_assign(sb, self.cfg)
        self.last_revenue = rev
        return ch, pw


class _ShareSeqEnv:
    """Sequential (channel, power-level) decisions for the links of one slot,
    strongest direct link first. Forecast-based state; TRUE marginal reward."""

    def __init__(self, sb: SlotBatch, cfg):
        self.sb, self.cfg = sb, cfg
        B, K, C = sb.B, sb.K, sb.C
        self.lv = torch.tensor(cfg["power_levels"], device=sb.dev)
        self.nl = len(cfg["power_levels"])
        self.Gd = torch.diagonal(sb.G, dim1=1, dim2=2)
        self.order = torch.argsort(-self.Gd, dim=1)
        self.A = torch.zeros(B, K, C, device=sb.dev)
        self.P = torch.zeros(B, K, device=sb.dev)
        self.I = torch.zeros(B, K, device=sb.dev)
        self.Nt = torch.ones(B, K, device=sb.dev)
        self.Ipu = torch.zeros(B, C, device=sb.dev)
        self.Ipu_hat = torch.zeros(B, C, device=sb.dev)
        self.ch = torch.full((B, K), C, dtype=torch.long, device=sb.dev)
        self.pw = torch.zeros(B, K, device=sb.dev)
        self.ar = torch.arange(B, device=sb.dev)
        self.i = 0

    def n_actions(self):
        return self.sb.C * self.nl + 1

    def state(self):
        sb, k = self.sb, self.order[:, self.i]
        Ikc = torch.einsum("bjc,bj->bc", self.A, self.P * sb.G[self.ar, :, k])
        g = 10 * torch.log10(self.Gd[self.ar, k] + 1e-12) / 40.0
        gs = 10 * torch.log10(sb.Gsp[self.ar, k] + 1e-12) / 40.0
        return torch.cat([g[:, None], gs[:, None], sb.phat, torch.log1p(sb.inr_hat) / 3,
                          torch.log1p(Ikc) / 3, torch.log1p(self.Ipu_hat) / 3,
                          torch.full_like(g[:, None], self.i / sb.K)], 1)

    def step(self, a):
        sb, cfg, C = self.sb, self.cfg, self.sb.C
        k, b = self.order[:, self.i], self.ar
        go = a < C * self.nl
        c = (a // self.nl).clamp(max=C - 1)
        p = self.lv[(a % self.nl).clamp(max=self.nl - 1)] * go.float()
        Ikc = torch.einsum("bjc,bj->bc", self.A, self.P * sb.G[b, :, k])
        Sk = self.Gd[b, k]
        own = _rates(p * Sk, 1 + sb.inr[b, c], Ikc[b, c])
        g_out = sb.G[b, k, :]
        on_c = self.A[b, :, c]
        harm = (on_c * (_rates(self.P * self.Gd, self.Nt, self.I)
                        - _rates(self.P * self.Gd, self.Nt, self.I + p[:, None] * g_out))).sum(1)
        gam = gamma_pu(cfg)
        busy = sb.y[b, c] > 0.5
        before = busy & (self.Ipu[b, c] > 0) & (sb.inr[b, c] / (1 + self.Ipu[b, c]) < gam)
        ipu_new = self.Ipu[b, c] + p * sb.Gsp[b, k]
        after = busy & go & (sb.inr[b, c] / (1 + ipu_new) < gam)
        r = torch.where(go, own - harm - cfg["violation_penalty"] * (after & ~before).float(),
                        torch.zeros_like(own))
        if go.any():
            bb, kk, cc, pp = b[go], k[go], c[go], p[go]
            self.I[bb] = self.I[bb] + on_c[go] * pp[:, None] * g_out[go]
            self.A[bb, kk, cc] = 1.0
            self.P[bb, kk] = pp
            self.I[bb, kk] = Ikc[bb, cc]
            self.Nt[bb, kk] = 1 + sb.inr[bb, cc]
            self.Ipu[bb, cc] = ipu_new[go]
            self.Ipu_hat[bb, cc] = self.Ipu_hat[bb, cc] + pp * sb.Gsp[bb, kk]
            self.ch[bb, kk] = cc
            self.pw[bb, kk] = pp
        self.i += 1
        return r, self.i >= sb.K


class DQNDSS(Allocator):
    learned = True
    label, ref = "DQN-DSS", "Challita & Sandberg, arXiv:2102.11176 (2021)"

    def __init__(self, cfg, dev, C):
        super().__init__(cfg, dev, C)
        self.nl = len(cfg["power_levels"])
        self.nets["q"] = QNet(4 * C + 3, C * self.nl + 1).to(dev)
        self.tgt = copy.deepcopy(self.nets["q"])
        self.replay: Dict[int, _Replay] = {}
        self.updates, self.episodes = 0, 0

    def act(self, sb):
        env = _ShareSeqEnv(sb, self.cfg)
        q = self.nets["q"]; q.eval()
        with torch.no_grad():
            done = False
            while not done:
                _, done = env.step(q(env.state()).argmax(1))
        return env.ch, env.pw

    def train_steps(self, src, n, rng, client=0):
        cfg = self.cfg
        rb = self.replay.setdefault(client, _Replay(cfg["dqn_replay"]))
        q = self.nets["q"]; opt = self.opt_for("q", q)
        self.tgt.load_state_dict(q.state_dict())
        e0, e1 = cfg["dqn_eps"]
        tot = max(1, cfg["train_steps"] // 5)
        nA = self.C * self.nl + 1
        for _ in range(max(1, n // 5)):
            eps = max(e1, e0 - (e0 - e1) * self.episodes / tot)
            env = _ShareSeqEnv(src.batch(rng, 64), cfg)
            s, done = env.state(), False
            while not done:
                with torch.no_grad():
                    a = q(s).argmax(1)
                rnd = torch.as_tensor(rng.random(a.shape[0]) < eps, device=a.device)
                a = torch.where(rnd, torch.as_tensor(rng.integers(0, nA, a.shape[0]), device=a.device), a)
                r, done = env.step(a)
                s2 = s if done else env.state()
                rb.add(s, a, r, s2, torch.full_like(r, float(done)))
                s = s2
            self.episodes += 1
            q.train()
            for _ in range(16):
                S, Aa, R, S2, D = rb.sample(rng, 512)
                with torch.no_grad():
                    y = R + cfg["dqn_gamma"] * (1 - D) * self.tgt(S2).max(1).values
                loss = F.smooth_l1_loss(q(S).gather(1, Aa[:, None]).squeeze(1), y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 1.0); opt.step()
                self.updates += 1
                if self.updates % cfg["dqn_target_every"] == 0:
                    self.tgt.load_state_dict(q.state_dict())


class MAIDQN(Allocator):
    """Independent DQN agents (shared weights) acting SIMULTANEOUSLY; each sees
    the forecast, its own gains and its own previous outcome (ACK / PU
    warning). Trained with FedAvg across scenarios this is Fed-MADRL."""
    learned = True
    sequential = True
    label, ref = "MA-IDQN", "Shin et al., IEEE Commun. Lett. 2024"

    def __init__(self, cfg, dev, C):
        super().__init__(cfg, dev, C)
        self.lv = torch.tensor(cfg["power_levels"], device=dev)
        self.nl = len(cfg["power_levels"])
        self.nets["q"] = QNet(3 * C + 5, C * self.nl + 1).to(dev)
        self.tgt = copy.deepcopy(self.nets["q"])
        self.replay: Dict[int, _Replay] = {}
        self.updates, self.episodes = 0, 0
        self.prev = None

    def set_mode(self, mode):
        if mode == "fedavg":
            self.label, self.ref = "Fed-MADRL", "Chang et al., IEEE TWC 2023 (FedAvg of MA-IDQN agents)"

    def _decode(self, a, C):
        go = a < C * self.nl
        ch = torch.where(go, a // self.nl, torch.full_like(a, C))
        pw = self.lv[(a % self.nl).clamp(max=self.nl - 1)] * go.float()
        return ch, pw

    def _obs(self, sb, prev):
        B, K, C = sb.B, sb.K, sb.C
        g = sb.gdirect_db() / 40.0
        gs = 10 * torch.log10(sb.Gsp + 1e-12) / 40.0
        if prev is None:
            pc = torch.full((B, K), C, dtype=torch.long, device=sb.dev)
            pr = torch.zeros(B, K, device=sb.dev); pv = torch.zeros(B, K, device=sb.dev)
        else:
            pc, pr, pv = prev
        return torch.cat([sb.phat[:, None, :].expand(B, K, C),
                          (torch.log1p(sb.inr_hat) / 3)[:, None, :].expand(B, K, C),
                          g[..., None], gs[..., None], F.one_hot(pc, C + 1).float(),
                          (pr / 5)[..., None], pv[..., None]], -1)

    def reset(self, B):
        self.prev = None

    def act(self, sb):
        q = self.nets["q"]; q.eval()
        with torch.no_grad():
            return self._decode(q(self._obs(sb, self.prev)).argmax(-1), sb.C)

    def feedback(self, sb, ch, out):
        self.prev = (ch, out["rate"], out["viol_link"].float())

    def train_steps(self, src, n, rng, client=0):
        cfg = self.cfg
        rb = self.replay.setdefault(client, _Replay(cfg["dqn_replay"]))
        q = self.nets["q"]; opt = self.opt_for("q", q)
        self.tgt.load_state_dict(q.state_dict())
        e0, e1 = cfg["dqn_eps"]
        tot = max(1, cfg["train_steps"] // 5)
        nA = self.C * self.nl + 1
        for _ in range(max(1, n // 5)):
            eps = max(e1, e0 - (e0 - e1) * self.episodes / tot)
            seq = src.sequence(rng, cfg["idqn_parallel"], 32)
            prev = None
            o = self._obs(seq[0], prev)
            for t, sb in enumerate(seq):
                with torch.no_grad():
                    a = q(o).argmax(-1)
                rnd = torch.as_tensor(rng.random(tuple(a.shape)) < eps, device=a.device)
                a = torch.where(rnd, torch.as_tensor(rng.integers(0, nA, tuple(a.shape)), device=a.device), a)
                ch, pw = self._decode(a, sb.C)
                out = evaluate_share(sb, ch, pw, cfg)
                r = out["rate"] - cfg["violation_penalty"] * out["viol_link"].float()
                prev = (ch, out["rate"], out["viol_link"].float())
                last = t == len(seq) - 1
                o2 = o if last else self._obs(seq[t + 1], prev)
                Fd = o.shape[-1]
                rb.add(o.reshape(-1, Fd), a.reshape(-1), r.reshape(-1), o2.reshape(-1, Fd),
                       torch.full((a.numel(),), float(last), device=a.device))
                o = o2
            self.episodes += 1
            q.train()
            for _ in range(16):
                S, Aa, R, S2, D = rb.sample(rng, 1024)
                with torch.no_grad():
                    y = R + cfg["idqn_gamma"] * (1 - D) * self.tgt(S2).max(1).values
                loss = F.smooth_l1_loss(q(S).gather(1, Aa[:, None]).squeeze(1), y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 1.0); opt.step()
                self.updates += 1
                if self.updates % cfg["dqn_target_every"] == 0:
                    self.tgt.load_state_dict(q.state_dict())


class AdapSharePolicy(nn.Module):
    def __init__(self, fin, levels, h=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(fin, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.pi, self.v = nn.Linear(h, levels), nn.Linear(h, 1)

    def forward(self, s):
        z = self.body(s)
        return self.pi(z), self.v(z).squeeze(-1)


class AdapSharePPO(Allocator):
    """Gopal, Griffith, Rouil, Liu, 'AdapShare: An RL-Based Dynamic Spectrum
    Sharing Solution for O-RAN', arXiv:2408.16842. ADAPTATION: a PPO agent
    observes the forecast and SU demand and sets the SHARE of sub-channels lent
    to the secondary tier; links are spread over the least-likely-busy ones."""
    learned = True
    label, ref = "AdapShare-PPO", "Gopal et al., arXiv:2408.16842 (2024)"

    def __init__(self, cfg, dev, C):
        super().__init__(cfg, dev, C)
        self.L = cfg["adapshare_levels"]
        self.nets["pi"] = AdapSharePolicy(C + 3, self.L).to(dev)

    def _state(self, sb):
        g = sb.gdirect_db() / 40.0
        return torch.cat([torch.sort(sb.phat, 1).values, sb.inr_hat.mean(1, keepdim=True) / 10,
                          torch.full_like(g[:, :1], sb.K / 64.0), g.mean(1, keepdim=True)], 1)

    def _alloc(self, sb, lev):
        C = sb.C
        n_ch = torch.round(lev.float() / (self.L - 1) * C).long()
        cand = torch.argsort(sb.phat, 1)
        rank = torch.argsort(torch.argsort(-torch.diagonal(sb.G, dim1=1, dim2=2), 1), 1)
        ch = torch.gather(cand, 1, rank % n_ch.clamp(min=1)[:, None])
        ch = torch.where(n_ch[:, None] > 0, ch, torch.full_like(ch, C))
        return ch, (ch < C).float()

    def act(self, sb):
        net = self.nets["pi"]; net.eval()
        with torch.no_grad():
            lg, _ = net(self._state(sb))
        return self._alloc(sb, lg.argmax(1))

    def train_steps(self, src, n, rng, client=0):
        cfg = self.cfg
        net = self.nets["pi"]; opt = self.opt_for("pi", net)
        for _ in range(n):
            sb = src.batch(rng, cfg["alloc_batch"])
            s = self._state(sb)
            with torch.no_grad():
                lg, v = net(s)
                dist = torch.distributions.Categorical(logits=lg)
                a = dist.sample(); lp0 = dist.log_prob(a)
                ch, pw = self._alloc(sb, a)
                r = evaluate_share(sb, ch, pw, cfg)["utility"] / sb.K
                adv = r - v
                adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            net.train()
            for _ in range(cfg["ppo_epochs"]):
                lg, v2 = net(s)
                d2 = torch.distributions.Categorical(logits=lg)
                ratio = torch.exp(d2.log_prob(a) - lp0)
                c = cfg["ppo_clip"]
                loss = (-torch.min(ratio * adv, ratio.clamp(1 - c, 1 + c) * adv).mean()
                        + 0.5 * F.mse_loss(v2, r) - 0.01 * d2.entropy().mean())
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()


SOTA_METHODS = {"lbt_sense": LBTSense, "auction": AuctionShare, "dqn_dss": DQNDSS,
                "adapshare_ppo": AdapSharePPO, "ma_idqn": MAIDQN}


# ==========================================================================
#          ABLATION (stage 2): client configurations of the forecaster
#          and of the federated policy training
# ==========================================================================
def ab_tasks(cfg) -> List[Tuple]:
    """Same task order as the forecasting ablation (SLURM aftercorr pairing):
    (K, partition) pairs, then the centralised reference."""
    return [("K", k, p) for k in cfg["ab_clients"] for p in cfg["ab_partitions"]] + [("ref", None, None)]


def task_tag(t) -> str:
    return "references" if t[0] == "ref" else f"K{t[1]:02d}-{t[2]}"


def task_configs(t, cfg) -> List[Tuple[str, int, str]]:
    """-> [(partition, n_clients, hand-off key)]"""
    if t[0] == "ref":
        return [("pooled", 1, "centralized")]
    return [(t[2], t[1], f"fedavg-{t[2]}-K{t[1]}")]


class AbBundle(Bundle):
    """stage2/ of 02_forecasting.py."""

    def best(self, mkey: str, family: str = "novel") -> Optional[str]:
        f = os.path.join(self.dir, f"best_{family.lower()}__{mkey}.json")
        if not os.path.isfile(f):
            return None
        return json.load(open(f))["model"]


class ShardSource:
    """One federated client: a subset of the training slots of one or more
    scenarios. A batch is drawn from one of its scenario parts (weighted by
    size), exactly as the pooled centralised source does."""

    def __init__(self, parts):
        self.parts = [(s, np.asarray(ix)) for s, ix in parts if len(ix) > 0]
        self.N = int(sum(len(ix) for _, ix in self.parts))
        self.w = np.array([len(ix) for _, ix in self.parts], np.float64) / max(self.N, 1)
        self.mix = {s.name: len(ix) for s, ix in self.parts}

    def batch(self, rng, B):
        s, ix = self.parts[int(rng.choice(len(self.parts), p=self.w))]
        G, Gsp = gen_topologies(rng, B, s.cfg)
        return s._sb(ix[rng.integers(0, len(ix), B)], G, Gsp)

    def sequence(self, rng, M, L):
        """Consecutive training slots of one scenario part. A sequence needs at
        least 2 slots, so parts with a single slot (few sweeps per band, many
        clients) are skipped; the draw is unchanged when every part has >= 2.
        If every part holds one slot, a one-step sequence is returned."""
        ok = [i for i, (_, ix) in enumerate(self.parts) if len(ix) >= 2]
        if not ok:
            s, ix = self.parts[int(rng.choice(len(self.parts), p=self.w))]
            G, Gsp = gen_topologies(rng, M, s.cfg)
            return [s._sb(np.repeat(ix[:1], M), G, Gsp)]
        w = self.w[ok] / self.w[ok].sum()
        s, ix = self.parts[ok[int(rng.choice(len(ok), p=w))]]
        L = min(L, len(ix) - 1)
        st = rng.integers(0, len(ix) - L, M)
        G, Gsp = gen_topologies(rng, M, s.cfg)
        return [s._sb(ix[st + t], G, Gsp) for t in range(L)]


def merge_until_balanced(n: int, pos: List[float], neg: List[float], min_per_class: int) -> List[int]:
    """_merge_until_balanced() of the sensing pipeline: merge ADJACENT time
    chunks until every client holds >= min_per_class occupied and vacant
    cells. -> chunk -> client map."""
    cl = [[i] for i in range(n)]
    pos, neg = list(pos), list(neg)
    changed = True
    while changed and len(cl) > 1:
        changed = False
        for i in range(len(cl)):
            if pos[i] < min_per_class or neg[i] < min_per_class:
                j = i + 1 if i + 1 < len(cl) else i - 1
                a, b = min(i, j), max(i, j)
                cl[a] = cl[a] + cl[b]; pos[a] += pos[b]; neg[a] += neg[b]
                del cl[b], pos[b], neg[b]
                changed = True
                break
    out = np.empty(n, int)
    for c, members in enumerate(cl):
        out[members] = c
    return out.tolist()


def temporal_shards(sources: List, K: int, min_per_class: int) -> List["ShardSource"]:
    """Non-IID TEMPORAL clients, same rule as 02_forecasting.py and the
    sensing pipeline: a client = one receiver during one contiguous stretch
    of the evening (equal numbers of training slots, all bands of it);
    several radios share the K clients; adjacent stretches merged until each
    client holds both classes."""
    radios = sorted({getattr(s, "radio", "radio") for s in sources})
    Kr = [K // len(radios) + (1 if i < K % len(radios) else 0) for i in range(len(radios))]
    out = []
    for r, kr in zip(radios, Kr):
        srcs = [s for s in sources if getattr(s, "radio", "radio") == r]
        kr = max(1, kr)
        allt = np.sort(np.concatenate([s.t_tr for s in srcs]))
        inner = np.quantile(allt, np.linspace(0, 1, kr + 1)[1:-1]) if kr > 1 else np.zeros(0)
        chunk = {id(s): np.searchsorted(inner, s.t_tr, "right") for s in srcs}
        pos = [float(sum(s.ypos[chunk[id(s)] == k].sum() for s in srcs)) for k in range(kr)]
        tot = [float(sum((chunk[id(s)] == k).sum() * s.ncell for s in srcs)) for k in range(kr)]
        cmap = merge_until_balanced(kr, pos, [t - p for t, p in zip(tot, pos)], min_per_class)
        for c in range(max(cmap) + 1):
            parts = [(s, np.flatnonzero(np.isin(chunk[id(s)], [k for k in range(kr) if cmap[k] == c])))
                     for s in srcs]
            sh = ShardSource(parts)
            if sh.N:
                out.append(sh)
    return out


def shard_sources(sources: List, partition: str, K: int, seed: int, cfg) -> List[ShardSource]:
    """Same partitions as 02_forecasting.py, on the training SLOTS:
    noniid = TEMPORAL clients (one stretch of the evening, all bands; the
             sensing pipeline's 'natural' partition), merged until balanced,
    band   = one band per client, contiguous time blocks of it (control),
    iid    = uniformly random equal shares of the pooled slots."""
    S = len(sources)
    if partition == "noniid":
        return temporal_shards(sources, K, cfg.get("min_per_class_per_client", 8))
    if partition == "band":
        partition = "noniid_band"
    if partition == "noniid_band" and K < S:            # fewer clients than bands: whole bands grouped
        return [ShardSource([(sources[i], np.arange(sources[i].N)) for i in range(S) if i % K == ci])
                for ci in range(K)]
    if partition == "noniid_band":
        per = [K // S + (1 if i < K % S else 0) for i in range(S)]
        out = []
        for i, src in enumerate(sources):
            cut = np.linspace(0, src.N, per[i] + 1).astype(int)
            out += [ShardSource([(src, np.arange(a, b))]) for a, b in zip(cut[:-1], cut[1:])]
        return out
    if partition == "iid":
        allp = np.concatenate([np.stack([np.full(s.N, i), np.arange(s.N)], 1) for i, s in enumerate(sources)])
        rng = np.random.default_rng(run_seed(cfg, "iid-slots", K, seed))
        out = []
        for sh in np.array_split(rng.permutation(len(allp)), K):
            sel = allp[np.sort(sh)]
            out.append(ShardSource([(sources[i], sel[sel[:, 0] == i, 1]) for i in range(S)]))
        return out
    raise ValueError(partition)


def shard_rows(clients: List[ShardSource], names, partition, K, seed) -> List[Dict]:
    rows = []
    for i, c in enumerate(clients):
        r = {"partition": partition, "n_clients": K, "n_clients_realized": len(clients), "seed": seed,
             "client": i, "n_train_slots": c.N}
        for s in names:
            r[f"frac_{s}"] = c.mix.get(s, 0) / max(c.N, 1)
        rows.append(r)
    return rows


def train_policy_k(alloc, clients: List[ShardSource], cfg, seed) -> Dict:
    """FedAvg with K clients. Every round each client takes
    max(fl_min_local_steps, fl_steps_per_round / K) local steps, so the total
    computation per round is the same for every K (K = 4 -> 50 steps = the
    main run); sample-weighted averaging."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    if not alloc.learned:
        return {"train_s": 0.0, "comm_bytes": 0}
    K = len(clients)
    steps = max(cfg["fl_min_local_steps"], int(round(cfg["fl_steps_per_round"] / K)))
    rounds = cfg["fl_rounds"]
    for _ in range(rounds):
        g = {k: copy.deepcopy(n.state_dict()) for k, n in alloc.nets.items()}
        states, ws = [], []
        for ci, src in enumerate(clients):
            for k, n in alloc.nets.items():
                n.load_state_dict(g[k])
            alloc.train_steps(src, steps, rng, client=ci)
            states.append({k: copy.deepcopy(n.state_dict()) for k, n in alloc.nets.items()})
            ws.append(src.N)
        for k, n in alloc.nets.items():
            n.load_state_dict(_avg_states([s[k] for s in states], ws))
    nb = sum(model_bytes(n) for n in alloc.nets.values())
    return {"train_s": time.time() - t0, "comm_bytes": 2 * nb * K * rounds, "rounds": rounds,
            "local_steps": steps, "comm_bytes_per_client_per_round": 2 * nb}


def method_family(mk) -> str:
    return "novel" if mk in METHODS else "SOTA"


def make_method(mk, cfg, dev, C, st_list, fsf, mode):
    """Novel methods use the novel run's settings; SOTA methods the SOTA run's
    settings (CONFIG['sota_overrides']: training steps, batch, FL rounds,
    power levels, ...), so every method is trained exactly as in its main run."""
    if mk in METHODS:
        a = METHODS[mk](cfg, dev, C)
        a.setup(st_list, fsf)
    else:
        a = SOTA_METHODS[mk]({**cfg, **cfg["sota_overrides"]}, dev, C)
        if hasattr(a, "set_mode"):
            a.set_mode(mode)
    a.key = mk
    return a


def _atomic_pickle(path, obj):
    import pickle
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)


# ---- resume after a crash / restart -------------------------------------------
# parts_v2/<part>.pkl            finished part -> skipped on the next run
# parts_v2/progress/<mkey>.pkl   rows of the seeds finished inside a running part
# Both carry run_fp(): methods, seeds, horizons, training budget and the
# stage-2 hand-off (manifest + forecast files). Any change -> recomputed.
# --redo ignores both.
_REDO = False


def run_fp(cfg) -> str:
    keys = ["methods", "sota_methods", "sota_methods_ablation", "seeds", "horizons", "forecasters",
            "train_steps", "fl_rounds", "fl_local_steps", "fl_steps_per_round", "su_links",
            "topo_block", "topo_pool_test", "violation_penalty", "sota_overrides", "seed"]
    h = hashlib.sha1(json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str).encode())
    h.update(b"engine_v2")                              # parts written before the engine fix are recomputed
    d, _ = find_stage2(cfg["forecast_dir"])
    if d:
        for f in sorted(glob.glob(os.path.join(d, "*.json")) + glob.glob(os.path.join(d, "*.npz"))):
            st = os.stat(f)
            h.update(f"{os.path.basename(f)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def _progress_path(cfg, mkey):
    return os.path.join(cfg["out_dir"], "parts_v2", "progress", f"{mkey}.pkl")


def load_progress(cfg, mkey, part, K, fp):
    """Rows of the seeds already finished inside this configuration."""
    import pickle
    p = _progress_path(cfg, mkey)
    if _REDO or not os.path.isfile(p):
        return [], [], [], [], {}, set()
    try:
        with open(p, "rb") as fh:
            z = pickle.load(fh)
    except Exception as e:
        log(f"  progress file unreadable ({e}) -> ignored"); return [], [], [], [], {}, set()
    if z.get("_fp") != fp:
        log(f"  progress of {mkey} was saved under other settings / forecasts -> recomputed")
        return [], [], [], [], {}, set()
    sel = lambda r: r.get("partition") == part and int(r.get("n_clients", -1)) == K
    R = [r for r in z["ROWS"] if sel(r)]
    done = {(int(r["seed"]), r["method_key"], r["forecaster_key"], int(r["horizon_step"])) for r in R}
    seeds = sorted({int(r["seed"]) for r in R})
    if R:
        log(f"  [{mkey}] resuming: seed(s) {seeds} already finished -> not run again")
    return (R, [r for r in z["SNR"] if sel(r)], [r for r in z["COMM"] if sel(r)],
            [r for r in z["CLI"] if sel(r)], dict(z["CX"]), done)


def save_progress(cfg, mkey, part, K, fp, ROWS, SNR, COMM, CLI, CX):
    sel = lambda r: r.get("partition") == part and int(r.get("n_clients", -1)) == K
    os.makedirs(os.path.dirname(_progress_path(cfg, mkey)), exist_ok=True)
    _atomic_pickle(_progress_path(cfg, mkey), {"_fp": fp, "ROWS": [r for r in ROWS if sel(r)],
                                               "SNR": [r for r in SNR if sel(r)],
                                               "COMM": [r for r in COMM if sel(r)],
                                               "CLI": [r for r in CLI if sel(r)], "CX": dict(CX)})


def merge_parts(out_dir, prefix, keys, snr_keys):
    import pickle
    files = sorted(glob.glob(os.path.join(out_dir, "parts_v2", "*.pkl")))
    if not files:
        log("  no parts to merge yet"); return
    R, SN, CM, CX, CL = [], [], [], {}, []
    for f in files:
        with open(f, "rb") as fh:
            p = pickle.load(fh)
        R += p["ROWS"]; SN += p["SNR"]; CM += p["COMM"]; CX.update(p.get("CX", {})); CL += p.get("CLI", [])
    W = lambda n, rows: write_csv(os.path.join(out_dir, n), rows)
    W(f"{prefix}_overall.csv", R)
    W(f"{prefix}_overall_agg.csv", aggregate(R, keys))
    W(f"{prefix}_per_snr.csv", aggregate(SN, snr_keys))
    W("fl_communication.csv", CM)
    W("fl_communication_agg.csv", aggregate(CM, ["method", "family", "mode", "partition", "n_clients"]))
    W("client_partitions.csv", CL)
    if CX:
        W(f"{prefix}_complexity.csv", list(CX.values()))
    log(f"  merged {len(files)} part(s): {', '.join(os.path.basename(f)[:-4] for f in files)}")


def ab_main(desc, prefix, run_task, keys, snr_keys, all_methods):
    global _LOG_FH
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--forecast-dir", help="out_dir of 02_forecasting.py --family sota")
    ap.add_argument("--out"); ap.add_argument("--seeds"); ap.add_argument("--methods")
    ap.add_argument("--sota-methods", help="SOTA methods of the main comparison ('' = none)")
    ap.add_argument("--sota-methods-ablation", help="SOTA methods of the off-main ablation parts")
    ap.add_argument("--list-tasks", action="store_true", help="print the part indices and exit")
    ap.add_argument("--clients"); ap.add_argument("--partitions")
    ap.add_argument("--task", type=int, help="run one part only (SLURM array index)")
    ap.add_argument("--merge", action="store_true", help="only rebuild the merged CSVs from parts/")
    ap.add_argument("--redo", action="store_true", help="recompute parts / seeds already finished")
    ap.add_argument("--device"); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = CONFIG
    if a.quick:
        cfg.update(QUICK)
    if a.forecast_dir:
        cfg["forecast_dir"] = a.forecast_dir
    if a.out:
        cfg["out_dir"] = a.out
    if a.seeds:
        cfg["seeds"] = [int(x) for x in a.seeds.split(",")]
    if a.methods:
        cfg["methods"] = [x.strip() for x in a.methods.split(",")]
    if a.sota_methods is not None:
        cfg["sota_methods"] = [x.strip() for x in a.sota_methods.split(",") if x.strip()]
    if a.sota_methods_ablation is not None:
        cfg["sota_methods_ablation"] = [x.strip() for x in a.sota_methods_ablation.split(",") if x.strip()]
    if a.clients:
        cfg["ab_clients"] = [int(x) for x in a.clients.split(",")]
    if a.partitions:
        cfg["ab_partitions"] = [x.strip() for x in a.partitions.split(",")]
    if a.device:
        cfg["device"] = a.device
    bad = ([m for m in cfg["methods"] if m not in METHODS]
           + [m for m in cfg["sota_methods"] + cfg["sota_methods_ablation"] if m not in SOTA_METHODS])
    if bad:
        sys.exit(f"unknown method(s) {bad}; novel: {list(METHODS)}; SOTA: {list(SOTA_METHODS)}")
    cfg["out_dir"] = resolve_out(cfg["out_dir"])
    out_dir = cfg["out_dir"]
    os.makedirs(os.path.join(out_dir, "parts_v2"), exist_ok=True)
    if cfg["ab_clients"] == "from_forecast":            # same K list (and task order) as the forecasting run
        d, tried = find_stage2(cfg["forecast_dir"])
        if d is None and not a.merge:
            sys.exit("no stage2/ hand-off found (looked in " + ", ".join(tried) + "); run "
                     "02_forecasting.py first or pass --forecast-dir / --clients")
        man = json.load(open(os.path.join(d, "manifest.json"))) if d else {}
        cfg["ab_clients"] = [int(k) for k in man.get("ab_clients", [])]
        cfg["main_clients"] = int(man.get("main_clients", cfg.get("main_clients", 8)))
        cfg["min_per_class_per_client"] = int(man.get("min_per_class_per_client",
                                                      cfg.get("min_per_class_per_client", 8)))
        if "ab_partitions" in man and not a.partitions:
            cfg["ab_partitions"] = list(man["ab_partitions"])
    tasks = ab_tasks(cfg)
    if a.list_tasks:
        print(", ".join(f"{i}={task_tag(t)}" for i, t in enumerate(tasks)))
        return
    if a.task is not None and not 0 <= a.task < len(tasks):
        sys.exit(f"--task must be 0..{len(tasks) - 1}: " + ", ".join(f"{i}={task_tag(t)}" for i, t in enumerate(tasks)))
    sel = tasks if a.task is None else [tasks[a.task]]
    tag = "merge" if a.merge else ("all" if a.task is None else task_tag(sel[0]))
    _LOG_FH = open(os.path.join(out_dir, f"run_log_{tag}.txt"), "a", encoding="utf-8")
    log("=" * 78)
    log(f"{desc}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"output folder: {out_dir}")
    if a.merge:
        merge_parts(out_dir, prefix, keys, snr_keys); return
    dev = get_device(cfg)
    log(f"device {dev} | parts: {', '.join(task_tag(t) for t in sel)} | novel {cfg['methods']} | "
        f"SOTA {cfg['sota_methods']} | forecasters {cfg['forecasters']} | seeds {cfg['seeds']}")
    log("=" * 78)
    with open(os.path.join(out_dir, f"config_used_{tag}.json"), "w") as fh:
        json.dump(cfg, fh, indent=1, default=str)
    global _REDO
    _REDO = bool(a.redo)
    cfg["_run_fp"] = run_fp(cfg)
    bd = AbBundle(cfg)
    for t in sel:
        log(f"\n[part {task_tag(t)}]")
        pf = os.path.join(out_dir, "parts_v2", f"{task_tag(t)}.pkl")
        if os.path.isfile(pf) and not _REDO:
            try:
                import pickle
                with open(pf, "rb") as fh:
                    old_fp = pickle.load(fh).get("_fp")
            except Exception:
                old_fp = "unreadable"
            if old_fp == cfg["_run_fp"]:
                log(f"  already finished -> skipped (--redo to recompute)")
                continue
            log(f"  saved by an earlier version of this file or under other settings / forecasts -> recomputed")
        part = run_task(t, cfg, dev, bd)
        if part is None:
            continue
        part["_fp"] = cfg["_run_fp"]
        _atomic_pickle(pf, part)
        merge_parts(out_dir, prefix, keys, snr_keys)
    log(f"\nDone. Results in {out_dir}")


# ==========================================================================
#            ABLATION PART: novel + SOTA methods, both forecasters
# ==========================================================================
def _valid(P, names):
    return P is not None and all("tr1" in P[s] for s in names)


def run_task_s2(t, cfg, dev, bd, kind):
    """One (K, partition) configuration (or the centralised reference):
      * learned policies (DQN-DSS, AdapShare, MA-IDQN) are trained on the
        training slots of the best SOTA forecaster of that configuration,
        with FedAvg over the K clients of the same partition, or centrally
        for the reference;
      * every method is evaluated with that forecaster ('best')."""
    names, C, edges = bd.names, bd.C, bd.edges
    pool = gen_topologies(np.random.default_rng(cfg["seed"] + 999), cfg["topo_pool_test"], cfg)
    st_list = [scenario_stats(bd.D[s]["y_tr1"], bd.D[s]["E_tr1"], bd.eta, C) for s in names]
    ROWS, SNR, COMM, CX, CLI = [], [], [], {}, []
    Ksu = cfg["su_links"]
    pers = bd.preds("persistence", "non-learned", 0)
    U_cache = {}
    for part, K, mkey in task_configs(t, cfg):
        mode = "centralized" if part == "pooled" else "fedavg"
        main = part == "pooled" or (part == "noniid" and K == int(cfg.get("main_clients", 8)))
        all_m = list(cfg["methods"]) + list(cfg["sota_methods"] if main else cfg["sota_methods_ablation"])
        cmp_tag = "main" if main else "ablation"
        bs, bn = bd.best(mkey, "SOTA"), bd.best(mkey, "novel")
        if bs is None and bn is None:
            log(f"  [{mkey}] no forecaster in the hand-off (run the forecasting ablation first) -> skipped")
            continue
        done = set()
        fp = cfg.get("_run_fp", "")
        gR, gS, gC, gL, gX, gdone = load_progress(cfg, mkey, part, K, fp)    # seeds finished before a restart
        ROWS += gR; SNR += gS; COMM += gC; CLI += gL; done |= gdone
        for k, v in gX.items():
            CX.setdefault(k, v)
        for seed in cfg["seeds"]:
            todo = lambda mk, fc, h: (seed, mk, fc, h) not in done
            Ps = bd.preds(bs, mkey, seed) if bs else None
            Pn = bd.preds(bn, mkey, seed) if bn else None
            Ptr, btr = (Ps, bs) if _valid(Ps, names) else (Pn, bn)
            if not _valid(Ptr, names):
                log(f"  [{mkey} seed {seed}] forecasts missing -> skipped")
                continue
            fsf_tr = fit_fsf_all(bd, Ptr, names, cfg, dev)       # per scenario (site), never communicated
            sources = [MemSource(s, i, Ptr[s]["tr1"], bd.D[s]["y_tr1"], bd.D[s]["E_tr1"],
                                 bd.D[s]["inr_busy"], bd.D[s]["inr_idle"], cfg, dev) for i, s in enumerate(names)]
            for src, s in zip(sources, names):
                src.t_tr, src.radio = bd.D[s]["t_tr1"], bd.D[s]["radio"]
                src.ypos, src.ncell = bd.D[s]["y_tr1"].sum(1), float(bd.C)
            clients = None
            if mode == "fedavg":
                clients = shard_sources(sources, part, K, seed, cfg)
                CLI += shard_rows(clients, names, part, K, seed)
            fcs_avail = [fc for fc, P in (("best", Ps), ("best_novel", Pn))
                         if fc in cfg["forecasters"] and _valid(P, names)]
            hs = [h for h in cfg["horizons"] if h <= bd.H]
            need = [mk for mk in all_m if any(todo(mk, fc, h) for fc in fcs_avail for h in hs)]
            if not need:
                log(f"  [{mkey} seed {seed}] everything already done in the previous run -> skipped")
                continue
            ms = {}
            for mk in need:
                a = make_method(mk, cfg, dev, C, st_list, fsf_tr, mode)
                set_seed(run_seed(cfg, mk, mkey, seed))
                rs = run_seed(cfg, mk, mkey, seed, "rng")
                t0 = time.time()
                if hasattr(a, "calibrate"):                       # forecast-interweave threshold (as in the SOTA run)
                    a.calibrate(PooledSource(sources), np.random.default_rng(rs))
                info = (train_policy_k(a, clients, a.cfg, rs) if clients
                        else train_policy(a, sources, "centralized", a.cfg, rs))
                info["trained_on"] = bd.label(btr)
                ms[mk] = (a, info)
                if a.learned:
                    log(f"  trained {a.label:<26s} {mkey:<18s} seed {seed} ({time.time() - t0:.0f} s)")
                    COMM.append({"method": a.label, "method_key": mk, "family": method_family(mk), "mode": mode,
                                 "partition": part, "n_clients": K, "seed": seed,
                                 "trained_on_forecaster": bd.label(btr),
                                 "rounds": info.get("rounds", ""), "local_steps_per_client": info.get("local_steps", ""),
                                 "comm_MB_total": info.get("comm_bytes", 0) / 1e6,
                                 "comm_MB_per_client_per_round": info.get("comm_bytes_per_client_per_round", 0) / 1e6,
                                 "train_s": info["train_s"]})
            fcs = []
            if "best" in cfg["forecasters"] and _valid(Ps, names):
                fcs.append(("best", f"{bd.label(bs)} ({mkey})", "SOTA", Ps))
            if "best_novel" in cfg["forecasters"] and _valid(Pn, names):
                fcs.append(("best_novel", f"{bd.label(bn)} ({mkey}, novel)", "novel", Pn))
            for fc, flab, ffam, P in fcs:
                fsf_fc = fsf_tr if P is Ptr else fit_fsf_all(bd, P, names, cfg, dev)
                for h in cfg["horizons"]:
                    if h > bd.H or not any(todo(mk, fc, h) for mk in ms):
                        continue
                    blk = test_blocks_mem(bd, lambda s, P=P: P[s]["te"], (lambda s: pers[s]["te"]) if pers else None,
                                          h, pool, cfg)
                    L = blk["PH"].shape[1]
                    if h not in U_cache:
                        U_cache[h] = oracle_utility(blk, cfg, dev)      # truth only: same for every configuration
                    U_or = U_cache[h]
                    for mk, (a, info) in ms.items():
                        if not todo(mk, fc, h):
                            continue
                        acc, sh, sv, sn, lat = eval_method(a, blk, a.cfg, dev, edges, bd.eta)
                        for si, s in enumerate(names + (["Combined"] if len(names) > 1 else [])):
                            mask = np.ones(len(blk["SID"]), bool) if s == "Combined" else blk["SID"] == si
                            if not mask.any():
                                continue
                            met = share_row(acc, mask, L, Ksu)
                            r = {"method": a.label, "method_key": mk, "family": method_family(mk),
                                 "comparison": cmp_tag, "reference": a.ref,
                                 "trained": mode if a.learned else "non-learned", "mode": mode,
                                 "partition": part, "n_clients": K,
                                 "forecaster": flab, "forecaster_key": fc, "forecaster_family": ffam,
                                 "forecaster_config": mkey, "seed": seed,
                                 "scenario": s, "horizon_step": h, **met,
                                 "decision_latency_ms_per_slot": lat * 1e3,
                                 "train_s": info.get("train_s", 0.0), "comm_MB": info.get("comm_bytes", 0) / 1e6,
                                 "policy_trained_on": (f"best SOTA forecaster ({info['trained_on']})" if a.learned else ""),
                                 "reused_from_previous_run": False}
                            uo = float(U_or[mask].sum()) / (mask.sum() * L)
                            r["oracle_utility_per_slot"] = uo
                            r["regret_vs_oracle"] = 1 - r["utility_per_slot"] / uo if uo > 0 else float("nan")
                            ROWS.append(r)
                            bk = {k: r[k] for k in ("method", "method_key", "family", "trained", "mode", "partition",
                                                    "n_clients", "forecaster_key", "forecaster_config", "seed",
                                                    "scenario", "horizon_step")}
                            idle_k = "idle_capture"
                            SNR.append({**bk, "snr_bin": f"<= {bd.eta:g} (idle)",
                                        "n_cells": float(acc["n_idle"][mask].sum()), idle_k: r[idle_k]})
                            for i in range(len(edges) - 1):
                                n_ = float(sn[mask, i].sum())
                                row = {**bk, "snr_bin": f"{edges[i]:g}-{edges[i + 1]:g}", "n_cells": n_,
                                       "pu_hit_rate": float(sh[mask, i].sum()) / n_ if n_ else float("nan")}
                                if sv is not None:
                                    row["pu_violation_rate"] = float(sv[mask, i].sum()) / n_ if n_ else float("nan")
                                SNR.append(row)
                    log(f"  evaluated {sum(todo(mk, fc, h) for mk in ms)} methods | {flab} | h={h} | seed {seed}")
            for mk, (a, _) in ms.items():
                if mk not in CX:
                    row = profile_policy(a, C, a.cfg)
                    CX[mk] = {**row, "family": method_family(mk)}
            save_progress(cfg, mkey, part, K, fp, ROWS, SNR, COMM, CLI, CX)
    if not ROWS:
        log("  nothing evaluated in this part")
        return None
    return {"ROWS": ROWS, "SNR": SNR, "COMM": COMM, "CX": CX, "CLI": CLI}


AB_KEYS = ["method", "method_key", "family", "trained", "mode", "partition", "n_clients",
           "forecaster_key", "scenario", "horizon_step"]
AB_SNR_KEYS = AB_KEYS + ["snr_bin"]


if __name__ == "__main__":
    ab_main("Stage 3: SOTA spectrum-sharing baselines (LBT-sense, BLAST auction, DQN-DSS, AdapShare, MA-IDQN)", "share", lambda t, cfg, dev, bd: run_task_s2(t, cfg, dev, bd, "share"),
            AB_KEYS, AB_SNR_KEYS, None)
