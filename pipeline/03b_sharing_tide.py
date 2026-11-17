#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
03b_sharing_tide.py -- Stage 3 (proposed) of the sense-forecast-share pipeline
=============================================================================
TIDE (Tracking Idle probability with Dynamic Experts), paper Sec. VII and
Algorithm 3: training-free primary / secondary spectrum sharing on the
Fireworks dataset (IEEE DataPort, DOI 10.21227/f2rv-ms26). Same secondary
network, PU protection rule, test slots, client grid and metrics as
03_sharing_baselines.py, so the rows of both scripts form one table.

TIDE, per slot and band
  1. beliefs   four experts predict P(busy) of each sub-channel h sweeps ahead:
               (i) logistic forecast-sensing fusion (36 parameters per band,
               fitted locally on training slots), (ii) the raw forecast,
               (iii) an online two-state Markov chain started from the training
               transition rates with kappa = 20 pseudo-observations, (iv) its
               stationary rate. Exponential weights on the log-loss of what
               has been sensed (eta = 1, fixed share 0.02, prior weights
               0.7 / 0.1 / 0.1 / 0.1); a forecast for sweep t is scored only
               after sweep t is sensed (causal for every horizon).
  2. PU strength  8 training quantiles of the busy-PU power + 4 residual
               atoms around the last sensed power, weighted by persistence.
  3. planner   exact expected utility (sum rate - 10 x expected violated
               sub-channels, gamma = 3 dB), power ladder {0.03, 0.1, 0.3, 1},
               greedy strongest-first + 2 best-response passes.
  Nothing is trained; band statistics and the fusion never leave the client.

INPUT   <results>/02_forecasting/stage2/       (best SOTA forecaster: 'best')
        <results>/02b_forecasting_gem/stage2/  (GEM / GEM-DCS: 'best_novel')
        Forecasters evaluated: best_novel (TIDE + GEM-DCS, the paper's chain),
        best (TIDE with the best SOTA forecaster), and in the main settings
        the oracle and historical-mean forecasts as references.
RUN GRID  references (centralized forecasts) and K in {4, 8, 12, 16, 24, 32}
        x {non-IID temporal, IID}: TIDE is evaluated with the forecasts of
        every federated configuration.

USAGE
  python 03b_sharing_tide.py                 # every part + merge
  python 03b_sharing_tide.py --quick         # smoke test
  python 03b_sharing_tide.py --list-tasks / --task i / --merge
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
    "forecast_dir": os.path.join(_RES, "02_forecasting"),           # 02_forecasting.py --family sota
    "novel_forecast_dir": os.path.join(_RES, "02b_forecasting_gem"),  # 02_forecasting.py --family gem
    "out_dir": os.path.join(_RES, "03b_sharing_tide"),
    "methods": ["tide"],
    "sota_methods": [], "sota_methods_ablation": [],   # baselines: 03_sharing_baselines.py
    "forecasters": ["best_novel", "best"],
    "reference_forecasters": ["oracle", "historical_mean"],   # main settings only
    "sota_overrides": {},
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
    # ---- TIDE planner -----------------------------------------------------------
    "power_levels": (0.03, 0.1, 0.3, 1.0),   # power ladder
    "ls_passes": 2,                   # best-response passes after the greedy placement
    "strength_atoms": 4,              # residual atoms around the last sensed PU power
    # ---- TIDE online expert mixture -------------------------------------------
    "tide_w0": [0.7, 0.1, 0.1, 0.1],  # prior weights: fused | forecast | Markov | rate
    "tide_eta": 1.0,                  # learning rate on the log-loss
    "tide_share": 0.02,               # fixed share: no expert is ever switched off
    "tide_prior": 20.0,               # pseudo-observations behind the training transition rates
    # ---- unused by TIDE (kept for the shared framework) -----------------------
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

QUICK["ls_passes"] = 1

_LOG_FH = None


def log(m: str = "") -> None:
    if CONFIG.get("verbose", True):
        print(m, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(m + "\n")
        _LOG_FH.flush()


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


def scenario_stats(y1: np.ndarray, e1: np.ndarray, eta: float, C: int, R: int) -> Dict:
    """From TRAINING slots (h = 1) of one scenario (local statistics, never
    communicated): busy-INR quantile atoms per sub-channel, idle INR mean,
    how often a busy PU keeps its strength (+ the residual quantiles when it
    does), and the per-sub-channel busy rate and 1-step transition rates."""
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
    d = (e1[1:] - e1[:-1])[both]
    kept = np.abs(d) <= 3.0
    keep = float(kept.mean()) if d.size else 0.0
    res = (np.quantile(d[kept], (np.arange(R) + 0.5) / R) if kept.sum() >= 8 else np.zeros(R)).astype(np.float32)
    a, b = busy[:-1], busy[1:]
    return {"atoms": atoms, "inr_idle": idle, "persist": keep, "res": res,
            "pi": ((busy.sum(0) + 1.0) / (len(y1) + 2.0)).astype(np.float32),
            "p01": (((~a & b).sum(0) + 1.0) / ((~a).sum(0) + 2.0)).astype(np.float32),
            "p10": (((a & ~b).sum(0) + 1.0) / (a.sum(0) + 2.0)).astype(np.float32)}


def memory_of(y: np.ndarray, e: np.ndarray, lag: int = 1):
    """Sensing memory of CONSECUTIVE windows: what was measured `lag` windows
    before the target (observed, causal). lag = 1: the state at the forecast
    time of a 1-step forecast."""
    ym = np.zeros_like(y); em = np.zeros_like(e); ok = np.zeros(len(y), np.float32)
    if lag < len(y):
        ym[lag:], em[lag:], ok[lag:] = y[:-lag], e[:-lag], 1.0
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
#        TIDE PLANNER: beliefs -> exact expected-utility assignment
# ==========================================================================
class Ctx:
    """Per-slot beliefs of the planner (B slots on device)."""

    def __init__(self, sb: SlotBatch, q, atoms, wts, n_idle):
        self.sb, self.q, self.atoms, self.wts, self.n_idle = sb, q, atoms, wts, n_idle


def build_ctx(sb: SlotBatch, st_list: List[Dict], sid: np.ndarray, cfg, fsf_list: Optional[List] = None,
              use_memory: bool = True, refine: bool = True) -> Ctx:
    """q = fused (or raw) occupancy probability. Busy-INR distribution = the
    training quantile atoms of the sub-channel, mixed (weight = persistence,
    only when the sub-channel was busy at forecast time) with the last sensed
    PU power: R residual atoms (refine) or the single value (refine=False)."""
    dev, B, C = sb.dev, sb.B, sb.C
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
    atoms = t(np.stack([st_list[i]["atoms"] for i in sid]))                     # [B,C,Q]
    n_idle = 1.0 + t(np.stack([st_list[i]["inr_idle"] for i in sid]))          # [B,C]
    persist = t([st_list[i]["persist"] for i in sid])[:, None]
    q = sb.phat
    if fsf_list is not None:
        q = torch.empty_like(sb.phat)
        for i in np.unique(sid):
            m = torch.as_tensor(sid == i, device=dev)
            with torch.no_grad():
                q[m] = fsf_list[i](sb.phat[m], sb.ym[m], sb.em[m], sb.mok[m], cfg["eta_db"])
    wq = torch.full((B, C, Q_ATOMS), 1.0 / Q_ATOMS, device=dev)
    if use_memory:
        w = (persist * (sb.ym > 0.5).float() * sb.mok[:, None])[..., None]      # [B,C,1]
        if refine:
            res = t(np.stack([st_list[i]["res"] for i in sid]))                 # [B,R] dB
            R = res.shape[1]
            last = torch.clamp(torch.pow(10.0, (sb.em[..., None] + res[:, None, :]) / 10.0) - 1.0, min=0.0)
            wl = (w / R).expand(-1, -1, R)
        else:
            last = torch.clamp(torch.pow(10.0, sb.em / 10.0) - 1.0, min=0.0)[..., None]
            wl = w
        atoms = torch.cat([atoms, last], -1)
        wts = torch.cat([wq * (1 - w), wl], -1)
    else:
        wts = wq
    return Ctx(sb, q.clamp(0, 1), atoms, wts, n_idle)


class Engine:
    """Exact-expectation state of one allocation (B slots, K links).
    Expected rate of link j on sub-channel c with co-channel interference I:
      (1-q_c) log2(1 + p S_j/(N_idle,c + I)) + q_c sum_a w_ca log2(1 + p S_j/(1 + X_ca + I))
    Expected PU violations of sub-channel c with SU interference I_pu,c > 0:
      q_c sum_a w_ca 1[X_ca/(1 + I_pu,c) < gamma]
    Objective = sum of expected rates - risk_scale * penalty * expected violations.
    It is separable over sub-channels and every move below is an exact
    marginal change of it."""

    def __init__(self, ctx: Ctx, cfg, risk_scale: float = 1.0, link_charge: bool = False,
                 self_term: bool = False):
        sb = ctx.sb
        self.ctx, self.cfg = ctx, cfg
        self.B, self.K, self.C = sb.B, sb.K, sb.C
        self.G, self.S = sb.G, torch.diagonal(sb.G, dim1=1, dim2=2)
        # cross gains only: a link never interferes with itself. self_term=True
        # includes the self term (diagnostic only; TIDE uses False).
        self.Gx = sb.G if self_term else sb.G * (1.0 - torch.eye(self.K, device=sb.dev))
        self.ar = torch.arange(self.B, device=sb.dev)
        self.ch = torch.full((self.B, self.K), self.C, dtype=torch.long, device=sb.dev)
        self.P = torch.zeros(self.B, self.K, device=sb.dev)
        self.Ipu = torch.zeros(self.B, self.C, device=sb.dev)
        self.lam = cfg["collision_penalty"]
        self.mu = cfg["violation_penalty"] * risk_scale
        self.gam = 10 ** (cfg["pu_sinr_min_db"] / 10.0)
        self.link_charge = link_charge                 # ablation: SOTA charge lam * q per link

    @staticmethod
    def _r(S, I, q, ni, X, W):
        """E[log2(1+SINR)]: idle branch + busy branch over the PU-power atoms."""
        r_idle = torch.log2(1 + S / (ni + I))
        r_busy = (W * torch.log2(1 + S[..., None] / (1 + X + I[..., None]))).sum(-1)
        return (1 - q) * r_idle + q * r_busy

    def _gather_ch(self, c):
        q = torch.gather(self.ctx.q, 1, c)
        ni = torch.gather(self.ctx.n_idle, 1, c)
        X = self.ctx.atoms[self.ar[:, None], c]
        Wt = self.ctx.wts[self.ar[:, None], c]
        return q, ni, X, Wt

    def _vp(self, Ipu):
        """Violation probability for Ipu [B,C,*] -> same shape."""
        extra = Ipu.dim() - 2
        X = self.ctx.atoms.reshape(self.B, self.C, *([1] * extra), -1)
        Wt = self.ctx.wts.reshape(self.B, self.C, *([1] * extra), -1)
        q = self.ctx.q.reshape(self.B, self.C, *([1] * extra))
        v = (Wt * (X / (1 + Ipu[..., None]) < self.gam).float()).sum(-1)
        return q * v * (Ipu > 0).float()

    def interference(self):
        A = F.one_hot(self.ch.clamp(max=self.C - 1), self.C).float() * (self.ch < self.C).float()[..., None]
        return torch.einsum("bjc,bjk->bkc", A * self.P[..., None], self.Gx), A

    def link_values(self, k, levels):
        """Marginal expected objective of putting link k ([B]) on each
        sub-channel at each power level -> [B, C, L]; link k must be unassigned."""
        C, ar = self.C, self.ar
        lv = torch.as_tensor(levels, dtype=torch.float32, device=self.G.device)
        I_all, A = self.interference()
        Ikc = I_all[ar, k]                                                  # [B,C]
        Sk = self.S[ar, k]
        S_own = Sk[:, None, None] * lv[None, None, :]                       # [B,1,L]
        own = self._r(S_own, Ikc[..., None], self.ctx.q[..., None], self.ctx.n_idle[..., None],
                      self.ctx.atoms[:, :, None, :], self.ctx.wts[:, :, None, :])   # [B,C,L]
        jch = self.ch.clamp(max=C - 1)
        on = (self.ch < C).float()
        q, ni, X, Wt = self._gather_ch(jch)                                 # [B,K], [B,K,Q]
        Ij = torch.gather(I_all, 2, jch[..., None]).squeeze(-1)
        Sj = self.P * self.S
        base = self._r(Sj, Ij, q, ni, X, Wt)                                # [B,K]
        g_out = self.G[ar, k, :]                                            # tx k -> rx j
        new = self._r(Sj[..., None], Ij[..., None] + g_out[..., None] * lv, q[..., None], ni[..., None],
                      X[:, :, None, :], Wt[:, :, None, :])                  # [B,K,L]
        harm = torch.einsum("bjc,bjl->bcl", A, (base[..., None] - new) * on[..., None])
        val = own - harm
        if self.link_charge:
            return val - self.lam * self.ctx.q[..., None]
        gsp = self.ctx.sb.Gsp[ar, k][:, None, None]
        return val - self.mu * (self._vp(self.Ipu[..., None] + gsp * lv) - self._vp(self.Ipu)[..., None])

    def place(self, k, c, p, mask):
        b, kk, cc, pp = self.ar[mask], k[mask], c[mask], p[mask]
        self.ch[b, kk] = cc
        self.P[b, kk] = pp
        self.Ipu[b, cc] += pp * self.ctx.sb.Gsp[b, kk]

    def remove(self, k):
        b = self.ar
        on = self.ch[b, k] < self.C
        if on.any():
            bb, kk = b[on], k[on]
            cc = self.ch[bb, kk]
            self.Ipu[bb, cc] = (self.Ipu[bb, cc] - self.P[bb, kk] * self.ctx.sb.Gsp[bb, kk]).clamp(min=0)
        self.ch[b, k] = self.C
        self.P[b, k] = 0.0

    def best_response(self, k, levels):
        vals = self.link_values(k, levels)                                 # [B,C,L]
        best, idx = vals.reshape(self.B, -1).max(1)
        c, li = idx // len(levels), idx % len(levels)
        p = torch.as_tensor(levels, dtype=torch.float32, device=self.G.device)[li]
        self.place(k, c, p, best > 0)

    def solve(self, levels, ls_passes=2):
        """Greedy (strongest link first) + best-response local search: every
        move strictly increases the expected objective, so it terminates."""
        order = torch.argsort(-self.S, dim=1)
        for i in range(self.K):
            self.best_response(order[:, i], levels)
        for _ in range(ls_passes):
            for i in range(self.K):
                k = order[:, i]
                self.remove(k)
                self.best_response(k, levels)
        return self.ch.clone(), self.P.clone()


# ==========================================================================
#                 TIDE  (paper Sec. VII, Algorithm 3)
# ==========================================================================
class NovelShare(Allocator):
    """Common part: beliefs (fusion + PU-strength atoms) -> corrected planner."""
    use_fsf, use_memory, refine = True, True, True
    link_charge, self_term, ls = False, False, None

    def __init__(self, cfg, dev, C):
        super().__init__(cfg, dev, C)
        self.st, self.fsf, self.oracle_fc, self.h = None, None, False, 1
        self.levels = list(cfg["power_levels"])

    def setup(self, st_list, fsf_list):
        self.st, self.fsf = st_list, fsf_list

    def ctx(self, sb):
        if self.oracle_fc:                              # the forecast IS the future: nothing to fuse
            return build_ctx(sb, self.st, sb.sid, self.cfg, None, True, self.refine)
        return build_ctx(sb, self.st, sb.sid, self.cfg, self.fsf if self.use_fsf else None,
                         self.use_memory, self.refine)

    def plan(self, ctx, risk_scale=1.0):
        ls = self.cfg["ls_passes"] if self.ls is None else self.ls
        return Engine(ctx, self.cfg, risk_scale, self.link_charge, self.self_term).solve(self.levels, ls)

    def stat(self, sb, key):
        return torch.as_tensor(np.stack([self.st[i][key] for i in sb.sid]), dtype=torch.float32, device=sb.dev)

    def act(self, sb):
        with torch.no_grad():
            return self.plan(self.ctx(sb))


class TIDE(NovelShare):
    """TIDE - Tracking Idle probability with Dynamic Experts (online, no training).
    Exact expected-utility planner; the busy probability is an online mixture of four
    experts whose weights follow their log-loss on the states the sensor has
    measured so far in the session (exponential weights with fixed share):
      fused forecast | raw forecast | online Markov chain (h-step) | online rate.
    The Markov chain starts from the training transition rates and is updated
    with every sensed transition, so a sub-channel whose behaviour drifts away
    from the training period is followed. Strictly causal for every horizon:
    a prediction made for slot t is scored only when slot t has been sensed."""
    label = "TIDE"
    ref = "proposed: online expert mixture (exponential weights, fixed share) + exact expected-utility planner"
    sequential = True

    def reset(self, B):
        self.s = None

    def _init(self, sb):
        B, C, dev = sb.B, sb.C, sb.dev
        kap = float(self.cfg["tide_prior"])
        w0 = torch.tensor(self.cfg["tide_w0"], dtype=torch.float32, device=dev)
        self.s = {"n0": torch.full((B, C), kap, device=dev), "n1": torch.full((B, C), kap, device=dev),
                  "n01": kap * self.stat(sb, "p01"), "n10": kap * self.stat(sb, "p10"),
                  "prev": None, "pok": None, "lw0": torch.log(w0 / w0.sum()),
                  "logw": torch.log(w0 / w0.sum()).expand(B, -1).clone(), "buf": []}

    def beliefs(self, sb, fused):
        if getattr(self, "s", None) is None or self.s["n0"].shape[0] != sb.B:
            self._init(sb)
        s, cfg = self.s, self.cfg
        ok, y0 = sb.mok[:, None], sb.ym                 # state sensed at forecast time
        if s["prev"] is not None:                       # one sensed 1-step transition per slot
            p, okk = s["prev"], ok * s["pok"]
            s["n0"] += okk * (1 - p); s["n01"] += okk * (1 - p) * y0
            s["n1"] += okk * p; s["n10"] += okk * p * (1 - y0)
        s["prev"], s["pok"] = y0, ok
        if len(s["buf"]) >= self.h:                     # predictions made h windows ago were for the state sensed now
            pe = s["buf"].pop(0)                        # [B,C,E]
            yy = y0[..., None]
            ll = yy * torch.log(pe) + (1 - yy) * torch.log(1 - pe)
            lw = s["logw"] + cfg["tide_eta"] * (ll * ok[..., None]).sum(1)
            lw = lw - lw.logsumexp(1, keepdim=True)
            a = float(cfg["tide_share"])
            s["logw"] = torch.logaddexp(lw + math.log(1 - a), s["lw0"][None] + math.log(a))
        a01, a10 = s["n01"] / s["n0"], s["n10"] / s["n1"]
        pi = a01 / (a01 + a10).clamp(min=1e-6)
        lam = (1 - a01 - a10).clamp(-0.99, 0.99) ** self.h
        mk = torch.where(ok > 0, pi + lam * (y0 - pi), pi)
        pe = torch.stack([fused, sb.phat, mk, pi], -1).clamp(1e-4, 1 - 1e-4)
        s["buf"].append(pe)
        return (pe * torch.softmax(s["logw"], 1)[:, None, :]).sum(-1)

    def act(self, sb):
        with torch.no_grad():
            ctx = self.ctx(sb)
            if not self.oracle_fc:
                ctx.q = self.beliefs(sb, ctx.q).clamp(0, 1)
            return self.plan(ctx)


def profile_novel_policy(a, C, cfg) -> Dict:
    try:
        rng = np.random.default_rng(0)
        G, Gsp = gen_topologies(rng, 1, cfg)
        ph = rng.uniform(0, 1, (1, C)).astype(np.float32)
        row = {"method": a.label, "method_key": a.key, "reference": a.ref, "su_links": cfg["su_links"],
               "subchannels": C, "params": int(sum(p.numel() for n in a.nets.values() for p in n.parameters())),
               "params_fsf_per_scenario": 4 + C}
        row["params_M"] = row["params"] / 1e6
        for devname in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
            dv = torch.device(devname)
            b = copy.copy(a); b.dev = dv
            b.nets = {k: copy.deepcopy(n).to(dv) for k, n in a.nets.items()}
            b.fsf = [copy.deepcopy(f).to(dv) for f in a.fsf] if a.fsf else None
            b.oracle_fc, b.h = False, 1
            b.reset(1)
            sb = SlotBatch(ph, (ph > 0.5).astype(np.float32), np.zeros((1, C), np.float32), G, Gsp,
                           np.full(C, 10.0, np.float32), np.zeros(C, np.float32), dv, ppers=ph)
            attach_memory(sb, (ph > 0.5).astype(np.float32), np.full((1, C), 10.0, np.float32),
                          np.ones(1, np.float32), np.zeros(1, int))
            sync = torch.cuda.synchronize if devname == "cuda" else None
            if devname == "cpu":
                row["flops_per_slot"] = count_flops(lambda: b.act(sb))
                row["mflops_per_slot"] = row["flops_per_slot"] / 1e6
            row[f"{devname}_latency_ms_per_slot"] = _time_call(lambda: b.act(sb), max(3, cfg["prof_iters"] // 5),
                                                              cfg["prof_warmup"], sync) * 1e3
        return row
    except Exception as e:
        log(f"  profiling {a.label} failed ({type(e).__name__}: {e})")
        return {"method": a.label, "method_key": getattr(a, "key", ""), "profiling_error": str(e)}


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


METHODS: Dict[str, type] = {"tide": TIDE}


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


def fit_fsf_all(bd, P, names, cfg, dev, lag: int = 1):
    """One fusion per scenario for forecasts that are `lag` steps ahead: the
    sensed state it sees is `lag` slots older than the target."""
    out = []
    for s in names:
        ym, em, ok = memory_of(bd.D[s]["y_tr1"], bd.D[s]["E_tr1"], lag)
        out.append(fit_fsf(P[s]["tr1"], bd.D[s]["y_tr1"], ym, em, ok, cfg["eta_db"], bd.C, dev))
    return out


# ==========================================================================
#   SOTA SHARING METHODS (identical classes to the SOTA sharing script)
# ==========================================================================


# ==========================================================================
#                                 METHODS
# ==========================================================================


SOTA_METHODS: Dict[str, type] = {}   # baselines: 03_sharing_baselines.py


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
    ap.add_argument("--novel-forecast-dir", help="out_dir of 02_forecasting.py --family gem")
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
    if getattr(a, "novel_forecast_dir", None):
        cfg["novel_forecast_dir"] = a.novel_forecast_dir
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
    bd.novel = None
    if cfg.get("novel_forecast_dir"):
        dn, _ = find_stage2(cfg["novel_forecast_dir"])
        if dn:
            bd.novel = AbBundle({**cfg, "forecast_dir": os.path.dirname(dn) if os.path.basename(dn) == "stage2" else dn})
            if bd.novel.names != bd.names or bd.novel.C != bd.C:
                log("  GEM hand-off has other scenarios / sub-channels -> not used"); bd.novel = None
        else:
            log("  no GEM forecasting hand-off found -> 'best_novel' rows skipped")
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


def run_task_s2(t, cfg, dev, bd, kind="share"):
    """One (K, partition) configuration (or the centralised reference): TIDE is
    evaluated with the forecasts of that configuration -- the best GEM model
    ('best_novel'), the best SOTA forecaster ('best') and, in the main
    settings, the oracle and historical-mean forecasts. The forecast-sensing
    fusion is fitted per forecaster and horizon on the training slots."""
    names, C, edges = bd.names, bd.C, bd.edges
    bdn = getattr(bd, "novel", None)
    pool = gen_topologies(np.random.default_rng(cfg["seed"] + 999), cfg["topo_pool_test"], cfg)
    st_list = [scenario_stats(bd.D[s]["y_tr1"], bd.D[s]["E_tr1"], bd.eta, C, cfg["strength_atoms"]) for s in names]
    ROWS, SNR, COMM, CX, CLI = [], [], [], {}, []
    Ksu = cfg["su_links"]
    pers = bd.preds("persistence", "non-learned", 0)
    hmean = bd.preds("historical_mean", "non-learned", 0)
    U_cache = {}
    for part, K, mkey in task_configs(t, cfg):
        mode = "centralized" if part == "pooled" else "fedavg"
        main = part == "pooled" or (part == "noniid" and K == int(cfg.get("main_clients", 8)))
        cmp_tag = "main" if main else "ablation"
        bs = bd.best(mkey, "SOTA")
        bn = bdn.best(mkey, "novel") if bdn is not None else None
        if bs is None and bn is None:
            log(f"  [{mkey}] no forecaster in the hand-offs (run 02_forecasting.py first) -> skipped")
            continue
        fp = cfg.get("_run_fp", "")
        gR, gS, gC, gL, gX, done = load_progress(cfg, mkey, part, K, fp)    # seeds finished before a restart
        ROWS += gR; SNR += gS; COMM += gC; CLI += gL
        for k, v in gX.items():
            CX.setdefault(k, v)
        for seed in cfg["seeds"]:
            todo = lambda mk, fc, h: (seed, mk, fc, h) not in done
            Ps = bd.preds(bs, mkey, seed) if bs else None
            Pn = bdn.preds(bn, mkey, seed) if bn else None
            fcs = []
            if "best_novel" in cfg["forecasters"] and _valid(Pn, names):
                fcs.append(("best_novel", f"{bdn.label(bn)} ({mkey}, GEM)", "novel", Pn, False))
            if "best" in cfg["forecasters"] and _valid(Ps, names):
                fcs.append(("best", f"{bd.label(bs)} ({mkey})", "SOTA", Ps, False))
            if main and "oracle" in cfg.get("reference_forecasters", []):
                fcs.append(("oracle", "Oracle forecast", "reference",
                            {s: {"te": bd.D[s]["y_te"]} for s in names}, True))
            if main and "historical_mean" in cfg.get("reference_forecasters", []) and hmean is not None:
                fcs.append(("historical_mean", bd.label("historical_mean"), "reference", hmean, False))
            if not fcs:
                log(f"  [{mkey} seed {seed}] forecasts missing -> skipped")
                continue
            hs = [h for h in cfg["horizons"] if h <= bd.H]
            if not any(todo(mk, fc[0], h) for mk in cfg["methods"] for fc in fcs for h in hs):
                log(f"  [{mkey} seed {seed}] everything already done -> skipped")
                continue
            if mode == "fedavg":                       # client composition of this configuration (for the record)
                srcs = [MemSource(s, i, fcs[0][3][s].get("tr1", bd.D[s]["y_tr1"]), bd.D[s]["y_tr1"], bd.D[s]["E_tr1"],
                                  bd.D[s]["inr_busy"], bd.D[s]["inr_idle"], cfg, dev) for i, s in enumerate(names)]
                for src, s in zip(srcs, names):
                    src.t_tr, src.radio = bd.D[s]["t_tr1"], bd.D[s]["radio"]
                    src.ypos, src.ncell = bd.D[s]["y_tr1"].sum(1), float(bd.C)
                CLI += shard_rows(shard_sources(srcs, part, K, seed, cfg), names, part, K, seed)
            ms = {}
            for mk in cfg["methods"]:
                a = make_method(mk, cfg, dev, C, st_list, None, mode)
                ms[mk] = (a, {"train_s": 0.0, "comm_bytes": 0})
            fsf_cache = {}
            for fc, flab, ffam, P, is_or in fcs:
                for h in hs:
                    if not any(todo(mk, fc, h) for mk in ms):
                        continue
                    if not is_or and (fc, h) not in fsf_cache:
                        if _valid(P, names):
                            fsf_cache[(fc, h)] = fit_fsf_all(bd, P, names, cfg, dev, lag=h)
                        else:                           # no training-slot forecasts (historical mean):
                            fl = next((f for f in fcs if not f[4] and _valid(f[3], names)), None)   # fusion of
                            fsf_cache[(fc, h)] = (fit_fsf_all(bd, fl[3], names, cfg, dev, lag=h)  # the main forecaster
                                                  if fl else None)
                    blk = test_blocks_mem(bd, lambda s, P=P: P[s]["te"], (lambda s: pers[s]["te"]) if pers else None,
                                          h, pool, cfg)
                    L = blk["PH"].shape[1]
                    if h not in U_cache:
                        U_cache[h] = oracle_utility(blk, cfg, dev)      # truth only: same for every configuration
                    U_or = U_cache[h]
                    for mk, (a, info) in ms.items():
                        if not todo(mk, fc, h):
                            continue
                        a.fsf, a.oracle_fc, a.h = (None if is_or else fsf_cache[(fc, h)]), is_or, h
                        acc, sh, sv, sn, lat = eval_method(a, blk, a.cfg, dev, edges, bd.eta)
                        for si, s in enumerate(names + (["Combined"] if len(names) > 1 else [])):
                            mask = np.ones(len(blk["SID"]), bool) if s == "Combined" else blk["SID"] == si
                            if not mask.any():
                                continue
                            met = share_row(acc, mask, L, Ksu)
                            r = {"method": a.label, "method_key": mk, "family": method_family(mk),
                                 "comparison": cmp_tag, "reference": a.ref, "trained": "non-learned", "mode": mode,
                                 "partition": part, "n_clients": K,
                                 "forecaster": flab, "forecaster_key": fc, "forecaster_family": ffam,
                                 "forecaster_config": mkey, "seed": seed,
                                 "scenario": s, "horizon_step": h, **met,
                                 "decision_latency_ms_per_slot": lat * 1e3, "train_s": 0.0, "comm_MB": 0.0}
                            uo = float(U_or[mask].sum()) / (mask.sum() * L)
                            r["oracle_utility_per_slot"] = uo
                            r["regret_vs_oracle"] = 1 - r["utility_per_slot"] / uo if uo > 0 else float("nan")
                            ROWS.append(r)
                            bk = {k: r[k] for k in ("method", "method_key", "family", "trained", "mode", "partition",
                                                    "n_clients", "forecaster_key", "forecaster_config", "seed",
                                                    "scenario", "horizon_step")}
                            SNR.append({**bk, "snr_bin": f"<= {bd.eta:g} (idle)",
                                        "n_cells": float(acc["n_idle"][mask].sum()), "idle_capture": r["idle_capture"]})
                            for i in range(len(edges) - 1):
                                n_ = float(sn[mask, i].sum())
                                SNR.append({**bk, "snr_bin": f"{edges[i]:g}-{edges[i + 1]:g}", "n_cells": n_,
                                            "pu_hit_rate": float(sh[mask, i].sum()) / n_ if n_ else float("nan"),
                                            "pu_violation_rate": float(sv[mask, i].sum()) / n_ if n_ else float("nan")})
                    log(f"  evaluated TIDE | {flab} | h={h} | seed {seed}")
            for mk, (a, _) in ms.items():
                if mk not in CX:
                    key0 = next(((fc, h) for (fc, h) in fsf_cache if h == 1), None)
                    a.fsf, a.oracle_fc, a.h = (fsf_cache[key0] if key0 else None), False, 1
                    CX[mk] = {**profile_novel_policy(a, C, cfg), "family": method_family(mk)}
            save_progress(cfg, mkey, part, K, fp, ROWS, SNR, COMM, CLI, CX)
    if not ROWS:
        log("  nothing evaluated in this part")
        return None
    return {"ROWS": ROWS, "SNR": SNR, "COMM": COMM, "CX": CX, "CLI": CLI}


AB_KEYS = ["method", "method_key", "family", "trained", "mode", "partition", "n_clients",
           "forecaster_key", "scenario", "horizon_step"]
AB_SNR_KEYS = AB_KEYS + ["snr_bin"]


if __name__ == "__main__":
    ab_main("Stage 3: TIDE forecast-driven spectrum sharing (proposed)", "share", lambda t, cfg, dev, bd: run_task_s2(t, cfg, dev, bd, "share"),
            AB_KEYS, AB_SNR_KEYS, None)
