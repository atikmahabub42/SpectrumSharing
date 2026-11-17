#!/usr/bin/env python3
"""
make_synthetic_export.py -- tiny SYNTHETIC stand-in for the Fireworks dataset
=============================================================================
--layout dataport (default) mimics the IEEE DataPort release after its
converter has run:
  session_meta.json, sweeps_<key>.csv, psd_<key>.f16     PSD tier (stage 2)
  derived/frames.h5  {iq [N,L,2] int16, sample_rate, band_name, center_hz,
                      utc, full_scale}                    raw-I/Q tier (stage 1)
--layout export writes one export .h5 instead:
  attrs radio, /iq_frames/{...}, /grid/<band>/{psd_db, freq_hz, utc}
The signals are random bursts in Gaussian noise. They exist only so that the
whole pipeline can be smoke-tested in minutes; they say NOTHING about the
real data. Never report numbers obtained from this file.

usage:  python tests/make_synthetic_export.py [OUT_DIR] [--seed 7]
"""
import argparse
import os

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default=os.path.join("data", "synthetic_export"))
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--layout", choices=["dataport", "export"], default="dataport")
ap.add_argument("--frame-len", type=int, default=4096)
ap.add_argument("--recordings", type=int, default=16, help="I/Q recordings per band")
ap.add_argument("--frames", type=int, default=10, help="frames per recording")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
os.makedirs(a.out, exist_ok=True)

FS, L, NSUB, NB = 25e6, a.frame_len, 32, 256
T0, DUR = 1.7e9, 51 * 60.0
# band: (f_lo, f_hi, sweeps, busy rate, persistence, has raw I/Q)
BANDS = {"ISM_2400": (2400e6, 2483.5e6, 171, 0.25, 0.3, True),
         "UNII_1": (5150e6, 5250e6, 171, 0.40, 0.2, True),
         "UNII_2A_DFS": (5250e6, 5350e6, 86, 0.30, 0.4, False),
         "UNII_3_5800": (5725e6, 5850e6, 171, 0.45, 0.5, True),
         "UNII_5_6E_EDGE": (5925e6, 6025e6, 86, 0.20, 0.4, False)}


def occupancy(T, rate, lam):
    """Two-state Markov chain per sub-channel with a per-channel raster."""
    off = rng.normal(0, 1.0, NSUB)
    pi = 1 / (1 + np.exp(-(np.log(rate / (1 - rate)) + off)))
    on = np.zeros((T, NSUB), bool)
    s = rng.random(NSUB) < pi
    for k in range(T):
        u = rng.random(NSUB)
        s = np.where(s, u > (1 - pi) * (1 - lam), u < pi * (1 - lam))
        on[k] = s
    return on


def frame(busy):
    """Complex noise + bursts; vacant frames sometimes carry a short strong burst."""
    x = (rng.normal(0, 1, L) + 1j * rng.normal(0, 1, L)) / np.sqrt(2)
    win = L // 32
    nw = L // win
    if busy:
        k = rng.choice(nw, max(1, int(nw * rng.uniform(0.15, 0.6))), replace=False)
    elif rng.random() < 0.2:
        k = rng.choice(nw, 1, replace=False)          # 1 window < 10 %: a hard negative
    else:
        k = []
    amp = 10 ** (rng.uniform(8, 25) / 20)
    for j in k:
        x[j * win:(j + 1) * win] += amp * np.exp(1j * rng.uniform(0, 2 * np.pi, win))
    x *= 300.0
    return np.clip(np.stack([x.real, x.imag], -1), -32767, 32767).astype(np.int16)


iq, sr, bn, cf, ut = [], [], [], [], []
grids = {}
for b, (lo, hi, T, rate, lam, has_iq) in BANDS.items():
    utc = T0 + np.linspace(0, DUR, T)
    fr = np.linspace(lo, hi, NB)
    on = occupancy(T, rate, lam)
    snr = rng.uniform(6.5, 25, (T, NSUB))
    idx = (np.arange(NB) * NSUB) // NB
    P = -50 + rng.normal(0, 1.0, (T, NB)) + np.where(on[:, idx], snr[:, idx], 0)
    grids[b] = (utc, fr, P, on, idx)
    if not has_iq:
        continue
    tiles = np.arange(lo + 10e6, hi - 9e6, 20e6)
    for r in range(a.recordings):
        k = int((r + 0.5) * T / a.recordings)
        c = float(rng.choice(tiles))
        subs = np.unique(idx[(fr >= c - 10e6) & (fr <= c + 10e6)])
        busy = bool(on[k, subs].mean() > 0.3)
        for j in range(a.frames):
            iq.append(frame(busy if rng.random() > 0.1 else not busy))
            sr.append(FS); bn.append(b); cf.append(c); ut.append(utc[k] + 0.2 + j * L / FS)

if a.layout == "export":
    path = os.path.join(a.out, "synthetic_fireworks.h5")
    with h5py.File(path, "w") as f:
        f.attrs["radio"] = "USRP B205mini-i"
        f.attrs["synthetic"] = 1
        g = f.create_group("grid")
        for b, (utc, fr, P, on, idx) in grids.items():
            gb = g.create_group(b)
            gb["psd_db"] = P.astype(np.float32); gb["freq_hz"] = fr; gb["utc"] = utc
        q = f.create_group("iq_frames")
        q.attrs["frame_len"] = L
        q["iq"] = np.stack(iq); q["sample_rate"] = np.array(sr)
        q["band_name"] = np.array([x.encode() for x in bn]); q["center_hz"] = np.array(cf)
        q["utc"] = np.array(ut); q["full_scale"] = np.full(len(iq), 32767.0)
else:
    import csv
    import json
    BINS, KEEP = 1024, 0.8
    json.dump({"config": {"psd_bins": BINS, "keep_frac": KEEP}, "radio": "USRP B205mini-i", "synthetic": True},
              open(os.path.join(a.out, "session_meta.json"), "w"), indent=1)
    rows, psd = [], []
    for b, (utc, fr, P, on, idx) in grids.items():
        lo, hi = BANDS[b][0], BANDS[b][1]
        for k in range(len(utc)):
            for ti, c in enumerate(np.arange(lo + 10e6, hi - 9e6, 20e6)):
                bf = c + (np.arange(BINS) - BINS / 2.0) * (FS / BINS)
                row = np.interp(bf, fr, P[k], left=-50.0, right=-50.0) + rng.normal(0, 0.3, BINS)
                rows.append({"sweep": k, "band_name": b, "utc_start": utc[k] + 0.05 * ti, "center_hz": c,
                             "samp_rate_hz": FS, "f_lo_hz": c - KEEP * FS / 2, "f_hi_hz": c + KEEP * FS / 2,
                             "psd_row": len(psd), "simulated": 0})
                psd.append(row.astype(np.float16))
    with open(os.path.join(a.out, "sweeps_wifi0_SYNTH.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.stack(psd).astype(np.float16).tofile(os.path.join(a.out, "psd_wifi0_SYNTH.f16"))
    os.makedirs(os.path.join(a.out, "derived"), exist_ok=True)
    path = os.path.join(a.out, "derived", "frames.h5")
    with h5py.File(path, "w") as f:                      # layout of code/sigmf_to_dataset.py
        f.attrs["frame_len"] = L
        f.attrs["synthetic"] = 1
        f["iq"] = np.stack(iq); f["sample_rate"] = np.array(sr); f["center_hz"] = np.array(cf)
        f["utc"] = np.array(ut); f["full_scale"] = np.full(len(iq), 32767.0)
        f.create_dataset("band_name", data=np.array(bn, dtype=object), dtype=h5py.string_dtype())
print(f"wrote {a.out} ({a.layout} layout): {len(iq)} I/Q frames of {L} samples, "
      f"{len(BANDS)} PSD bands -- SYNTHETIC, for testing only")
