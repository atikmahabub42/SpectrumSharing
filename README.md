# Sense, Forecast, Share

**A lightweight federated spectrum-sharing framework on a real-world mass-gathering RF dataset**

This repository contains the code of the paper *"Sense, Forecast, Share: A Lightweight Federated Spectrum-Sharing Framework on a Real-World Mass-Gathering RF Dataset"* (submitted to IEEE DySPAN 2027). It implements a three-stage dynamic spectrum-sharing chain, trained centrally and with federated learning (FedAvg, IID and non-IID clients), on the public Fireworks multi-band RF dataset ([IEEE DataPort, DOI 10.21227/f2rv-ms26](https://ieee-dataport.org/documents/real-world-multi-band-rf-dataset-mass-gathering-event-spectrum-analysis-framework)).

| Stage | Task | Proposed model | Baselines included |
|---|---|---|---|
| 1 | Spectrum **sensing** on raw I/Q frames | **OSCAR-Net**: differentiable order-statistic CFAR with a learned noise floor and a soft duty-cycle spectrum (20.6k parameters, gain invariant) | DeepSense, T-PRIME, SpecFormer, CNN-Former, DnFormer |
| 2 | Sub-channel occupancy **forecasting** | **GEM**: gated mixture of nine analytic occupancy experts (2.7k parameters); **GEM-DCS**: GEM with a streaming drift-compensation offset | N-HiTS, PatchTST, iTransformer, Bi-ConvLSTM, SpectrumFM |
| 3 | Primary/secondary spectrum **sharing** | **TIDE**: training-free online expert beliefs + PU-strength atoms + exact expected-utility planner | LBT-sense, BLAST auction, DQN-DSS, AdapShare, MA-IDQN |

```
 raw I/Q frames ──► [1] 01_sensing.py ──► sensing_handoff/ (OSCAR-Net decisions)
                                                  │  calibrates the per-band occupancy threshold
 PSD sweeps ──────────────────────────► [2] 02_forecasting.py ──► stage2/ (labels + forecasts)
                                                                      │
                        [3] 03_sharing_baselines.py   (SOTA methods) ◄┤
                        [3] 03b_sharing_tide.py       (TIDE)         ◄┘
```

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Installation](#2-installation)
3. [Dataset preparation](#3-dataset-preparation)
4. [Quick check: smoke test on synthetic data](#4-quick-check-smoke-test-on-synthetic-data)
5. [Running the full pipeline](#5-running-the-full-pipeline)
6. [Stage 1: sensing (`01_sensing.py`)](#6-stage-1-sensing-01_sensingpy)
7. [Stage 2: forecasting (`02_forecasting.py`)](#7-stage-2-forecasting-02_forecastingpy)
8. [Stage 3: sharing (`03_sharing_baselines.py`, `03b_sharing_tide.py`)](#8-stage-3-sharing)
9. [Federated learning settings](#9-federated-learning-settings)
10. [Resuming, parallel runs and clusters](#10-resuming-parallel-runs-and-clusters)
11. [Configuration and reproducibility](#11-configuration-and-reproducibility)
12. [Troubleshooting](#12-troubleshooting)
13. [Citation, license and acknowledgment](#13-citation-license-and-acknowledgment)

---

## 1. Repository layout

```
.
├── run_pipeline.py                 # runs the four scripts below in order
├── pipeline/
│   ├── 01_sensing.py               # stage 1: OSCAR-Net + 5 SOTA detectors, centralized + FedAvg
│   ├── 02_forecasting.py           # stage 2: GEM / GEM-DCS + 5 SOTA forecasters
│   ├── 03_sharing_baselines.py     # stage 3: 5 SOTA sharing methods
│   └── 03b_sharing_tide.py         # stage 3: TIDE (proposed)
├── tests/
│   ├── make_synthetic_export.py    # tiny synthetic stand-in for the dataset (testing only)
│   └── smoke_test.py               # end-to-end check of the code in a few minutes
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

Every script is a single self-contained file: it runs with no arguments (defaults reproduce the paper's settings), seeds every random generator deterministically and resumes from its own checkpoints.

## 2. Installation

Python 3.9 or newer. A CUDA GPU is recommended for stage 1; stages 2 and 3 run comfortably on a CPU.

```bash
git clone https://github.com/<your-account>/sense-forecast-share.git
cd sense-forecast-share

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# PyTorch first, matching your CUDA version (see https://pytorch.org/get-started/locally/), e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CPU only:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

Dependencies: `torch`, `numpy`, `h5py`, `scikit-learn`.

## 3. Dataset preparation

1. **Download** the release from [IEEE DataPort](https://ieee-dataport.org/documents/real-world-multi-band-rf-dataset-mass-gathering-event-spectrum-analysis-framework) (CC BY 4.0) and unpack it, e.g. to `data/fireworks_rf_dataset/`:

   ```
   data/fireworks_rf_dataset/
   ├── session_meta.json
   ├── sweeps_wifi0_<serial>.csv      # per-dwell index of the PSD tier
   ├── psd_wifi0_<serial>.f16         # float16 PSD rows
   ├── iq/<name>.sigmf-data / .sigmf-meta   # raw I/Q recordings (ISM 2.4 GHz, U-NII-1, U-NII-3)
   └── code/sigmf_to_dataset.py       # converter shipped with the dataset
   ```

2. **Convert the raw I/Q recordings into frames** with the converter shipped with the dataset. Keep **every** frame (`--keep-ambiguous`): the pipeline relabels each frame itself with the paper's rule (noise floor = 10th percentile, η = 6 dB, β₀ = 0.10), so frames must not be pre-filtered.

   ```bash
   python data/fireworks_rf_dataset/code/sigmf_to_dataset.py \
          --root data/fireworks_rf_dataset \
          --frame-len 32768 --crop all --keep-ambiguous --no-spectrogram
   ```

   This writes `data/fireworks_rf_dataset/derived/frames.h5` (about 17 GB of int16 I/Q).

3. **Point the pipeline at the folder**, either per command (`--data` / `--root`) or once through environment variables:

   ```bash
   export SFS_DATA=$PWD/data/fireworks_rf_dataset     # Windows: set SFS_DATA=...
   export SFS_RESULTS=$PWD/results
   ```

Where each stage reads its input:

| Stage | Reads | Found automatically under the data folder |
|---|---|---|
| 1 | raw-I/Q frames | `derived/*.h5` (converter output), `*.h5`, or `*/export/*.h5` |
| 2 | PSD tier | the session folder itself (`session_meta.json` + `sweeps_*.csv` + `psd_*.f16`), or an export `.h5` with `/grid/<band>` or `/sweeps` + `/psd` |
| 2 | sensing hand-off | `<results>/01_sensing/sensing_handoff/` |
| 3 | forecasts | `<results>/02_forecasting/stage2/` and `<results>/02b_forecasting_gem/stage2/` |

Disk: plan for about 35 GB free (converter output plus the stage-1 frame cache, both float16/int16 memory-mapped; RAM use stays small).

## 4. Quick check: smoke test on synthetic data

Before the full run, check that everything works on your machine:

```bash
python tests/smoke_test.py            # a few minutes on a laptop CPU
```

It creates a tiny **synthetic** dataset in the DataPort layout (`tests/make_synthetic_export.py`) and runs all four scripts with `--quick` and a reduced grid. It only tests the code; never use its numbers.

## 5. Running the full pipeline

All stages in order, with the paper's settings:

```bash
python run_pipeline.py --data data/fireworks_rf_dataset --results results
```

Options of `run_pipeline.py`:

| Option | Meaning |
|---|---|
| `--data DIR` | dataset folder (default `$SFS_DATA` or `./data/fireworks_export`) |
| `--results DIR` | results root, one sub-folder per stage (default `$SFS_RESULTS` or `./results`) |
| `--stages LIST` | subset of `sensing,forecasting,sharing,tide` |
| `--quick` | few epochs / rounds (smoke test only) |
| `--device cpu\|cuda` | device for stages 2 and 3 (default: automatic) |
| `--sensing-args "..."`, `--forecasting-args "..."`, `--sharing-args "..."`, `--tide-args "..."` | extra options passed to one stage |

Example: only OSCAR-Net and DeepSense in stage 1, then everything else:

```bash
python run_pipeline.py --data data/fireworks_rf_dataset --results results \
       --sensing-args "--models oscar_net_p1,deepsense_s4"
```

The same can be done stage by stage, as described next.

## 6. Stage 1: sensing (`01_sensing.py`)

```bash
python pipeline/01_sensing.py --root data/fireworks_rf_dataset --out results/01_sensing
```

**What it does**

* **Labels** (paper Sec. III-B): each frame of 32,768 samples (1.31 ms at 25 MS/s) is cut into 20-µs windows; noise floor N₀ = 10th percentile of the window powers; a frame is occupied when at least 10 % of its windows exceed N₀ + 6 dB. The rule is gain invariant. The SNR proxy of a frame is Q₀.₉₉ − N₀.
* **Split**: chronological 65 / 15 / 20 % inside 32 time blocks per band (no temporal leakage).
* **Training**: AdamW (lr 1e-3, weight decay 5e-4), batch 128, label smoothing 0.05, early stopping, up to 100 epochs; FedAvg with 8 non-IID temporal clients, half of them per round, 2 local epochs, up to 100 rounds, GroupNorm in every federated model.
* **Operating point**: the validation threshold whose false-alarm rate is closest below 10 %.
* **Complexity**: parameters, FLOPs and latency, always profiled on the CPU.
* **Hand-off**: OSCAR-Net (FedAvg) is run over every frame; its decisions are written to `results/01_sensing/sensing_handoff/` for stage 2.

**Stages** (`--stages`, default `main,clients,handoff`)

| Stage | Content |
|---|---|
| `main` | every model × {centralized, federated} |
| `clients` | federated client-count sweep K ∈ {4, 8, 12, 16, 32} × {non-IID temporal, IID} |
| `handoff` | sensing → forecasting hand-off |
| `lobo`, `labels` | optional audits (leave-one-band-out, label-scheme comparison); not used in the paper |

**Model keys** (`--models`, `--ablation-models`)

| Key | Model |
|---|---|
| `oscar_net_p1` | OSCAR-Net (proposed) |
| `deepsense_s4` | DeepSense (IEEE INFOCOM 2021) |
| `tprime_s9` | T-PRIME (IEEE INFOCOM 2024) |
| `spectrum_transformer_s5` | SpecFormer (IEEE TWC 2024) |
| `cnn_transformer_s6` | CNN-Former (IEEE WCL 2025) |
| `denoise_transformer_s7` | DnFormer (IEEE COMML 2025) |

**Main options**

| Option | Meaning |
|---|---|
| `--root DIR` / `--out DIR` | data folder / stage-1 results folder |
| `--stages LIST` | see above |
| `--models LIST` | detectors of the main table |
| `--ablation-models LIST` | detectors of the client sweep (`''` = all) |
| `--counts 4,8,12,16,32` | client counts of the sweep |
| `--partitions natural,iid` | `natural` = contiguous time blocks (non-IID), `iid` = random shares |
| `--seeds S1,S2,...` | repeat the main table over several training seeds |
| `--frame-len N` | model input length (default 32768) |
| `--cache DIR` | location of the frame cache (default `<out>/_frame_cache`) |
| `--max-frames N` | frames per file (0 = all), e.g. for a fast trial |
| `--quick` | 5 epochs / 5 rounds |

## 7. Stage 2: forecasting (`02_forecasting.py`)

```bash
python pipeline/02_forecasting.py --root data/fireworks_rf_dataset --results results
```

**What it does**

* **Sub-channels and labels** (paper Sec. IV): each band is divided into C = 32 sub-channels; the excess of a sub-channel over its noise floor (10th percentile of the training span) is thresholded at a per-band η_b. η_b is **calibrated to OSCAR-Net**: the value in [0, 20] dB that agrees best (balanced accuracy) with the detector's decisions on co-located I/Q recordings of the training span. Bands without I/Q recordings keep η = 6 dB. `--input psd` uses a fixed 6 dB everywhere and needs no stage 1.
* **Windows**: the last W = 6 sweeps predict the next H = 2 sweeps; chronological 65 / 15 / 20 % split.
* **Training**: 40 epochs (centralized) or 40 FedAvg rounds of 2 local epochs; three seeds (0, 1, 2). Per-band parameters (prefix `loc_`) are personal and never averaged.
* **GEM-DCS**: the drift-compensation step γ_b is selected per band on validation from {0, 0.01, 0.03, 0.1, 0.3}.
* **Hand-off**: forecasts of the best model of every configuration (by validation Brier score) are written to `stage2/` for stage 3.

**Model families** (`--family`)

| Family | Models | Output folder |
|---|---|---|
| `gem` | GEM, GEM-DCS (`gem`, `gem_dcs`) | `results/02b_forecasting_gem/` |
| `sota` | N-HiTS, PatchTST, iTransformer, Bi-ConvLSTM, SpectrumFM (`nhits`, `patchtst`, `itransformer`, `biconvlstm_s2s`, `spectrumfm_f`) | `results/02_forecasting/` |
| `all` (default) | both, one after the other | both folders |

Persistence, seasonal-naive and historical-mean baselines are always included.

**Main options**

| Option | Meaning |
|---|---|
| `--root DIR` | data folder (PSD tier) |
| `--results DIR` | results root (reads `01_sensing/`, writes the two folders above) |
| `--family all\|gem\|sota` | model family |
| `--input sensing\|psd` | labels calibrated to stage 1 (default) or fixed 6 dB |
| `--clients 4,8,12,16,24,32` | client counts of the federated grid |
| `--partitions noniid,iid` | partitions of the grid |
| `--algos fedavg,fedprox` | federated algorithms of the grid |
| `--seeds 0,1,2` | seeds |
| `--bands ISM_2400,UNII_1` | subset of bands |
| `--list-tasks`, `--task i`, `--merge` | split the grid into parts (Section 10) |
| `--quick` | 2 epochs / 2 rounds, one seed |

## 8. Stage 3: sharing

Both scripts share the same model of the problem (paper Sec. VII):

* **Measured**: PU occupancy and PU power of every test sweep of every band.
* **Simulated**: only the secondary network: K = 48 SU links of 2–8 m in a 15 m × 15 m area, path-loss exponent 3, 4 dB shadowing, 40 dB reference SNR, and the gain of each SU towards the PU receiver.
* **PU protection**: on a busy sub-channel the PU SINR S_pu / (1 + I_pu) must stay above γ = 3 dB; every violated sub-channel costs μ = 10 bit/s/Hz.
* **Utility** per slot: Σ log₂(1 + SINR_k) − μ · (number of violated sub-channels).
* **Grid**: centralized forecasts plus every federated configuration of stage 2 (K ∈ {4, 8, 12, 16, 24, 32} × {non-IID, IID}); three seeds.

**Baselines** (`03_sharing_baselines.py`)

```bash
python pipeline/03_sharing_baselines.py --forecast-dir results/02_forecasting --out results/03_sharing_baselines
```

| Key | Method |
|---|---|
| `lbt_sense` | listen-before-talk on the last sensed slot |
| `auction` | BLAST second-price auction mechanism (no LLM agents, no ledger) |
| `dqn_dss` | DQN dynamic spectrum sharing |
| `adapshare_ppo` | AdapShare (PPO) |
| `ma_idqn` | multi-agent independent DQN |

The learned policies are trained on the training sweeps with the best SOTA forecaster of each configuration, with FedAvg over the same clients as the forecaster. Options: `--sota-methods`, `--sota-methods-ablation`, `--seeds`, `--clients`, `--partitions`, `--list-tasks`, `--task`, `--merge`, `--redo`, `--quick`.

**TIDE** (`03b_sharing_tide.py`)

```bash
python pipeline/03b_sharing_tide.py --forecast-dir results/02_forecasting \
       --novel-forecast-dir results/02b_forecasting_gem --out results/03b_sharing_tide
```

TIDE is evaluated with the forecasts of every configuration: the best GEM model (`best_novel`, the paper's chain), the best SOTA forecaster (`best`) and, in the main settings, the oracle and historical-mean forecasts as references. Its parameters (`tide_w0`, `tide_eta`, `tide_share`, `tide_prior`, `power_levels`, `ls_passes`, `strength_atoms`) are at the top of the script. Nothing is trained; the forecast–sensing fusion and the band statistics are fitted locally and never shared.

## 9. Federated learning settings

| | Sensing | Forecasting | Sharing |
|---|---|---|---|
| Algorithm | FedAvg | FedAvg (FedProx in the grid) | FedAvg (learned baselines only) |
| Main setting | 8 non-IID temporal clients | 8 non-IID temporal clients | same clients as the forecaster |
| Client sweep | K ∈ {4, 8, 12, 16, 32} × {non-IID, IID} | K ∈ {4, 8, 12, 16, 24, 32} × {non-IID, IID} × {FedAvg, FedProx} | K grid of the forecaster |
| Normalization | GroupNorm | GroupNorm | – |
| Never averaged | – | per-band `loc_` parameters | fusion, band statistics |

A **non-IID client** is one contiguous stretch of the evening (all bands), so the clients see different phases of the crowd; adjacent stretches are merged until every client holds both classes. **IID clients** receive uniformly drawn equal shares.

## 10. Resuming, parallel runs and clusters

* **Resume**: every script checkpoints each finished unit (stage 1: `_completed/`; stage 2: `parts_v2/` and a model cache; stage 3: `parts_v2/`). After a crash, reboot or Ctrl+C, rerun the same command: finished work is loaded, not recomputed. A unit is recomputed only when a setting that changes results was changed. `--redo` (stage 3) forces recomputation.
* **Parallel parts** (stages 2 and 3): `--list-tasks` prints the parts of the grid (one per (K, partition) plus the references); `--task i` runs part *i* only; `--merge` rebuilds the merged outputs from the finished parts.

  ```bash
  python pipeline/02_forecasting.py --family gem --list-tasks
  python pipeline/02_forecasting.py --family gem --task 0
  python pipeline/02_forecasting.py --family gem --merge
  ```

* **SLURM** array example for stage 2:

  ```bash
  #SBATCH --array=0-12
  python pipeline/02_forecasting.py --family sota --task $SLURM_ARRAY_TASK_ID --results $SFS_RESULTS
  ```

  Run `--merge` once all array jobs have finished, and start stage 3 afterwards.

## 11. Configuration and reproducibility

* Every setting lives in the `CONFIG` dictionary at the top of each script; command-line options override it. The defaults are the paper's settings.
* All random generators are seeded (`seed` in `CONFIG`); each training unit is reseeded from a stable hash of its name, so a unit gives the same result whether it runs alone or after others. Stage 1 enables deterministic cuDNN / cuBLAS kernels; small differences across GPU models remain possible, so run all units of a comparison on the same hardware.
* Every run writes the configuration it used (`config.json` / `config_used*.json`) and a log next to its outputs.
* The chain is causal: the sensing threshold is calibrated on the training span only, forecasters see only past sweeps, and TIDE scores a forecast only after the target sweep has been sensed.

## 12. Troubleshooting

| Message | Fix |
|---|---|
| `No .h5 found` (stage 1) | run the converter (Section 3) or pass `--root` pointing at the folder that contains `derived/` |
| `NO SENSING HAND-OFF FOUND` (stage 2) | run stage 1 with the `handoff` stage, pass `--results`, or use `--input psd` |
| `no stage2/ hand-off found` (stage 3) | run stage 2 first, or pass `--forecast-dir` / `--novel-forecast-dir` |
| `Not enough disk for the frame cache` | use `--cache` on a larger drive or `--max-frames` |
| CUDA out of memory (stage 1) | lower `batch_size` in `CONFIG` or run `--models` one at a time |
| Windows paths with spaces | quote them: `--root "D:\My Data\fireworks_rf_dataset"` |

## 13. Citation, license and acknowledgment

If you use this code, please cite the paper and the dataset (see also `CITATION.cff`):

```bibtex
@misc{mahabub2027sfs,
  author = {A. Mahabub and T. Djerafi and S. Vakili},
  title  = {Sense, Forecast, Share: A Lightweight Federated Spectrum-Sharing Framework
            on a Real-World Mass-Gathering {RF} Dataset},
  note   = {Submitted to IEEE DySPAN 2027},
  year   = {2026}
}

@misc{mahabub2026dataset,
  author       = {A. Mahabub and S. Vakili},
  title        = {A Real-World Multi-Band {RF} Dataset from a Mass Gathering Event:
                  Spectrum Analysis Framework},
  howpublished = {IEEE DataPort},
  year         = {2026},
  doi          = {10.21227/f2rv-ms26}
}
```

The SOTA baselines are re-implementations for this dataset; please also cite their original papers when you use them.

**License**: MIT (see `LICENSE`). The dataset is distributed separately under CC BY 4.0.

**Acknowledgment**: Supported by the Natural Sciences and Engineering Research Council of Canada (NSERC), grant RGPIN-2024-06358.
