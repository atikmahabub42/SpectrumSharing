#!/usr/bin/env python3
"""
run_pipeline.py -- run the sense -> forecast -> share pipeline in order
=======================================================================
  stage 1  pipeline/01_sensing.py            OSCAR-Net + 5 SOTA detectors
  stage 2  pipeline/02_forecasting.py        GEM / GEM-DCS + 5 SOTA forecasters
  stage 3  pipeline/03_sharing_baselines.py  5 SOTA sharing methods
           pipeline/03b_sharing_tide.py      TIDE (proposed)

usage
  python run_pipeline.py --data DATA_DIR --results RESULTS_DIR
  python run_pipeline.py --data DATA_DIR --results RESULTS_DIR --stages forecasting,sharing,tide
  python run_pipeline.py ... --quick          # few epochs / rounds (smoke test)
Extra options can be passed to one stage, e.g.
  --sensing-args "--models oscar_net_p1,deepsense_s4"
Every stage resumes from its checkpoints, so the same command can be rerun
after an interruption.
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda name: os.path.join(HERE, "pipeline", name)          # noqa: E731
STAGES = ("sensing", "forecasting", "sharing", "tide")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("SFS_DATA", os.path.join("data", "fireworks_export")),
                    help="folder with the export .h5 file(s) of the dataset")
    ap.add_argument("--results", default=os.environ.get("SFS_RESULTS", "results"),
                    help="pipeline results root (one sub-folder per stage)")
    ap.add_argument("--stages", default=",".join(STAGES), help=f"subset of {','.join(STAGES)}")
    ap.add_argument("--quick", action="store_true", help="few epochs / rounds: smoke test only")
    ap.add_argument("--device", help="cpu | cuda (default: auto)")
    for s in STAGES:
        ap.add_argument(f"--{s}-args", default="", help=f"extra options for the {s} stage")
    a = ap.parse_args()
    data, res = os.path.abspath(a.data), os.path.abspath(a.results)
    os.makedirs(res, exist_ok=True)
    env = dict(os.environ, SFS_DATA=data, SFS_RESULTS=res)
    q = ["--quick"] if a.quick else []
    dv = ["--device", a.device] if a.device else []
    cmds = {
        "sensing": [P("01_sensing.py"), "--root", data, "--out", os.path.join(res, "01_sensing")] + q,
        "forecasting": [P("02_forecasting.py"), "--root", data, "--results", res] + q + dv,
        "sharing": [P("03_sharing_baselines.py"), "--forecast-dir", os.path.join(res, "02_forecasting"),
                    "--out", os.path.join(res, "03_sharing_baselines")] + q + dv,
        "tide": [P("03b_sharing_tide.py"), "--forecast-dir", os.path.join(res, "02_forecasting"),
                 "--novel-forecast-dir", os.path.join(res, "02b_forecasting_gem"),
                 "--out", os.path.join(res, "03b_sharing_tide")] + q + dv,
    }
    want = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in want if s not in STAGES]
    if bad:
        sys.exit(f"unknown stage(s) {bad}; choose from {STAGES}")
    for s in STAGES:
        if s not in want:
            continue
        cmd = [sys.executable] + cmds[s] + shlex.split(getattr(a, f"{s}_args"))
        print(f"\n{'=' * 78}\n[{s}] {' '.join(cmd)}\n{'=' * 78}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, env=env)
        if r.returncode:
            sys.exit(f"[{s}] failed with exit code {r.returncode}")
        print(f"[{s}] done in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"\nAll requested stages finished. Results under {res}")


if __name__ == "__main__":
    main()
