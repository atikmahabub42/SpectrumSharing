#!/usr/bin/env python3
"""
smoke_test.py -- end-to-end check of the code on a tiny SYNTHETIC dataset
=========================================================================
Generates a synthetic export file (tests/make_synthetic_export.py) and runs
all four stages with --quick and a reduced grid. Takes a few minutes on a
laptop CPU. It only checks that the code runs; the numbers mean nothing.

usage:  python tests/smoke_test.py [WORK_DIR]      (default ./smoke_test_run)
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
work = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "smoke_test_run")
data, res = os.path.join(work, "data"), os.path.join(work, "results")


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run([sys.executable] + cmd, check=True)


run([os.path.join(HERE, "make_synthetic_export.py"), data])
run([os.path.join(ROOT, "run_pipeline.py"), "--data", data, "--results", res, "--quick",
     "--sensing-args", "--frame-len 4096 --models oscar_net_p1,deepsense_s4 --counts 4 --partitions natural,iid",
     "--forecasting-args", "--clients 4,8 --partitions noniid,iid --algos fedavg"])
need = [os.path.join(res, "01_sensing", "sensing_handoff", "manifest.json"),
        os.path.join(res, "02_forecasting", "stage2", "manifest.json"),
        os.path.join(res, "02b_forecasting_gem", "stage2", "manifest.json"),
        os.path.join(res, "03_sharing_baselines", "share_overall_agg.csv"),
        os.path.join(res, "03b_sharing_tide", "share_overall_agg.csv")]
miss = [p for p in need if not os.path.isfile(p)]
if miss:
    sys.exit("SMOKE TEST FAILED, missing:\n  " + "\n  ".join(miss))
print("\nSMOKE TEST PASSED: every stage ran and wrote its hand-off / results.")
