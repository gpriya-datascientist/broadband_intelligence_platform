"""
Broadband Intelligence Platform — Full Pipeline Runner
======================================================
Runs the complete ML pipeline in order:
  1. Generate synthetic data
  2. P1 — prepare + train
  3. P2 — prepare + train
  4. P3 — prepare + train

Usage:
    python run_pipeline.py                    # full run
    python run_pipeline.py --steps generate   # only data generation
    python run_pipeline.py --steps p1 p2 p3   # skip data gen (data exists)
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.shared.logger import get_logger

log = get_logger("pipeline", log_file="data/reports/pipeline.log")


def run_generate():
    log.info("━" * 50)
    log.info("STAGE: Data Generation")
    log.info("━" * 50)
    from src.shared.generate_data import main
    main()


def run_p1():
    log.info("━" * 50)
    log.info("STAGE: P1 — HFC Anomaly  (prepare → train)")
    log.info("━" * 50)
    from src.p1_hfc_anomaly.prepare_data import run_pipeline as p1_prep
    from src.p1_hfc_anomaly.train_model  import train       as p1_train
    p1_prep(sample_rows=200_000)
    p1_train()


def run_p2():
    log.info("━" * 50)
    log.info("STAGE: P2 — Wi-Fi Experience  (prepare → train)")
    log.info("━" * 50)
    from src.p2_wifi_anomaly.prepare_data import run_pipeline as p2_prep
    from src.p2_wifi_anomaly.train_model  import train        as p2_train
    sample_cids = [f"CUS-{i:04d}" for i in range(20)]
    p2_prep(customer_ids=sample_cids)
    p2_train()


def run_p3():
    log.info("━" * 50)
    log.info("STAGE: P3 — Churn Prediction  (prepare → train)")
    log.info("━" * 50)
    from src.p3_churn.prepare_data import run_pipeline as p3_prep
    from src.p3_churn.train_model  import train        as p3_train
    p3_prep()
    p3_train()


def main():
    parser = argparse.ArgumentParser(description="Broadband Intelligence Pipeline")
    parser.add_argument(
        "--steps",
        nargs="*",
        choices=["generate", "p1", "p2", "p3"],
        default=["generate", "p1", "p2", "p3"],
        help="Which stages to run (default: all)",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Broadband Intelligence Platform — Pipeline Run")
    log.info("Steps: %s", args.steps)
    log.info("=" * 60)

    t0 = time.time()

    stage_map = {
        "generate": run_generate,
        "p1":       run_p1,
        "p2":       run_p2,
        "p3":       run_p3,
    }

    for step in args.steps:
        t_step = time.time()
        stage_map[step]()
        log.info("  ✓ %s completed in %.1fs", step, time.time() - t_step)

    log.info("=" * 60)
    log.info("Pipeline complete — total time: %.1fs", time.time() - t0)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
