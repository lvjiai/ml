"""Run the whole experiment pipeline end to end, from a clean checkout.

    python scripts/run_all.py            # everything (W1-W5 + summary)
    python scripts/run_all.py --fast     # smoke test: W1 + W2 on FD001 only

Every step is a separate process, so a failure stops the chain with the step's
name and its own exit code instead of a half-written artifact.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def steps(fast: bool) -> list[tuple[str, list[str]]]:
    py = sys.executable
    if fast:
        return [
            ("W1 data + EDA + baselines", [py, "scripts/w1_data_baseline.py"]),
            ("W2 features + GBDT (FD001)", [py, "scripts/w2_features_gbdt.py", "--subsets", "FD001"]),
        ]
    return [
        ("W1 data + EDA + baselines", [py, "scripts/w1_data_baseline.py"]),
        ("W2 features + GBDT", [py, "scripts/w2_features_gbdt.py"]),
        ("W3 LSTM", [py, "scripts/w3_lstm.py"]),
        ("W4 conformal intervals", [py, "scripts/w4_intervals.py"]),
        ("W5 domain shift", [py, "scripts/w5_domain_shift.py"]),
        ("summary tables", [py, "scripts/compare_models.py"]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="W1 + W2/FD001 only")
    args = parser.parse_args()

    plan = steps(args.fast)
    started = time.time()
    for i, (name, cmd) in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] {name}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            print(f"\nFAILED: {name} (exit {proc.returncode})", file=sys.stderr)
            return proc.returncode
        print(f"    done in {time.time() - t0:.1f}s", flush=True)

    print(f"\nAll {len(plan)} steps finished in {time.time() - started:.1f}s")
    print("Next: streamlit run app/streamlit_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
