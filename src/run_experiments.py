"""
run_experiments.py
------------------
Automation script for running multiple (c,k)-anonymisation experiments.

Experiment definitions are read from configs/config.json ("experiments" key),
not hardcoded here. Add, remove, or change experiments there.

For each (k, c) combination:
  1. Calls the existing pipeline functions from ck_anonymisation.py
  2. Saves output to results/Exp_{set}_k{k}_c{c}/
  3. Collects metrics from the output CSVs

After all runs:
  - Saves results/experiment_summary.csv
  - Prints a formatted summary table

Can be invoked via:
  python cli.py --experiments
  python src/run_experiments.py          (standalone, from project root)
"""

import json
import sys
from pathlib import Path

# ── Path bootstrap ─────────────────────────────────────────────────────────────
# Works whether run standalone or imported by cli.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR      = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd
import ck_anonymisation as ck

# ── Project paths ──────────────────────────────────────────────────────────────
CONFIG_PATH  = _PROJECT_ROOT / "configs" / "config.json"
RESULTS_ROOT = _PROJECT_ROOT / "results"
SUMMARY_CSV  = RESULTS_ROOT / "experiment_summary.csv"

NAN = float("nan")
METRIC_COLS = [
    "total_records", "num_buckets", "groups_at_k", "groups_above_k",
    "min_group_size", "max_group_size", "avg_group_size", "safb_buckets",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def exp_folder_name(exp: dict) -> str:
    return f"Exp_{exp['set']}_k{exp['k']}_c{exp['c']}"


def collect_metrics(run_folder: Path, k: int, c: int, exp_set: str, fname: str) -> dict:
    """Read output CSVs and return a metrics dict for this experiment."""
    gt_df   = pd.read_csv(run_folder / "gt_partition.csv",   index_col=0)
    safb_df = pd.read_csv(run_folder / "safb_partition.csv")

    group_sizes   = gt_df.groupby("BucketID").size()
    total_records = len(gt_df)
    num_buckets   = len(group_sizes)

    return {
        "experiment_set": exp_set,
        "k":              k,
        "c":              c,
        "folder_name":    fname,
        "total_records":  total_records,
        "num_buckets":    num_buckets,
        "groups_at_k":    int((group_sizes == k).sum()),
        "groups_above_k": int((group_sizes >  k).sum()),
        "min_group_size": int(group_sizes.min()),
        "max_group_size": int(group_sizes.max()),
        "avg_group_size": round(total_records / num_buckets, 2),
        "safb_buckets":   len(safb_df),
    }


def nan_row(exp_set: str, k: int, c: int, fname: str) -> dict:
    """Return a metrics dict populated with NaN for a failed experiment."""
    return {
        "experiment_set": exp_set,
        "k":              k,
        "c":              c,
        "folder_name":    fname,
        **{col: NAN for col in METRIC_COLS},
    }


def print_run_metrics(metrics: dict) -> None:
    print(f"  num_buckets    : {metrics['num_buckets']}")
    print(f"  avg_group_size : {metrics['avg_group_size']:.2f}")
    print(f"  min_group_size : {metrics['min_group_size']}")
    print(f"  max_group_size : {metrics['max_group_size']}")


def print_summary_table(rows: list) -> None:
    df = pd.DataFrame(rows)
    col_order = [
        "experiment_set", "k", "c", "folder_name",
        "total_records", "num_buckets", "groups_at_k", "groups_above_k",
        "min_group_size", "max_group_size", "avg_group_size", "safb_buckets",
    ]
    df = df[col_order]
    print("\n" + "=" * 100)
    print("  EXPERIMENT SUMMARY")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load config — column definitions, input CSV path, and experiment list
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    qi_cols     = cfg["attributes"]["quasi_identifiers"]
    sa_cols     = cfg["attributes"]["sensitive_attributes"]
    experiments = cfg.get("experiments", [])
    input_csv   = _PROJECT_ROOT / cfg["file_paths"]["input_csv"]

    if not experiments:
        print("[ERROR] No experiments defined in configs/config.json under 'experiments' key.")
        sys.exit(1)

    RESULTS_ROOT.mkdir(exist_ok=True)

    # Load the dataset once — shared across all runs
    df = ck.load_dataset(input_csv, qi_cols, sa_cols)

    all_metrics: list = []
    n_total = len(experiments)

    for idx, exp in enumerate(experiments, start=1):
        k        = exp["k"]
        c        = exp["c"]
        exp_set  = exp["set"]
        fname    = exp_folder_name(exp)
        run_folder = RESULTS_ROOT / fname

        print(f"\n{'=' * 60}")
        print(f"  Experiment {idx}/{n_total}: {fname}  (k={k}, c={c})")
        print(f"{'=' * 60}")

        # Skip if folder already exists
        if run_folder.exists():
            print(f"  [SKIP] Folder '{fname}' already exists — skipping run.")
            try:
                metrics = collect_metrics(run_folder, k, c, exp_set, fname)
                all_metrics.append(metrics)
                print_run_metrics(metrics)
            except Exception as e:
                print(f"  [WARN] Could not read existing results: {e}")
                all_metrics.append(nan_row(exp_set, k, c, fname))
            continue

        # Run pipeline steps
        try:
            run_folder.mkdir(parents=True, exist_ok=True)

            gt_df   = ck.build_gt_partition(df, qi_cols, k)
            st_df   = ck.build_st_partition(df, gt_df, sa_cols)
            safb_df = ck.enforce_c_diversity(st_df, sa_cols, c)
            ck.verify_bucket_integrity(gt_df, safb_df)
            st_upd  = ck.enforce_c_constraint(gt_df, st_df, c)
            ck.verify_c_constraint(gt_df, st_upd, c)
            ck.validate_k_anonymity(gt_df, k)
            ck.export_partitions(gt_df, safb_df, run_folder)
            ck.validate_st_diversity(st_df, sa_cols, c)

            metrics = collect_metrics(run_folder, k, c, exp_set, fname)
            all_metrics.append(metrics)
            print_run_metrics(metrics)

        except Exception as e:
            print(f"  [ERROR] Experiment {fname} failed: {e}")
            all_metrics.append(nan_row(exp_set, k, c, fname))
            # Remove empty folder so a re-run can retry this experiment
            if run_folder.exists() and not any(run_folder.iterdir()):
                run_folder.rmdir()

    # Save summary CSV
    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\n[Done] Summary saved to '{SUMMARY_CSV}'")

    print_summary_table(all_metrics)


if __name__ == "__main__":
    main()
