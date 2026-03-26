"""
cli.py
------
Command-line interface for the (c,k)-Anonymisation pipeline.

Project layout expected:
  <project_root>/
  ├── cli.py                   ← this file
  ├── configs/config.json
  ├── dataset/*.csv
  ├── src/ck_anonymisation.py
  └── src/run_experiments.py

Run modes
---------
  Anonymise — single run, auto k/c derivation:
    python cli.py --auto
    python cli.py --file dataset/sample_medical_dataset.csv --auto

  Anonymise — single run, manual k/c:
    python cli.py --file dataset/sample_medical_dataset.csv --manual --k 10 --c 3

  Experiments — run all batches defined in configs/config.json:
    python cli.py --experiments

  Fully interactive (prompts for everything):
    python cli.py

  Help:
    python cli.py --help
"""

import argparse
import sys
from pathlib import Path

# ── Path bootstrap — must happen before importing src modules ──────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR      = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ck_anonymisation import (   # noqa: E402  (import after sys.path setup)
    QI_COLS,
    SA_COLS,
    CONFIG_PATH,
    load_config,
    run_pipeline,
)
import run_experiments           # noqa: E402

_DATASET_DIR = _PROJECT_ROOT / "dataset"

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

_BANNER = """
╔══════════════════════════════════════════════════════════╗
║         (c,k)-Anonymisation Pipeline — CLI               ║
╚══════════════════════════════════════════════════════════╝
"""


def _list_csv_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.csv"))


def _prompt_run_mode() -> str:
    """Ask whether to run a single anonymisation job or the experiment batch."""
    print("\n── Choose run mode ─────────────────────────────────────────")
    print("    [1]  Anonymise   — run a single anonymisation job")
    print("    [2]  Experiments — run all batches defined in configs/config.json")
    while True:
        raw = input("\n  Your choice [1/2]: ").strip()
        if raw == "1":
            return "anonymise"
        if raw == "2":
            return "experiments"
        print("  Please enter 1 or 2.")


def _prompt_file_selection(directory: Path) -> Path:
    """Interactively prompt the researcher to select a CSV file."""
    csv_files = _list_csv_files(directory)

    print("\n── Step 1: Select the dataset to anonymise ─────────────────")

    if csv_files:
        print(f"  CSV files found in '{directory}':\n")
        for i, fp in enumerate(csv_files, start=1):
            print(f"    [{i}]  {fp.name}")
        print(f"    [0]  Enter a custom file path")
    else:
        print(f"  No CSV files found in '{directory}'.")
        print(f"    [0]  Enter a custom file path")

    while True:
        raw = input("\n  Your choice (number or 0 for custom path): ").strip()

        if raw == "0" or not csv_files:
            custom = input("  Enter the full path to the CSV file: ").strip()
            chosen = Path(custom)
        else:
            try:
                idx = int(raw)
                if 1 <= idx <= len(csv_files):
                    chosen = csv_files[idx - 1]
                else:
                    print(f"  Please enter a number between 0 and {len(csv_files)}.")
                    continue
            except ValueError:
                print("  Invalid input — please enter a number.")
                continue

        if not chosen.exists():
            print(f"  File not found: '{chosen}'. Please try again.")
            continue

        print(f"\n  Selected file: {chosen}")
        return chosen


def _prompt_kc_mode() -> str:
    print("\n── Step 2: Choose how k and c are determined ───────────────")
    print("    [1]  Auto   — derive k and c from the dataset properties")
    print("    [2]  Manual — specify k and c values yourself")

    while True:
        raw = input("\n  Your choice [1/2]: ").strip()
        if raw == "1":
            return "auto"
        if raw == "2":
            return "manual"
        print("  Please enter 1 or 2.")


def _prompt_manual_kc() -> tuple[int, int]:
    print("\n── Step 3: Enter k and c values ────────────────────────────")
    print("  k  — minimum equivalence class size (integer ≥ 2)")
    print("  c  — minimum distinct SA values / ST buckets per GT bucket (integer ≥ 1)")

    while True:
        try:
            k = int(input("\n  k value: ").strip())
            if k < 2:
                print("  k must be ≥ 2. Please try again.")
                continue
            break
        except ValueError:
            print("  Please enter a valid integer.")

    while True:
        try:
            c = int(input("  c value: ").strip())
            if c < 1:
                print("  c must be ≥ 1. Please try again.")
                continue
            break
        except ValueError:
            print("  Please enter a valid positive integer (e.g. 3).")

    return k, c


def _confirm_run(filepath: Path, mode: str, k: int | None, c: int | None) -> bool:
    print("\n── Run summary ──────────────────────────────────────────────")
    print(f"  File  : {filepath}")
    print(f"  Mode  : {mode}")
    if mode == "manual":
        print(f"  k     : {k}")
        print(f"  c     : {c}")
    else:
        print("  k, c  : will be derived automatically")

    cfg     = load_config(CONFIG_PATH)
    qi_cols = cfg["attributes"]["quasi_identifiers"]
    sa_cols = cfg["attributes"]["sensitive_attributes"]
    print(f"  QI    : {qi_cols}")
    print(f"  SA    : {sa_cols}")

    raw = input("\n  Proceed? [Y/n]: ").strip().lower()
    return raw in ("", "y", "yes")


def _confirm_experiments() -> bool:
    import json
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    exps = cfg.get("experiments", [])
    print("\n── Experiment batch summary ─────────────────────────────────")
    print(f"  Config  : {CONFIG_PATH}")
    print(f"  Dataset : {_PROJECT_ROOT / cfg['file_paths']['input_csv']}")
    print(f"  Runs    : {len(exps)} experiment(s)")
    for exp in exps:
        print(f"    Exp_{exp['set']}_k{exp['k']}_c{exp['c']}")
    raw = input("\n  Proceed? [Y/n]: ").strip().lower()
    return raw in ("", "y", "yes")


# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="(c,k)-Anonymisation Pipeline — command-line interface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python cli.py                                                         # interactive\n"
            "  python cli.py --experiments                                           # run all batches from config\n"
            "  python cli.py --file dataset/data.csv --auto                          # single run, auto k/c\n"
            "  python cli.py --file dataset/data.csv --manual --k 10 --c 3          # single run, manual k/c"
        ),
    )

    parser.add_argument(
        "--file", "-f",
        metavar="PATH",
        type=Path,
        help="Path to the CSV dataset to anonymise (single-run modes only).",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--experiments",
        action="store_true",
        default=False,
        help="Run all experiment batches defined in configs/config.json.",
    )
    mode_group.add_argument(
        "--auto",
        action="store_true",
        default=False,
        help="Derive k and c automatically from dataset properties.",
    )
    mode_group.add_argument(
        "--manual",
        action="store_true",
        default=False,
        help="Supply k and c values manually (requires --k and --c).",
    )

    parser.add_argument(
        "--k",
        metavar="INT",
        type=int,
        help="Minimum equivalence class size (required with --manual).",
    )
    parser.add_argument(
        "--c",
        metavar="INT",
        type=int,
        help="Minimum distinct SA values / ST buckets per GT bucket (required with --manual).",
    )

    return parser


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Parse arguments, fill in missing inputs interactively, and run the pipeline.

    Behaviour by argument combination
    ----------------------------------
    No arguments          → interactive: choose mode, then guided through options.
    --experiments         → run all batches from configs/config.json (non-interactive).
    --auto                → single run, auto k/c, file from config (non-interactive).
    --file F --auto       → single run, auto k/c, specific file.
    --file F --manual     → prompts for k and c if --k/--c not supplied.
    --file F --manual --k K --c C → fully non-interactive single run.
    """
    print(_BANNER)

    parser         = build_parser()
    args           = parser.parse_args()
    is_interactive = len(sys.argv) == 1

    # ── Experiments mode ──────────────────────────────────────────────────────
    if args.experiments or (is_interactive and _prompt_run_mode() == "experiments"):
        if is_interactive and not args.experiments:
            if not _confirm_experiments():
                print("\n  Run cancelled.")
                sys.exit(0)
        run_experiments.main()
        return

    # ── Single-run anonymisation mode ─────────────────────────────────────────

    # Resolve file path
    if args.file is not None:
        filepath = args.file
        if not filepath.exists():
            parser.error(f"File not found: '{filepath}'")
    elif not is_interactive:
        # Non-interactive without --file → use config default
        cfg      = load_config(CONFIG_PATH)
        filepath = _PROJECT_ROOT / cfg["file_paths"]["input_csv"]
        if not filepath.exists():
            parser.error(f"Default dataset not found: '{filepath}'")
    else:
        filepath = _prompt_file_selection(_DATASET_DIR)

    # Resolve k/c mode
    if args.manual:
        kc_mode = "manual"
    elif args.auto or not is_interactive:
        kc_mode = "auto"
    else:
        kc_mode = _prompt_kc_mode()

    # Resolve k and c
    k: int | None = None
    c: int | None = None

    if kc_mode == "manual":
        if args.k is not None and args.c is not None:
            k, c = args.k, args.c
        elif args.k is not None or args.c is not None:
            parser.error("--manual requires both --k and --c to be specified.")
        else:
            k, c = _prompt_manual_kc()

        if k < 2:
            parser.error(f"k must be an integer ≥ 2 (got {k}).")
        if c < 1:
            parser.error(f"c must be a positive integer ≥ 1 (got {c}).")

    # Confirm and run
    if is_interactive:
        if not _confirm_run(filepath, kc_mode, k, c):
            print("\n  Run cancelled.")
            sys.exit(0)

    cfg     = load_config(CONFIG_PATH)
    qi_cols = cfg["attributes"]["quasi_identifiers"]
    sa_cols = cfg["attributes"]["sensitive_attributes"]

    run_pipeline(filepath, qi_cols, sa_cols, k=k, c=c)


if __name__ == "__main__":
    main()
