"""
ck_anonymisation.py
--------------------
Implements the (c,k)-Anonymisation model on a medical dataset.

Steps implemented:
  0.  Automatic derivation of k and c from dataset properties
  1.  Dataset and attribute configuration (via config.json)
  2.  k-Anonymity via Mondrian partitioning  → GT partition
  3.  ST partition extraction
  4.  c-Diversity enforcement                → SAFB partition
  5.  c-Constraint on GT-to-ST correspondence
  6.  k-Anonymity validation
  7.  Export of all partitions to a timestamped results folder
  8.  Summary test report

Usage:
    python ck_anonymisation.py
"""

import json
import math
import random
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 – LOAD CONFIGURATION FROM config.json
# ──────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH   = _PROJECT_ROOT / "configs" / "config.json"
RESULTS_ROOT  = _PROJECT_ROOT / "results"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """
    Load the pipeline configuration from a JSON file.

    All tunable parameters — the input file path, privacy parameters, and
    attribute lists — are read from this file so that no source-code changes
    are needed when the dataset schema changes.

    Parameters
    ----------
    config_path : Path
        Path to the JSON configuration file (default: config.json).

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist at the given path.
    KeyError
        If any required top-level key is absent from the configuration.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at '{config_path}'. "
            "Please create config.json before running the pipeline."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    required_keys = {"file_paths", "attributes"}
    missing = required_keys - cfg.keys()
    if missing:
        raise KeyError(f"config.json is missing required top-level keys: {missing}")

    return cfg


def make_run_folder(results_root: Path = RESULTS_ROOT) -> Path:
    """
    Create a timestamped subfolder inside the results directory for this run.

    The folder name uses the format YYYY-MM-DD_HH-MM-SS so that each pipeline
    run produces its own isolated output directory.

    Parameters
    ----------
    results_root : Path
        Parent results directory (created if it does not exist).

    Returns
    -------
    Path
        The newly created timestamped run folder.
    """
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = results_root / timestamp
    run_folder.mkdir(parents=True, exist_ok=True)
    print(f"[Setup] Results folder created: {run_folder}")
    return run_folder


# ── Load configuration at module level ────────────────────────────────────────
_CFG = load_config()

INPUT_CSV = _PROJECT_ROOT / _CFG["file_paths"]["input_csv"]

# Attribute lists — edit quasi_identifiers / sensitive_attributes in config.json
QI_COLS = _CFG["attributes"]["quasi_identifiers"]
SA_COLS = _CFG["attributes"]["sensitive_attributes"]

RANDOM_SEED = int(_CFG.get("random_seed", 42))
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def print_config() -> None:
    """Print the currently loaded configuration to stdout."""
    print(
        f"[Config] Loaded from '{CONFIG_PATH}':\n"
        f"  Input file          : {INPUT_CSV}\n"
        f"  Quasi-identifiers   : {QI_COLS}\n"
        f"  Sensitive attributes: {SA_COLS}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 – STEP 1: LOAD DATASET
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(
    filepath: Path,
    qi_cols: list[str] | None = None,
    sa_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load the dataset from a CSV file and validate that all required columns exist.

    Parameters
    ----------
    filepath : Path
        Path to the input CSV file.
    qi_cols : list[str] or None
        Quasi-identifier column names to validate. Defaults to the module-level
        QI_COLS loaded from config.json when None.
    sa_cols : list[str] or None
        Sensitive attribute column names to validate. Defaults to the module-level
        SA_COLS loaded from config.json when None.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all required columns present.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the specified path.
    ValueError
        If any required QI or SA column is missing from the dataset.
    """
    qi_cols = qi_cols if qi_cols is not None else QI_COLS
    sa_cols = sa_cols if sa_cols is not None else SA_COLS

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Please provide a valid CSV file path."
        )

    df = pd.read_csv(filepath)

    missing = [col for col in qi_cols + sa_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    print(f"[Step 1] Dataset loaded: {len(df)} records, {len(df.columns)} columns.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 – AUTOMATIC DERIVATION OF k AND c
# ──────────────────────────────────────────────────────────────────────────────

def derive_k(df: pd.DataFrame, qi_cols: list[str]) -> int:
    """
    Automatically derive an appropriate k value from the dataset properties.

    The derivation follows three sequential steps:

    Step 1 — Base k from dataset size:
        base_k = ceil(0.01 × n), clamped to [2, 20].

    Step 2 — Dimensionality adjustment:
        For every two QI attributes beyond a baseline of 2, add 1 to base_k.

    Step 3 — Uniqueness rate adjustment:
        Compute the fraction of records that form a unique QI combination.
        If uniqueness rate > 0.8, increase adjusted_k by 50%.
        If uniqueness rate > 0.5 (but ≤ 0.8), increase adjusted_k by 20%.
        Re-apply the maximum ceiling of 20 after adjustment.

    A derivation summary is printed so the researcher can see exactly how k
    was determined.

    Parameters
    ----------
    df : pd.DataFrame
        The full raw dataset.
    qi_cols : list[str]
        Quasi-identifier column names.

    Returns
    -------
    int
        The final derived k value.
    """
    n = len(df)

    # Step 1: base k from dataset size
    base_k = max(2, min(20, math.ceil(0.01 * n)))

    # Step 2: adjust for QI dimensionality
    extra_dims   = max(0, len(qi_cols) - 2)
    dim_k        = base_k + (extra_dims // 2)

    # Step 3: adjust for QI uniqueness rate
    n_unique_qi     = df[qi_cols].drop_duplicates().shape[0]
    uniqueness_rate = n_unique_qi / n

    if uniqueness_rate > 0.8:
        adjusted_k = math.ceil(dim_k * 1.5)
    elif uniqueness_rate > 0.5:
        adjusted_k = math.ceil(dim_k * 1.2)
    else:
        adjusted_k = dim_k

    final_k = min(20, adjusted_k)

    print(
        f"\n[k Derivation]\n"
        f"  Total records          : {n}\n"
        f"  QI attributes          : {len(qi_cols)}\n"
        f"  Unique QI combinations : {n_unique_qi}\n"
        f"  Uniqueness rate        : {uniqueness_rate:.4f}\n"
        f"  Base k (1% of n)       : {base_k}\n"
        f"  Dimensionality-adj. k  : {dim_k}\n"
        f"  Final k (after unique) : {final_k}"
    )
    return final_k


def derive_c(df: pd.DataFrame, sa_cols: list[str]) -> int:
    """
    Automatically derive an appropriate c value from the sensitive attribute properties.

    c is a positive integer that directly sets the minimum number of distinct
    sensitive-attribute values required per bucket and the minimum number of ST
    buckets each GT bucket must be spread across.

    The derivation follows three sequential steps:

    Step 5 — Base c from the number of sensitive attributes:
        ≤ 2 SA columns → base_c = 2
        3–4 SA columns → base_c = 3
        5–6 SA columns → base_c = 4
        > 6 SA columns → base_c = 5

    Step 6 — Adjust c for sensitive attribute diversity:
        Compute the diversity ratio for each SA column (unique values / n).
        Average the ratios across all SA columns.
        avg_ratio < 0.05  → increase base_c by 2 (low diversity needs stricter enforcement).
        avg_ratio < 0.10  → increase base_c by 1.
        avg_ratio ≥ 0.10  → keep base_c unchanged.

    Step 7 — Enforce boundaries:
        Clamp final_c to [2, 10].

    A derivation summary is printed so the researcher can see exactly how c
    was determined and what it means structurally for the ST partition.

    Parameters
    ----------
    df : pd.DataFrame
        The full raw dataset.
    sa_cols : list[str]
        Sensitive attribute column names.

    Returns
    -------
    int
        The final derived c value (minimum distinct SA values / ST buckets per GT bucket).
    """
    n        = len(df)
    n_sa     = len(sa_cols)

    # Step 5: base c from number of sensitive attributes
    if n_sa <= 2:
        base_c = 2
    elif n_sa <= 4:
        base_c = 3
    elif n_sa <= 6:
        base_c = 4
    else:
        base_c = 5

    # Step 6: adjust for SA diversity
    diversity_ratios = [df[col].nunique() / n for col in sa_cols]
    avg_diversity    = sum(diversity_ratios) / len(diversity_ratios)

    if avg_diversity < 0.05:
        adjusted_c = base_c + 2
    elif avg_diversity < 0.10:
        adjusted_c = base_c + 1
    else:
        adjusted_c = base_c

    # Step 7: enforce boundaries [2, 10]
    final_c = max(2, min(10, adjusted_c))

    print(
        f"\n[c Derivation]\n"
        f"  Sensitive attributes   : {n_sa}\n"
        f"  Average diversity ratio: {avg_diversity:.4f}\n"
        f"  Base c                 : {base_c}\n"
        f"  Adjusted c             : {adjusted_c}\n"
        f"  Final c (after bounds) : {final_c}\n"
        f"  Min ST buckets per GT  : {final_c}"
    )
    return final_c


def initialise_privacy_parameters(
    df: pd.DataFrame,
    qi_cols: list[str],
    sa_cols: list[str],
) -> tuple[int, float]:
    """
    Derive and validate k and c from dataset properties before any anonymisation.

    This is the sole entry point for determining k and c. No other part of the
    pipeline defines or overrides these values. The function runs derive_k and
    derive_c in sequence and then validates the results.

    Parameters
    ----------
    df : pd.DataFrame
        The full raw dataset.
    qi_cols : list[str]
        Quasi-identifier column names.
    sa_cols : list[str]
        Sensitive attribute column names.

    Returns
    -------
    tuple[int, float]
        (k, c) — the final derived privacy parameters ready for use throughout
        the pipeline.

    Raises
    ------
    ValueError
        If the derived k is less than 2, or if the derived c is not a positive integer ≥ 1.
    """
    print("\n" + "─" * 60)
    print("  Initialising Privacy Parameters")
    print("─" * 60)

    k = derive_k(df, qi_cols)
    c = derive_c(df, sa_cols)

    # Validate derived values before proceeding
    if not isinstance(k, int) or k < 2:
        raise ValueError(
            f"Derived k={k} is invalid. k must be an integer ≥ 2."
        )
    if not (isinstance(c, int) and c >= 1):
        raise ValueError(
            f"Derived c={c} is invalid. c must be a positive integer ≥ 1."
        )

    print(
        f"\n[Init] Privacy parameters confirmed: k={k}, c={c}\n"
        f"{'─' * 60}"
    )
    return k, c


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 – STEP 2: MONDRIAN k-ANONYMITY → GT PARTITION
# ──────────────────────────────────────────────────────────────────────────────

def _choose_split_attribute(partition: pd.DataFrame, qi_cols: list[str]) -> str | None:
    """
    Select the QI attribute with the highest number of distinct values in the
    current partition.

    Parameters
    ----------
    partition : pd.DataFrame
        The current subset of records being considered for splitting.
    qi_cols : list[str]
        List of quasi-identifier column names.

    Returns
    -------
    str or None
        The name of the attribute to split on, or None if no valid split exists.
    """
    distinct_counts = {col: partition[col].nunique() for col in qi_cols}
    valid = {col: cnt for col, cnt in distinct_counts.items() if cnt > 1}
    if not valid:
        return None
    return max(valid, key=valid.get)


def _generalise_qi(partition: pd.DataFrame, qi_cols: list[str]) -> dict:
    """
    Generalise QI attribute values in a partition into range or set strings.

    Numeric attributes are represented as "min-max".
    Categorical attributes are represented as a sorted, comma-separated list
    of unique values.

    Parameters
    ----------
    partition : pd.DataFrame
        The finalised (non-splittable) partition.
    qi_cols : list[str]
        List of quasi-identifier column names.

    Returns
    -------
    dict
        Mapping of column name → generalised string representation.
    """
    generalised = {}
    for col in qi_cols:
        if pd.api.types.is_numeric_dtype(partition[col]):
            lo, hi = partition[col].min(), partition[col].max()
            generalised[col] = f"{lo}-{hi}" if lo != hi else str(lo)
        else:
            unique_vals = sorted(partition[col].dropna().unique().tolist())
            generalised[col] = ", ".join(str(v) for v in unique_vals)
    return generalised


def _mondrian_split(
    partition: pd.DataFrame,
    qi_cols: list[str],
    k: int,
    results: list,
    bucket_counter: list,
) -> None:
    """
    Recursively split a partition using the Mondrian algorithm.

    Selects the QI attribute with the most distinct values, sorts by that
    attribute, and splits at the median. Recursion stops when a split would
    produce a sub-partition smaller than k, at which point the current
    partition is finalised as an equivalence class.

    Parameters
    ----------
    partition : pd.DataFrame
        Current subset of records to partition.
    qi_cols : list[str]
        Quasi-identifier column names.
    k : int
        Minimum equivalence class size.
    results : list
        Accumulator list; each element is a dict with generalised QI values,
        BucketID, and the original DataFrame indices.
    bucket_counter : list
        Single-element list used as a mutable integer counter for BucketID
        assignment across recursive calls.
    """
    split_attr = _choose_split_attribute(partition, qi_cols)

    if split_attr is None:
        bucket_id = bucket_counter[0]
        bucket_counter[0] += 1
        gen = _generalise_qi(partition, qi_cols)
        results.append({"indices": partition.index.tolist(), "BucketID": bucket_id, "generalised": gen})
        return

    sorted_partition = partition.sort_values(by=split_attr)
    median_idx = len(sorted_partition) // 2
    left  = sorted_partition.iloc[:median_idx]
    right = sorted_partition.iloc[median_idx:]

    if len(left) >= k and len(right) >= k:
        _mondrian_split(left,  qi_cols, k, results, bucket_counter)
        _mondrian_split(right, qi_cols, k, results, bucket_counter)
    else:
        bucket_id = bucket_counter[0]
        bucket_counter[0] += 1
        gen = _generalise_qi(partition, qi_cols)
        results.append({"indices": partition.index.tolist(), "BucketID": bucket_id, "generalised": gen})


def build_gt_partition(df: pd.DataFrame, qi_cols: list[str], k: int) -> pd.DataFrame:
    """
    Apply Mondrian k-anonymity to produce the GT (Generalised Table) partition.

    Each record in the output has its QI values replaced by generalised ranges
    or sets, and is assigned a BucketID identifying its equivalence class.

    Parameters
    ----------
    df : pd.DataFrame
        The full raw dataset.
    qi_cols : list[str]
        Quasi-identifier column names.
    k : int
        Minimum equivalence class size.

    Returns
    -------
    pd.DataFrame
        GT partition with columns for each QI attribute (generalised) plus
        'BucketID'. Row order matches the original DataFrame index.

    Raises
    ------
    ValueError
        If the dataset is too small to form even one equivalence class of size k.
    """
    if len(df) < k:
        raise ValueError(
            f"Dataset has only {len(df)} records, which is fewer than k={k}. "
            "Cannot build a GT partition."
        )

    results: list = []
    bucket_counter = [0]
    _mondrian_split(df, qi_cols, k, results, bucket_counter)

    rows = []
    for bucket in results:
        gen = bucket["generalised"]
        for idx in bucket["indices"]:
            row = {"BucketID": bucket["BucketID"]}
            row.update(gen)
            rows.append((idx, row))

    rows.sort(key=lambda x: x[0])
    gt_df = pd.DataFrame([r for _, r in rows], index=[i for i, _ in rows])
    gt_df = gt_df[["BucketID"] + qi_cols]

    num_buckets = gt_df["BucketID"].nunique()
    print(
        f"[Step 2] GT partition built: {len(gt_df)} records "
        f"across {num_buckets} equivalence classes (k={k})."
    )
    return gt_df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 – STEP 3: BUILD ST PARTITION
# ──────────────────────────────────────────────────────────────────────────────

def build_st_partition(df: pd.DataFrame, gt_df: pd.DataFrame, sa_cols: list[str]) -> pd.DataFrame:
    """
    Extract the ST (Sensitive Table) partition from the original dataset.

    The ST contains only sensitive attributes and the BucketID inherited from
    the GT partition. No QI attributes or explicit identifiers are included.

    Parameters
    ----------
    df : pd.DataFrame
        The full raw dataset.
    gt_df : pd.DataFrame
        The GT partition (must contain 'BucketID' and share the same index as df).
    sa_cols : list[str]
        Sensitive attribute column names.

    Returns
    -------
    pd.DataFrame
        ST partition with columns for each SA attribute plus 'BucketID'.
    """
    st_df = df[sa_cols].copy()
    st_df.insert(0, "BucketID", gt_df["BucketID"].values)

    print(
        f"[Step 3] ST partition built: {len(st_df)} records, "
        f"columns: {list(st_df.columns)}."
    )
    return st_df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 – STEP 4: c-DIVERSITY → SAFB PARTITION
# ──────────────────────────────────────────────────────────────────────────────

def enforce_c_diversity(
    st_df: pd.DataFrame,
    sa_cols: list[str],
    c: float,
) -> pd.DataFrame:
    """
    Enforce c-diversity within each ST bucket and produce the SAFB partition.

    The BucketID used here is inherited directly from the GT partition via the
    ST partition. Each unique BucketID in the ST corresponds to exactly one
    equivalence class in the GT. The resulting SAFB therefore carries the same
    BucketID set as the GT, making BucketID the sole structural link between
    the two published partitions.

    For each bucket (grouped by BucketID):
      1. Remove exact duplicate rows when more than two duplicates exist,
         keeping one instance.
      2. For each SA column, if the number of unique values is less than the
         required minimum (c), sample additional values with
         replacement to satisfy the diversity requirement.
      3. Concatenate all unique SA values per column into a comma-separated
         string. This consolidated row is the single SAFB record for that
         BucketID.

    Parameters
    ----------
    st_df : pd.DataFrame
        The ST partition with BucketID (inherited from GT) and SA columns.
    sa_cols : list[str]
        Sensitive attribute column names.
    c : int
        Minimum number of distinct SA values required per bucket.

    Returns
    -------
    pd.DataFrame
        SAFB partition: exactly one row per BucketID, with each SA column
        containing a comma-separated string of unique values in that bucket.
        The BucketID set in this output is identical to the BucketID set in
        the source ST partition.

    Raises
    ------
    ValueError
        If the SAFB BucketID set does not exactly match the ST BucketID set
        after construction, indicating a structural integrity failure.
    """
    min_distinct  = c
    safb_records  = []
    expected_ids  = set(st_df["BucketID"].unique())

    for bucket_id, group in st_df.groupby("BucketID"):
        sa_group = group[sa_cols].copy()

        # Remove excessive exact duplicates — keep at most 2 of each combination
        deduped = (
            sa_group
            .groupby(sa_cols, dropna=False)
            .head(2)
            .reset_index(drop=True)
        )

        # Ensure each SA column has at least min_distinct unique values
        for col in sa_cols:
            unique_vals = deduped[col].dropna().unique().tolist()
            if len(unique_vals) < min_distinct:
                shortage = min_distinct - len(unique_vals)
                extra = pd.Series(
                    np.random.choice(unique_vals, size=shortage, replace=True),
                    name=col,
                )
                patch = pd.DataFrame({c_: [""] * shortage for c_ in sa_cols})
                patch[col] = extra.values
                deduped = pd.concat([deduped, patch], ignore_index=True)

        # Concatenate unique values per SA column into a single string.
        # The BucketID is preserved unchanged — it is the linking key to GT.
        record = {"BucketID": bucket_id}
        for col in sa_cols:
            unique_vals = deduped[col].replace("", np.nan).dropna().unique().tolist()
            record[col] = ", ".join(str(v) for v in sorted(set(str(x) for x in unique_vals)))

        safb_records.append(record)

    safb_df      = pd.DataFrame(safb_records)
    produced_ids = set(safb_df["BucketID"].unique())

    # Structural integrity check: SAFB BucketID set must match ST exactly
    if produced_ids != expected_ids:
        missing = expected_ids - produced_ids
        extra   = produced_ids - expected_ids
        raise ValueError(
            f"[Step 4] SAFB BucketID integrity failure after construction.\n"
            f"  Missing from SAFB : {missing}\n"
            f"  Extra in SAFB     : {extra}"
        )

    print(
        f"[Step 4] SAFB partition built: {len(safb_df)} bucket records "
        f"(minimum distinct SA values per bucket per column: {min_distinct}).\n"
        f"  BucketID integrity confirmed: SAFB IDs match GT IDs."
    )
    return safb_df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 – STEP 5: c-CONSTRAINT ON GT-TO-ST CORRESPONDENCE
# ──────────────────────────────────────────────────────────────────────────────

def enforce_c_constraint(
    gt_df: pd.DataFrame,
    st_df: pd.DataFrame,
    c: float,
) -> pd.DataFrame:
    """
    Enforce the c-correspondence constraint between GT and ST buckets.

    For each GT bucket of size n, records are distributed across at least
    c ST buckets such that no single GT-to-ST link covers more
    than n/c records. The original GT BucketID is retained as 'GT_BucketID'.

    Parameters
    ----------
    gt_df : pd.DataFrame
        The GT partition (must contain 'BucketID').
    st_df : pd.DataFrame
        The ST partition (must contain 'BucketID' matching gt_df).
    c : int
        Minimum number of ST buckets each GT bucket must be distributed across.
        Equivalently, no ST bucket may contain more than 1/c of a GT bucket's records.

    Returns
    -------
    pd.DataFrame
        Updated ST partition where 'BucketID' reflects the new ST bucket
        assignment, and 'GT_BucketID' records the original GT bucket.
    """
    min_st_buckets    = c
    all_st_bucket_ids = sorted(st_df["BucketID"].unique().tolist())
    num_available     = len(all_st_bucket_ids)

    if num_available < min_st_buckets:
        warnings.warn(
            f"Only {num_available} ST buckets available but the c-constraint requires "
            f"at least {min_st_buckets}. Distribution may violate the c-constraint.",
            stacklevel=2,
        )

    st_updated = st_df.copy()
    st_updated["GT_BucketID"] = st_updated["BucketID"]
    new_bucket_assignments: dict = {}

    for _, group in gt_df.groupby("BucketID"):
        n     = len(group)
        k_st  = min(min_st_buckets, num_available)
        selected_st = random.sample(all_st_bucket_ids, k_st)

        indices = group.index.tolist()
        random.shuffle(indices)
        chunk_size = math.ceil(n / k_st)
        for i, st_bucket in enumerate(selected_st):
            for idx in indices[i * chunk_size: (i + 1) * chunk_size]:
                new_bucket_assignments[idx] = st_bucket

    st_updated.loc[list(new_bucket_assignments.keys()), "BucketID"] = [
        new_bucket_assignments[idx] for idx in new_bucket_assignments
    ]

    print(
        f"[Step 5] c-Constraint enforced: each GT bucket distributed across "
        f"≥{min_st_buckets} ST buckets (c={c})."
    )
    return st_updated


def verify_c_constraint(
    gt_df: pd.DataFrame,
    st_updated: pd.DataFrame,
    c: float,
) -> bool:
    """
    Verify that the c-correspondence constraint holds for every GT-ST bucket pair.

    For each GT bucket of size n and each ST bucket it maps to, the fraction of
    shared records must not exceed c.

    Parameters
    ----------
    gt_df : pd.DataFrame
        GT partition with 'BucketID'.
    st_updated : pd.DataFrame
        Updated ST partition with 'BucketID' (ST assignment) and 'GT_BucketID'.
    c : int
        Minimum number of ST buckets each GT bucket must be spread across.
        No single ST bucket may contain more than 1/c of a GT bucket's records.

    Returns
    -------
    bool
        True if all GT-ST pairs satisfy the constraint; False if any violation
        is detected (an alert is printed for each violation).
    """
    max_fraction = 1 / c
    violations = []
    for gt_bucket_id, group in gt_df.groupby("BucketID"):
        n    = len(group)
        mask = st_updated["GT_BucketID"] == gt_bucket_id
        dist = st_updated.loc[mask, "BucketID"].value_counts()
        for st_bucket, count in dist.items():
            fraction = count / n
            if fraction > max_fraction:
                violations.append(
                    f"  GT bucket {gt_bucket_id} → ST bucket {st_bucket}: "
                    f"{count}/{n} = {fraction:.3f} > 1/c={max_fraction:.3f}"
                )

    if violations:
        print(
            f"[Step 5] ALERT – c-Constraint VIOLATIONS detected: "
            f"{len(violations)} GT-ST bucket pair(s) exceed 1/c={max_fraction:.3f} (c={c})."
        )
        return False

    print("[Step 5] c-Constraint verification PASSED for all GT-ST bucket pairs.")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 – STEP 6: VALIDATE k-ANONYMITY
# ──────────────────────────────────────────────────────────────────────────────

def validate_k_anonymity(gt_df: pd.DataFrame, k: int) -> None:
    """
    Validate that every equivalence class in the GT partition satisfies k-anonymity.

    Iterates over all GT groups by BucketID and raises an error for any group
    smaller than k. Prints a summary of group size statistics.

    Parameters
    ----------
    gt_df : pd.DataFrame
        GT partition with 'BucketID'.
    k : int
        Minimum required group size.

    Raises
    ------
    ValueError
        If any GT equivalence class contains fewer than k records.
    """
    group_sizes  = gt_df.groupby("BucketID").size()
    total_groups = len(group_sizes)
    exact_k      = (group_sizes == k).sum()
    larger_k     = (group_sizes > k).sum()
    violations   = group_sizes[group_sizes < k]

    if len(violations) > 0:
        raise ValueError(
            f"[Step 6] k-Anonymity VIOLATED: {len(violations)} bucket(s) have "
            f"fewer than k={k} records (smallest: {violations.min()} records)."
        )

    print(
        f"[Step 6] k-Anonymity validation PASSED.\n"
        f"  Total groups  : {total_groups}\n"
        f"  Groups = k    : {exact_k}\n"
        f"  Groups > k    : {larger_k}\n"
        f"  Min group size: {group_sizes.min()}\n"
        f"  Max group size: {group_sizes.max()}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 – STEP 7: EXPORT PARTITIONS
# ──────────────────────────────────────────────────────────────────────────────

def export_partitions(
    gt_df: pd.DataFrame,
    safb_df: pd.DataFrame,
    run_folder: Path,
) -> None:
    """
    Export the GT and SAFB partitions to CSV files inside the timestamped run folder.

    Only two files are written. They are joined on BucketID, which is the sole
    structural link between them. BucketID integrity is verified immediately
    before writing to guarantee that the exported files are consistent.

    Files written:
      - gt_partition.csv   — one row per original record, with generalised QI
                             attribute values and BucketID.
      - safb_partition.csv — one row per BucketID, with concatenated unique
                             sensitive attribute values for that equivalence class.

    Parameters
    ----------
    gt_df : pd.DataFrame
        GT partition (generalised QI attributes + BucketID).
    safb_df : pd.DataFrame
        SAFB partition (concatenated SA strings + BucketID).
    run_folder : Path
        Timestamped output folder created by make_run_folder().

    Raises
    ------
    ValueError
        If BucketID integrity between GT and SAFB fails immediately before export.
    """
    # Final integrity gate — halt export if BucketIDs are inconsistent
    verify_bucket_integrity(gt_df, safb_df)

    gt_path   = run_folder / "gt_partition.csv"
    safb_path = run_folder / "safb_partition.csv"

    gt_df.to_csv(gt_path,     index=True)
    safb_df.to_csv(safb_path, index=False)

    print(
        f"[Step 7] Exports complete:\n"
        f"  GT partition   → {gt_path}\n"
        f"  SAFB partition → {safb_path}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10 – STEP 8: VALIDATION AND SUMMARY REPORT
# ──────────────────────────────────────────────────────────────────────────────

def verify_bucket_integrity(gt_df: pd.DataFrame, safb_df: pd.DataFrame) -> None:
    """
    Verify that the BucketID sets in the GT and SAFB partitions are identical.

    This is the core structural invariant of the (c,k)-anonymisation model:
      - Every BucketID in the GT partition must exist in the SAFB partition,
        so that every equivalence class has a corresponding SA record.
      - Every BucketID in the SAFB partition must have at least one record in
        the GT partition, so that no orphaned SAFB rows exist.

    This check must pass before any output file is written. A mismatch
    indicates that a pipeline step has broken the GT-SAFB link and the results
    must not be exported.

    Parameters
    ----------
    gt_df : pd.DataFrame
        GT partition containing at least a 'BucketID' column.
    safb_df : pd.DataFrame
        SAFB partition containing at least a 'BucketID' column.

    Raises
    ------
    ValueError
        If any BucketID present in GT is absent from SAFB, or if any BucketID
        present in SAFB has no corresponding records in GT.
    """
    gt_ids   = set(gt_df["BucketID"].unique())
    safb_ids = set(safb_df["BucketID"].unique())

    in_gt_not_safb   = gt_ids - safb_ids
    in_safb_not_gt   = safb_ids - gt_ids

    errors = []
    if in_gt_not_safb:
        errors.append(
            f"  BucketIDs in GT but missing from SAFB: {sorted(in_gt_not_safb)}"
        )
    if in_safb_not_gt:
        errors.append(
            f"  BucketIDs in SAFB with no records in GT: {sorted(in_safb_not_gt)}"
        )

    if errors:
        raise ValueError(
            f"[BucketID Integrity] FAILED — "
            f"{len(in_gt_not_safb)} BucketID(s) in GT missing from SAFB, "
            f"{len(in_safb_not_gt)} BucketID(s) in SAFB with no GT records."
        )

    print(
        f"[BucketID Integrity] PASSED — {len(gt_ids)} BucketIDs present and "
        f"consistent across GT and SAFB."
    )


def validate_st_diversity(
    st_df: pd.DataFrame,
    sa_cols: list[str],
    c: float,
) -> bool:
    """
    Validate that every ST bucket has at least c distinct values
    per sensitive attribute.

    Parameters
    ----------
    st_df : pd.DataFrame
        ST partition with 'BucketID' and SA columns.
    sa_cols : list[str]
        Sensitive attribute column names.
    c : int
        Minimum number of distinct SA values required per bucket.

    Returns
    -------
    bool
        True if all buckets satisfy the diversity requirement, False otherwise.
    """
    min_distinct = c
    violations   = []

    for bucket_id, group in st_df.groupby("BucketID"):
        for col in sa_cols:
            n_unique = group[col].nunique()
            if n_unique < min_distinct:
                violations.append(
                    f"  BucketID={bucket_id}, column='{col}': "
                    f"{n_unique} unique values (required ≥ {min_distinct})"
                )

    if violations:
        print(
            f"[Validation] ALERT – ST diversity: {len(violations)} bucket/column "
            f"combination(s) have fewer than {min_distinct} distinct SA values."
        )
        return False

    print(
        f"[Validation] ST diversity check PASSED: all buckets have "
        f"≥{min_distinct} distinct values per SA column."
    )
    return True


def print_summary_report(
    df: pd.DataFrame,
    gt_df: pd.DataFrame,
    run_folder: Path,
    k: int,
    c: float,
    filepath: Path | None = None,
) -> None:
    """
    Print a comprehensive summary report of the (c,k)-anonymisation results.

    Parameters
    ----------
    df : pd.DataFrame
        Original raw dataset.
    gt_df : pd.DataFrame
        GT partition.
    run_folder : Path
        Timestamped output folder for this run.
    k : int
        k-Anonymity parameter.
    c : float
        c-Correspondence constraint parameter.
    filepath : Path or None
        Path to the input file. Defaults to the module-level INPUT_CSV when None.
    """
    filepath       = filepath or INPUT_CSV
    total_records  = len(df)
    num_buckets    = gt_df["BucketID"].nunique()   # identical in GT and SAFB
    avg_group_size = gt_df.groupby("BucketID").size().mean()

    print("\n" + "=" * 60)
    print("       (c,k)-ANONYMISATION SUMMARY REPORT")
    print("=" * 60)
    print(f"  Input dataset          : {filepath}")
    print(f"  Output folder          : {run_folder}")
    print(f"  Total records          : {total_records}")
    print(f"  k value                : {k}")
    print(f"  c value                : {c}")
    print(f"  Number of buckets      : {num_buckets}  (GT and SAFB share this key set)")
    print(f"  Average GT group size  : {avg_group_size:.2f}")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11 – SHARED PIPELINE RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    filepath: Path,
    qi_cols: list[str],
    sa_cols: list[str],
    k: int | None = None,
    c: float | None = None,
) -> None:
    """
    Execute the full (c,k)-anonymisation pipeline with the given parameters.

    This is the primary entry point used by both main() and the CLI. When k
    and c are both supplied they are validated and used directly (manual mode).
    When either is None the values are derived automatically from the dataset
    (auto mode).

    Parameters
    ----------
    filepath : Path
        Path to the input CSV dataset.
    qi_cols : list[str]
        Quasi-identifier column names.
    sa_cols : list[str]
        Sensitive attribute column names.
    k : int or None
        Manual k value. When None, k is derived automatically.
    c : float or None
        Manual c value. When None, c is derived automatically.

    Raises
    ------
    ValueError
        If manually supplied k or c fail the validity constraints, or if any
        anonymisation invariant is violated during the run.
    """
    print("\n" + "=" * 60)
    print("  Starting (c,k)-Anonymisation Pipeline")
    print("=" * 60)

    # Load dataset — columns are validated here
    df = load_dataset(filepath, qi_cols, sa_cols)

    # Determine k and c
    if k is not None and c is not None:
        # Manual mode: validate the user-supplied values before proceeding
        if not isinstance(k, int) or k < 2:
            raise ValueError(f"Supplied k={k} is invalid. k must be an integer ≥ 2.")
        if not (isinstance(c, int) and c >= 1):
            raise ValueError(f"Supplied c={c} is invalid. c must be a positive integer ≥ 1.")
        print(
            f"\n[Init] Using manually supplied parameters: k={k}, c={c}\n"
            f"{'─' * 60}"
        )
    else:
        # Auto mode: derive k and c from the dataset
        k, c = initialise_privacy_parameters(df, qi_cols, sa_cols)

    # Create timestamped output folder now that parameters are confirmed
    run_folder = make_run_folder()

    # Build GT partition via Mondrian k-anonymity
    gt_df = build_gt_partition(df, qi_cols, k)

    # Build ST partition
    st_df = build_st_partition(df, gt_df, sa_cols)

    # Enforce c-diversity → SAFB partition (BucketID integrity asserted inside)
    safb_df = enforce_c_diversity(st_df, sa_cols, c)

    # Mid-pipeline BucketID integrity check before the c-constraint step
    verify_bucket_integrity(gt_df, safb_df)

    # Enforce c-correspondence constraint and verify
    st_updated = enforce_c_constraint(gt_df, st_df, c)
    verify_c_constraint(gt_df, st_updated, c)

    # Validate k-anonymity
    validate_k_anonymity(gt_df, k)

    # Export only GT and SAFB (BucketID integrity re-checked inside export)
    export_partitions(gt_df, safb_df, run_folder)

    # Final validation checks and summary
    print("\n--- Running Final Validation Checks ---")
    verify_bucket_integrity(gt_df, safb_df)
    validate_st_diversity(st_df, sa_cols, c)
    verify_c_constraint(gt_df, st_updated, c)
    validate_k_anonymity(gt_df, k)
    print_summary_report(df, gt_df, run_folder, k, c, filepath)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Execute the pipeline using settings from config.json (auto k/c derivation).

    To customise the input file or use manual k/c values, use cli.py instead.
    """
    print_config()
    run_pipeline(INPUT_CSV, QI_COLS, SA_COLS)


if __name__ == "__main__":
    main()
