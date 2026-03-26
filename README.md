# (c,k)-Anonymisation Pipeline for Medical Data

> A research-grade, end-to-end privacy-preserving anonymisation pipeline combining **k-Anonymity** and **c-Diversity** for structured medical datasets.

> **Implementation note:** This project is an independent Python implementation based on the paper *"Privacy Preserving for Multiple Sensitive Attributes Against Fingerprint Correlation Attack Satisfying c-Diversity."* Since no official code was provided by the authors, this implementation was developed from scratch from the methods described in the paper.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Privacy Model](#2-privacy-model)
3. [Project Structure](#3-project-structure)
4. [Installation & Setup](#4-installation--setup)
5. [Configuration](#5-configuration)
6. [Usage & Reproducibility](#6-usage--reproducibility)
   - [Interactive Mode](#61-interactive-mode)
   - [Single Run — Auto k/c](#62-single-run--auto-kc)
   - [Single Run — Manual k/c](#63-single-run--manual-kc)
   - [Experiment Batches](#64-experiment-batches)
7. [Output Structure](#7-output-structure)
8. [Algorithm Details](#8-algorithm-details)
9. [Extending the Pipeline](#9-extending-the-pipeline)
10. [Dataset Provenance](#10-dataset-provenance)
11. [Credits & Collaborators](#11-credits--collaborators)
12. [License](#12-license)
13. [Citation](#13-citation)
14. [References](#14-references)

---

## 1. Overview

The **(c,k)-Anonymisation Pipeline** is a research tool for applying rigorous, auditable privacy-preserving transformations to structured medical datasets. It is designed to satisfy two complementary privacy guarantees simultaneously:

- **k-Anonymity** — every individual record is indistinguishable from at least *k* − 1 others based on quasi-identifier (QI) attributes, protecting against linkage attacks.
- **c-Diversity** — within each equivalence class, at least *c* distinct values are present for every sensitive attribute (SA), protecting against inference and homogeneity attacks.

The pipeline was developed as part of a research investigation into privacy-utility trade-offs in clinical data sharing. It is intended for use by researchers, data custodians, and privacy engineers who need a transparent, reproducible anonymisation workflow that meets the standards expected in medical informatics and data governance literature.

**Key properties:**

| Property | Detail |
|---|---|
| Privacy model | (c,k)-Anonymisation |
| Partitioning algorithm | Mondrian multidimensional k-anonymity |
| Parameter derivation | Automatic from dataset properties, or manual override |
| Output format | Two published CSV partitions (GT + SAFB) |
| Reproducibility | Configurable random seed |
| Configuration | Fully externalised to `configs/config.json` |
| Compliance context | Relevant to GDPR Art. 89, HIPAA Safe Harbour |

---

## 2. Privacy Model

This pipeline implements a two-partition publication model:

### Generalised Table (GT)

- One row per original record
- Quasi-identifier columns **generalised**: numerics become ranges (`"25–35"`), categoricals become sorted value sets (`"F, M"`)
- No sensitive attributes present
- Records within each equivalence class are k-anonymous

### Sensitive Attribute Feature Buckets (SAFB)

- One row per equivalence class (BucketID)
- Each SA column contains a comma-separated string of all unique values in that bucket
- Every bucket satisfies c-diversity: at least *c* distinct values per SA column
- No quasi-identifiers present

The two partitions are linked **only** by `BucketID`. There is no record-level linkage between them, providing structural separation between identity and sensitivity.

### c as a Positive Integer

In this implementation, **c is a positive integer** (not a fraction). A value of *c* = 3 means:

- Each equivalence class must contain at least **3 distinct values** for every sensitive attribute column.
- No single equivalence class may direct more than **1/c** of its records to any single ST bucket (c-correspondence constraint).

This reframing makes c directly interpretable as a diversity count, analogous to the *l* parameter in *l*-Diversity.

---

## 3. Project Structure

```
ck-anonymisation/
│
├── cli.py                        # Main entry point — all modes accessible from here
│
├── configs/
│   └── config.json               # All configuration: paths, columns, experiments
│
├── dataset/
│   └── sample_medical_dataset.csv   # Input dataset (not committed — see note)
│
├── src/
│   ├── ck_anonymisation.py       # Core 8-step anonymisation pipeline
│   └── run_experiments.py        # Batch experiment runner (reads config)
│
├── results/                      # Auto-created; one subfolder per run/experiment
│   ├── Exp_A_k3_c2/
│   │   ├── gt_partition.csv
│   │   └── safb_partition.csv
│   └── experiment_summary.csv
│
├── README.md
└── LICENSE
```

> **Note on data:** Raw medical datasets should never be committed to version control. The `dataset/` folder is included in `.gitignore`. See [Section 6](#6-usage--reproducibility) for instructions on preparing your own dataset.

---

## 4. Installation & Setup

### Prerequisites

- Python **3.10** or higher
- pip

### Clone the Repository

```bash
git clone https://github.com/srgdevs/ck-anonymisation.git
cd ck-anonymisation
```

### Install Dependencies

```bash
pip install pandas numpy
```

All other dependencies (`json`, `pathlib`, `argparse`, `math`, `random`, `warnings`, `datetime`) are part of the Python standard library.

### Verify Installation

```bash
python -c "import pandas, numpy; print('Dependencies OK')"
```

### Prepare Your Dataset

Place your CSV dataset in the `dataset/` folder and update `configs/config.json` accordingly:

```bash
cp /path/to/your/data.csv dataset/your_data.csv
```

Then edit `configs/config.json`:

```json
"file_paths": {
  "input_csv": "dataset/your_data.csv"
}
```

---

## 5. Configuration

All pipeline behaviour is controlled by `configs/config.json`. **No source code changes are required** when switching datasets or changing experiment designs.

```json
{
  "file_paths": {
    "input_csv": "dataset/sample_medical_dataset.csv"
  },

  "attributes": {
    "quasi_identifiers": ["gender", "age", "zip", "weight", "height"],
    "sensitive_attributes": [
      "occupation", "cancer_type", "cancer_treatment", "symptoms", "diagnosis_method"
    ]
  },

  "random_seed": 42,

  "experiments": [
    { "set": "A", "k": 3,  "c": 2 },
    { "set": "A", "k": 5,  "c": 2 },
    { "set": "A", "k": 7,  "c": 2 },
    { "set": "A", "k": 10, "c": 2 },
    { "set": "A", "k": 15, "c": 2 },
    { "set": "B", "k": 7,  "c": 3 },
    { "set": "B", "k": 7,  "c": 4 }
  ]
}
```

### Field Reference

| Field | Type | Description |
|---|---|---|
| `file_paths.input_csv` | string | Path to CSV input, relative to project root |
| `attributes.quasi_identifiers` | array | Columns that could identify individuals when linked to external data |
| `attributes.sensitive_attributes` | array | Columns containing private information to protect |
| `random_seed` | integer | Seed for reproducible Mondrian partitioning and c-constraint distribution |
| `experiments` | array | List of `{set, k, c}` objects for batch experiment runs |

### Experiment Sets

Each entry in `experiments` defines one batch run:

```json
{ "set": "A", "k": 10, "c": 3 }
```

- `set` — label used in the output folder name (e.g., `Exp_A_k10_c3`)
- `k` — minimum equivalence class size (integer ≥ 2)
- `c` — minimum distinct SA values per bucket per column (integer ≥ 1)

Add, remove, or modify entries freely. No code changes required.

---

## 6. Usage & Reproducibility

All modes are accessible through a single entry point:

```bash
python cli.py [OPTIONS]
```

### 6.1 Interactive Mode

The simplest option. Prompts guide you through every step.

```bash
python cli.py
```

**Session example:**

```
╔══════════════════════════════════════════════════════════╗
║         (c,k)-Anonymisation Pipeline — CLI               ║
╚══════════════════════════════════════════════════════════╝

── Choose run mode ─────────────────────────────────────────
    [1]  Anonymise   — run a single anonymisation job
    [2]  Experiments — run all batches defined in configs/config.json

  Your choice [1/2]: 1

── Step 1: Select the dataset to anonymise ─────────────────

  CSV files found in 'dataset/':

    [1]  sample_medical_dataset.csv
    [0]  Enter a custom file path

  Your choice (number or 0 for custom path): 1

── Step 2: Choose how k and c are determined ───────────────
    [1]  Auto   — derive k and c from the dataset properties
    [2]  Manual — specify k and c values yourself

  Your choice [1/2]: 1

── Run summary ──────────────────────────────────────────────
  File  : dataset/sample_medical_dataset.csv
  Mode  : auto
  k, c  : will be derived automatically
  QI    : ['gender', 'age', 'zip', 'weight', 'height']
  SA    : ['occupation', 'cancer_type', 'cancer_treatment', 'symptoms', 'diagnosis_method']

  Proceed? [Y/n]: y
```

---

### 6.2 Single Run — Auto k/c

k and c are derived automatically from dataset properties (size, dimensionality, SA diversity).

```bash
# Uses dataset from config.json
python cli.py --auto

# Explicit file override
python cli.py --file dataset/sample_medical_dataset.csv --auto
```

**Derivation logic:**

| Factor | Adjustment |
|---|---|
| Dataset size | base k = ceil(0.01 × n), clamped to [2, 20] |
| QI count | +1 per additional pair beyond 2 |
| QI uniqueness rate > 0.8 | ×1.5 |
| SA count ≤ 2 | base c = 2 |
| SA count 3–4 | base c = 3 |
| SA count 5–6 | base c = 4 |
| SA count > 6 | base c = 5 |
| Low SA diversity (avg ratio < 0.05) | +2 to c |

**Example output:**

```
[k Derivation]
  Total records          : 50000
  QI attributes          : 5
  Unique QI combinations : 42000
  Uniqueness rate        : 0.8400
  Base k (1% of n)       : 20
  Dimensionality-adj. k  : 21
  Final k (after unique) : 20

[c Derivation]
  Sensitive attributes   : 5
  Average diversity ratio: 0.0850
  Base c                 : 4
  Adjusted c             : 5
  Final c (after bounds) : 5
  Min ST buckets per GT  : 5

[Init] Privacy parameters confirmed: k=20, c=5
```

---

### 6.3 Single Run — Manual k/c

For precise parameter control in research experiments.

```bash
python cli.py --file dataset/sample_medical_dataset.csv --manual --k 10 --c 3
```

**Parameter constraints:**

| Parameter | Type | Constraint | Meaning |
|---|---|---|---|
| `--k` | integer | ≥ 2 | Minimum equivalence class size |
| `--c` | integer | ≥ 1 | Minimum distinct SA values per bucket |

**Prompt-assisted manual entry** (if `--k`/`--c` are omitted with `--manual`):

```bash
python cli.py --file dataset/data.csv --manual
# → Prompts for k then c interactively
```

---

### 6.4 Experiment Batches

Run all `(k, c)` combinations defined in `configs/config.json` in a single command. Results are saved to named folders and a summary CSV is generated automatically.

```bash
python cli.py --experiments
```

**What happens:**

1. Reads all experiment definitions from `configs/config.json`
2. For each `{set, k, c}`:
   - Checks if `results/Exp_{set}_k{k}_c{c}/` already exists → skips if so
   - Runs the full pipeline with those parameters
   - Saves `gt_partition.csv` + `safb_partition.csv` to the named folder
   - Collects metrics (bucket count, group sizes, etc.)
3. Saves `results/experiment_summary.csv` with all metrics side by side
4. Prints a formatted comparison table

**Example console output:**

```
============================================================
  Experiment 1/7: Exp_A_k3_c2  (k=3, c=2)
============================================================
  num_buckets    : 412
  avg_group_size : 121.36
  min_group_size : 3
  max_group_size : 1840

============================================================
  Experiment 2/7: Exp_A_k5_c2  (k=5, c=2)
============================================================
  num_buckets    : 289
  avg_group_size : 173.01
  min_group_size : 5
  max_group_size : 2104

...

====================================================================================================
  EXPERIMENT SUMMARY
====================================================================================================
 experiment_set   k  c      folder_name  total_records  num_buckets  ...
              A   3  2     Exp_A_k3_c2          50000          412  ...
              A   5  2     Exp_A_k5_c2          50000          289  ...
```

**Re-running experiments:**

Experiments whose output folder already exists are automatically skipped. To re-run a specific experiment, delete its folder:

```bash
rm -rf results/Exp_A_k10_c2
python cli.py --experiments
```

---

### Reproducibility

Set `random_seed` in `configs/config.json` before your first run. The same seed, dataset, and config will always produce identical GT and SAFB partitions.

```json
"random_seed": 42
```

To verify reproducibility:

```bash
# Run 1
python cli.py --file dataset/data.csv --auto

# Run 2 (should produce identical output)
python cli.py --file dataset/data.csv --auto
```

---

## 7. Output Structure

Each run produces a subfolder under `results/`. Single runs use a timestamp; experiment runs use the naming pattern `Exp_{set}_k{k}_c{c}`.

```
results/
├── 2026-03-26_15-30-45/          # Single run (auto/manual mode)
│   ├── gt_partition.csv
│   └── safb_partition.csv
│
├── Exp_A_k3_c2/                  # Experiment batch run
│   ├── gt_partition.csv
│   └── safb_partition.csv
│
├── Exp_A_k5_c2/
│   ├── gt_partition.csv
│   └── safb_partition.csv
│
└── experiment_summary.csv        # Aggregated metrics for all experiment runs
```

### `gt_partition.csv`

One row per original record. QI values are generalised; no sensitive attributes are present.

```csv
,BucketID,gender,age,zip,weight,height
0,1,M,25-35,120-125,60-70,160-170
1,1,M,25-35,120-125,60-70,160-170
2,1,M,25-35,120-125,60-70,160-170
3,2,F,36-45,200-210,55-65,150-160
```

### `safb_partition.csv`

One row per equivalence class. SA values are comma-separated strings of unique values in that bucket.

```csv
BucketID,occupation,cancer_type,cancer_treatment,symptoms,diagnosis_method
1,"Doctor, Engineer, Nurse","Melanoma, Lymphoma","Surgery, Chemo","Pain, Nausea","Biopsy, Imaging"
2,"Teacher, Professor","Breast, Ovarian","Surgery, Radio","Pain, Weakness","Palpation, Imaging"
```

### Joining the Partitions

```python
import pandas as pd

gt   = pd.read_csv('results/Exp_A_k10_c3/gt_partition.csv', index_col=0)
safb = pd.read_csv('results/Exp_A_k10_c3/safb_partition.csv')

joined = gt.join(safb.set_index('BucketID'), on='BucketID')
```

### `experiment_summary.csv`

| Column | Description |
|---|---|
| `experiment_set` | Set label from config (`A`, `B`, …) |
| `k` | k value used |
| `c` | c value used |
| `folder_name` | Output folder name |
| `total_records` | Total rows in GT |
| `num_buckets` | Number of equivalence classes |
| `groups_at_k` | Buckets with exactly k records |
| `groups_above_k` | Buckets with more than k records |
| `min_group_size` | Smallest bucket size |
| `max_group_size` | Largest bucket size |
| `avg_group_size` | Mean bucket size |
| `safb_buckets` | Rows in SAFB (should equal `num_buckets`) |

---

## 8. Algorithm Details

The pipeline runs 8 sequential steps on every invocation.

| Step | Function | Description |
|---|---|---|
| 0 | `initialise_privacy_parameters` | Derive or validate k and c |
| 1 | `load_dataset` | Load CSV, validate required columns |
| 2 | `build_gt_partition` | Mondrian recursive k-anonymity partitioning |
| 3 | `build_st_partition` | Extract sensitive attributes + BucketID |
| 4 | `enforce_c_diversity` | Build SAFB; ensure ≥ c distinct values per SA per bucket |
| 5 | `enforce_c_constraint` | Distribute GT records across ≥ c ST buckets |
| 6 | `validate_k_anonymity` | Assert every equivalence class has ≥ k records |
| 7 | `export_partitions` | Write GT and SAFB to CSV (with integrity gate) |
| 8 | `print_summary_report` | Final validation + summary statistics |

### Mondrian Partitioning (Step 2)

The GT partition is built using the **Mondrian multidimensional k-anonymity** algorithm (LeFevre et al., 2006):

1. Choose the QI attribute with the most distinct values in the current partition
2. Sort by that attribute and split at the median
3. Recurse on both halves if each has ≥ k records; otherwise stop
4. Generalise QI values: numeric → `"min–max"` range; categorical → sorted comma-separated list

### c-Diversity Enforcement (Step 4)

For each equivalence class:
- If any SA column has fewer than *c* distinct values, the deficit is patched by sampling additional values from existing unique values in that bucket
- The SAFB row for that class consolidates all unique values into a single comma-separated string per SA column

### c-Correspondence Constraint (Step 5)

For each GT bucket of size *n*:
- Records are distributed across at least *c* ST buckets
- No single ST bucket may receive more than ⌊n/c⌋ records from one GT bucket
- This prevents an adversary from correlating GT and ST buckets probabilistically

---

## 9. Extending the Pipeline

### Adding a New Dataset

1. Place the CSV in `dataset/`
2. Update `configs/config.json`: set `file_paths.input_csv`, `quasi_identifiers`, and `sensitive_attributes`
3. Run as normal — no code changes required

### Adding New Experiments

Add entries to the `experiments` array in `configs/config.json`:

```json
"experiments": [
  { "set": "C", "k": 20, "c": 5 },
  { "set": "C", "k": 20, "c": 7 }
]
```

Then run:

```bash
python cli.py --experiments
```

### Using the Pipeline Programmatically

```python
import sys
sys.path.insert(0, 'src')
import ck_anonymisation as ck

df      = ck.load_dataset('dataset/data.csv', qi_cols, sa_cols)
gt_df   = ck.build_gt_partition(df, qi_cols, k=10)
st_df   = ck.build_st_partition(df, gt_df, sa_cols)
safb_df = ck.enforce_c_diversity(st_df, sa_cols, c=3)
ck.export_partitions(gt_df, safb_df, output_folder)
```

---

## 10. Dataset Provenance

The sample medical dataset used in this repository was created synthetically by the project author.

Its structure and generation approach were informed by the IBM Telco Customer Churn sample documentation:

- https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn

No real patient records are included in this project dataset.

---

## 11. Credits & Collaborators

### Authors

| Name | Role | Institution |
|---|---|---|
| **Seyed R. Ghorashi** | Lead researcher & software developer | Charles Sturt University |
| **Tanveer Zia** | Principal Supervisor (research supervision, methodological guidance, conceptual feedback, and manuscript review) | University of Notre Dame Australia |
| **Michael Bewong** | Co-supervisor (research guidance, conceptual feedback, and manuscript review) | Charles Sturt University |

### Acknowledgements

This research was conducted as part of postgraduate research at **Charles Sturt University**. The author acknowledges the academic supervision and manuscript support provided by **Tanveer Zia** and **Michael Bewong**.

The work has been supported by the Cyber Security Research Centre Limited, whose activities are partially funded by the Australian Government's Cooperative Research Centres Programme P23_000_19_0027.

The synthetic medical dataset used for testing was generated independently and contains no real patient data.

> To add yourself as a contributor, please open a pull request or contact the corresponding author.

---

## 12. License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Ramin Ghorashi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

See the [LICENSE](LICENSE) file for the full text.

---

## 13. Citation

If you use this pipeline in your research, please cite it as:

```bibtex
@software{seyedrghorashi2026ck,
  author    = {Seyed R. Ghorashi},
  title     = {(c,k)-Anonymisation Pipeline for Medical Data},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/srgdevs/ck-anonymisation},
  note      = {Version 1.3}
}
```

Or in APA format:

> Ghorashi, S. R. (2026). *(c,k)-Anonymisation Pipeline for Medical Data* [Software]. GitHub. https://github.com/srgdevs/ck-anonymisation

---

## 14. References

The following works underpin the theoretical foundations of this pipeline:

**k-Anonymity:**

> Sweeney, L. (2002). k-anonymity: A model for protecting privacy. *International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems*, *10*(5), 557–570. https://doi.org/10.1142/S0218488502001648

> Samarati, P., & Sweeney, L. (1998). Protecting privacy when disclosing information: k-anonymity and the aggregation problem. *Technical Report, SRI International*.

**Mondrian Multidimensional k-Anonymity:**

> LeFevre, K., DeWitt, D. J., & Ramakrishnan, R. (2006). Mondrian multidimensional k-anonymity. In *Proceedings of the 22nd International Conference on Data Engineering (ICDE 2006)* (p. 25). IEEE. https://doi.org/10.1109/ICDE.2006.101

**l-Diversity (conceptual basis for c as a diversity count):**

> Machanavajjhala, A., Kifer, D., Gehrke, J., & Venkitasubramaniam, M. (2007). l-diversity: Privacy beyond k-anonymity. *ACM Transactions on Knowledge Discovery from Data (TKDD)*, *1*(1), Article 3. https://doi.org/10.1145/1217299.1217302

**c-Diversity with Multiple Sensitive Attributes / Fingerprint Correlation Attack:**

> Khan, R., Tao, X., Anjum, A., Sajjad, H., Malik, S. ur R., Khan, A., & Amiri, F. (2020). Privacy preserving for multiple sensitive attributes against fingerprint correlation attack satisfying c-diversity. *Wireless Communications and Mobile Computing*, *2020*, 8416823. https://doi.org/10.1155/2020/8416823

**Privacy in Medical Data / Regulatory Context:**

> El Emam, K., & Malin, B. (2015). Appendix B: Concepts and methods for de-identifying clinical trial data. In *Clinical and Translational Science: Principles of Human Research* (2nd ed.). Academic Press.

> European Parliament. (2016). *Regulation (EU) 2016/679 of the European Parliament and of the Council (General Data Protection Regulation)*. Official Journal of the European Union.

> U.S. Department of Health and Human Services. (2012). *Guidance regarding methods for de-identification of protected health information in accordance with the Health Insurance Portability and Accountability Act (HIPAA) Privacy Rule*. HHS.

---

<p align="center">
  Built for reproducible medical data privacy research · MIT Licensed · 2026
</p>
