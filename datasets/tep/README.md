# TEP dataset provenance and layout

## Paper dataset decision

The paper experiments use only `extended500` because the framework requires
independent-run train/validation/test partitions. This fixes the task to 21
classes (normal + IDV1--IDV20), 500 samples per source-training run, and 960
samples per source-test run. The `standard` files remain a loader-compatibility
target and must not be mixed into an `extended500` experiment to add IDV21.

## Active extended 500-run dataset

- Official source: Harvard Dataverse, DOI `10.7910/DVN/6C3JR1`
- Official originals: `extended500/raw/*.RData`
- Verified Parquet mirror: `extended500/parquet/*.parquet`
- Run-indexed working copies: `extended500/run_parquet/*.parquet`
- Variables: 41 XMEAS + 11 XMV = 52
- Classes: normal + IDV1--IDV20 (the extended release does not contain IDV21)
- Runs: 500 independent runs per class in both source train and source test
- Samples/run: 500 for training and 960 for testing

The four downloaded mirror files were checked against their published SHA-256
digests.  The fault-free files were additionally compared cell-for-cell with
the official RData objects.  `scripts/prepare_tep_extended.py` verifies the
hashes and repacks each complete run as one Parquet row group.  This enables
the loader to fetch a run lazily without materializing all 15.75 million rows.
The source files are not modified.

```text
extended500/
|-- raw/          # official Harvard Dataverse RData files
|-- parquet/      # immutable, hash-verified Parquet mirror
`-- run_parquet/  # memory-efficient files used by configs/data.yaml
```

Preparation and validation commands:

```powershell
python scripts/prepare_tep_extended.py
python scripts/check_data_pipeline.py --config configs/data.yaml --check-dataset
```

## Standard reference dataset (retained)

- Source: https://github.com/camaramm/tennessee-eastman-profBraatz
- Source revision: `0643318858ce7c87292bd59820b459e12a13d009`
- License: [upstream LICENSE](https://github.com/camaramm/tennessee-eastman-profBraatz/blob/0643318858ce7c87292bd59820b459e12a13d009/LICENSE) (also available at `source/tennessee-eastman-profBraatz/LICENSE` when the upstream repository is downloaded locally).
- Variables: 41 XMEAS + 11 XMV = 52
- Classes: normal (`d00`) + IDV1--IDV21

Directory layout:

```text
standard/train/d00.dat ... d21.dat
standard/test/d00_te.dat ... d21_te.dat
source/tennessee-eastman-profBraatz/   # unmodified source checkout
```

The source contains 22 training sequences and 22 testing sequences. Faulty
training sequences have shape `480 x 52`; `d00.dat` is stored transposed and is
loaded as `500 x 52`. Every testing sequence has shape `960 x 52`. The loader
handles the transposed normal-training file automatically.

This source does not contain a 22-train/6-test layout. Any experiment requiring
that count must provide a separate, provenance-documented manifest; silently
selecting six of these 22 test conditions would change class coverage.

Fault labels use onset sample 21 for training and 161 for testing (1-indexed).
Samples before onset are labeled normal; a window label is the label of its
last sample.

## Reproducibility warning

The standard dataset has only one training sequence per class. Therefore, a
strict run-level train/validation split cannot retain every class in both
subsets. Do not work around this by randomly splitting overlapping windows;
use a genuinely multi-run dataset or generate additional independent simulator
runs for final run-level experiments.

Validation command:

```powershell
python scripts/check_data_pipeline.py --config configs/data.yaml --check-dataset
```
