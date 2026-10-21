# Fixed case-level splits

| Dataset | Train | Validation | Test | Unit |
|---|---:|---:|---:|---|
| ACDC | 70 | 15 | 15 | Patients; both time frames stay together |
| BTCV_cervix | 20 | 4 | 6 | Cases |
| PROMISE12 | 40 | 10 | 30 | Source-qualified cases |

Each `<dataset>_<train|val|test>.txt` contains one identifier per line. `case_split_inventory.csv` combines all 210 entries with columns `dataset,split,case_id`. Read identifiers as strings to retain leading zeros. `split_manifest.json` contains the same memberships without machine-specific paths. `SHA256SUMS` identifies these generated files.

ACDC retains the provided expert-scribble split. BTCV_cervix reserves four validation cases from its 24-case development set. PROMISE12 uses a fixed 40/10 partition of its 50-case development source and 30 cases from the separate local test source. Its validation IDs are `Case04`, `Case05`, `Case09`, `Case18`, `Case21`, `Case28`, `Case29`, `Case34`, `Case35`, and `Case36`.

PROMISE12 training/validation IDs use `train_source/CaseNN`; test IDs use `test_source/CaseNN`, since the two sources reuse local names. These prefixes identify sources, not verified clinical patient identities. The original validation selection procedure and seed were not retained; reproduce the experiment using these fixed memberships without resampling.

These files are generated from the working experiment records under `gq_paper/cmpb/splits/`. Do not edit generated memberships independently. Regenerate or verify from the repository root:

```bash
python reproducibility/prepare_splits.py
python reproducibility/prepare_splits.py --check
```
