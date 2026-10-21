# Data preparation

Use the fixed case identifiers in `splits/` without resampling. ACDC retains its supplied expert-scribble partition. PROMISE12 uses the supplied fixed 40/10 development partition and 30 test cases; source prefixes in the lists identify the corresponding source subsets.

The preparation entry point converts existing annotations; it does not regenerate ScribbleBench curves. Obtain the expert annotations for ACDC and the existing foreground/background scribble labels for BTCV_cervix and PROMISE12. Generator settings and label meanings are described in `scribbles.md`.

Expected inputs:

- ACDC processed expert source: `train/`, `val/`, `test/`, each with `images/`, `masks/`, `full_annotations.coco.json` (or `image_annotations.coco.json`), and `spacing_map.json`; train/val also contain `scribble_tmi_annotations.coco.json`. COCO paths must resolve within the corresponding split.
- BTCV source: `train/` and `test/`, each with `img/<id>-Image.nii.gz`, `gt/<id>-Mask.nii.gz`, `scribble_bench/<id>-Mask.nii.gz`, and `spacing_map.json`.
- PROMISE12 source: `train/` and `test/`, each with `img/CaseXX.nii.gz`, `gt/CaseXX_segmentation.nii.gz`, `scribble_bench/CaseXX_segmentation.nii.gz`, and `spacing_map.json`.

```bash
python reproducibility/prepare_data.py \
  --acdc-source /path/to/acdc_processed_expert \
  --btcv-source /path/to/btcv_nifti \
  --promise12-source /path/to/promise12_nifti \
  --output-root /path/to/new/prepared_data
```

The output root must be absent or empty. Outputs are `acdc/`, `btcv/` and `promise12/`. Existing source files are read and linked, so retain the source directories. Image/label conversion follows the original builder with all slices retained. MRI uses percentile clipping, CT uses level 40/width 400; source unknown labels remain ignored. Spacing files preserve the source NIfTI axis order as consumed by the supplied evaluator.

Training uses `train/scribble_annotations.coco.json`; dense validation/test use `image_annotations.coco.json`. Do not substitute dense labels for training scribbles. The original ACDC expert COCO source is a prerequisite for this converter; raw-to-expert annotation acquisition is handled by the data provider.
