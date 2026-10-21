# Isolated slice error audit

This post-hoc audit uses the retained seed-123 main SAM3-Scribble and SAM3-Full predictions at confidence threshold 0.7. It does not run inference, retrain models, alter predictions or apply smoothing. Both models use slice-wise 2D inference; SAM3-Full is a dense-supervision reference, not a 3D control.

## Definitions

A mask is nonempty if it contains at least one positive pixel. Only centers with both neighboring slices present at consecutive slice indices (z-1, z, z+1) are examined; the first and last stored slices are excluded. Categories are evaluated independently, with the same volume-class inclusion rules as the original 3D evaluation. True positives occurring only on a single slice are not counted as false-positive islands.

- `positive_gt_triplet`: eligible centers where the ground-truth target is nonempty on all three consecutive slices.
- `single_slice_gap`: among positive GT triplets, the prediction is nonempty on both neighbors and empty only at the center. Longer gaps, partial undersegmentation and border-slice misses are outside this definition.
- `absent_gt_center`: eligible interior centers where the ground-truth target is empty at the center.
- `isolated_false_positive`: among absent-GT centers, the prediction is nonempty only at the center and empty on both neighbors.
- `gap_percent`: 100 * single_slice_gap / positive_gt_triplet.
- `isolated_fp_percent`: 100 * isolated_false_positive / absent_gt_center.
- `pairs_with_gap` and `pairs_with_isolated_fp`: numbers of volume-class pairs containing at least one corresponding event. These are not numbers of distinct patients.
- `slices` and `interior_centers` count slice-class queries, not unique image slices. ACDC has 30 time-frame volumes from 15 patients; BTCV_cervix has 6 volumes, and PROMISE12 has 30 volumes.

## SAM3-Scribble results

| Dataset | Single-slice gaps / eligible positive-GT centers | Isolated false positives / eligible GT-negative centers |
|---|---:|---:|
| ACDC | 0/545 (0.00%) | 0/14 (0.00%) |
| BTCV_cervix | 6/999 (0.60%) | 2/3153 (0.06%) |
| PROMISE12 | 0/358 (0.00%) | 2/320 (0.63%) |

`slice_continuity_summary.csv` includes both models. The full local audit additionally provides `per_slice.csv`, `per_volume_class.csv`, and `source_manifest.json`. For every volume-class pair in all six groups, foreground voxel counts and reconstructed 3D Dice were verified against the retained evaluation JSON (Dice error below 1e-12).

These descriptive counts quantify two specific error types. They do not measure general boundary roughness, all disconnected 3D components, or a causal penalty attributable solely to 2D processing. No inference of a 3D-model improvement or a statistical difference between models is made.

Reproducing the analysis requires the retained prediction files and dense test annotations; these are not bundled with the summary CSV. The CSV reports slice-class event counts and their explicit denominators, without confidence intervals or patient-independence assumptions.
