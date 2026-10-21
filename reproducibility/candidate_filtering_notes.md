# Candidate filtering counts

Counts refer to the retained seed-123 SAM3-Scribble and SAM3-Full checkpoints on all three fixed test datasets. A separate counting inference pass reproduced all 11,682 saved final slice-class masks exactly. The CSV reports each model/dataset separately; no original predictions were replaced.

Each slice-class query produces 200 decoder candidates. Candidate scores are the sigmoid query probability multiplied by the sigmoid presence probability. A candidate is retained only when this score is strictly greater than 0.7. If several candidates survive, only the highest-scoring candidate is selected. The selected mask is binarized at probability greater than 0.5. If no candidate is retained, an empty prediction is returned.

CSV fields:

- `slice_class_queries`: slice-class combinations, including slices where the queried class is absent; these are not distinct image counts.
- `candidates`: all decoder candidates before filtering, equal to 200 times `slice_class_queries`.
- `threshold_discarded`: candidates with scores at or below 0.7.
- `retained`: candidates surviving the threshold.
- `top1_discarded`: additional surviving candidates discarded by highest-score selection.
- `selected`: selected candidates, at most one per slice-class query.
- `no_candidate`: queries with no candidate above threshold.
- `final_empty`: queries whose final binary prediction is empty, including target-absent slices. These are not necessarily errors. In this audit, `final_empty` equals `no_candidate` in every group.
- `volume_classes` and `empty_volume_classes`: evaluated 3D volume-class combinations and those with no predicted foreground on any slice. These use a different denominator from slice-class counts. ACDC has 30 time-frame volumes from 15 patients, contributing 90 volume-class pairs; BTCV_cervix has 24 pairs, and PROMISE12 has 30 pairs. All evaluated volume-class pairs have ground-truth target foreground somewhere in the volume; whole-volume absent classes are outside this evaluation.
- `compared_slice_masks` and `different_slice_masks`: number of final masks compared against the retained predictions and the number that differed (zero in all six groups).

Accounting identities: candidates = threshold_discarded + retained; retained = top1_discarded + selected; slice_class_queries = selected + no_candidate. Rejection proportions are calculated over all candidates; empty slice proportions use slice-class queries. Neither is a segmentation accuracy measure. Table standard deviations describe test-volume variation; these counts do not measure training-run stability.

Across three datasets, each model has 5,841 slice-class queries and 1,168,200 candidates. SAM3-Scribble has 1,165,923 threshold rejections (99.81%), 102 additional top-1 rejections and 3,666 empty slice-class outputs. SAM3-Full has 1,166,270 threshold rejections (99.83%), 3 additional top-1 rejections and 3,914 empty slice-class outputs. Both models have zero entirely empty 3D volume-class predictions.
