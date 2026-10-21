# Static-baseline protocols

ScribFormer, EFFDNet and the ScribbleBench nnUNet+pL baseline are trained separately for each dataset, whereas SAM3 is jointly adapted across datasets. All use the fixed memberships in [`splits/`](splits/README.md). Author-released code and method-specific training configurations are used with adaptations for data loading, fixed partitions and evaluation.

| Baseline | Retained training settings | Validation / selection |
|---|---|---|
| ScribFormer | 2D, 256×256 patches; batch 8; 200 epochs; seed 2022; AdamW, learning rate 0.001, weight decay 0.0005; CAM weight 0.1, subweights 0.25/0.5/0.75/1.0 | Dense validation masks; highest native mean Dice |
| EFFDNet | 2D U-Net, 256×256 patches; batch 12; 60000 iterations; seed 2022; SGD, learning rate 0.01, momentum 0.9, weight decay 0.0001; polynomial exponent 0.9; FADC/FBSL enabled, lambda 0.6, contrastive weight 0.3 | Dense validation masks, every 200 iterations; highest native mean Dice |
| ScribbleBench nnUNet+pL | nnU-Net v2 ignore-label baseline, 3d_fullres, fixed fold 0; 1000 epochs, 250 iterations/epoch; SGD, learning rate 0.01, momentum 0.99 with Nesterov, weight decay 0.00003; polynomial exponent 0.9 | Sparse validation masks with ignore masking; native online EMA pseudo-Dice |

nnU-Net planned batch sizes are 4/2/2 and patch sizes are 20×256×224, 64×192×160 and 28×256×256 for ACDC/BTCV_cervix/PROMISE12. The fixed fold excludes validation cases from gradient updates. SAM3 uses dense validation masks; common partitions do not imply identical checkpoint-selection metrics.

Code sources: ScribFormer author implementation (`HUANGLIZI/ScribFormer`), EFFDNet author implementation (`Aurora-003-web/EFFDNet`), and nnU-Net (`MIC-DKFZ/nnUNet`) with ignore-label supervision as specified by ScribbleBench. Available local base commits are `55d1a92abe26`, `3640367f38f9` and `6ed222f6c41f`, respectively; local adaptations must accompany the final release.

## Preparation status

These settings describe the retained comparison runs. The nnU-Net procedure specifies the best checkpoint, but the saved prediction arguments do not establish whether the retained predictions loaded best or final weights; this identity remains to be verified before the release is finalized. Full source snapshots and portable baseline commands are still being assembled. This document does not establish equal hyperparameter-search budgets across methods.
