# Native forward complexity comparison

| Model | Native input | Total (M) | Updated (M) | Counted GFLOPs | Peak GPU (GiB) |
|---|---|---:|---:|---:|---:|
| sam3_scribble | 1x3x1008x1008 | 848.16 | 11.13 | 6108.06 | 3.95 |
| sam3_full | 1x3x1008x1008 | 848.16 | 11.13 | 6108.06 | 3.95 |
| scribformer | 1x1x256x256 | 47.92 | 47.92 | 89.85 | 0.35 |
| effdnet | 1x1x256x256 | 1.81 | 1.81 | 5.94 | 0.04 |
| nnunet | 1x1x20x256x224 | 30.38 | 30.38 | 630.46 | 1.10 |
| biomedparse | 1x1x512x512 | 371.80 | 0.00 | 8804.07 | 1.58 |
| sat_pro | 1x3x288x288x96 | 586.23 | 0.00 | 54079.21 | 26.34 |
| voxtell | 1x1x192x192x192 | 4461.80 | 0.00 | 5057.43 | 7.47 |

ACDC-reference architectures; batch size 1, synthetic zero inputs. Input sizes are native and differ across methods, so these are workload-specific costs, not an equal-workload speed ranking. Static methods predict all classes; text-prompted measurements use one target. SAM3 includes image and text encoding; SAT-Pro/VoxTell use cached text queries. Total parameters include their auxiliary text encoders, while VoxTell's cached-query patch measurement excludes its separately offloaded text-encoding pass. Updated parameters describe adaptation performed in this study; pretrained text-prompted baselines were not fine-tuned.

FLOPs count two operations per multiply-accumulate using PyTorch's registered convolution, matrix multiplication and attention formulas; unregistered/elementwise operations are excluded. GPU memory is peak allocated memory for the benchmark process, including resident model/input allocations; it is not whole-device used or reserved memory. Disk IO, volume tiling/TTA and label resampling are excluded. Precision and framework version are listed in the CSV. No inference-speed claim is made from shared-server timings.
