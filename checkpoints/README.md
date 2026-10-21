# Trained checkpoints

The retained main-table checkpoints are selected by validation Dice. They contain the complete adapted model state, including LoRA and task heads; they are not standalone LoRA deltas. Use the supplied inference loader and the protocol in the root README.

| Model | Download filename | Size (bytes) | Access |
|---|---|---:|---|
| SAM3-Scribble, seed 123 | `sam3_scribble_seed123.pt` | 3483214347 | [Quark Drive folder](https://pan.quark.cn/s/291caf207228) |
| SAM3-Full, seed 123 | `sam3_full_seed123.pt` | 3483214347 | [Quark Drive folder](https://pan.quark.cn/s/291caf207228) |

The folder is shared without an extraction code and is configured not to expire. It contains both checkpoints, their checksums and the license. If Quark prompts for sign-in or its client to download large files, follow its download instructions.

Download the two `.pt` files into this directory. `manifest.json` records their exact SHA-256 values. From this directory, verify the downloaded files before inference:

```bash
sha256sum -c SHA256SUMS
```

Obtain the official SAM3 initialization separately from its provider for training. These adapted checkpoints are distributed under the SAM License; a copy of the repository's `LICENSE` accompanies the separate weight-upload folder. Preserve that license with redistributed files.
