"""Check package integrity and fixed partitions without loading models or data."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    if not args.skip_hashes:
        for line in (ROOT / "SHA256SUMS").read_text().splitlines():
            expected, relative = line.split("  ", 1)
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            if expected != actual:
                raise ValueError(f"Checksum mismatch: {relative}")
    manifest = json.loads((ROOT / "reproducibility/splits/split_manifest.json").read_text())
    # The manifest's format is checked against the plain memberships below.
    for dataset, counts in {"acdc": (70, 15, 15), "btcv": (20, 4, 6), "promise12": (40, 10, 30)}.items():
        groups = []
        for split, count in zip(["train", "val", "test"], counts):
            ids = (ROOT / f"reproducibility/splits/{dataset}_{split}.txt").read_text().splitlines()
            if len(ids) != count or len(set(ids)) != count:
                raise ValueError(f"Invalid split: {dataset}/{split}")
            if manifest[dataset][split] != ids:
                raise ValueError(f"Manifest mismatch: {dataset}/{split}")
            groups.append(set(ids))
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"Overlapping source-qualified IDs: {dataset}")
    for path in (ROOT / "sam3/train/configs/final").glob("*.yaml"):
        text = path.read_text()
        if "/home/" in text:
            raise ValueError(f"Nonportable configuration: {path.name}")
    weak = (ROOT / "sam3/train/configs/final/weak_lora.yaml").read_text()
    if "PartialMasks" not in weak or "ignore_index: 255" not in weak:
        raise ValueError("Weak-supervision configuration must ignore unknown pixels")
    metadata = json.loads((ROOT / "reproducibility/release_metadata.json").read_text())
    checkpoints = json.loads((ROOT / "checkpoints/manifest.json").read_text())
    if len(checkpoints) != 2 or {x["model"] for x in checkpoints} != {"SAM3-Scribble", "SAM3-Full"}:
        raise ValueError("The release must identify both main checkpoints")
    expected_checksums = "".join(f"{x['sha256']}  {x['filename']}\n" for x in checkpoints)
    if (ROOT / "checkpoints/SHA256SUMS").read_text() != expected_checksums:
        raise ValueError("Checkpoint checksum list differs from the manifest")
    for item in checkpoints:
        if item["download_url"] != metadata["checkpoint_share_url"] or item["bundled"]:
            raise ValueError("Checkpoints must use the documented external download link")
    for relative in ["README.md", "checkpoints/README.md"]:
        if metadata["checkpoint_share_url"] not in (ROOT / relative).read_text():
            raise ValueError(f"Checkpoint link missing from {relative}")
    if "SAM License" not in (ROOT / "LICENSE").read_text():
        raise ValueError("The upstream SAM License must accompany this release")
    print("Package checks passed; no training or inference was run.")


if __name__ == "__main__":
    main()
