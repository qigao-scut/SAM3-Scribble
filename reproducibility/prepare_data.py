"""Run the original CMPB conversion with explicit source/output paths."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["acdc-source", "btcv-source", "promise12-source", "output-root"]:
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    for src in [args.acdc_source, args.btcv_source, args.promise12_source]:
        if not src.is_dir():
            parser.error(f"Missing source directory: {src}")
    out = args.output_root.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Output root must be new or empty; existing data will not be replaced")
    if any(out == src.resolve() or out in src.resolve().parents or src.resolve() in out.parents
           for src in [args.acdc_source, args.btcv_source, args.promise12_source]):
        parser.error("Output and source directories must not overlap")
    from gq_scripts.preprocess import build_cmpb_clean_datasets as builder
    builder.ACDC_PROCESSED = args.acdc_source.resolve()
    builder.BTCV_NIFTI = args.btcv_source.resolve()
    builder.PROMISE_NIFTI = args.promise12_source.resolve()
    builder.ACDC_OUT, builder.BTCV_OUT, builder.PROMISE_OUT = (
        out / "acdc", out / "btcv", out / "promise12")
    builder.main()


if __name__ == "__main__":
    main()
