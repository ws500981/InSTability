#!/usr/bin/env python3
"""Download and filter DLPFC samples 151671 and 151673."""

from argparse import ArgumentParser
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import scanpy as sc

from functions import get_stats, qc_spot_and_gene_removal


SAMPLES = ("151671", "151673")
BASE_URL = (
    "https://huggingface.co/datasets/han-shu/st_datasets/"
    "resolve/main/DLPFC"
)
TRUTH_BASE_URL = (
    "https://zenodo.org/records/7591162/files"
)
GROUND_TRUTH_SOURCE = "https://zenodo.org/records/7591162"


def read_ground_truth(path: Path) -> pd.Series:
    """Read a two-column, headerless barcode-to-layer file."""
    truth = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["barcode", "annotation"],
        dtype=str,
        keep_default_na=False,
    )
    if truth["barcode"].duplicated().any():
        duplicates = truth.loc[
            truth["barcode"].duplicated(), "barcode"
        ].tolist()
        raise ValueError(
            f"Duplicate barcodes in {path}: {duplicates[:5]}"
        )

    truth["annotation"] = truth["annotation"].replace("", pd.NA)
    return truth.set_index("barcode")["annotation"]


def add_ground_truth(adata, annotations: pd.Series, sample: str):
    missing_barcodes = adata.obs_names.difference(annotations.index)
    if len(missing_barcodes):
        raise ValueError(
            f"{len(missing_barcodes)} spots in DLPFC {sample} are absent "
            f"from the ground-truth file. Examples: "
            f"{missing_barcodes[:5].tolist()}"
        )

    adata.obs["annotation"] = annotations.reindex(
        adata.obs_names
    ).to_numpy()
    adata.uns["ground_truth_source"] = GROUND_TRUTH_SOURCE

    annotated = int(adata.obs["annotation"].notna().sum())
    print(
        f"Attached ground truth to {len(adata.obs):,} spots; "
        f"{annotated:,} have a layer label."
    )
    return adata


def filter_sample(adata):
    adata.var_names_make_unique()

    sc.pp.calculate_qc_metrics(
        adata, log1p=False, percent_top=None, inplace=True
    )
    adata.raw = adata.copy()
    return qc_spot_and_gene_removal(
        adata,
        umi_threshold=600,
        gene_threshold=20,
        remove_mt=True,
        max_pct_mt=25,
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use raw files already present under DATA_DIR/raw.",
    )
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw"
    truth_dir = args.data_dir / "ground_truth"
    processed_dir = args.data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for sample in SAMPLES:
        raw_path = raw_dir / f"DLPFC_{sample}.h5ad"
        truth_path = truth_dir / f"{sample}_truth.txt"
        output_path = processed_dir / f"dlpfc_{sample}_filtered.h5ad"

        if not raw_path.exists():
            if args.no_download:
                raise FileNotFoundError(raw_path)
            print(f"Downloading DLPFC {sample}...")
            urlretrieve(f"{BASE_URL}/DLPFC_{sample}.h5ad", raw_path)

        if not truth_path.exists():
            if args.no_download:
                raise FileNotFoundError(truth_path)
            print(f"Downloading ground truth for DLPFC {sample}...")
            urlretrieve(
                f"{TRUTH_BASE_URL}/{sample}_truth.txt?download=1",
                truth_path,
            )

        print(f"\n{sample} before filtering:")
        adata = sc.read_h5ad(raw_path)
        annotations = read_ground_truth(truth_path)
        adata = add_ground_truth(adata, annotations, sample)
        get_stats(adata)

        adata = filter_sample(adata)
        print(f"{sample} after filtering:")
        get_stats(adata)
        adata.write_h5ad(output_path)


if __name__ == "__main__":
    main()
