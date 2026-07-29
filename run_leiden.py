#!/usr/bin/env python3
"""Generate repeated Leiden partitions for DLPFC 151671 and 151673."""

from argparse import ArgumentParser
from pathlib import Path
import warnings

import joblib
import numpy as np
import scanpy as sc

from functions import preprocessing, search_leiden_resolution


SAMPLE_CLUSTERS = {"151671": 5, "151673": 7}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/leiden")
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sample, target_clusters in SAMPLE_CLUSTERS.items():
        input_path = args.data_dir / f"dlpfc_{sample}_filtered.h5ad"
        print("\n" + "=" * 70)
        print(f"Starting DLPFC {sample}")
        print(f"Loading: {input_path}")
        adata = sc.read_h5ad(input_path)
        print(f"Loaded {adata.n_obs:,} spots x {adata.n_vars:,} genes")

        adata.raw = adata.copy()
        print("Selecting highly variable genes and preprocessing...")
        adata = preprocessing(
            adata, highly_var=True, flavor="seurat_v3", n_hvg=3000
        )
        print(
            f"Preprocessing complete: "
            f"{adata.n_obs:,} spots x {adata.n_vars:,} HVGs"
        )

        print("Applying PCA with up to 50 components...")
        rng = np.random.RandomState(args.base_seed)
        sc.pp.pca(
            adata,
            n_comps=min(50, adata.n_obs - 1, adata.n_vars - 1),
            svd_solver="arpack",
            random_state=rng,
        )
        print(f"PCA complete: {adata.obsm['X_pca'].shape[1]} components")

        print("Constructing the 50-nearest-neighbor graph...")
        sc.pp.neighbors(
            adata,
            use_rep="X_pca",
            n_neighbors=50,
            random_state=rng,
        )
        print("Neighbor graph complete.")

        print(
            f"Searching for a Leiden resolution near "
            f"{target_clusters} clusters..."
        )
        resolution = search_leiden_resolution(
            adata.copy(),
            target_clusters,
            use_rep="X_pca",
            start=0.1,
            end=1.7,
            increment=0.01,
        )
        print(f"Selected Leiden resolution: {resolution:.4f}")
        print(
            f"Running Leiden {args.iterations} times with seeds "
            f"{args.base_seed} through "
            f"{args.base_seed + args.iterations - 1}..."
        )

        partitions = []
        for run in range(args.iterations):
            seed = args.base_seed + run
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        "In the future, the default backend for leiden "
                        "will be igraph.*"
                    ),
                    category=FutureWarning,
                )
                sc.tl.leiden(
                    adata,
                    key_added="leiden",
                    random_state=seed,
                    resolution=resolution,
                    flavor="leidenalg",
                )
            labels = adata.obs["leiden"].astype(int).to_numpy(copy=True)
            partitions.append(labels)
            print(
                f"{sample}: run {run + 1}/{args.iterations}, "
                f"seed={seed}, clusters={np.unique(labels).size}"
            )

        joblib.dump(
            partitions,
            args.output_dir
            / f"{sample}_leiden_results_{args.iterations}iterations.pkl",
        )
        joblib.dump(
            {
                "resolution": resolution,
                "target_clusters": target_clusters,
                "iterations": args.iterations,
                "base_seed": args.base_seed,
            },
            args.output_dir / f"{sample}_leiden_metadata.pkl",
        )
        print(f"Finished DLPFC {sample}.")


if __name__ == "__main__":
    main()
