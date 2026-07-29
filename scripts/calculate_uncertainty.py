#!/usr/bin/env python3
"""Calculate spot-level clustering instability from repeated partitions."""

from argparse import ArgumentParser
from pathlib import Path

import joblib
import numpy as np
import scanpy as sc


SAMPLES = ("151671", "151673")


def coassociation_counts(partitions):
    partitions = [np.asarray(labels) for labels in partitions]
    n_spots = len(partitions[0])
    counts = np.zeros((n_spots, n_spots), dtype=np.uint8)

    if len(partitions) > np.iinfo(counts.dtype).max:
        raise ValueError("This implementation supports at most 255 runs.")

    for labels in partitions:
        if len(labels) != n_spots:
            raise ValueError("All partitions must contain the same spots.")
        for cluster in np.unique(labels):
            members = np.flatnonzero(labels == cluster)
            counts[np.ix_(members, members)] += 1

    return counts


def calculate_instability(partitions):
    """Return per-spot and sample-level instability.

    For every spot in every run, stability is its mean co-association with
    spots inside its current cluster minus its mean co-association with spots
    outside that cluster. Instability is one minus stability, averaged over
    runs.
    """
    partitions = [np.asarray(labels) for labels in partitions]
    n_runs = len(partitions)
    n_spots = len(partitions[0])
    coassoc = coassociation_counts(partitions).astype(np.float32) / n_runs
    scores = np.empty((n_runs, n_spots), dtype=np.float32)
    global_stability = []

    for run, labels in enumerate(partitions):
        run_intra_sum = 0.0
        run_intra_count = 0
        run_inter_sum = 0.0
        run_inter_count = 0

        for cluster in np.unique(labels):
            inside = np.flatnonzero(labels == cluster)
            outside = np.flatnonzero(labels != cluster)

            if len(inside) <= 1:
                mean_intra = np.ones(len(inside), dtype=np.float32)
            else:
                block = coassoc[np.ix_(inside, inside)]
                mean_intra = (block.sum(axis=1) - 1.0) / (len(inside) - 1)
                run_intra_sum += block.sum() - len(inside)
                run_intra_count += len(inside) * (len(inside) - 1)

            if len(outside) == 0:
                mean_inter = np.zeros(len(inside), dtype=np.float32)
            else:
                between = coassoc[np.ix_(inside, outside)]
                mean_inter = between.mean(axis=1)
                run_inter_sum += between.sum()
                run_inter_count += between.size

            scores[run, inside] = 1.0 - (mean_intra - mean_inter)

        if run_intra_count and run_inter_count:
            global_stability.append(
                run_intra_sum / run_intra_count
                - run_inter_sum / run_inter_count
            )

    spot_instability = scores.mean(axis=0)
    sample_instability = 1.0 - float(np.mean(global_stability))
    return spot_instability, sample_instability


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed")
    )
    parser.add_argument(
        "--clustering-dir", type=Path, default=Path("results/leiden")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/uncertainty")
    )
    parser.add_argument(
        "--run-counts",
        type=int,
        nargs="+",
        default=[20],
        help="Numbers of leading runs to use.",
    )
    parser.add_argument("--max-iterations", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sample in SAMPLES:
        partition_path = (
            args.clustering_dir
            / f"{sample}_leiden_results_{args.max_iterations}iterations.pkl"
        )
        all_partitions = joblib.load(partition_path)
        adata = sc.read_h5ad(
            args.data_dir / f"dlpfc_{sample}_filtered.h5ad"
        )

        for n_runs in args.run_counts:
            if n_runs < 1 or n_runs > len(all_partitions):
                raise ValueError(
                    f"Requested {n_runs} runs, but {partition_path} "
                    f"contains {len(all_partitions)}."
                )

            spot, sample_score = calculate_instability(
                all_partitions[:n_runs]
            )
            result = {
                "spot_uncertainty": spot,
                "uncertainty": sample_score,
                "iterations": n_runs,
            }
            joblib.dump(
                result,
                args.output_dir
                / f"dlpfc_{sample}_leiden_uncertainty_{n_runs}runs.pkl",
            )

            adata.obs[f"leiden_uncertainty_{n_runs}runs"] = spot
            adata.uns[f"leiden_uncertainty_{n_runs}runs"] = sample_score
            print(
                f"{sample}, {n_runs} runs: "
                f"mean instability={sample_score:.4f}"
            )

        adata.write_h5ad(
            args.output_dir / f"dlpfc_{sample}_leiden_uncertainty.h5ad"
        )


if __name__ == "__main__":
    main()
