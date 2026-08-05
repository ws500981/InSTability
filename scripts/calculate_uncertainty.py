#!/usr/bin/env python3
"""Calculate spot-level clustering instability from repeated partitions."""

from argparse import ArgumentParser
from pathlib import Path

import joblib
import numpy as np
import scanpy as sc


SAMPLES = ("151671", "151673")


def calculate_instability(partitions):
    """Exact instability calculation without an N x N co-association matrix."""

    n_runs = len(partitions)
    n_spots = len(partitions[0])

    # Encode arbitrary cluster labels as compact integers.
    codes = []
    for labels in partitions:
        labels = np.asarray(labels)

        if len(labels) != n_spots:
            raise ValueError("All partitions must contain the same spots.")

        _, c = np.unique(labels, return_inverse=True)
        codes.append(c.astype(np.int32, copy=False))

    # uint32 is sufficient if R * N < 2**32.
    if n_runs * n_spots >= np.iinfo(np.uint32).max:
        raise ValueError("n_runs * n_spots exceeds uint32 capacity.")

    row_sums_counts = np.zeros(n_spots, dtype=np.uint32)
    intra_sum_counts = np.zeros(
        (n_runs, n_spots),
        dtype=np.uint32,
    )

    # --------------------------------------------------------------
    # Direct exact count calculation: O(R^2 N)
    # --------------------------------------------------------------

    for q in range(n_runs):
        cq = codes[q]

        cluster_sizes_q = np.bincount(cq).astype(
            np.uint32,
            copy=False,
        )
        row_sums_counts += cluster_sizes_q[cq]

        n_q_clusters = len(cluster_sizes_q)

        for r in range(q + 1):
            cr = codes[r]

            n_r_clusters = int(cr.max()) + 1

            if (
                n_r_clusters * n_q_clusters
                <= np.iinfo(np.int32).max
            ):
                pair_key = cr * np.int32(n_q_clusters)
                pair_key += cq
            else:
                pair_key = cr.astype(np.int64) * n_q_clusters
                pair_key += cq

            pair_counts = np.bincount(pair_key).astype(
                np.uint32,
                copy=False,
            )

            contribution = pair_counts[pair_key]
            contribution -= np.uint32(1)

            intra_sum_counts[r] += contribution

            if r != q:
                intra_sum_counts[q] += contribution

    # --------------------------------------------------------------
    # Instability calculation
    # --------------------------------------------------------------

    row_sums_f = row_sums_counts.astype(np.float32)
    spot_score_sum = np.zeros(n_spots, dtype=np.float32)
    global_stability = []

    for run in range(n_runs):
        labels = codes[run]

        intra_sum_f = intra_sum_counts[run].astype(np.float32)

        intra_sum = intra_sum_f / n_runs
        inter_sum = (
            row_sums_f - intra_sum_f - n_runs
        ) / n_runs

        cluster_sizes = np.bincount(labels).astype(
            np.uint32,
            copy=False,
        )
        sizes_per_spot = cluster_sizes[labels]

        valid = sizes_per_spot > 1

        intra_count = sizes_per_spot - np.uint32(1)
        inter_count = np.uint32(n_spots) - sizes_per_spot

        # Singleton behavior
        inter_count[~valid] = 0

        mean_intra = np.divide(
            intra_sum,
            intra_count,
            out=np.ones_like(intra_sum),
            where=intra_count != 0,
        )

        mean_inter = np.divide(
            inter_sum,
            inter_count,
            out=np.zeros_like(inter_sum),
            where=inter_count != 0,
        )

        spot_score_sum += 1.0 - (mean_intra - mean_inter)

        valid_intra = intra_count > 0
        valid_inter = inter_count > 0

        if valid_intra.any() and valid_inter.any():
            mean_intra_global = (
                intra_sum[valid_intra].sum()
                / intra_count[valid_intra].sum()
            )

            mean_inter_global = (
                inter_sum[valid_inter].sum()
                / inter_count[valid_inter].sum()
            )

            global_stability.append(
                mean_intra_global - mean_inter_global
            )

    if not global_stability:
        raise ValueError("No valid intra/inter comparisons found.")

    spot_instability = spot_score_sum / np.float32(n_runs)
    sample_instability = 1.0 - float(np.mean(global_stability))

    return spot_instability, sample_instability


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--clustering-dir",
        type=Path,
        default=Path("results/leiden"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/uncertainty"),
    )
    parser.add_argument(
        "--run-counts",
        type=int,
        nargs="+",
        default=[20],
        help="Numbers of leading runs to use.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sample in SAMPLES:
        partition_path = (
            args.clustering_dir
            / f"{sample}_leiden_results_"
              f"{args.max_iterations}iterations.pkl"
        )

        all_partitions = joblib.load(partition_path)

        adata = sc.read_h5ad(
            args.data_dir / f"dlpfc_{sample}_filtered.h5ad"
        )

        for n_runs in args.run_counts:
            if n_runs < 1 or n_runs > len(all_partitions):
                raise ValueError(
                    f"Requested {n_runs} runs, but "
                    f"{partition_path} contains "
                    f"{len(all_partitions)}."
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
                / (
                    f"dlpfc_{sample}_leiden_"
                    f"uncertainty_{n_runs}runs.pkl"
                ),
            )

            adata.obs[
                f"leiden_uncertainty_{n_runs}runs"
            ] = spot

            adata.uns[
                f"leiden_uncertainty_{n_runs}runs"
            ] = sample_score

            print(
                f"{sample}, {n_runs} runs: "
                f"mean instability={sample_score:.4f}"
            )

        adata.write_h5ad(
            args.output_dir
            / f"dlpfc_{sample}_leiden_uncertainty.h5ad"
        )


if __name__ == "__main__":
    main()
