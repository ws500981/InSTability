"""Shared preprocessing helpers required by the DLPFC examples."""

import numpy as np
import scanpy as sc
import warnings
from scipy import sparse


def get_stats(adata) -> None:
    """Print UMI and detected-gene summary statistics per spot."""
    if "total_counts" not in adata.obs:
        sc.pp.calculate_qc_metrics(
            adata, log1p=False, percent_top=None, inplace=True
        )

    total = adata.obs["total_counts"]
    genes = adata.obs["n_genes_by_counts"]
    print(f"{adata.n_obs:,} spots x {adata.n_vars:,} genes")
    print(
        "UMIs: "
        f"sum={total.sum():.0f}, mean={total.mean():.2f}, "
        f"median={total.median():.2f}, min={total.min():.0f}, "
        f"max={total.max():.0f}"
    )
    print(
        "Detected genes: "
        f"mean={genes.mean():.2f}, median={genes.median():.2f}, "
        f"min={genes.min():.0f}, max={genes.max():.0f}"
    )


def qc_spot_and_gene_removal(
    adata,
    umi_threshold: int = 200,
    gene_threshold: int | None = None,
    remove_mt: bool = False,
    max_pct_mt: float | None = None,
):
    """Filter low-count spots, rare genes, mitochondrial genes and MT outliers."""
    print(f"Initial: {adata.n_obs:,} spots x {adata.n_vars:,} genes")

    if {"total_counts", "n_genes_by_counts"} <= set(adata.obs):
        adata.obs["total_counts_original"] = adata.obs[
            "total_counts"
        ].copy()
        adata.obs["n_genes_by_counts_original"] = adata.obs[
            "n_genes_by_counts"
        ].copy()

    if "mt" not in adata.var:
        adata.var["mt"] = adata.var_names.str.contains(
            r"^MT[-.]", case=False, regex=True
        )

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        log1p=False,
        percent_top=None,
        inplace=True,
    )
    adata.obs["total_cnts_mt_b4_rmval"] = adata.obs[
        "total_counts_mt"
    ].copy()
    adata.obs["pct_cnts_mt_b4_rmval"] = adata.obs[
        "pct_counts_mt"
    ].copy()

    sc.pp.filter_cells(adata, min_counts=umi_threshold)

    if gene_threshold is not None:
        detected = np.asarray((adata.X > 0).sum(axis=0)).ravel()
        adata = adata[:, detected >= gene_threshold].copy()

    if remove_mt:
        adata.uns["excluded_mt_genes"] = adata.var_names[
            adata.var["mt"]
        ].tolist()
        adata = adata[:, ~adata.var["mt"]].copy()
        del adata.var["mt"]

    if max_pct_mt is not None:
        adata = adata[
            adata.obs["pct_cnts_mt_b4_rmval"] < max_pct_mt
        ].copy()

    sc.pp.calculate_qc_metrics(
        adata, log1p=False, percent_top=None, inplace=True
    )
    print(f"Final: {adata.n_obs:,} spots x {adata.n_vars:,} genes")
    return adata


def preprocessing(
    adata,
    highly_var: bool = True,
    flavor: str = "seurat_v3",
    n_hvg: int = 3000,
):
    """Select HVGs, normalize, log-transform and scale expression."""
    if adata.n_obs == 0:
        raise ValueError("Cannot preprocess an AnnData object with no spots.")

    if highly_var:
        if flavor == "seurat_v3":
            if adata.raw is None:
                raise ValueError("Set adata.raw to raw counts first.")
            adata.layers["counts"] = adata.raw.X.copy()
            sc.pp.highly_variable_genes(
                adata,
                flavor="seurat_v3",
                n_top_genes=min(n_hvg, adata.n_vars),
                layer="counts",
            )
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        elif flavor == "seurat":
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(
                adata,
                flavor="seurat",
                n_top_genes=min(n_hvg, adata.n_vars),
            )
        else:
            sc.pp.highly_variable_genes(
                adata,
                flavor=flavor,
                n_top_genes=min(n_hvg, adata.n_vars),
            )

        print(f"Selected {adata.var['highly_variable'].sum():,} HVGs")
        print("Subsetting the expression matrix to the selected HVGs...")
        adata = adata[:, adata.var["highly_variable"]].copy()
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Scaling zero-centers the matrix. Convert explicitly so Scanpy does not
    # emit a warning about implicitly densifying sparse data.
    if sparse.issparse(adata.X):
        print("Converting the HVG matrix from sparse to dense format...")
        adata.X = adata.X.toarray()
    print("Scaling expression values with a maximum absolute value of 10...")
    sc.pp.scale(adata, max_value=10)
    print("Expression preprocessing complete.")
    return adata


def search_leiden_resolution(
    adata,
    n_clusters: int,
    use_rep: str = "X_pca",
    start: float = 0.1,
    end: float = 1.7,
    increment: float = 0.01,
    tolerance: int = 2,
) -> float:
    """Adaptively search downward for the requested cluster count."""
    print("Searching resolution...")
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)

    best_resolution = None
    best_difference = float("inf")
    best_count = None

    previous_resolution = None
    previous_count = None
    resolution = end
    coarse = True

    while resolution > start:
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
                random_state=0,
                resolution=resolution,
                key_added="_resolution_search",
                flavor="leidenalg",
            )
        count = int(adata.obs["_resolution_search"].nunique())
        print(
            f"resolution={resolution:.4f}, "
            f"cluster number={count}"
        )

        if count == n_clusters:
            print(
                f"Found exact match: resolution={resolution:.4f}, "
                f"clusters={count}"
            )
            return float(resolution)

        difference = count - n_clusters
        if 0 <= difference <= tolerance and difference < best_difference:
            best_resolution = resolution
            best_difference = difference
            best_count = count

        # A coarse jump crossed from too many to too few clusters. Return to
        # the previous resolution and continue with the fine increment.
        if (
            previous_resolution is not None
            and previous_count is not None
            and previous_count > n_clusters
            and count < n_clusters
        ):
            resolution = previous_resolution
            previous_resolution = None
            previous_count = None
            coarse = False
            continue

        if coarse and difference >= 5:
            step = 30 * increment
        elif coarse and difference >= 3:
            step = 10 * increment
        elif coarse and difference >= 2:
            step = 5 * increment
        else:
            step = increment

        if coarse and step > increment:
            previous_resolution = resolution
            previous_count = count

        step = min(step, resolution - start)
        resolution -= step

    if best_resolution is not None:
        print(
            "Exact match not found. Using closest: "
            f"resolution={best_resolution:.4f}, "
            f"clusters={best_count} (target={n_clusters})."
        )
        return float(best_resolution)

    print(
        f"Resolution not found within tolerance={tolerance}. "
        "Returning default resolution 1.0."
    )
    return 1.0
