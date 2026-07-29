"""Optional removal of spatially isolated spots using DBSCAN."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN


def remove_free_lying_spots(
    adata,
    sample_name: str,
    threshold: int = 8000,
    eps: float = 30,
    min_samples: int = 5,
    plot: bool = False,
    output_dir: str | Path = "figures/spot_removal",
):
    """Keep DBSCAN components containing more than ``threshold`` spots.

    This helper is retained from the broader pipeline but is not applied to
    DLPFC 151671 or 151673 by the example workflow.
    """
    labels = (
        DBSCAN(eps=eps, min_samples=min_samples)
        .fit_predict(adata.obsm["spatial"])
        + 1
    )
    counts = np.bincount(labels)
    large_components = np.flatnonzero(counts > threshold)
    keep = np.isin(labels, large_components)
    filtered = adata[keep].copy()

    if plot:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        library = next(iter(adata.uns["spatial"]))
        spatial = adata.uns["spatial"][library]
        scale = spatial["scalefactors"]["tissue_hires_scalef"]
        image = spatial["images"]["hires"]
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, current, title in (
            (axes[0], adata, "Before removal"),
            (axes[1], filtered, "After DBSCAN removal"),
        ):
            ax.imshow(gray, cmap="gray")
            coords = current.obsm["spatial"] * scale
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                s=1,
                c=current.obs["total_counts"],
                cmap="viridis",
            )
            ax.set_title(title)
            ax.set_axis_off()

        fig.tight_layout()
        fig.savefig(
            output_dir / f"{sample_name}_spot_removal.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    return filtered

