#!/usr/bin/env python3
"""Create Figure 2-style InSTability panels using spatialdata-plot."""

from argparse import ArgumentParser
from pathlib import Path
import warnings

import anndata as ad
import geopandas as gpd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
import spatialdata_plot  # noqa: F401; registers sdata.pl
from anndata import ImplicitModificationWarning
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from spatialdata.models import Image2DModel, ShapesModel, TableModel

warnings.filterwarnings(
    "ignore",
    message="Transforming to str index.*",
    category=ImplicitModificationWarning,
)


SAMPLES = ("151671", "151673")

CLUSTER_COLORS = [
    "#3049ad", "#fe8011", "#1b7837", "#d62a2b", "#ab43fc",
    "#8d574c", "#e187c4", "#b8bd6c", "#23bed0", "#bc510a",
    "#0aac00", "#ff008c", "#057dff", "#a7d1e6",
]

CLUSTER_STABILITY_BINS = [
    ("[0.0,0.1)", 0.0, 0.1, "#08306bea"),
    ("[0.1,0.2)", 0.1, 0.2, "#4292c6"),
    ("[0.2,0.3)", 0.2, 0.3, "#9ecae1"),
    ("[0.3,0.4)", 0.3, 0.4, "#deebf7"),
    ("[0.4,0.5)", 0.4, 0.5, "#fee391"),
    ("[0.5,0.6)", 0.5, 0.6, "#fec44f"),
    ("[0.6,0.7)", 0.6, 0.7, "#fdae6b"),
    ("[0.7,0.8)", 0.7, 0.8, "#fd8d3c"),
    ("[0.8,0.9)", 0.8, 0.9, "#fb6a4a"),
    ("[0.9,1.0]", 0.9, 1.0, "#cb181d"),
]

THRES_PALETTE = {
    label: color
    for label, _, _, color in CLUSTER_STABILITY_BINS
}
BIN_LABELS = [
    label for label, _, _, _ in CLUSTER_STABILITY_BINS
]
BIN_EDGES = np.array(
    [lower for _, lower, _, _ in CLUSTER_STABILITY_BINS]
    + [np.nextafter(CLUSTER_STABILITY_BINS[-1][2], np.inf)]
)

IMAGE_ALPHA = 0.55
CROP_PADDING = 0.035


def example_run_index(partitions) -> int:
    """Choose the earliest run among those with the fewest clusters."""
    cluster_counts = [len(set(labels)) for labels in partitions]
    return cluster_counts.index(min(cluster_counts))


def instability_bins(values) -> pd.Categorical:
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return pd.cut(
        values,
        bins=BIN_EDGES,
        labels=BIN_LABELS,
        right=False,
        include_lowest=True,
    )


def add_plot_columns(adata, labels, spotwise):
    adata.obs["example_leiden"] = pd.Categorical(
        np.asarray(labels).astype(str)
    )
    adata.obs["spotwise_instability"] = np.asarray(
        spotwise, dtype=float
    )
    adata.obs["spotwise_bin"] = instability_bins(spotwise)

    cluster_means = adata.obs.groupby(
        "example_leiden", observed=True
    )["spotwise_instability"].mean()
    per_spot_cluster_mean = (
        adata.obs["example_leiden"]
        .map(cluster_means)
        .astype(float)
    )
    adata.obs["clusterwise_bin"] = instability_bins(
        per_spot_cluster_mean
    )
    return cluster_means


def grayscale(image):
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    rgb = image[..., :3].astype(float)
    gray = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )
    if np.issubdtype(image.dtype, np.integer):
        gray = np.clip(gray, 0, np.iinfo(image.dtype).max)
    else:
        gray = np.clip(gray, 0.0, 1.0)
    return gray.astype(image.dtype)


def image_and_coordinates(adata):
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    spatial = adata.uns["spatial"]
    library = spatial[next(iter(spatial))]
    images = library["images"]
    scales = library["scalefactors"]

    for key in ("hires", "lowres"):
        if key in images:
            scale = float(scales.get(f"tissue_{key}_scalef", 1.0))
            return np.asarray(images[key]), coords * scale
    raise KeyError("No hires or lowres tissue image was found.")


def pointy_hexagons(coords):
    unique = np.unique(coords, axis=0)
    distances = cKDTree(unique).query(unique, k=2)[0][:, 1]
    spacing = float(np.median(distances[np.isfinite(distances)]))
    radius = spacing / np.sqrt(3)
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    return [
        Polygon(
            [
                (
                    x + radius * np.cos(angle),
                    y + radius * np.sin(angle),
                )
                for angle in angles
            ]
        )
        for x, y in coords
    ]


def build_spatialdata(adata):
    image, coords = image_and_coordinates(adata)
    index = pd.Index(
        adata.obs_names.astype(str), name="instance_id"
    )
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame(
            geometry=pointy_hexagons(coords),
            index=index,
        )
    )

    table_obs = pd.DataFrame(index=index)
    table_obs["region"] = pd.Categorical(["spots"] * adata.n_obs)
    table_obs["instance_id"] = index
    table = TableModel.parse(
        ad.AnnData(obs=table_obs),
        region="spots",
        region_key="region",
        instance_key="instance_id",
    )
    spatial_data = sd.SpatialData(
        images={
            "histology": Image2DModel.parse(
                grayscale(image)[None, ...],
                dims=("c", "y", "x"),
                c_coords=["gray"],
            )
        },
        shapes={"spots": shapes},
        tables={"table": table},
    )
    return spatial_data, shapes.total_bounds


def first_appearance_palette(values, colors=CLUSTER_COLORS):
    categories = list(pd.unique(values.dropna().astype(str)))
    if len(categories) > len(colors):
        raise ValueError("Not enough colors for all categories.")
    return dict(zip(categories, colors[: len(categories)]))


def render_categorical(
    spatial_data,
    bounds,
    values,
    name,
    palette,
    ax,
    title,
):
    values = pd.Series(values, index=spatial_data.tables["table"].obs_names)
    if isinstance(values.dtype, pd.CategoricalDtype):
        categories = list(values.cat.categories.astype(str))
        ordered = values.cat.ordered
    else:
        categories = list(pd.unique(values.dropna().astype(str)))
        ordered = True

    spatial_data.tables["table"].obs[name] = pd.Categorical(
        values.astype("string"),
        categories=categories,
        ordered=ordered,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Transforming to str index.*",
            category=ImplicitModificationWarning,
        )
        (
            spatial_data.pl.render_images(
                element="histology",
                cmap="gray",
                alpha=IMAGE_ALPHA,
            )
            .pl.render_shapes(
                element="spots",
                color=name,
                palette={
                    str(key): color for key, color in palette.items()
                },
                fill_alpha=0.98,
                outline_width=0,
                table_name="table",
                method="matplotlib",
            )
            .pl.show(
                ax=ax,
                title=title,
                frameon=True,
                legend_loc=None,
                colorbar=False,
                show=False,
            )
        )

    x0, y0, x1, y1 = bounds
    px = CROP_PADDING * (x1 - x0)
    py = CROP_PADDING * (y1 - y0)
    ax.set_xlim(x0 - px, x1 + px)
    ax.set_ylim(y1 + py, y0 - py)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)


def draw_grouped_instability_legend(fig, small=True):
    range_fontsize = 10 if small else 12
    group_fontsize = 12 if small else 15

    legend_ax = fig.add_axes([0.04, -0.06, 0.92, 0.13])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.set_axis_off()

    groups = [
        ("Stable", CLUSTER_STABILITY_BINS[0:2]),
        ("Modestly Stable", CLUSTER_STABILITY_BINS[2:4]),
        ("Unstable", CLUSTER_STABILITY_BINS[4:6]),
        ("Highly Unstable", CLUSTER_STABILITY_BINS[6:10]),
    ]

    group_widths = [2, 2, 2, 4]
    total_units = sum(group_widths)
    x_cursor = 0.0

    for (group_label, bins_here), units in zip(groups, group_widths):
        box_width = units / total_units
        x0 = x_cursor + 0.006
        y0 = 0.42
        width = box_width - 0.012
        height = 0.45

        legend_ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor="black",
                linewidth=1.0,
            )
        )

        inner_padding = 0.012
        item_width = (
            width - 2 * inner_padding
        ) / len(bins_here)

        for index, (
            bin_name,
            _,
            _,
            bin_color,
        ) in enumerate(bins_here):
            item_x = x0 + inner_padding + index * item_width

            color_width = item_width * 0.30
            color_height = 0.16
            color_y = y0 + 0.15

            legend_ax.add_patch(
                plt.Rectangle(
                    (item_x, color_y),
                    color_width,
                    color_height,
                    facecolor=bin_color,
                    edgecolor="none",
                )
            )

            legend_ax.text(
                item_x + color_width + 0.006,
                color_y + color_height / 2,
                bin_name,
                ha="left",
                va="center",
                fontsize=range_fontsize,
                fontweight="bold",
            )

        legend_ax.text(
            x0 + width / 2,
            0.14,
            group_label,
            ha="center",
            va="center",
            fontsize=group_fontsize,
            fontweight="bold",
        )

        x_cursor += box_width

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed")
    )
    parser.add_argument(
        "--clustering-dir", type=Path, default=Path("results/leiden")
    )
    parser.add_argument(
        "--uncertainty-dir",
        type=Path,
        default=Path("results/uncertainty"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("figures")
    )
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sample in SAMPLES:
        adata = sc.read_h5ad(
            args.data_dir / f"dlpfc_{sample}_filtered.h5ad"
        )
        if "annotation" not in adata.obs:
            raise KeyError(
                f"DLPFC {sample} has no annotation column. "
                "Rerun scripts/prepare_data.py."
            )
        adata.obs["annotation"] = pd.Categorical(
            adata.obs["annotation"]
        )

        partitions = joblib.load(
            args.clustering_dir
            / f"{sample}_leiden_results_{args.iterations}iterations.pkl"
        )
        result = joblib.load(
            args.uncertainty_dir
            / (
                f"dlpfc_{sample}_leiden_uncertainty_"
                f"{args.iterations}runs.pkl"
            )
        )
        selected = example_run_index(partitions)
        cluster_means = add_plot_columns(
            adata,
            partitions[selected],
            result["spot_uncertainty"],
        )

        spatial_data, bounds = build_spatialdata(adata)
        cluster_palette = first_appearance_palette(
            adata.obs["example_leiden"]
        )
        reference_palette = first_appearance_palette(
            adata.obs["annotation"]
        )

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(16, 4.6),
            gridspec_kw={"wspace": 0.25},
        )
        render_categorical(
            spatial_data,
            bounds,
            adata.obs["example_leiden"],
            "example_leiden",
            cluster_palette,
            axes[0],
            f"Leiden example run",
        )
        render_categorical(
            spatial_data,
            bounds,
            adata.obs["spotwise_bin"],
            "spotwise_bin",
            THRES_PALETTE,
            axes[1],
            "Spotwise InSTability",
        )
        render_categorical(
            spatial_data,
            bounds,
            adata.obs["clusterwise_bin"],
            "clusterwise_bin",
            THRES_PALETTE,
            axes[2],
            "Clusterwise InSTability",
        )
        render_categorical(
            spatial_data,
            bounds,
            adata.obs["annotation"],
            "annotation",
            reference_palette,
            axes[3],
            "Reference labels",
        )

        for label, ax in zip("ABCD", axes):
            ax.text(
                -0.08,
                1.06,
                label,
                transform=ax.transAxes,
                fontsize=18,
                fontweight="bold",
            )

        draw_grouped_instability_legend(fig, small=True)
        fig.suptitle(f"DLPFC {sample}", fontsize=17, fontweight="bold")
        fig.subplots_adjust(top=0.80, bottom=0.14)

        output_stem = args.output_dir / f"dlpfc_{sample}"
        fig.savefig(
            output_stem.with_suffix(".png"),
            dpi=600,
            bbox_inches="tight",
        )
        fig.savefig(
            output_stem.with_suffix(".pdf"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        cluster_means.rename("cluster_instability").to_csv(
            args.output_dir
            / f"dlpfc_{sample}_cluster_instability.csv"
        )
        print(f"Saved {output_stem}.png and {output_stem}.pdf")


if __name__ == "__main__":
    main()
