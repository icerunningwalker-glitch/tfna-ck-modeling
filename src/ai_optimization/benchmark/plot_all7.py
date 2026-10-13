"""Plot existing seven-model five-fold OOF predictions; no model fitting.

Inputs: results/predictions.csv and results/metrics.csv beside this script.
CSV training rows average each sequence's four outer-model in-sample fitted
predictions. They are fit diagnostics and are validated but not plotted.
The figure uses one outer OOF prediction per independent sequence.
Previously analysed data are reused; no external validation was performed.
There are identical independent sequence assignments across all seven models.
The OOF uses one prediction per independent sequence, with identical
sequence assignments across all seven models. ROC endpoint: observed_count >=
1; confusion-matrix prediction cutoff: predicted_score >= 0.5. Regression
scores are site-count predictions, not probabilities or energies.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from layout import PROJECT_ROOT, RESULTS, locations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

MODELS = ["MLP", "SVR", "Ridge", "Random forest", "ExtraTrees", "GPR", "TabPFN"]
ROSE = "#E27783"
BLUE = "#083F77"
BAR_COLORS = ["#F29DAB", "#E27783", "#044875", "#738B64", "#DBA997", "#7E0909", "#C7C3AC"]
SPLITS = ["train", "test"]


def configure_fonts(font_file: Path | None = None) -> None:
    if font_file is not None:
        if not font_file.is_file():
            raise FileNotFoundError('Provided --font-file does not exist.')
        font_manager.fontManager.addfont(str(font_file))
        family = font_manager.FontProperties(fname=str(font_file)).get_name()
    else:
        try:
            font_manager.findfont('Arial', fallback_to_default=False)
            family = 'Arial'
        except ValueError:
            family = 'DejaVu Sans'
            print('Arial is unavailable; using bundled sans-serif DejaVu Sans. Supply --font-file for Arial.')
    plt.rcParams.update({
        'font.family': family, 'font.size': 15,
        'text.color': 'black', 'axes.labelcolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black',
        'axes.edgecolor': 'black', 'svg.fonttype': 'none',
        'axes.unicode_minus': False, 'savefig.facecolor': 'white',
    })


def confusion(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, (y.astype(int), prediction.astype(int)), 1)
    assert cm.sum() == len(y)
    return cm


def roc(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    y = y.astype(bool)
    assert y.any() and (~y).any(), "ROC requires both observed classes."
    order = np.argsort(-score, kind="stable")
    sorted_score, ordered_y = score[order], y[order]
    end_of_tie = np.r_[np.flatnonzero(np.diff(sorted_score) != 0), len(y) - 1]
    tp, fp = np.cumsum(ordered_y), np.cumsum(~ordered_y)
    tpr = np.r_[0, tp[end_of_tie] / y.sum()]
    fpr = np.r_[0, fp[end_of_tie] / (~y).sum()]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def validate(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    fields = {"model", "split", "sequence_id", "observed_count", "observed_positive",
              "predicted_score", "predicted_positive"}
    assert fields <= set(predictions.columns), "Prediction fields are incomplete."
    assert {"model", "split", "n", "AUC", "MAE", "MSE", "RMSE", "TN", "FP", "FN", "TP"} <= set(metrics.columns)
    assert set(predictions["model"]) == set(MODELS), "Expected exactly the seven specified models."
    assert set(predictions["split"]) == set(SPLITS)
    assert not predictions[list(fields)].isna().any().any()
    counts = predictions["observed_count"].to_numpy(float)
    scores = predictions["predicted_score"].to_numpy(float)
    assert np.isfinite(counts).all() and np.isfinite(scores).all()
    assert (counts >= 0).all() and (counts == np.floor(counts)).all()
    assert (scores >= 0).all()
    assert np.array_equal(predictions["observed_positive"], counts >= 1)
    assert np.array_equal(predictions["predicted_positive"], scores >= 0.5)
    reference = {}
    for model in MODELS:
        ids = {}
        for split in SPLITS:
            q = predictions.query("model == @model and split == @split").sort_values("sequence_id")
            assert q["sequence_id"].is_unique, "Repeated sequence prediction within split."
            ids[split] = set(q["sequence_id"])
            keys = q[["sequence_id", "observed_count"]].reset_index(drop=True)
            if model == MODELS[0]:
                reference[split] = keys
            else:
                pd.testing.assert_frame_equal(keys, reference[split])
            y = q["observed_positive"].to_numpy(int)
            score = q["predicted_score"].to_numpy(float)
            pred = q["predicted_positive"].to_numpy(int)
            cm = confusion(y, pred)
            auc = roc(y, score)[2]
            residual = score - q["observed_count"].to_numpy(float)
            expected = {"n": len(q), "AUC": auc, "MAE": np.abs(residual).mean(),
                        "MSE": (residual ** 2).mean(), "RMSE": np.sqrt((residual ** 2).mean()),
                        "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1]}
            m = metrics.query("model == @model and split == @split")
            assert len(m) == 1, "Metrics table lacks unique model/split row."
            for key, value in expected.items():
                assert np.isclose(float(m.iloc[0][key]), value, atol=1e-10, rtol=1e-8), f"Metric mismatch: {model}/{split}/{key}"
        assert ids["train"] == ids["test"], "Fit diagnostic and OOF sequence IDs differ."
        assert len(ids["test"]) == 123, "Expected 123 unique outer OOF sequences."


def top_axes(fig, x, top, width, height, canvas_width, canvas_height):
    return fig.add_axes([x / canvas_width, (canvas_height - top - height) / canvas_height,
                         width / canvas_width, height / canvas_height])


def arrows(ax, fontsize=15, width=1.5):
    for name in ["top", "right"]:
        ax.spines[name].set_visible(False)
    for name in ["left", "bottom"]:
        ax.spines[name].set_linewidth(width)
        ax.spines[name].set_color("black")
    ax.tick_params(labelsize=fontsize, width=width, length=4, direction="out", colors="black")
    ax.annotate("", xy=(1.055, 0), xytext=(0.96, 0), xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": "black", "lw": width}, annotation_clip=False)
    ax.annotate("", xy=(0, 1.055), xytext=(0, .96), xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": "black", "lw": width}, annotation_clip=False)


def save(fig, output_dir: Path, stem: str) -> None:
    fig.canvas.draw()
    # A canvas-bounds check includes every visible text element before export.
    bounds = fig.bbox
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        assert box.x0 >= bounds.x0 - 2 and box.x1 <= bounds.x1 + 2, f"Horizontal text crop: {artist.get_text()}"
        assert box.y0 >= bounds.y0 - 2 and box.y1 <= bounds.y1 + 2, f"Vertical text crop: {artist.get_text()}"
    for annotation in fig.findobj(matplotlib.text.Annotation):
        if not annotation.get_visible() or annotation.arrow_patch is None:
            continue
        box = annotation.get_window_extent(renderer)
        assert box.x0 >= bounds.x0 - 2 and box.x1 <= bounds.x1 + 2, "Horizontal arrow crop."
        assert box.y0 >= bounds.y0 - 2 and box.y1 <= bounds.y1 + 2, "Vertical arrow crop."
    fig.savefig(output_dir / f"{stem}.png", dpi=600)
    fig.savefig(output_dir / f"{stem}.svg")
    fig.savefig(output_dir / f"{stem}_preview.png", dpi=100)
    plt.close(fig)


def plot_matrix(predictions: pd.DataFrame, output_dir: Path) -> None:
    width, height = 2010, 590
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    left, right = 200, 1930
    column_width = (right - left) / len(MODELS)
    cmap = LinearSegmentedColormap.from_list("count_blue", ["#F5F9FD", BLUE])
    maxima = {}
    for split in ["test"]:
        maxima[split] = max(confusion(q["observed_positive"].to_numpy(int),
                                     q["predicted_positive"].to_numpy(int)).max()
                            for _, q in predictions.query("split == @split").groupby("model"))
    for split_index, split in enumerate(["test"]):
        label = "5-fold OOF"
        label_y = 290
        fig.text(95 / width, 1 - label_y / height, label, fontweight="bold", fontsize=20,
                 ha="center", va="center", color="black")
        roc_top, cm_top = 105, 326
        for column, model in enumerate(MODELS):
            x0 = left + column * column_width
            if split_index == 0:
                fig.add_artist(Rectangle((x0 / width, (height - 57) / height),
                                         (column_width - 2) / width, 36 / height,
                                         transform=fig.transFigure, facecolor=BLUE, edgecolor="none"))
                fig.text((x0 + column_width / 2) / width, (height - 39) / height,
                         model, ha="center", va="center", color="white", fontsize=21, fontweight="bold")
            q = predictions.query("model == @model and split == @split")
            y, score, yp = [q[k].to_numpy() for k in ["observed_positive", "predicted_score", "predicted_positive"]]
            fpr, tpr, auc = roc(y, score)
            ax = top_axes(fig, x0 + 48, roc_top, 158, 150, width, height)
            ax.plot(fpr, tpr, color=ROSE, lw=1.7)
            ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .5, 1])
            ax.set_xticklabels(["0.0", "0.5", "1.0"])
            ax.set_yticklabels(["0.0", "0.5", "1.0"])
            ax.set_xlabel("False positive rate", fontsize=15, labelpad=9)
            ax.set_ylabel("True positive rate", fontsize=15, labelpad=7)
            arrows(ax, fontsize=13)
            ax.set_title(f"AUC {auc:.3f}", fontsize=19, pad=12, color="black")
            cm = confusion(y, yp)
            axc = top_axes(fig, x0 + 45, cm_top, 170, 170, width, height)
            im = axc.imshow(cm, cmap=cmap, vmin=0, vmax=maxima[split], interpolation="nearest")
            for r in range(2):
                for c in range(2):
                    axc.text(c, r, str(cm[r, c]), fontsize=26, ha="center", va="center",
                             color="white" if cm[r, c] / maxima[split] >= .55 else "black")
            axc.set_xticks([0, 1], ["0 sites", "≥1 site"], fontsize=14)
            axc.set_yticks([0, 1], ["0 sites", "≥1 site"], fontsize=14, rotation=90, va="center")
            axc.set_xlabel("Predicted label", fontsize=16, labelpad=8)
            axc.set_ylabel("Calculated label", fontsize=14, labelpad=9)
            axc.tick_params(length=0, pad=8)
            for spine in axc.spines.values():
                spine.set_visible(False)
            axc.set_xticks([.5], minor=True)
            axc.set_yticks([.5], minor=True)
            axc.grid(which="minor", color="white", linewidth=.8)
            axc.tick_params(which="minor", length=0)
        cax = top_axes(fig, 1958, cm_top, 12, 170, width, height)
        count_ticks = [0, int(maxima[split] // 2), int(maxima[split])]
        cb = fig.colorbar(im, cax=cax, ticks=count_ticks)
        cb.set_ticklabels([str(v) for v in count_ticks])
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=12, length=2, pad=4, colors="black")
        cb.ax.set_title("n", fontsize=15, pad=10)
    save(fig, output_dir, "dry002_all7_benchmark")


def plot_error(metrics: pd.DataFrame, output_dir: Path, field: str, stem: str) -> None:
    q = metrics.query("split == 'test'").set_index("model").loc[MODELS]
    values = q[field].to_numpy(float)
    fig, ax = plt.subplots(figsize=(11.6, 6.4), dpi=100, facecolor="white")
    fig.subplots_adjust(left=.13, bottom=.18, right=.94, top=.92)
    ax.bar(np.arange(len(MODELS)), values, width=.58, color=BAR_COLORS, edgecolor="none", linewidth=0)
    target_limit = float(values.max() * 1.26)
    raw_step = target_limit / 5
    magnitude = 10 ** np.floor(np.log10(raw_step))
    factors = np.array([1, 2, 2.5, 5, 10])
    step = float(factors[np.argmin(np.abs(np.log(factors / (raw_step / magnitude))))] * magnitude)
    limit = float(np.ceil(target_limit / step) * step)
    ax.set(xlim=(-.6, len(MODELS) - .4), ylim=(0, limit), xticks=np.arange(len(MODELS)),
           yticks=np.arange(0, limit + step * .5, step))
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:g}"))
    ax.set_xticklabels([m.replace("Random forest", "Random\nforest") for m in MODELS], fontsize=19)
    ax.set_ylabel("MSE (count²)" if field == "MSE" else "RMSE (count)", fontsize=25, fontweight="bold", labelpad=12)
    ax.tick_params(axis="x", labelsize=19)
    ax.tick_params(axis="y", labelsize=18)
    arrows(ax, fontsize=18, width=1.8)
    ax.tick_params(axis="x", labelsize=19)
    for x, value in enumerate(values):
        ax.text(x, value + limit * .035, f"{value:.3f}", fontsize=21, ha="center", va="bottom", color="black")
    save(fig, output_dir, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument('--output-dir', type=Path, default=RESULTS / 'runs' / 'plots', help='New directory; saved figures are never overwritten.')
    parser.add_argument('--font-file', type=Path, help='Optional user-provided Arial/sans-serif TTF file.')
    args = parser.parse_args()
    root = args.root.resolve()
    result_dir = locations(root)['results']
    configure_fonts(args.font_file)
    # Round-trip parsing preserves exact saved prediction scores and ROC ties.
    predictions = pd.read_csv(result_dir / "predictions.csv",
                              dtype={"sequence_id": str}, float_precision="round_trip")
    metrics = pd.read_csv(result_dir / "metrics.csv", float_precision="round_trip")
    assert set(MODELS) <= set(predictions["model"]), "A requested algorithm is missing."
    # Preserve any baseline records in the source CSV, but keep seven algorithms
    # in the comparison figure and error charts.
    predictions = predictions[predictions["model"].isin(MODELS)].copy()
    metrics = metrics[metrics["model"].isin(MODELS)].copy()
    validate(predictions, metrics)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.error('Output directory already exists; choose a new directory.')
    output_dir.mkdir(parents=True, exist_ok=False)
    plot_matrix(predictions, output_dir)
    plot_error(metrics, output_dir, "MSE", "dry003_all7_MSE")
    plot_error(metrics, output_dir, "RMSE", "dry004_all7_RMSE")
    print("Generated benchmark, MSE and RMSE PNG (600 dpi), SVG and preview PNG.")


if __name__ == "__main__":
    main()
