import os
import logging
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
CACHE_ROOT = MODULE_DIR / ".cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)

FORCE_TEX = True
BASE_FONT_SIZE = 16
AXIS_LABEL_SIZE = 18
AXIS_TITLE_SIZE = 18
TICK_LABEL_SIZE = 14
LEGEND_FONT_SIZE = 13
DEFAULT_FIGSIZE = (8.4, 4.8)
DEFAULT_BINS = 22
GAMMA_COL = "Gamma"
MODEL_COL = "model"
DEFINED_COL = "Gamma_defined"

SERIES_STYLES = [
    {
        "color": "#111111",
        "linestyle": "-",
        "marker": "o",
        "markerfacecolor": "#111111",
        "markeredgecolor": "#111111",
    },
    {
        "color": "#B22222",
        "linestyle": "-",
        "marker": "s",
        "markerfacecolor": "white",
        "markeredgecolor": "#B22222",
    },
    {
        "color": "#6A6A6A",
        "linestyle": "-.",
        "marker": "D",
        "markerfacecolor": "white",
        "markeredgecolor": "#111111",
    },
]

try:
    from scipy.stats import gaussian_kde

    SCIPY_AVAILABLE = True
except ImportError:
    gaussian_kde = None
    SCIPY_AVAILABLE = False


@lru_cache(maxsize=1)
def latex_ready():
    required_commands = ("latex", "dvipng")
    if any(shutil.which(command) is None for command in required_commands):
        return False

    if shutil.which("kpsewhich") is not None:
        latex_fmt = subprocess.run(
            ["kpsewhich", "latex.fmt"],
            capture_output=True,
            text=True,
            check=False,
        )
        if latex_fmt.returncode != 0 or not latex_fmt.stdout.strip():
            return False

    latex_source = "\n".join([
        r"\documentclass{article}",
        r"\begin{document}",
        r"lp",
        r"\end{document}",
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir) / "matplotlib_tex_probe.tex"
        tex_path.write_text(latex_source, encoding="utf-8")
        result = subprocess.run(
            ["latex", "-interaction=nonstopmode", "--halt-on-error", tex_path.name],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and (Path(tmpdir) / "matplotlib_tex_probe.dvi").exists()


def has_tex_package(package_name):
    if shutil.which("kpsewhich") is None:
        return False
    return subprocess.run(
        ["kpsewhich", f"{package_name}.sty"],
        capture_output=True,
        text=True,
        check=False,
    ).returncode == 0


@lru_cache(maxsize=2)
def configure_plot_style(force_tex=FORCE_TEX):
    use_tex = force_tex and latex_ready()
    preamble = r"\usepackage{fontawesome5}" if use_tex and has_tex_package("fontawesome5") else ""

    if force_tex and not use_tex:
        print(
            "LaTeX was requested but the current environment cannot compile a minimal "
            "TeX document. Falling back to Matplotlib serif text."
        )

    mpl.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "STIXGeneral",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "text.latex.preamble": preamble,
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": AXIS_TITLE_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    return use_tex


def running_inside_notebook():
    try:
        from IPython import get_ipython
    except ImportError:
        return False

    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def show_or_close(fig):
    backend = mpl.get_backend().lower()
    if "agg" in backend and not running_inside_notebook():
        plt.close(fig)
        return
    plt.show()


def clean_model_name(name):
    label = str(name).strip()
    lower = label.lower()
    if "resnet_large" in lower or "large" in lower:
        return "ResNet-Large"
    if "resnet_medium" in lower or "medium" in lower:
        return "ResNet-Medium"
    if "vgg" in lower:
        return "VGG16"
    return label.replace("_", " ")


def load_gamma_data(csv_path, gamma_col=GAMMA_COL, defined_col=DEFINED_COL):
    df = pd.read_csv(csv_path).copy()

    if defined_col in df.columns:
        df = df[df[defined_col].fillna(False).astype(bool)].copy()

    df = df[df[gamma_col].notna()].copy()
    df[gamma_col] = pd.to_numeric(df[gamma_col], errors="coerce")
    df = df[df[gamma_col].notna()].copy()
    return df


def model_sort_key(name):
    lower = str(name).strip().lower()
    if "large" in lower:
        return (0, lower)
    if "medium" in lower:
        return (1, lower)
    if "vgg" in lower:
        return (2, lower)
    return (3, lower)


def build_series(df, merge_models=False, merged_label=None, model_col=MODEL_COL, gamma_col=GAMMA_COL):
    if merge_models:
        label = merged_label or "Merged"
        values = df[gamma_col].astype(float).to_numpy()
        return [(label, values)]

    unique_models = sorted(df[model_col].dropna().unique(), key=model_sort_key)
    return [
        (clean_model_name(model), df.loc[df[model_col] == model, gamma_col].dropna().astype(float).to_numpy())
        for model in unique_models
    ]


def density_curve(values, x_grid, bins=DEFAULT_BINS):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.array([]), np.array([])

    if SCIPY_AVAILABLE and values.size >= 2 and np.std(values) > 0:
        kde = gaussian_kde(values)
        return x_grid, kde(x_grid)

    hist_y, hist_x = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (hist_x[:-1] + hist_x[1:])
    return centers, hist_y


def compute_x_grid(series_values, pad_ratio=0.08, n_points=500):
    all_values = np.concatenate([values for _, values in series_values if len(values) > 0])
    x_min = float(np.min(all_values))
    x_max = float(np.max(all_values))
    span = x_max - x_min
    pad = pad_ratio * span if span > 0 else 0.5
    return np.linspace(x_min - pad, x_max + pad, n_points)


def plot_gamma_density(
    series_values,
    out_pdf,
    x_label=r"$\Gamma$",
    y_label="Density",
    figsize=DEFAULT_FIGSIZE,
    bins=DEFAULT_BINS,
):
    configure_plot_style()

    series_values = [(label, np.asarray(values, dtype=float)) for label, values in series_values if len(values) > 0]
    if not series_values:
        raise ValueError("No defined gamma values available to plot.")

    x_grid = compute_x_grid(series_values)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    for index, (label, values) in enumerate(series_values):
        style = SERIES_STYLES[index % len(SERIES_STYLES)]
        ax.hist(
            values,
            bins=bins,
            density=True,
            color=style["color"],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.16,
            zorder=2,
        )
        x_curve, y_curve = density_curve(values, x_grid, bins=bins)
        marker_stride = max(len(x_curve) // 10, 1) if len(x_curve) else 1
        ax.plot(
            x_curve,
            y_curve,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.6,
            marker=style["marker"],
            markersize=5.4,
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["markeredgecolor"],
            markeredgewidth=1.0,
            markevery=marker_stride,
            label=f"{label} ({len(values)})",
            zorder=4,
        )

    ax.axvline(0.0, color="#999999", linewidth=2.6, linestyle="--", alpha=0.95, zorder=3)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.55)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    legend_columns = min(len(series_values), 3)
    ax.legend(
        frameon=False,
        ncol=legend_columns,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        handlelength=2.4,
        columnspacing=1.4,
    )

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    show_or_close(fig)

    print(f"Saved: {out_pdf}")


def plot_gamma_density_from_csv(
    csv_path,
    out_pdf,
    merge_models=False,
    merged_label=None,
    x_label=r"$\Gamma$",
    y_label="Density",
    figsize=DEFAULT_FIGSIZE,
    bins=DEFAULT_BINS,
):
    df = load_gamma_data(csv_path)
    series_values = build_series(df, merge_models=merge_models, merged_label=merged_label)
    plot_gamma_density(
        series_values=series_values,
        out_pdf=out_pdf,
        x_label=x_label,
        y_label=y_label,
        figsize=figsize,
        bins=bins,
    )
    return df
