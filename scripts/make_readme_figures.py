"""Draw the README figures from the committed result files (light and dark SVGs in docs/figures/).

Usage: python scripts/make_readme_figures.py      (needs matplotlib, which is not a package dependency)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

root = Path(__file__).resolve().parents[1]
out = root / "docs/figures"
res = root / "docs/results"

THEMES = {  # dataviz reference palette: categorical slots 1-2, chart chrome and ink
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        ours="#2a78d6",
        paper="#eb6834",
        band="#f0efec",
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#f0efec",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        ours="#3987e5",
        paper="#d95926",
        band="#2c2c2a",
    ),
}
GATE1_TOLERANCE = 0.02


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], labelsize=9, length=0)
    ax.grid(color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def heading(fig, t, title, subtitle):
    fig.text(0.01, 0.97, title, color=t["ink"], fontsize=12, weight="bold", va="top")
    fig.text(0.01, 0.905, subtitle, color=t["ink2"], fontsize=9.5, va="top")


def gate1(t, name):
    c = pd.read_csv(res / "gate1_comparison.csv", index_col=0)
    c = c.loc[sorted(c.index, key=lambda s: (s.split()[0], s.split()[1] == "AE"))]  # finder, then NF before AE
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4), sharey=True, facecolor=t["surface"])
    y = range(len(c))
    for ax, metric in zip(axes, ("Accuracy", "F1"), strict=True):
        style(ax, t)
        ax.grid(axis="y", visible=False)
        ax.axvspan(-GATE1_TOLERANCE, GATE1_TOLERANCE, color=t["band"], zorder=0, linewidth=0)
        ax.axvline(0, color=t["axis"], linewidth=1)
        ax.scatter(c[f"{metric}_diff"], y, s=64, color=t["ours"], edgecolor=t["surface"], linewidth=1.5, zorder=3)
        ax.set_xlim(-0.03, 0.03)
        ax.set_xticks([-0.02, -0.01, 0, 0.01, 0.02])
        ax.set_xticklabels(["−0.02", "−0.01", "0", "+0.01", "+0.02"])
        ax.set_title(f"{metric}: pyMIRAW − paper", color=t["ink"], fontsize=10, loc="left")
        ax.text(
            GATE1_TOLERANCE - 0.001, -0.75, "Gate 1 tolerance", color=t["muted"], fontsize=8, ha="right", va="bottom"
        )
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels([s.replace("miRAW-", "") for s in c.index])
    axes[0].set_ylim(len(c) - 0.5, -1.1)  # top to bottom, with room for the band label
    heading(
        fig,
        t,
        "With the original weights, pyMIRAW reproduces the published benchmark",
        "Mean over the paper's 10 balanced test sets per configuration "
        "(finder, NF = no filter, AE = accessibility filter)",
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.77, bottom=0.08, wspace=0.08)
    fig.savefig(
        out / f"{name}-{'dark' if t is THEMES['dark'] else 'light'}.svg",
        facecolor=t["surface"],
        metadata={"Date": None},
    )
    plt.close(fig)


def gate2(t, name):
    cv = pd.read_csv(res / "cv_paper_regime.csv")
    site_ours = cv[(cv.orientation == "old-default") & ~cv.collapsed].groupby("fold").Accuracy.mean()
    site_paper = pd.read_csv(root / "tests/fixtures/paper_cv_xent_models_rescored.csv").set_index("fold").Accuracy
    g = pd.read_csv(res / "gene_level_retrained.csv")
    gene_ours = g[g.orientation == "old-default"].groupby("fold").Accuracy.mean()
    gene_paper = pd.read_csv(res / "gene_level_paper_folds.csv").groupby("fold").Accuracy.mean()
    published = pd.read_csv(root / "tests/fixtures/paper_global_summary.csv").Accuracy.mean()

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), facecolor=t["surface"])
    panels = (
        (axes[0], site_ours, site_paper, "Site level (CV test fold)"),
        (axes[1], gene_ours, gene_paper, "Gene level (all configurations and test sets)"),
    )
    for ax, ours, paper, title in panels:
        style(ax, t)
        ax.grid(axis="x", visible=False)
        x = ours.index.to_numpy()
        ax.scatter(
            x - 0.12,
            paper.loc[x],
            s=64,
            color=t["paper"],
            edgecolor=t["surface"],
            linewidth=1.5,
            zorder=3,
            label="paper's fold model",
        )
        ax.scatter(
            x + 0.12,
            ours,
            s=64,
            color=t["ours"],
            edgecolor=t["surface"],
            linewidth=1.5,
            zorder=3,
            label="pyMIRAW, retrained",
        )
        ax.set_xticks(x)
        ax.set_xlabel("cross-validation fold", color=t["ink2"], fontsize=9)
        ax.set_title(f"{title}: accuracy", color=t["ink"], fontsize=10, loc="left")
    axes[1].axhline(published, color=t["muted"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    axes[1].text(
        8.5,
        published + 0.0015,
        "published (paper's fold 6 model)",
        color=t["muted"],
        fontsize=8,
        ha="right",
        va="bottom",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(
        handles[::-1],
        labels[::-1],
        loc="upper left",
        bbox_to_anchor=(0.005, 0.86),
        ncol=2,
        frameon=False,
        fontsize=9,
        handletextpad=0.3,
        columnspacing=1.5,
    )
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    heading(
        fig,
        t,
        "Retrained from scratch, pyMIRAW models vary from fold to fold as much as the paper's own",
        "Each dot is one fold's model (pyMIRAW: mean over its seeds). Same folds, pipeline and ViennaRNA for both.",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.72, bottom=0.12, wspace=0.16)
    fig.savefig(
        out / f"{name}-{'dark' if t is THEMES['dark'] else 'light'}.svg",
        facecolor=t["surface"],
        metadata={"Date": None},
    )
    plt.close(fig)


if __name__ == "__main__":
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.family": "sans-serif", "svg.fonttype": "none", "font.size": 9, "svg.hashsalt": "pymiraw"}
    )
    for theme in THEMES.values():
        gate1(theme, "gate1_parity")
        gate2(theme, "gate2_folds")
    print("wrote", sorted(p.name for p in out.glob("*.svg")))
