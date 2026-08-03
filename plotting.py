"""
plotting.py

Produces the thesis figures from output/tables/bsadf_results.csv:

  1. output/figures/bsadf_full_sample.png
     Two panels sharing the x-axis: log price-dividend ratio (context) on
     top, BSADF statistic vs both critical value sequences (Monte Carlo and
     wild bootstrap, at 90% and 95%) on the bottom. Two-tier shading marks
     weeks exceeding the wild-bootstrap 90% threshold (light) and the
     stricter 95% threshold (dark).

  2. output/figures/bsadf_2021_episode.png
     Same bottom panel, zoomed on the 2020-2022 window that contains the
     main flagged episode, for a detail figure in the results chapter.

Colors follow the categorical palette slots 1 (blue), 2 (orange), 8 (red)
from the studio data-viz reference palette (colorblind-validated ordering).
Hue encodes method (orange=Monte Carlo, aqua=wild bootstrap); line style
encodes significance level (dashed=95%, dotted=90%) so the figure stays
legible if printed in grayscale.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

RESULTS_PATH = "output/tables/bsadf_results.csv"
FIG_DIR = "output/figures"

COLOR_STAT = "#2a78d6"      # blue  -- BSADF statistic
COLOR_MC = "#eb6834"        # orange -- Monte Carlo critical values
COLOR_WB = "#1baf7a"        # aqua  -- wild bootstrap critical values
COLOR_FLAG_STRONG = "#d03b3b"   # red (status: critical) -- exceeds WB 95%
COLOR_FLAG_WEAK = "#eda100"     # yellow (status: warning) -- exceeds WB 90% only
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_PRICE = "#52514e"     # secondary ink -- context panel line

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.edgecolor": COLOR_MUTED,
        "axes.labelcolor": COLOR_INK,
        "text.color": COLOR_INK,
        "xtick.color": COLOR_MUTED,
        "ytick.color": COLOR_MUTED,
        "axes.grid": True,
        "grid.color": COLOR_GRID,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _plot_stat_and_cvs(ax, d: pd.DataFrame):
    """Draws the BSADF statistic plus MC/WB critical values at 90% and 95%,
    with two-tier shading for weeks exceeding the wild-bootstrap thresholds.
    Returns (ymin, ymax) so callers can reuse the scale.
    """
    ax.plot(d["date"], d["bsadf_stat"], color=COLOR_STAT, linewidth=1.4,
             label="BSADF statistic", zorder=5)

    ax.plot(d["date"], d["cv_mc_95"], color=COLOR_MC, linewidth=1.2,
             linestyle="--", label="95% CV (Monte Carlo)", zorder=3)
    ax.plot(d["date"], d["cv_mc_90"], color=COLOR_MC, linewidth=1.0,
             linestyle=":", alpha=0.75, label="90% CV (Monte Carlo)", zorder=3)

    ax.plot(d["date"], d["cv_wb_95"], color=COLOR_WB, linewidth=1.2,
             linestyle="--", label="95% CV (wild bootstrap)", zorder=4)
    ax.plot(d["date"], d["cv_wb_90"], color=COLOR_WB, linewidth=1.0,
             linestyle=":", alpha=0.75, label="90% CV (wild bootstrap)", zorder=4)

    all_vals = pd.concat([d["bsadf_stat"], d["cv_mc_90"], d["cv_mc_95"],
                           d["cv_wb_90"], d["cv_wb_95"]])
    ymin, ymax = all_vals.min() - 0.2, all_vals.max() + 0.2

    exceed_95 = d["bsadf_stat"] > d["cv_wb_95"]
    exceed_90_only = (d["bsadf_stat"] > d["cv_wb_90"]) & ~exceed_95

    ax.fill_between(d["date"], ymin, ymax, where=exceed_90_only, color=COLOR_FLAG_WEAK,
                     alpha=0.18, linewidth=0, label="Exceeds wild bootstrap 90% CV")
    ax.fill_between(d["date"], ymin, ymax, where=exceed_95, color=COLOR_FLAG_STRONG,
                     alpha=0.18, linewidth=0, label="Exceeds wild bootstrap 95% CV")

    ax.axhline(0, color=COLOR_MUTED, linewidth=0.8)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("BSADF statistic")
    return ymin, ymax


def plot_full_sample(df: pd.DataFrame):
    fig, (ax_price, ax_stat) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, height_ratios=[1, 1.8],
        gridspec_kw={"hspace": 0.08},
    )

    ax_price.plot(df["date"], df["log_pd_ratio"], color=COLOR_PRICE, linewidth=1.4)
    ax_price.set_ylabel("log(Price / Dividend)")
    ax_price.set_title(
        "NASDAQ-100: log price-dividend ratio and BSADF bubble statistic, 2005–2025",
        loc="left", fontsize=12, color=COLOR_INK, pad=10,
    )

    _plot_stat_and_cvs(ax_stat, df)
    ax_stat.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=3)

    ax_stat.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_stat.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/bsadf_full_sample.png", dpi=200)
    plt.close(fig)


def plot_episode_zoom(df: pd.DataFrame, start="2020-01-01", end="2022-06-30"):
    mask = (df["date"] >= start) & (df["date"] <= end)
    d = df[mask]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    _plot_stat_and_cvs(ax, d)
    ax.set_title("Detail: 2020–2022 pandemic-era rally and reversal", loc="left", fontsize=12, pad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/bsadf_2021_episode.png", dpi=200)
    plt.close(fig)


def main():
    df = pd.read_csv(RESULTS_PATH, parse_dates=["date"])
    plot_full_sample(df)
    plot_episode_zoom(df)
    print(f"Wrote {FIG_DIR}/bsadf_full_sample.png and {FIG_DIR}/bsadf_2021_episode.png")


if __name__ == "__main__":
    main()
