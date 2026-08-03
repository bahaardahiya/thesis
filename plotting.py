"""
plotting.py

Produces the thesis figures from output/tables/bsadf_results.csv:

  1. output/figures/bsadf_full_sample.png
     Two panels sharing the x-axis: log price-dividend ratio (context) on
     top, BSADF statistic vs both critical value sequences (Monte Carlo and
     wild bootstrap, 95%) on the bottom, with periods where BSADF exceeds
     the wild-bootstrap threshold shaded.

  2. output/figures/bsadf_2021_episode.png
     Same bottom panel, zoomed on the 2020-2022 window that contains the
     main flagged episode, for a detail figure in the results chapter.

Colors follow the categorical palette slots 1 (blue), 2 (orange), 8 (red)
from the studio data-viz reference palette (colorblind-validated ordering);
line style is also varied (solid / dashed / dotted) so the figure remains
legible if printed in grayscale.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

RESULTS_PATH = "output/tables/bsadf_results.csv"
FIG_DIR = "output/figures"

COLOR_STAT = "#2a78d6"      # blue  -- BSADF statistic
COLOR_MC = "#eb6834"        # orange -- Monte Carlo 95% CV
COLOR_WB = "#1baf7a"        # aqua  -- wild bootstrap 95% CV
COLOR_FLAG = "#d03b3b"      # red (status: critical) -- flagged/exceedance shading
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


def _shade_exceedance(ax, dates, stat, cv, color=COLOR_FLAG, alpha=0.15):
    exceed = stat > cv
    ax.fill_between(dates, ax.get_ylim()[0], ax.get_ylim()[1], where=exceed,
                     color=color, alpha=alpha, linewidth=0, step=None)


def plot_full_sample(df: pd.DataFrame):
    fig, (ax_price, ax_stat) = plt.subplots(
        2, 1, figsize=(11, 7.5), sharex=True, height_ratios=[1, 1.6],
        gridspec_kw={"hspace": 0.08},
    )

    ax_price.plot(df["date"], df["log_pd_ratio"], color=COLOR_PRICE, linewidth=1.4)
    ax_price.set_ylabel("log(Price / Dividend)")
    ax_price.set_title(
        "NASDAQ-100: log price-dividend ratio and BSADF bubble statistic, 2005–2025",
        loc="left", fontsize=12, color=COLOR_INK, pad=10,
    )

    ax_stat.plot(df["date"], df["bsadf_stat"], color=COLOR_STAT, linewidth=1.3,
                 label="BSADF statistic", zorder=4)
    ax_stat.plot(df["date"], df["cv_mc_95"], color=COLOR_MC, linewidth=1.2,
                 linestyle="--", label="95% critical value (Monte Carlo)", zorder=3)
    ax_stat.plot(df["date"], df["cv_wb_95"], color=COLOR_WB, linewidth=1.2,
                 linestyle=":", label="95% critical value (wild bootstrap)", zorder=3)

    ymin = min(df["bsadf_stat"].min(), df["cv_mc_95"].min(), df["cv_wb_95"].min()) - 0.2
    ymax = max(df["bsadf_stat"].max(), df["cv_mc_95"].max(), df["cv_wb_95"].max()) + 0.2
    ax_stat.set_ylim(ymin, ymax)

    exceed_wb = df["bsadf_stat"] > df["cv_wb_95"]
    ax_stat.fill_between(df["date"], ymin, ymax, where=exceed_wb, color=COLOR_FLAG,
                          alpha=0.15, linewidth=0, label="Flagged by wild bootstrap (95%)")

    ax_stat.axhline(0, color=COLOR_MUTED, linewidth=0.8)
    ax_stat.set_ylabel("BSADF statistic")
    ax_stat.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)

    ax_stat.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_stat.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/bsadf_full_sample.png", dpi=200)
    plt.close(fig)


def plot_episode_zoom(df: pd.DataFrame, start="2020-01-01", end="2022-06-30"):
    mask = (df["date"] >= start) & (df["date"] <= end)
    d = df[mask]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(d["date"], d["bsadf_stat"], color=COLOR_STAT, linewidth=1.6, label="BSADF statistic")
    ax.plot(d["date"], d["cv_mc_95"], color=COLOR_MC, linewidth=1.3, linestyle="--",
            label="95% critical value (Monte Carlo)")
    ax.plot(d["date"], d["cv_wb_95"], color=COLOR_WB, linewidth=1.3, linestyle=":",
            label="95% critical value (wild bootstrap)")

    ymin = min(d["bsadf_stat"].min(), d["cv_mc_95"].min(), d["cv_wb_95"].min()) - 0.2
    ymax = max(d["bsadf_stat"].max(), d["cv_mc_95"].max(), d["cv_wb_95"].max()) + 0.2
    exceed_wb = d["bsadf_stat"] > d["cv_wb_95"]
    ax.fill_between(d["date"], ymin, ymax, where=exceed_wb, color=COLOR_FLAG,
                     alpha=0.15, linewidth=0, label="Flagged by wild bootstrap (95%)")
    ax.set_ylim(ymin, ymax)

    ax.axhline(0, color=COLOR_MUTED, linewidth=0.8)
    ax.set_ylabel("BSADF statistic")
    ax.set_title("Detail: 2020–2022 pandemic-era rally and reversal", loc="left", fontsize=12, pad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
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
