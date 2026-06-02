"""Plot benchmark results.

Reads benchmark/results/*.csv and writes:

  * docs/assets/benchmark.png            — N-sweep, threads=1 where
                                            available, full HiGHS solve.
  * docs/assets/benchmark_threads.png    — fixed N, sweep across thread
                                            counts (linear x-axis).
  * docs/assets/benchmark_buildonly.png  — N-sweep, HiGHS short-circuited
                                            via time_limit=1e-6.
  * docs/assets/benchmark_network.png    — N-sweep on the irregular
                                            network-flow LP (separate
                                            CSV, separate scale).

Conventions used by all three "dense LP" figures:

  * build_s, solve_s panels: log y-axis with a *shared* range across
    the three figures so they're directly comparable.
  * peak_rss_mb panel: linear y-axis starting at zero with a shared
    upper bound across the three figures.

The network figure has its own scale (very different LP size).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent

TOOL_COLORS = {
    "polar": "#1f77b4",
    "polar_sm": "#1f77b4",
    "polar_da": "#16a085",
    "linopy": "#d97706",
    "pyomo": "#a91b0d",
    "pulp": "#6b21a8",
    "polar_net": "#1f77b4",
    "polar_sm_net": "#1f77b4",
    "polar_da_net": "#16a085",
    "linopy_net": "#d97706",
    "pyomo_net": "#a91b0d",
}
# Same colour family for the polar variants where one is "the same tool
# with a knob flipped" (regular vs save_memory). polar_da uses a separate
# colour because it's a different code path (block-COO dense-axis arm),
# not the same path with a knob.
TOOL_LINESTYLES = {
    "polar": "-",
    "polar_sm": "--",
    "polar_da": "-",
    "linopy": "-",
    "pyomo": "-",
    "pulp": "-",
    "polar_net": "-",
    "polar_sm_net": "--",
    "polar_da_net": "-",
    "linopy_net": "-",
    "pyomo_net": "-",
}
TOOL_MARKERS = {
    "polar": "o",
    "polar_sm": "s",
    "polar_da": "D",
    "linopy": "o",
    "pyomo": "o",
    "pulp": "^",
    "polar_net": "o",
    "polar_sm_net": "s",
    "polar_da_net": "D",
    "linopy_net": "o",
    "pyomo_net": "o",
}
TOOL_LABELS = {
    "polar": "polar-high (regular)",
    "polar_sm": "polar-high (save_memory)",
    "polar_da": "polar-high (dense_axes)",
    "linopy": "linopy",
    "pyomo": "Pyomo",
    "pulp": "PuLP (HiGHS_CMD)",
    "polar_net": "polar-high (regular)",
    "polar_sm_net": "polar-high (save_memory)",
    "polar_da_net": "polar-high (dense_axes)",
    "linopy_net": "linopy",
    "pyomo_net": "Pyomo",
}
TOOL_ORDER_DENSE = ["polar", "polar_sm", "polar_da", "linopy", "pyomo", "pulp"]
TOOL_ORDER_NET = ["polar_net", "polar_sm_net", "polar_da_net", "linopy_net", "pyomo_net"]


def _load_all(in_csv_glob: list[str]) -> pd.DataFrame:
    frames = []
    for path in in_csv_glob:
        p = Path(path)
        if not p.exists():
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["build_s", "solve_s", "peak_rss_mb"])
    if "threads" not in df.columns:
        df["threads"] = -1
    # Legacy data may still carry the now-removed ``polar_lean`` /
    # ``polar_lean_net`` tool ids — fold them into the canonical
    # ``polar`` / ``polar_net`` since the lean settings are now the
    # engine defaults.
    df["tool"] = df["tool"].replace({"polar_lean": "polar", "polar_lean_net": "polar_net"})
    df = df.drop_duplicates(
        subset=["tool", "N", "threads", "rep"],
        keep="last",
    )
    return df


def _aggregate(df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    # ``total_s`` is build_s + solve_s per *rep* (the apples-to-apples
    # cross-tool measurement: full time from no-model to solution).
    # ``peak_mb`` is cgroup-level memory.peak when available (kernel
    # cgroup v2 accounting, what OOM would actually use), and falls back
    # to ru_maxrss for legacy CSVs that pre-date the cgroup column.
    df = df.assign(total_s=df["build_s"] + df["solve_s"])
    if "cgroup_peak_mb" in df.columns:
        peak = df["cgroup_peak_mb"].where(df["cgroup_peak_mb"] > 0, df["peak_rss_mb"])
        df = df.assign(peak_mb=peak)
    else:
        df = df.assign(peak_mb=df["peak_rss_mb"])
    return df.groupby(group, as_index=False).agg(
        build_s=("build_s", "median"),
        solve_s=("solve_s", "median"),
        total_s=("total_s", "median"),
        peak_rss_mb=("peak_mb", "median"),
    )


def _shared_limits(*aggs: pd.DataFrame) -> dict:
    """Compute shared y-axis limits across the apples-to-apples figures.

    Returns: {"time_s": (lo, hi), "peak_rss_mb": (lo, hi)}

    ``time_s`` is the shared y-range for the build / solve / total
    panels — all in seconds, on the same log axis so values are
    eyeball-comparable.

    ``peak_rss_mb`` is also log-scaled (consistent with the time
    panels) with both bounds shared so memory values are
    eyeball-comparable across files.
    """
    builds, solves, totals, peaks = [], [], [], []
    for agg in aggs:
        if agg is None or agg.empty:
            continue
        builds.append(agg["build_s"])
        solves.append(agg["solve_s"])
        totals.append(agg["total_s"])
        peaks.append(agg["peak_rss_mb"])
    if not builds:
        return {}
    all_times = pd.concat(builds + solves + totals)
    all_peak = pd.concat(peaks)
    return {
        "time_s": (all_times.min() * 0.6, all_times.max() * 1.5),
        "peak_rss_mb": (all_peak.min() * 0.6, all_peak.max() * 1.5),
    }


def _draw_three_panels(
    agg: pd.DataFrame,
    *,
    x_col: str,
    x_label: str,
    title_suffix: str,
    out_path: Path,
    tool_order: list[str],
    x_log: bool = True,
    y_limits: dict | None = None,
    panel_titles: tuple[str, str, str] = (
        "Time in build()",
        "Time in solve()",
        "Peak memory",
    ),
) -> None:
    """3-panel layout: build_s | solve_s | peak_rss_mb. Used for the
    headline figure that replicates linopy's benchmark format."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True)

    tools_present = [t for t in tool_order if t in set(agg["tool"])]
    for tool in tools_present:
        sub = agg[agg["tool"] == tool].sort_values(x_col)
        if sub.empty:
            continue
        kw = dict(
            marker=TOOL_MARKERS.get(tool, "o"),
            linestyle=TOOL_LINESTYLES.get(tool, "-"),
            linewidth=1.6,
            markersize=5,
            color=TOOL_COLORS[tool],
            label=TOOL_LABELS[tool],
        )
        axes[0].plot(sub[x_col], sub["build_s"], **kw)
        axes[1].plot(sub[x_col], sub["solve_s"], **kw)
        axes[2].plot(sub[x_col], sub["peak_rss_mb"], **kw)

    for ax in axes:
        if x_log:
            ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.grid(True, which="both", linestyle=":", alpha=0.45)

    for ax_i in (0, 1):
        axes[ax_i].set_yscale("log")
        if y_limits and "time_s" in y_limits:
            axes[ax_i].set_ylim(*y_limits["time_s"])

    axes[2].set_yscale("log")
    if y_limits and "peak_rss_mb" in y_limits:
        axes[2].set_ylim(*y_limits["peak_rss_mb"])

    axes[0].set_ylabel("seconds")
    axes[1].set_ylabel("seconds")
    axes[2].set_ylabel("MB")

    for ax, t in zip(axes, panel_titles):
        ax.set_title(t)
    axes[0].legend(loc="best", fontsize=9)

    fig.suptitle(
        f"polar-high vs linopy vs Pyomo — LP benchmark {title_suffix}",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _draw_threading_benefit(
    df_net: pd.DataFrame,
    *,
    out_path: Path,
    y_limits: dict,
) -> None:
    """Three-line figure for the network LP: polar at 1 thread, polar
    at 32 threads, linopy at 1 thread. Shows where polars parallelism
    starts paying as N grows."""
    df = df_net.assign(total_s=df_net["build_s"] + df_net["solve_s"])
    if "cgroup_peak_mb" in df.columns:
        peak = df["cgroup_peak_mb"].where(df["cgroup_peak_mb"] > 0, df["peak_rss_mb"])
        df = df.assign(peak_mb=peak)
    else:
        df = df.assign(peak_mb=df["peak_rss_mb"])
    agg = df.groupby(["tool", "threads", "N"], as_index=False).agg(
        total_s=("total_s", "median"),
        peak_rss_mb=("peak_mb", "median"),
    )

    series = [
        ("polar_net", 1, "polar-high (1 thread)", "#1f77b4", "-"),
        ("polar_net", 32, "polar-high (32 threads)", "#1f77b4", "--"),
        ("linopy_net", 1, "linopy (1 thread)", "#d97706", "-"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True)
    for tool, threads, label, color, linestyle in series:
        sub = agg[(agg["tool"] == tool) & (agg["threads"] == threads)].sort_values("N")
        if sub.empty:
            continue
        kw = dict(
            marker="o",
            linewidth=1.6,
            markersize=5,
            color=color,
            linestyle=linestyle,
            label=label,
        )
        axes[0].plot(sub["N"], sub["total_s"], **kw)
        axes[1].plot(sub["N"], sub["peak_rss_mb"], **kw)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.45)
        ax.set_xlabel("N (nodes; edges = 5·N, T = 168)")

    if y_limits.get("time_s"):
        axes[0].set_ylim(*y_limits["time_s"])
    if y_limits.get("peak_rss_mb"):
        axes[1].set_ylim(*y_limits["peak_rss_mb"])
    axes[0].set_ylabel("seconds")
    axes[1].set_ylabel("MB")
    axes[0].set_title("Time in build() + solve()  (HiGHS short-circuited)")
    axes[1].set_title("Peak memory")
    axes[0].legend(loc="best", fontsize=9)

    fig.suptitle(
        "polar-high vs linopy — network LP, threading benefit on polars",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _draw_two_panels(
    agg: pd.DataFrame,
    *,
    x_col: str,
    x_label: str,
    title_suffix: str,
    out_path: Path,
    tool_order: list[str],
    x_log: bool = True,
    y_limits: dict | None = None,
    time_panel_title: str = "Time in build() + solve()",
) -> None:
    """2-panel layout: total_s (build+solve) | peak_rss_mb. The total
    is the apples-to-apples cross-tool measurement — each tool draws
    the build/solve boundary in a different place, so summing the two
    is what's meaningful to compare."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True)

    tools_present = [t for t in tool_order if t in set(agg["tool"])]
    for tool in tools_present:
        sub = agg[agg["tool"] == tool].sort_values(x_col)
        if sub.empty:
            continue
        kw = dict(
            marker=TOOL_MARKERS.get(tool, "o"),
            linestyle=TOOL_LINESTYLES.get(tool, "-"),
            linewidth=1.6,
            markersize=5,
            color=TOOL_COLORS[tool],
            label=TOOL_LABELS[tool],
        )
        axes[0].plot(sub[x_col], sub["total_s"], **kw)
        axes[1].plot(sub[x_col], sub["peak_rss_mb"], **kw)

    for ax in axes:
        if x_log:
            ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.grid(True, which="both", linestyle=":", alpha=0.45)
        ax.set_yscale("log")

    if y_limits and "time_s" in y_limits:
        axes[0].set_ylim(*y_limits["time_s"])
    if y_limits and "peak_rss_mb" in y_limits:
        axes[1].set_ylim(*y_limits["peak_rss_mb"])

    axes[0].set_ylabel("seconds")
    axes[1].set_ylabel("MB")
    axes[0].set_title(time_panel_title)
    axes[1].set_title("Peak memory")
    axes[0].legend(loc="best", fontsize=9)

    fig.suptitle(
        f"polar-high vs linopy vs Pyomo — LP benchmark {title_suffix}",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="in_csvs",
        nargs="+",
        default=[
            str(HERE / "results" / "cgroup" / "dense_fullsolve.csv"),
            str(HERE / "results" / "results.csv"),
            str(HERE / "results" / "results_v1_1.csv"),
        ],
    )
    ap.add_argument(
        "--in-buildonly",
        nargs="+",
        default=[
            str(HERE / "results" / "cgroup" / "dense_buildonly.csv"),
            str(HERE / "results" / "cgroup" / "threads_dense.csv"),
            str(HERE / "results" / "results_buildonly.csv"),
        ],
    )
    ap.add_argument(
        "--in-network",
        nargs="+",
        default=[
            str(HERE / "results" / "cgroup" / "network.csv"),
            str(HERE / "results" / "cgroup" / "network_threads.csv"),
            str(HERE / "results" / "results_network.csv"),
        ],
    )
    ap.add_argument(
        "--out-main",
        default=str(REPO_ROOT / "docs" / "assets" / "benchmark.png"),
        help=(
            "Headline (2-panel, build-only): tool-side time + memory. "
            "HiGHS short-circuited so the comparison is purely the "
            "modelling-layer cost."
        ),
    )
    ap.add_argument(
        "--out-replication",
        default=str(REPO_ROOT / "docs" / "assets" / "benchmark_replication.png"),
        help="3-panel linopy-style replication (build / solve / memory) with full HiGHS.",
    )
    ap.add_argument(
        "--out-threads",
        default=str(REPO_ROOT / "docs" / "assets" / "benchmark_threads.png"),
    )
    ap.add_argument(
        "--out-network",
        default=str(REPO_ROOT / "docs" / "assets" / "benchmark_network.png"),
    )
    ap.add_argument(
        "--out-threading-benefit",
        default=str(REPO_ROOT / "docs" / "assets" / "benchmark_threading_benefit.png"),
        help=(
            "Three-line figure (polar @1 thread, polar @32 threads, "
            "linopy @1 thread) on the network LP — shows where polars's "
            "parallelism starts paying."
        ),
    )
    args = ap.parse_args()

    # --- Dense-LP family --------------------------------------------------
    df = _load_all(args.in_csvs)

    # Headline: per-(tool, N) cell, pick the lowest available thread
    # count. Polar/polar_lean/linopy will mostly resolve to threads=1
    # (their fastest+leanest); Pyomo (single-threaded by design) keeps
    # its threads=32 rows where that's all we ran. Result: full N
    # coverage per tool, no rows silently dropped.
    if not df.empty:
        df = df.copy()
        df["_min_t"] = df.groupby(["tool", "N"])["threads"].transform("min")
        df_main = df[df["threads"] == df["_min_t"]].drop(columns="_min_t")
    else:
        df_main = df
    agg_main = _aggregate(df_main, group=["tool", "N"])

    # Threading: read from the *build-only* CSV at a fixed N. We use
    # build-only because the threads scaling story is about the
    # modelling layer, not HiGHS — HiGHS is single-threaded the same
    # way everywhere. N=300 is small enough that pyomo can run at
    # every thread count cheaply, but big enough to amortize startup.
    df_thr_src = _load_all(list(args.in_buildonly))
    N_thr_target = 300
    if not df_thr_src.empty and (df_thr_src["N"] == N_thr_target).any():
        df_thr = df_thr_src[df_thr_src["N"] == N_thr_target]
        agg_thr = _aggregate(df_thr, group=["tool", "threads"])
        counts = agg_thr.groupby("tool").size()
        agg_thr = agg_thr[agg_thr["tool"].isin(set(counts[counts > 1].index))]
        N_thr = N_thr_target
    else:
        agg_thr = pd.DataFrame()
        N_thr = None

    # Build-only — apply the same lowest-threads-per-cell rule so the
    # headline figure cleanly reflects threads=1 (single-thread) data
    # for polar and linopy, threads=32 (= threads=1 effectively) for
    # Pyomo. Without this the median would mix threads=1 and 32 reps.
    df_b_full = _load_all(list(args.in_buildonly))
    if not df_b_full.empty:
        df_b = df_b_full.copy()
        df_b["_min_t"] = df_b.groupby(["tool", "N"])["threads"].transform("min")
        df_b = df_b[df_b["threads"] == df_b["_min_t"]].drop(columns="_min_t")
        agg_b = _aggregate(df_b, group=["tool", "N"])
    else:
        df_b = df_b_full
        agg_b = pd.DataFrame()

    # Shared y-axis limits across the three dense-LP figures.
    y_lims = _shared_limits(agg_main, agg_thr, agg_b)

    # Headline figure: build-only (HiGHS short-circuited). This
    # isolates the modelling-layer cost — HiGHS is identical across
    # all three tools, so the build-only timing IS the comparison
    # we actually care about.
    if not agg_b.empty:
        _draw_two_panels(
            agg_b,
            x_col="N",
            x_label="N (variable grid is N × N)",
            title_suffix=("(build-only — HiGHS time-limited to ~1 µs; modelling-layer cost only)"),
            out_path=Path(args.out_main),
            tool_order=TOOL_ORDER_DENSE,
            y_limits=y_lims,
            time_panel_title=("Time in build() + solve()  (HiGHS short-circuited)"),
        )

    # Replication: 3-panel linopy-style format with full HiGHS.
    # Sits in a methodology section of the page, not the headline.
    if not agg_main.empty:
        y_lims_repl = _shared_limits(agg_main)
        _draw_three_panels(
            agg_main,
            x_col="N",
            x_label="N (variable grid is N × N)",
            title_suffix=("(linopy-format replication: full HiGHS solve included)"),
            out_path=Path(args.out_replication),
            tool_order=TOOL_ORDER_DENSE,
            y_limits=y_lims_repl,
        )

    if not agg_thr.empty:
        _draw_two_panels(
            agg_thr,
            x_col="threads",
            x_label="threads",
            title_suffix=(f"(N = {N_thr}, scaling with threads, build-only)"),
            out_path=Path(args.out_threads),
            tool_order=TOOL_ORDER_DENSE,
            x_log=False,
            y_limits=y_lims,
            time_panel_title=("Time in build() + solve()  (HiGHS short-circuited)"),
        )

    # --- Network LP family (build-only too) ------------------------------
    df_net = _load_all(list(args.in_network))
    if not df_net.empty:
        # Standard network plot uses lowest-threads-per-cell rule
        # (gives polar threads=1 if available, threads=32 if not).
        df_net_main = df_net.copy()
        df_net_main["_min_t"] = df_net_main.groupby(["tool", "N"])["threads"].transform("min")
        df_net_main = df_net_main[df_net_main["threads"] == df_net_main["_min_t"]].drop(
            columns="_min_t"
        )
        agg_net = _aggregate(df_net_main, group=["tool", "N"])
        y_lims_net = _shared_limits(agg_net)
        _draw_two_panels(
            agg_net,
            x_col="N",
            x_label="N (nodes; edges = 5·N, T = 168)",
            title_suffix=("(network LP — irregular topology, build-only, 1 thread)"),
            out_path=Path(args.out_network),
            tool_order=TOOL_ORDER_NET,
            y_limits=y_lims_net,
            time_panel_title=("Time in build() + solve()  (HiGHS short-circuited)"),
        )

        # --- Threading-benefit figure: same network LP, three series.
        # If threads=32 polar_net data is present, show where polars
        # parallelism starts paying.
        thr_pairs = df_net.groupby(["tool", "threads"]).size().reset_index()
        has_polar_t1 = ((thr_pairs["tool"] == "polar_net") & (thr_pairs["threads"] == 1)).any()
        has_polar_t32 = ((thr_pairs["tool"] == "polar_net") & (thr_pairs["threads"] == 32)).any()
        has_linopy_t1 = ((thr_pairs["tool"] == "linopy_net") & (thr_pairs["threads"] == 1)).any()
        if has_polar_t1 and has_polar_t32 and has_linopy_t1:
            _draw_threading_benefit(
                df_net,
                out_path=Path(args.out_threading_benefit),
                y_limits=y_lims_net,
            )


if __name__ == "__main__":
    main()
