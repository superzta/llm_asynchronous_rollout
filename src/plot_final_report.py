"""
Publication-quality plots for the final report / poster.

Reads per-run `summary.json` (and optionally `results.jsonl`) under
results/experiments/<exp>/ and produces:

    plots/fig_overview_poster.png     -- 2x2 panel for the poster
    plots/fig_reward_curves.png       -- reward vs step, mean +/- CI across seeds
    plots/fig_pass_rate_curves.png    -- pass_rate vs step, smoothed, per mode
    plots/fig_pareto.png              -- throughput vs final pass-rate Pareto
    plots/fig_time_to_threshold.png   -- wall-clock to reach target pass-rate
    plots/fig_staleness_violin.png    -- staleness distribution per async mode
    plots/fig_reward_per_ktoken.png   -- sample efficiency
    plots/fig_updates_per_sec.png     -- effective updates/s per mode

    tables/summary_headline.csv       -- one-row-per-run summary
    tables/summary_headline.md        -- same in markdown
    tables/mode_aggregate.csv         -- seed-averaged per (mode, k, decoupled)
    tables/mode_aggregate.md          -- same in markdown

A small INDEX.md in the same directory tells you which figure to use where.
"""
import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODE_COLOR = {
    "sync_train": "#264653",
    "async_train": "#E76F51",
    "async_areal_style": "#2A9D8F",
}
MODE_LABEL = {
    "sync_train": "Sync (oracle)",
    "async_train": "Async (non-interruptible)",
    "async_areal_style": "Async AReaL (interruptible)",
}
MODE_ORDER = ["sync_train", "async_train", "async_areal_style"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _iter_runs(exp_dir):
    # type: (Path) -> List[Tuple[str, Dict[str, Any], Path]]
    out = []
    for d in sorted(exp_dir.iterdir()):
        if not d.is_dir():
            continue
        sj = d / "summary.json"
        if not sj.exists():
            continue
        try:
            summary = json.loads(sj.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append((d.name, summary, d))
    return out


def _cfg(s):
    return s.get("config", {}) or {}


def _mode(s):
    return s.get("mode") or _cfg(s).get("mode") or "unknown"


def _staleness(s):
    v = s.get("staleness_k")
    if v is None:
        v = _cfg(s).get("staleness_k")
    try:
        return int(v) if v is not None else 0
    except Exception:
        return 0


def _decoupled(s):
    v = _cfg(s).get("decoupled_objective", 1)
    try:
        return int(v)
    except Exception:
        return 1


def _seed(s):
    return int(_cfg(s).get("seed", 0))


def _effective_tokens_per_sec(s):
    tps = float(s.get("avg_tokens_per_sec") or 0.0)
    acc = s.get("accepted_fraction")
    if acc is None:
        acc = 1.0
    return tps * float(acc)


def _final_pass_rate(s):
    # Prefer pass_rate_per_update (reported in every runner); fall back to avg pass_rate_by_step.
    val = s.get("pass_rate_per_update")
    if val is not None:
        return float(val)
    ps = [x.get("pass_rate", 0.0) for x in s.get("pass_rate_by_step", [])]
    return float(sum(ps) / len(ps)) if ps else 0.0


def _smooth(xs, window):
    if not xs:
        return xs
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        s = xs[lo:i + 1]
        out.append(sum(s) / len(s))
    return out


def _stack_for_mean_ci(series_list):
    """Given list of lists (one per seed), returns (x, mean, lo, hi) with seed-stddev band."""
    if not series_list:
        return [], [], [], []
    L = min(len(s) for s in series_list)
    if L == 0:
        return [], [], [], []
    xs = list(range(1, L + 1))
    mean = []
    lo = []
    hi = []
    for i in range(L):
        col = [s[i] for s in series_list]
        m = sum(col) / len(col)
        if len(col) > 1:
            sd = statistics.stdev(col)
        else:
            sd = 0.0
        mean.append(m)
        lo.append(m - sd)
        hi.append(m + sd)
    return xs, mean, lo, hi


# ---------------------------------------------------------------------------
# Fig: reward curves per mode (mean across seeds, std band)
# ---------------------------------------------------------------------------
def _plot_reward_curves_by_mode(ax, runs, k_filter=None, decoupled_filter=1, smooth_window=5):
    by_mode_seeds = defaultdict(list)
    for _, s, _ in runs:
        if _decoupled(s) != decoupled_filter:
            continue
        if k_filter is not None and _mode(s) != "sync_train" and _staleness(s) != k_filter:
            continue
        rewards = [r["reward"] for r in s.get("reward_by_step", [])]
        if rewards:
            by_mode_seeds[_mode(s)].append(rewards)
    any_plotted = False
    for mode in MODE_ORDER:
        series = by_mode_seeds.get(mode, [])
        if not series:
            continue
        xs, mean, lo, hi = _stack_for_mean_ci(series)
        if not xs:
            continue
        win = max(1, len(xs) // 20) if smooth_window is None else smooth_window
        mean_s = _smooth(mean, win)
        lo_s = _smooth(lo, win)
        hi_s = _smooth(hi, win)
        color = MODE_COLOR.get(mode, "#888888")
        ax.plot(xs, mean_s, color=color, label=MODE_LABEL.get(mode, mode), linewidth=2)
        ax.fill_between(xs, lo_s, hi_s, color=color, alpha=0.18)
        any_plotted = True
    ax.set_xlabel("Training step")
    ax.set_ylabel("Per-step reward (smoothed, mean ± stdev)")
    title = "Learning curves by mode"
    if k_filter is not None:
        title += " (k=%d)" % k_filter
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if any_plotted:
        ax.legend(loc="lower right", fontsize=9)


def _plot_pass_rate_curves_by_mode(ax, runs, k_filter=None, decoupled_filter=1, smooth_window=5):
    by_mode_seeds = defaultdict(list)
    for _, s, _ in runs:
        if _decoupled(s) != decoupled_filter:
            continue
        if k_filter is not None and _mode(s) != "sync_train" and _staleness(s) != k_filter:
            continue
        prs = [x.get("pass_rate", 0.0) for x in s.get("pass_rate_by_step", [])]
        if prs:
            by_mode_seeds[_mode(s)].append(prs)
    for mode in MODE_ORDER:
        series = by_mode_seeds.get(mode, [])
        if not series:
            continue
        xs, mean, lo, hi = _stack_for_mean_ci(series)
        if not xs:
            continue
        win = max(1, len(xs) // 20) if smooth_window is None else smooth_window
        mean_s = _smooth(mean, win)
        color = MODE_COLOR.get(mode, "#888888")
        ax.plot(xs, mean_s, color=color, label=MODE_LABEL.get(mode, mode), linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Pass rate (smoothed)")
    title = "Task pass rate by mode"
    if k_filter is not None:
        title += " (k=%d)" % k_filter
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)


# ---------------------------------------------------------------------------
# Fig: Pareto (throughput vs final pass-rate)
# ---------------------------------------------------------------------------
def _plot_pareto(ax, runs):
    markers = {0: "o", 1: "D"}
    seen_labels = set()
    xs_all, ys_all = [], []
    for _, s, _ in runs:
        m = _mode(s)
        color = MODE_COLOR.get(m, "#888888")
        dec = _decoupled(s)
        x = _effective_tokens_per_sec(s)
        y = _final_pass_rate(s)
        k = _staleness(s)
        lbl = "%s%s" % (MODE_LABEL.get(m, m),
                        " (naive)" if dec == 0 else "")
        if lbl in seen_labels:
            lbl = None
        else:
            seen_labels.add(lbl)
        ax.scatter(x, y, s=60, color=color, marker=markers.get(dec, "o"),
                   edgecolor="black", linewidth=0.4, label=lbl, zorder=3)
        if k and m != "sync_train":
            ax.annotate("k=%d" % k, (x, y), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color="#333")
        xs_all.append(x)
        ys_all.append(y)
    ax.set_xlabel("Effective throughput (tokens/s, accepted)")
    ax.set_ylabel("Final pass rate (GSM8K)")
    ax.set_title("Throughput vs accuracy (Pareto view)")
    ax.grid(True, alpha=0.3)
    if seen_labels:
        ax.legend(fontsize=8, loc="best")


# ---------------------------------------------------------------------------
# Fig: Wall-clock time to reach a target pass-rate threshold
# ---------------------------------------------------------------------------
def _time_to_threshold(s, threshold):
    wall = float(s.get("wall_clock_sec") or 0.0)
    prs = [x.get("pass_rate", 0.0) for x in s.get("pass_rate_by_step", [])]
    n = len(prs)
    if n == 0 or wall <= 0:
        return None
    running = 0.0
    for i, pr in enumerate(prs, start=1):
        running += pr
        mean_so_far = running / i
        if mean_so_far >= threshold:
            return wall * (i / n)
    return None  # never reached


def _plot_time_to_threshold(ax, runs, threshold=0.10, decoupled_filter=1):
    by_mode = defaultdict(list)
    for _, s, _ in runs:
        if _decoupled(s) != decoupled_filter:
            continue
        t = _time_to_threshold(s, threshold)
        if t is not None:
            by_mode[_mode(s)].append(t)
    labels = []
    values = []
    colors = []
    errs = []
    for mode in MODE_ORDER:
        if by_mode.get(mode):
            xs = by_mode[mode]
            labels.append(MODE_LABEL.get(mode, mode))
            values.append(sum(xs) / len(xs))
            errs.append(statistics.stdev(xs) if len(xs) > 1 else 0.0)
            colors.append(MODE_COLOR.get(mode, "#888"))
    if not values:
        ax.text(0.5, 0.5, "No run reached threshold %.2f" % threshold,
                ha="center", va="center")
        return
    xs = list(range(len(values)))
    bars = ax.bar(xs, values, color=colors, yerr=errs, capsize=4)
    for i, (v, e) in enumerate(zip(values, errs)):
        ax.text(i, v + 0.02 * max(values), "%.1fs" % v, ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Wall-clock seconds to mean pass-rate ≥ %.2f" % threshold)
    ax.set_title("Time-to-threshold speedup")
    ax.grid(True, axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Fig: staleness distribution (violin) for async modes
# ---------------------------------------------------------------------------
def _plot_staleness_violin(ax, runs):
    data = []
    labels = []
    colors = []
    for _, s, _ in runs:
        m = _mode(s)
        if m not in ("async_train", "async_areal_style"):
            continue
        hist = s.get("staleness_histogram") or {}
        if not hist:
            continue
        samples = []
        for key, count in hist.items():
            try:
                samples.extend([int(key)] * int(count))
            except Exception:
                continue
        if not samples:
            continue
        data.append(samples)
        labels.append("%s\nk=%d, seed=%d" % (MODE_LABEL.get(m, m),
                                             _staleness(s), _seed(s)))
        colors.append(MODE_COLOR.get(m, "#888"))
    if not data:
        ax.text(0.5, 0.5, "No staleness histograms found",
                ha="center", va="center")
        return
    positions = list(range(1, len(data) + 1))
    vp = ax.violinplot(data, positions=positions, showmedians=True,
                       showextrema=False)
    for pc, color in zip(vp["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.55)
        pc.set_edgecolor("black")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("Per-sample staleness")
    ax.set_title("Realized staleness distribution (async runs)")
    ax.grid(True, axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Fig: reward per 1K tokens (sample efficiency)
# ---------------------------------------------------------------------------
def _plot_reward_per_ktoken(ax, runs, decoupled_filter=1):
    by_mode = defaultdict(list)
    for _, s, _ in runs:
        if _decoupled(s) != decoupled_filter:
            continue
        rewards = [r.get("reward", 0.0) for r in s.get("reward_by_step", [])]
        tokps = [r.get("tokens_per_sec", 0.0) for r in s.get("tokens_per_sec_by_step", [])]
        wall = float(s.get("wall_clock_sec") or 0.0)
        if not rewards or wall <= 0:
            continue
        total_reward = sum(rewards)
        # Approximate total tokens generated:
        # each step has avg_tokens_per_sec * latency, but we have tokens_per_sec per step;
        # use sum of tokens_per_sec * wall/n as a proxy.
        n = len(tokps)
        if n == 0:
            continue
        total_tokens = (wall / n) * sum(tokps)
        if total_tokens <= 0:
            continue
        by_mode[_mode(s)].append(total_reward / (total_tokens / 1000.0))
    labels = []
    values = []
    colors = []
    errs = []
    for mode in MODE_ORDER:
        if by_mode.get(mode):
            xs = by_mode[mode]
            labels.append(MODE_LABEL.get(mode, mode))
            values.append(sum(xs) / len(xs))
            errs.append(statistics.stdev(xs) if len(xs) > 1 else 0.0)
            colors.append(MODE_COLOR.get(mode, "#888"))
    if not values:
        ax.text(0.5, 0.5, "Insufficient token-rate data", ha="center", va="center")
        return
    xs = list(range(len(values)))
    ax.bar(xs, values, color=colors, yerr=errs, capsize=4)
    for i, (v, e) in enumerate(zip(values, errs)):
        ax.text(i, v + 0.02 * max(values), "%.3f" % v, ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Total reward / 1K generated tokens")
    ax.set_title("Sample efficiency")
    ax.grid(True, axis="y", alpha=0.3)


def _plot_updates_per_sec(ax, runs):
    by_mode = defaultdict(list)
    for _, s, _ in runs:
        v = float(s.get("effective_updates_per_second") or 0.0)
        if v > 0:
            by_mode[_mode(s)].append(v)
    labels, values, colors, errs = [], [], [], []
    for mode in MODE_ORDER:
        if by_mode.get(mode):
            xs = by_mode[mode]
            labels.append(MODE_LABEL.get(mode, mode))
            values.append(sum(xs) / len(xs))
            errs.append(statistics.stdev(xs) if len(xs) > 1 else 0.0)
            colors.append(MODE_COLOR.get(mode, "#888"))
    if not values:
        ax.text(0.5, 0.5, "No update-rate data", ha="center", va="center")
        return
    xs = list(range(len(values)))
    ax.bar(xs, values, color=colors, yerr=errs, capsize=4)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02 * max(values), "%.2f" % v, ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Effective updates / second")
    ax.set_title("Trainer throughput")
    ax.grid(True, axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Overview 2x2 panel (poster)
# ---------------------------------------------------------------------------
def _plot_overview(runs, out_path, k_headline):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    _plot_reward_curves_by_mode(axes[0][0], runs, k_filter=k_headline, decoupled_filter=1)
    _plot_pareto(axes[0][1], runs)
    _plot_time_to_threshold(axes[1][0], runs, threshold=0.10, decoupled_filter=1)
    _plot_staleness_violin(axes[1][1], runs)
    fig.suptitle("Async AReaL vs Sync RL on GSM8K + Qwen2.5-0.5B (2x V100)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _summary_headline_rows(runs):
    rows = []
    for name, s, _ in runs:
        rows.append({
            "run": name,
            "mode": _mode(s),
            "staleness_k": _staleness(s),
            "decoupled": _decoupled(s),
            "seed": _seed(s),
            "update_count": int(s.get("update_count") or 0),
            "wall_clock_sec": round(float(s.get("wall_clock_sec") or 0.0), 2),
            "avg_tokens_per_sec": round(float(s.get("avg_tokens_per_sec") or 0.0), 2),
            "effective_tokens_per_sec": round(_effective_tokens_per_sec(s), 2),
            "accepted_fraction": round(float(s.get("accepted_fraction") or 1.0), 3),
            "pass_rate_per_update": round(_final_pass_rate(s), 3),
            "mean_staleness": round(float(s.get("mean_staleness") or 0.0), 3),
            "max_staleness_observed": int(s.get("max_staleness") or 0),
        })
    return rows


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_md_table(path, rows, header_title):
    if not rows:
        path.write_text("# %s\n\n(no data)\n" % header_title)
        return
    cols = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write("# %s\n\n" % header_title)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")


def _aggregate_by_mode_k_dec(runs):
    buckets = defaultdict(list)
    for name, s, _ in runs:
        key = (_mode(s), _staleness(s), _decoupled(s))
        buckets[key].append(s)
    rows = []
    for (mode, k, dec), ss in sorted(buckets.items()):
        def _mean(fn):
            vals = [fn(x) for x in ss]
            vals = [v for v in vals if v is not None]
            return (sum(vals) / len(vals)) if vals else 0.0
        rows.append({
            "mode": mode,
            "staleness_k": k,
            "decoupled": dec,
            "n_seeds": len(ss),
            "pass_rate": round(_mean(_final_pass_rate), 3),
            "effective_tokens_per_sec": round(_mean(_effective_tokens_per_sec), 2),
            "avg_tokens_per_sec": round(_mean(lambda x: float(x.get("avg_tokens_per_sec") or 0.0)), 2),
            "wall_clock_sec": round(_mean(lambda x: float(x.get("wall_clock_sec") or 0.0)), 2),
            "accepted_fraction": round(_mean(lambda x: float(x.get("accepted_fraction") or 1.0)), 3),
            "mean_staleness": round(_mean(lambda x: float(x.get("mean_staleness") or 0.0)), 3),
        })
    return rows


# ---------------------------------------------------------------------------
# INDEX.md
# ---------------------------------------------------------------------------
INDEX_TEMPLATE = """# Report & poster asset index

Experiment root: `{exp_dir}`

## How to use these in the final report

| Section of report | Asset |
|---|---|
| Headline teaser figure | `plots/fig_overview_poster.png` |
| Section "Learning under staleness" | `plots/fig5a_naive_learning_curves.png`, `plots/fig5b_decoupled_learning_curves.png` |
| Section "Throughput vs staleness" | `plots/fig5c_throughput_vs_staleness.png` |
| Section "Interruptible generation" | `plots/fig6b_interruptible_throughput.png` |
| Section "End-to-end comparison"  | `plots/fig_reward_curves.png`, `plots/fig_pass_rate_curves.png` |
| Section "Speedup analysis" | `plots/fig_time_to_threshold.png`, `plots/fig_updates_per_sec.png` |
| Section "Throughput/accuracy tradeoff" | `plots/fig_pareto.png` |
| Section "Sample efficiency" | `plots/fig_reward_per_ktoken.png` |
| Section "Realized staleness" | `plots/fig_staleness_violin.png` |
| Appendix: full results table | `tables/summary_headline.csv/.md` |
| Appendix: per-config aggregates | `tables/mode_aggregate.csv/.md` |
| Paper-style Table 2 reproduction | `plots/table2_staleness_vs_pass_rate.csv/.md` |

## How to use these on the poster

- **Main panel (top):** `plots/fig_overview_poster.png` (4-in-1 figure, publication-ready).
- **Secondary panel (bottom-left):** `plots/fig5b_decoupled_learning_curves.png` — shows AReaL tolerates staleness when paired with the decoupled objective.
- **Secondary panel (bottom-right):** `plots/fig5c_throughput_vs_staleness.png` — shows throughput scales with allowed staleness.
- **Sidebar stat:** the speedup ratio can be read directly from `plots/fig_time_to_threshold.png`.

## Raw inputs
- `manifest.json` — grid configuration that produced these runs.
- `merged_summary.{{json,csv}}` — every run summary concatenated.
- `failures.json` — any runs that did not produce a `summary.json`.
- per-run directories hold `summary.json`, `results.jsonl`, `command.txt`, `config.json`, `stdout.log`, `stderr.log`.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--k-headline", type=int, default=2,
                        help="Staleness value used for the headline reward-curve panel.")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Pass-rate threshold for time-to-threshold plot.")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    plots_dir = exp_dir / "plots"
    tables_dir = exp_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    runs = _iter_runs(exp_dir)
    print("Loaded %d runs from %s" % (len(runs), exp_dir))
    if not runs:
        raise SystemExit("No runs found.")

    # Individual plots -------------------------------------------------------
    def _save_single(name, size, plot_fn):
        fig, ax = plt.subplots(figsize=size)
        plot_fn(ax)
        fig.tight_layout()
        fig.savefig(plots_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  wrote %s" % (plots_dir / name))

    _save_single("fig_reward_curves.png", (7.5, 4.5),
                 lambda ax: _plot_reward_curves_by_mode(ax, runs,
                                                       k_filter=args.k_headline,
                                                       decoupled_filter=1))
    _save_single("fig_pass_rate_curves.png", (7.5, 4.5),
                 lambda ax: _plot_pass_rate_curves_by_mode(ax, runs,
                                                          k_filter=args.k_headline,
                                                          decoupled_filter=1))
    _save_single("fig_pareto.png", (7.5, 5.0),
                 lambda ax: _plot_pareto(ax, runs))
    _save_single("fig_time_to_threshold.png", (6.8, 4.5),
                 lambda ax: _plot_time_to_threshold(ax, runs,
                                                   threshold=args.threshold,
                                                   decoupled_filter=1))
    _save_single("fig_staleness_violin.png", (8.5, 4.5),
                 lambda ax: _plot_staleness_violin(ax, runs))
    _save_single("fig_reward_per_ktoken.png", (6.8, 4.5),
                 lambda ax: _plot_reward_per_ktoken(ax, runs, decoupled_filter=1))
    _save_single("fig_updates_per_sec.png", (6.8, 4.5),
                 lambda ax: _plot_updates_per_sec(ax, runs))

    # Overview panel ---------------------------------------------------------
    _plot_overview(runs, plots_dir / "fig_overview_poster.png", k_headline=args.k_headline)
    print("  wrote %s" % (plots_dir / "fig_overview_poster.png"))

    # Tables -----------------------------------------------------------------
    headline = _summary_headline_rows(runs)
    _write_csv(tables_dir / "summary_headline.csv", headline)
    _write_md_table(tables_dir / "summary_headline.md", headline,
                    "Headline results (one row per run)")
    print("  wrote %s" % (tables_dir / "summary_headline.csv"))

    aggr = _aggregate_by_mode_k_dec(runs)
    _write_csv(tables_dir / "mode_aggregate.csv", aggr)
    _write_md_table(tables_dir / "mode_aggregate.md", aggr,
                    "Per-config aggregates (seed-averaged)")
    print("  wrote %s" % (tables_dir / "mode_aggregate.csv"))

    # INDEX ------------------------------------------------------------------
    (exp_dir / "INDEX.md").write_text(
        INDEX_TEMPLATE.format(exp_dir=str(exp_dir)),
        encoding="utf-8",
    )
    print("  wrote %s" % (exp_dir / "INDEX.md"))


if __name__ == "__main__":
    main()
