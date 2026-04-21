"""
Generate paper-style plots from an experiment directory.

Reads every per-run `summary.json` under results/experiments/<exp>/ and
produces:

    plots/fig5a_naive_learning_curves.png     -- reward vs step, naive PPO
    plots/fig5b_decoupled_learning_curves.png -- reward vs step, decoupled
    plots/fig5c_throughput_vs_staleness.png   -- effective tokens/s vs k
    plots/fig6b_interruptible_throughput.png  -- interruptible on vs off
    plots/table2_staleness_vs_pass_rate.csv   -- pass rate vs k, W/o vs With
    plots/table2_staleness_vs_pass_rate.md    -- markdown rendering

Usage:
    python3 -m src.plot_paper_repro --experiment-dir results/experiments/<name>
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _iter_run_summaries(exp_dir):
    for d in sorted(Path(exp_dir).iterdir()):
        if not d.is_dir():
            continue
        sj = d / "summary.json"
        if sj.exists():
            try:
                yield d.name, json.loads(sj.read_text(encoding="utf-8"))
            except Exception:
                continue


def _get_cfg(summary):
    cfg = summary.get("config", {}) or {}
    return cfg


def _staleness_from_summary(summary):
    val = summary.get("staleness_k")
    if val is None:
        cfg = _get_cfg(summary)
        val = cfg.get("staleness_k")
    if val is None:
        return 0
    try:
        return int(val)
    except Exception:
        return 0


def _decoupled_from_summary(summary):
    cfg = _get_cfg(summary)
    dv = cfg.get("decoupled_objective")
    if dv is None:
        return 1
    try:
        return int(dv)
    except Exception:
        return 1


def _mode_from_summary(summary):
    return summary.get("mode") or _get_cfg(summary).get("mode") or "unknown"


def _smooth(xs, window=5):
    if not xs:
        return xs
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        s = xs[lo:i + 1]
        out.append(sum(s) / len(s))
    return out


def _group_by(items, key_fn):
    grouped = defaultdict(list)
    for it in items:
        grouped[key_fn(it)].append(it)
    return grouped


def _plot_learning_curves(ax, runs, title):
    by_k = defaultdict(list)
    for name, summary in runs:
        k = _staleness_from_summary(summary)
        rewards = [r["reward"] for r in summary.get("reward_by_step", [])]
        if not rewards:
            continue
        by_k[k].append(rewards)
    for k in sorted(by_k.keys()):
        lists = by_k[k]
        # pad to common length
        min_len = min(len(r) for r in lists)
        if min_len == 0:
            continue
        truncated = [r[:min_len] for r in lists]
        mean = [sum(col) / len(col) for col in zip(*truncated)]
        smoothed = _smooth(mean, window=max(1, min_len // 20))
        ax.plot(range(1, len(smoothed) + 1), smoothed, label="k=%d" % k)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training reward (smoothed)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)


def _effective_throughput(summary):
    """Effective training throughput: (num generated tokens actually consumed
    by PPO updates) / wall_clock_sec.
    We approximate via avg_tokens_per_sec * accepted_fraction * wall_clock_sec / wall_clock_sec.
    Cleanest proxy available: accepted_tokens_per_sec = avg_tokens_per_sec * accepted_fraction.
    """
    tps = float(summary.get("avg_tokens_per_sec") or 0.0)
    accepted = summary.get("accepted_fraction")
    if accepted is None:
        accepted = 1.0
    return tps * float(accepted)


def _plot_throughput_vs_staleness(ax, runs_decoupled, runs_naive):
    def _agg(runs):
        by_k = defaultdict(list)
        for _, s in runs:
            k = _staleness_from_summary(s)
            by_k[k].append(_effective_throughput(s))
        ks = sorted(by_k.keys())
        vals = [sum(by_k[k]) / len(by_k[k]) for k in ks]
        return ks, vals

    ks_d, vals_d = _agg(runs_decoupled)
    ks_n, vals_n = _agg(runs_naive)
    if ks_d:
        ax.plot(ks_d, vals_d, "o-", label="+decoupled objective")
    if ks_n:
        ax.plot(ks_n, vals_n, "s--", label="naive PPO")
    ax.set_xlabel("Max staleness (k)")
    ax.set_ylabel("Effective throughput (tokens/s)")
    ax.set_title("Figure 5c: Throughput vs max staleness")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)


def _plot_interruptible(ax, summaries):
    """Bar chart of avg_tokens_per_sec grouped by mode: async_train (non-interruptible)
    vs async_areal_style (interruptible), averaged over seeds at matching k."""
    by_mode = defaultdict(list)
    for _, s in summaries:
        mode = _mode_from_summary(s)
        if mode not in ("async_train", "async_areal_style"):
            continue
        by_mode[mode].append(float(s.get("avg_tokens_per_sec") or 0.0))
    labels = []
    values = []
    colors = []
    for mode, color, label in [
        ("async_train", "#6c8ebf", "non-interruptible (async_train)"),
        ("async_areal_style", "#82b366", "interruptible (async_areal_style)"),
    ]:
        if by_mode.get(mode):
            labels.append(label)
            values.append(sum(by_mode[mode]) / len(by_mode[mode]))
            colors.append(color)
    if not values:
        ax.text(0.5, 0.5, "No matching runs", ha="center", va="center")
        return
    xs = list(range(len(values)))
    ax.bar(xs, values, color=colors)
    for x, v in zip(xs, values):
        ax.text(x, v, "%.1f" % v, ha="center", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Average generation throughput (tokens/s)")
    ax.set_title("Figure 6b: Interruptible vs non-interruptible generation")
    ax.grid(True, axis="y", alpha=0.3)


def _table2(summaries):
    """Pass rate vs staleness, W/o vs With decoupled objective."""
    rows = defaultdict(lambda: {"naive": [], "decoupled": []})
    for _, s in summaries:
        k = _staleness_from_summary(s)
        d = _decoupled_from_summary(s)
        pr = s.get("pass_rate_per_update")
        if pr is None:
            # Fallback: average pass_rate_by_step
            prs = [x.get("pass_rate", 0.0) for x in s.get("pass_rate_by_step", [])]
            pr = sum(prs) / len(prs) if prs else 0.0
        rows[k]["decoupled" if d == 1 else "naive"].append(float(pr))
    ks = sorted(rows.keys())
    table = []
    for k in ks:
        naive_list = rows[k]["naive"]
        decoupled_list = rows[k]["decoupled"]
        table.append({
            "max_staleness": k,
            "pass_rate_naive": sum(naive_list) / len(naive_list) if naive_list else None,
            "n_naive": len(naive_list),
            "pass_rate_decoupled": sum(decoupled_list) / len(decoupled_list) if decoupled_list else None,
            "n_decoupled": len(decoupled_list),
        })
    return table


def _write_table2(out_dir, table):
    csv_path = out_dir / "table2_staleness_vs_pass_rate.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("max_staleness,pass_rate_naive,n_naive,pass_rate_decoupled,n_decoupled\n")
        for row in table:
            f.write("%s,%s,%d,%s,%d\n" % (
                row["max_staleness"],
                ("%.4f" % row["pass_rate_naive"]) if row["pass_rate_naive"] is not None else "",
                row["n_naive"],
                ("%.4f" % row["pass_rate_decoupled"]) if row["pass_rate_decoupled"] is not None else "",
                row["n_decoupled"],
            ))
    md_path = out_dir / "table2_staleness_vs_pass_rate.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Max stale | naive PPO pass rate | n | decoupled pass rate | n |\n")
        f.write("|---|---|---|---|---|\n")
        for row in table:
            f.write("| %s | %s | %d | %s | %d |\n" % (
                row["max_staleness"],
                ("%.3f" % row["pass_rate_naive"]) if row["pass_rate_naive"] is not None else "—",
                row["n_naive"],
                ("%.3f" % row["pass_rate_decoupled"]) if row["pass_rate_decoupled"] is not None else "—",
                row["n_decoupled"],
            ))
    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    exp_dir = Path(args.experiment_dir)
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = list(_iter_run_summaries(exp_dir))
    print("Loaded %d run summaries from %s" % (len(summaries), exp_dir))
    if not summaries:
        raise SystemExit("No run summaries found.")

    # Partition
    decoupled_runs = [(n, s) for (n, s) in summaries if _decoupled_from_summary(s) == 1
                      and _mode_from_summary(s) in ("async_train", "async_areal_style")]
    naive_runs = [(n, s) for (n, s) in summaries if _decoupled_from_summary(s) == 0
                  and _mode_from_summary(s) in ("async_train", "async_areal_style")]
    sync_runs = [(n, s) for (n, s) in summaries if _mode_from_summary(s) == "sync_train"]

    # Figure 5a: naive PPO learning curves
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _plot_learning_curves(ax, naive_runs, "Figure 5a: naive PPO (learning curves)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig5a_naive_learning_curves.png", dpi=140)
    plt.close(fig)

    # Figure 5b: decoupled PPO learning curves
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _plot_learning_curves(ax, decoupled_runs, "Figure 5b: +decoupled objective (learning curves)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig5b_decoupled_learning_curves.png", dpi=140)
    plt.close(fig)

    # Figure 5c: throughput vs staleness
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _plot_throughput_vs_staleness(ax, decoupled_runs, naive_runs)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5c_throughput_vs_staleness.png", dpi=140)
    plt.close(fig)

    # Figure 6b: interruptible vs non-interruptible
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    # Prefer decoupled runs so algorithm is held fixed
    _plot_interruptible(ax, decoupled_runs or summaries)
    fig.tight_layout()
    fig.savefig(out_dir / "fig6b_interruptible_throughput.png", dpi=140)
    plt.close(fig)

    # Table 2
    tbl = _table2([(n, s) for (n, s) in summaries
                   if _mode_from_summary(s) in ("async_train", "async_areal_style")])
    csv_path, md_path = _write_table2(out_dir, tbl)

    # Summary banner
    print("Wrote:")
    for p in ["fig5a_naive_learning_curves.png",
              "fig5b_decoupled_learning_curves.png",
              "fig5c_throughput_vs_staleness.png",
              "fig6b_interruptible_throughput.png"]:
        print("  %s" % (out_dir / p))
    print("  %s" % csv_path)
    print("  %s" % md_path)


if __name__ == "__main__":
    main()
