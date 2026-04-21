import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = [
    "avg_reward",
    "pass_rate",
    "wall_clock_sec",
    "update_count",
    "dropped_stale_count",
    "dropped_interrupted_count",
    "mean_staleness",
    "max_staleness",
    "avg_tokens_per_sec",
    "accepted_fraction",
    "dropped_fraction",
    "accepted_fraction_terminal",
    "dropped_fraction_terminal",
    "effective_updates_per_second",
    "reward_per_second",
    "reward_per_update",
    "pass_rate_per_update",
]


def _safe_float(v, default=0.0):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    m = sum(values) / float(len(values))
    if len(values) == 1:
        return m, 0.0
    var = sum([(x - m) ** 2 for x in values]) / float(len(values))
    return m, math.sqrt(var)


def _load_runs(experiment_dir):
    runs = []
    for run_dir in sorted([p for p in Path(experiment_dir).iterdir() if p.is_dir()]):
        summary_path = run_dir / "summary.json"
        config_path = run_dir / "config.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        row = {"run_name": run_dir.name}
        row.update(config)
        row.update(summary)
        runs.append(row)
    return runs


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted(set().union(*[set(r.keys()) for r in rows]))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate(runs):
    grouped = defaultdict(list)
    for r in runs:
        key = (
            r.get("mode", ""),
            r.get("staleness_k", None),
            r.get("seed", None),
        )
        grouped[key].append(r)

    agg_rows = []
    for key, items in grouped.items():
        mode, staleness_k, seed = key
        out = {
            "mode": mode,
            "staleness_k": staleness_k,
            "seed": seed,
            "n_runs": len(items),
        }
        for metric in METRICS:
            vals = [_safe_float(i.get(metric, 0.0)) for i in items]
            m, s = _mean_std(vals)
            out["%s_mean" % metric] = m
            out["%s_std" % metric] = s
        agg_rows.append(out)
    agg_rows.sort(key=lambda x: (str(x["mode"]), str(x["staleness_k"]), str(x["seed"])))
    return agg_rows


def _group_mode_staleness(runs):
    grouped = defaultdict(list)
    for r in runs:
        key = (r.get("mode", ""), r.get("staleness_k", None))
        grouped[key].append(r)
    out = {}
    for key, items in grouped.items():
        out[key] = {
            metric: _mean_std([_safe_float(i.get(metric, 0.0)) for i in items])
            for metric in METRICS
        }
    return out


def _plot_all(experiment_dir, runs):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot generation.")
        return

    plots_dir = Path(experiment_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    grouped = _group_mode_staleness(runs)

    # Reward vs staleness bound
    async_modes = ["async_train", "async_areal_style"]
    if any([k[0] in async_modes for k in grouped.keys()]):
        plt.figure(figsize=(7, 4))
        for mode in async_modes:
            mode_keys = sorted(
                [k for k in grouped.keys() if k[0] == mode and k[1] is not None],
                key=lambda x: int(x[1]),
            )
            if not mode_keys:
                continue
            xs = [int(k[1]) for k in mode_keys]
            ys = [grouped[k]["avg_reward"][0] for k in mode_keys]
            es = [grouped[k]["avg_reward"][1] for k in mode_keys]
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=mode)
        sync_vals = [grouped[k]["avg_reward"][0] for k in grouped.keys() if k[0] == "sync_train"]
        if sync_vals:
            sync_mean = sum(sync_vals) / float(len(sync_vals))
            plt.axhline(sync_mean, linestyle="--", label="sync mean")
        plt.xlabel("staleness_k")
        plt.ylabel("avg_reward")
        plt.title("Reward vs Staleness Bound")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(plots_dir / "reward_vs_staleness.png"), dpi=150)
        plt.close()

    # Pass rate vs staleness bound
    if any([k[0] in async_modes for k in grouped.keys()]):
        plt.figure(figsize=(7, 4))
        for mode in async_modes:
            mode_keys = sorted(
                [k for k in grouped.keys() if k[0] == mode and k[1] is not None],
                key=lambda x: int(x[1]),
            )
            if not mode_keys:
                continue
            xs = [int(k[1]) for k in mode_keys]
            ys = [grouped[k]["pass_rate"][0] for k in mode_keys]
            es = [grouped[k]["pass_rate"][1] for k in mode_keys]
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=mode)
        sync_vals = [grouped[k]["pass_rate"][0] for k in grouped.keys() if k[0] == "sync_train"]
        if sync_vals:
            sync_mean = sum(sync_vals) / float(len(sync_vals))
            plt.axhline(sync_mean, linestyle="--", label="sync mean")
        plt.xlabel("staleness_k")
        plt.ylabel("pass_rate")
        plt.title("Pass Rate vs Staleness Bound")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(plots_dir / "pass_rate_vs_staleness.png"), dpi=150)
        plt.close()

    # Dropped stale count vs staleness bound
    if any([k[0] in async_modes for k in grouped.keys()]):
        plt.figure(figsize=(7, 4))
        for mode in async_modes:
            mode_keys = sorted(
                [k for k in grouped.keys() if k[0] == mode and k[1] is not None],
                key=lambda x: int(x[1]),
            )
            if not mode_keys:
                continue
            xs = [int(k[1]) for k in mode_keys]
            ys = [grouped[k]["dropped_stale_count"][0] for k in mode_keys]
            es = [grouped[k]["dropped_stale_count"][1] for k in mode_keys]
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=mode)
        plt.xlabel("staleness_k")
        plt.ylabel("dropped_stale_count")
        plt.title("Dropped Stale Count vs Staleness Bound")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(plots_dir / "dropped_stale_vs_staleness.png"), dpi=150)
        plt.close()

    # Wall-clock vs mode
    mode_groups = defaultdict(list)
    for r in runs:
        mode_groups[r.get("mode", "")].append(_safe_float(r.get("wall_clock_sec", 0.0)))
    if mode_groups:
        modes = sorted(mode_groups.keys())
        means = []
        stds = []
        for m in modes:
            mean, std = _mean_std(mode_groups[m])
            means.append(mean)
            stds.append(std)
        plt.figure(figsize=(6, 4))
        plt.bar(modes, means, yerr=stds, capsize=4)
        plt.ylabel("wall_clock_sec")
        plt.title("Wall-Clock Runtime by Mode")
        plt.tight_layout()
        plt.savefig(str(plots_dir / "wall_clock_vs_mode.png"), dpi=150)
        plt.close()

    # Updates completed vs mode
    upd_groups = defaultdict(list)
    for r in runs:
        upd_groups[r.get("mode", "")].append(_safe_float(r.get("update_count", 0.0)))
    if upd_groups:
        modes = sorted(upd_groups.keys())
        means = []
        stds = []
        for m in modes:
            mean, std = _mean_std(upd_groups[m])
            means.append(mean)
            stds.append(std)
        plt.figure(figsize=(6, 4))
        plt.bar(modes, means, yerr=stds, capsize=4)
        plt.ylabel("update_count")
        plt.title("Updates Completed by Mode")
        plt.tight_layout()
        plt.savefig(str(plots_dir / "updates_vs_mode.png"), dpi=150)
        plt.close()

    # Queue depth traces for representative async runs (one per staleness_k)
    rep_async = {}
    for r in runs:
        if r.get("mode") not in ("async_train", "async_areal_style"):
            continue
        k = "%s|k=%s" % (str(r.get("mode")), str(r.get("staleness_k", None)))
        if k not in rep_async:
            rep_async[k] = r
    if rep_async:
        plt.figure(figsize=(8, 5))
        for k in sorted(rep_async.keys()):
            trace = rep_async[k].get("queue_depth_trace", [])
            if not trace:
                continue
            xs = [float(t.get("t_sec", 0.0)) for t in trace]
            ys = [float(t.get("sample_queue_depth", t.get("rollout_queue_depth", 0.0))) for t in trace]
            plt.plot(xs, ys, label="sample_queue %s" % str(k))
        plt.xlabel("time (sec)")
        plt.ylabel("queue depth")
        plt.title("Representative Async Sample Queue Depth Traces")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(plots_dir / "queue_depth_traces_async_sample_queue.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        for k in sorted(rep_async.keys()):
            trace = rep_async[k].get("queue_depth_trace", [])
            if not trace:
                continue
            xs = [float(t.get("t_sec", 0.0)) for t in trace]
            ys = [float(t.get("train_queue_depth", 0.0)) for t in trace]
            plt.plot(xs, ys, label="train_queue %s" % str(k))
        plt.xlabel("time (sec)")
        plt.ylabel("queue depth")
        plt.title("Representative Async Train Queue Depth Traces")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(plots_dir / "queue_depth_traces_async_train_queue.png"), dpi=150)
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate and plot experiment summaries.")
    parser.add_argument("--experiment-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    exp_dir = Path(args.experiment_dir)
    runs = _load_runs(exp_dir)
    if not runs:
        raise RuntimeError("No run summaries found under %s" % str(exp_dir))

    agg = _aggregate(runs)
    per_run_json = exp_dir / "analysis_per_run.json"
    per_run_csv = exp_dir / "analysis_per_run.csv"
    agg_json = exp_dir / "analysis_aggregated.json"
    agg_csv = exp_dir / "analysis_aggregated.csv"
    per_run_json.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    agg_json.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    _write_csv(per_run_csv, runs)
    _write_csv(agg_csv, agg)

    _plot_all(exp_dir, runs)
    print("Wrote:")
    print(" - %s" % str(per_run_json))
    print(" - %s" % str(per_run_csv))
    print(" - %s" % str(agg_json))
    print(" - %s" % str(agg_csv))
    print(" - %s" % str(exp_dir / "plots"))


if __name__ == "__main__":
    main()
