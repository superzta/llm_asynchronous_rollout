import argparse
import csv
import datetime
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path


def _parse_csv_list(raw, cast_fn=str):
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    return [cast_fn(x) for x in items]


def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_name_from_config(cfg):
    mode = cfg["mode"]
    staleness = cfg.get("staleness_k", "na")
    seed = cfg["seed"]
    upd = cfg["update_batch_size"]
    q = cfg.get("queue_maxsize", "na")
    ep = cfg["epochs"]
    backend = cfg["backend"]
    lr = cfg["lr"]
    p_delay = cfg.get("producer_delay_sec", 0.0)
    l_delay = cfg.get("learner_delay_sec", 0.0)
    nr = cfg.get("num_rollout_workers", "na")
    nt = cfg.get("num_trainer_workers", "na")
    return (
        "mode-{m}__k-{k}__seed-{s}__ub-{ub}__q-{q}__ep-{ep}__b-{b}__lr-{lr}"
        "__nr-{nr}__nt-{nt}__pdelay-{pd}__ldelay-{ld}"
    ).format(
        m=mode,
        k=staleness,
        s=seed,
        ub=upd,
        q=q,
        ep=ep,
        b=backend,
        lr=lr,
        nr=nr,
        nt=nt,
        pd=p_delay,
        ld=l_delay,
    )


def _build_command(cfg, run_dir):
    results_jsonl = run_dir / "results.jsonl"
    summary_json = run_dir / "summary.json"
    module_name = "src.run_sync_baseline"
    if cfg["mode"] == "async_train":
        module_name = "src.run_async_baseline"
    elif cfg["mode"] == "async_areal_style":
        module_name = "src.run_async_areal_style"

    base = [
        sys.executable,
        "-m",
        module_name,
        "--dataset",
        cfg["dataset"],
        "--epochs",
        str(cfg["epochs"]),
        "--seed",
        str(cfg["seed"]),
        "--backend",
        cfg["backend"],
        "--lr",
        str(cfg["lr"]),
        "--update-batch-size",
        str(cfg["update_batch_size"]),
        "--max-new-tokens",
        str(cfg["max_new_tokens"]),
        "--reward-timeout-sec",
        str(cfg["reward_timeout_sec"]),
        "--results-jsonl",
        str(results_jsonl),
        "--summary-json",
        str(summary_json),
    ]
    if cfg.get("hf_model_id"):
        base.extend(["--hf-model-id", cfg["hf_model_id"]])
    if cfg["mode"] == "async_train":
        base.extend(
            [
                "--staleness-k",
                str(cfg["staleness_k"]),
                "--queue-maxsize",
                str(cfg["queue_maxsize"]),
                "--queue-trace-interval-sec",
                str(cfg["queue_trace_interval_sec"]),
                "--producer-delay-sec",
                str(cfg["producer_delay_sec"]),
                "--learner-delay-sec",
                str(cfg["learner_delay_sec"]),
            ]
        )
    if cfg["mode"] == "async_areal_style":
        base.extend(
            [
                "--staleness-k",
                str(cfg["staleness_k"]),
                "--queue-maxsize",
                str(cfg["queue_maxsize"]),
                "--queue-trace-interval-sec",
                str(cfg["queue_trace_interval_sec"]),
                "--producer-delay-sec",
                str(cfg["producer_delay_sec"]),
                "--learner-delay-sec",
                str(cfg["learner_delay_sec"]),
                "--num-rollout-workers",
                str(cfg["num_rollout_workers"]),
                "--num-trainer-workers",
                str(cfg["num_trainer_workers"]),
                "--rollout-devices",
                cfg["rollout_devices"],
                "--trainer-devices",
                cfg["trainer_devices"],
                "--interrupt-check-interval-sec",
                str(cfg["interrupt_check_interval_sec"]),
                "--generation-chunk-size",
                str(cfg["generation_chunk_size"]),
            ]
        )
    return base


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


def parse_args():
    parser = argparse.ArgumentParser(description="Run sync/async experiment grid with structured outputs.")
    parser.add_argument("--output-root", default="results/experiments")
    parser.add_argument("--experiment-name", default="", help="Optional fixed name; defaults to timestamp.")
    parser.add_argument("--dataset", default="data/tiny_coding_train.jsonl")
    parser.add_argument("--modes", default="sync_train,async_train,async_areal_style")
    parser.add_argument("--staleness-k-values", default="0,1,2,4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--update-batch-sizes", default="4")
    parser.add_argument("--queue-maxsizes", default="64")
    parser.add_argument("--epochs-values", default="2")
    parser.add_argument("--backend-values", default="tiny_policy")
    parser.add_argument("--lr-values", default="0.1")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--reward-timeout-sec", type=float, default=2.0)
    parser.add_argument("--queue-trace-interval-sec", type=float, default=0.1)
    parser.add_argument("--producer-delay-sec", type=float, default=0.0)
    parser.add_argument("--learner-delay-sec", type=float, default=0.0)
    parser.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--num-rollout-workers-values", default="1")
    parser.add_argument("--num-trainer-workers-values", default="1")
    parser.add_argument("--rollout-devices", default="cpu")
    parser.add_argument("--trainer-devices", default="cpu")
    parser.add_argument("--interrupt-check-interval-sec", type=float, default=0.02)
    parser.add_argument("--generation-chunk-size", type=int, default=4)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_name = args.experiment_name or _timestamp()
    exp_dir = Path(args.output_root) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    modes = _parse_csv_list(args.modes, str)
    staleness_values = _parse_csv_list(args.staleness_k_values, int)
    seeds = _parse_csv_list(args.seeds, int)
    update_batch_sizes = _parse_csv_list(args.update_batch_sizes, int)
    queue_maxsizes = _parse_csv_list(args.queue_maxsizes, int)
    epochs_values = _parse_csv_list(args.epochs_values, int)
    backend_values = _parse_csv_list(args.backend_values, str)
    lr_values = _parse_csv_list(args.lr_values, float)
    num_rollout_workers_values = _parse_csv_list(args.num_rollout_workers_values, int)
    num_trainer_workers_values = _parse_csv_list(args.num_trainer_workers_values, int)

    grid_rows = []
    for mode in modes:
        if mode not in ("sync_train", "async_train", "async_areal_style"):
            raise ValueError("Unsupported mode: %s" % mode)
        if mode == "sync_train":
            combos = itertools.product(seeds, update_batch_sizes, epochs_values, backend_values, lr_values)
            for (seed, ub, epochs, backend, lr) in combos:
                grid_rows.append(
                    {
                        "mode": mode,
                        "dataset": args.dataset,
                        "seed": seed,
                        "update_batch_size": ub,
                        "epochs": epochs,
                        "backend": backend,
                        "lr": lr,
                        "max_new_tokens": args.max_new_tokens,
                        "reward_timeout_sec": args.reward_timeout_sec,
                        "hf_model_id": args.hf_model_id,
                    }
                )
        elif mode == "async_train":
            combos = itertools.product(
                staleness_values,
                seeds,
                update_batch_sizes,
                queue_maxsizes,
                epochs_values,
                backend_values,
                lr_values,
            )
            for (k, seed, ub, q, epochs, backend, lr) in combos:
                grid_rows.append(
                    {
                        "mode": mode,
                        "dataset": args.dataset,
                        "staleness_k": k,
                        "seed": seed,
                        "update_batch_size": ub,
                        "queue_maxsize": q,
                        "epochs": epochs,
                        "backend": backend,
                        "lr": lr,
                        "max_new_tokens": args.max_new_tokens,
                        "reward_timeout_sec": args.reward_timeout_sec,
                        "queue_trace_interval_sec": args.queue_trace_interval_sec,
                        "producer_delay_sec": args.producer_delay_sec,
                        "learner_delay_sec": args.learner_delay_sec,
                        "hf_model_id": args.hf_model_id,
                    }
                )
        else:
            combos = itertools.product(
                staleness_values,
                seeds,
                update_batch_sizes,
                queue_maxsizes,
                epochs_values,
                backend_values,
                lr_values,
                num_rollout_workers_values,
                num_trainer_workers_values,
            )
            for (k, seed, ub, q, epochs, backend, lr, nr, nt) in combos:
                grid_rows.append(
                    {
                        "mode": mode,
                        "dataset": args.dataset,
                        "staleness_k": k,
                        "seed": seed,
                        "update_batch_size": ub,
                        "queue_maxsize": q,
                        "epochs": epochs,
                        "backend": backend,
                        "lr": lr,
                        "max_new_tokens": args.max_new_tokens,
                        "reward_timeout_sec": args.reward_timeout_sec,
                        "queue_trace_interval_sec": args.queue_trace_interval_sec,
                        "producer_delay_sec": args.producer_delay_sec,
                        "learner_delay_sec": args.learner_delay_sec,
                        "hf_model_id": args.hf_model_id,
                        "num_rollout_workers": nr,
                        "num_trainer_workers": nt,
                        "rollout_devices": args.rollout_devices,
                        "trainer_devices": args.trainer_devices,
                        "interrupt_check_interval_sec": args.interrupt_check_interval_sec,
                        "generation_chunk_size": args.generation_chunk_size,
                    }
                )

    manifest = {
        "experiment_name": experiment_name,
        "created_at": datetime.datetime.now().isoformat(),
        "num_planned_runs": len(grid_rows),
        "args": vars(args),
    }
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    merged = []
    failures = []
    for idx, cfg in enumerate(grid_rows, start=1):
        run_name = "%03d__%s" % (idx, _run_name_from_config(cfg))
        run_dir = exp_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = _build_command(cfg, run_dir)
        cmd_str = " ".join(cmd)
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        (run_dir / "command.txt").write_text(cmd_str + "\n", encoding="utf-8")

        print("[%d/%d] Running %s" % (idx, len(grid_rows), run_name), flush=True)
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        (run_dir / "stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (run_dir / "stderr.log").write_text(proc.stderr or "", encoding="utf-8")

        summary_path = run_dir / "summary.json"
        run_record = dict(cfg)
        run_record["run_name"] = run_name
        run_record["run_dir"] = str(run_dir)
        run_record["return_code"] = proc.returncode
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for k, v in summary.items():
                if isinstance(v, (dict, list)):
                    continue
                run_record[k] = v
        else:
            failures.append({"run_name": run_name, "reason": "missing summary", "return_code": proc.returncode})

        merged.append(run_record)
        if proc.returncode != 0 and args.fail_fast:
            break

    merged_json = exp_dir / "merged_summary.json"
    merged_csv = exp_dir / "merged_summary.csv"
    merged_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    _write_csv(merged_csv, merged)
    (exp_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    print("Experiment directory: %s" % str(exp_dir))
    print("Merged summary JSON: %s" % str(merged_json))
    print("Merged summary CSV: %s" % str(merged_csv))
    if failures:
        print("Failures: %d (see failures.json)" % len(failures))


if __name__ == "__main__":
    main()
