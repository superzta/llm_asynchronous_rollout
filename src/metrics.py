import json
from pathlib import Path
from typing import Any, Dict, List, Union


def _safe_mean(values):
    # type: (List[float]) -> float
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize_sync(results, wall_clock_sec):
    # type: (List[Dict[str, Any]], float) -> Dict[str, Any]
    rewards = [float(r.get("reward", 0.0)) for r in results]
    latencies = [float(r.get("latency_sec", 0.0)) for r in results]
    passed = [bool(r.get("pass", False)) for r in results]
    return {
        "mode": "sync",
        "num_samples": len(results),
        "wall_clock_sec": wall_clock_sec,
        "avg_reward": _safe_mean(rewards),
        "pass_rate": _safe_mean([1.0 if x else 0.0 for x in passed]),
        "avg_latency_sec": _safe_mean(latencies),
    }


def summarize_async(
    accepted_results,
    dropped_results,
    wall_clock_sec,
    queue_depth_trace,
):
    # type: (List[Dict[str, Any]], List[Dict[str, Any]], float, List[Dict[str, Any]]) -> Dict[str, Any]
    rewards = [float(r.get("reward", 0.0)) for r in accepted_results]
    latencies = [float(r.get("latency_sec", 0.0)) for r in accepted_results]
    passed = [bool(r.get("pass", False)) for r in accepted_results]
    staleness = [int(r.get("staleness", 0)) for r in accepted_results]
    return {
        "mode": "async",
        "num_accepted": len(accepted_results),
        "num_dropped": len(dropped_results),
        "num_total_seen": len(accepted_results) + len(dropped_results),
        "wall_clock_sec": wall_clock_sec,
        "avg_reward": _safe_mean(rewards),
        "pass_rate": _safe_mean([1.0 if x else 0.0 for x in passed]),
        "avg_latency_sec": _safe_mean(latencies),
        "mean_staleness": _safe_mean([float(x) for x in staleness]),
        "max_staleness": max(staleness) if staleness else 0,
        "queue_depth_trace": queue_depth_trace,
    }


def write_json(path, payload):
    # type: (Union[str, Path], Dict[str, Any]) -> None
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    # type: (Union[str, Path], List[Dict[str, Any]]) -> None
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
