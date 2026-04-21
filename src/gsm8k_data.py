"""
Download GSM8K into local JSONL files compatible with CodingTask.from_dict.

Usage:
    python3 -m src.gsm8k_data --train-out data/gsm8k_train.jsonl --eval-out data/gsm8k_eval.jsonl --train-size 256 --eval-size 128
"""
import argparse
import json
import re
from pathlib import Path


_HASH_RE = re.compile(r"####\s*([\-+]?\d[\d,]*\.?\d*)")


def _extract_gold(answer_text):
    m = _HASH_RE.search(answer_text or "")
    if m:
        return m.group(1).replace(",", "")
    return None


def _load_gsm8k():
    from datasets import load_dataset

    return load_dataset("openai/gsm8k", "main")


def build_records(split_rows, prefix, limit):
    out = []
    for idx, row in enumerate(split_rows):
        if limit is not None and idx >= limit:
            break
        gold = _extract_gold(row["answer"])
        if gold is None:
            continue
        out.append(
            {
                "task_id": "%s_%04d" % (prefix, idx),
                "prompt": row["question"].strip(),
                "reference_tests": [],
                "starter_code": None,
                "task_type": "gsm8k",
                "gold_answer": gold,
            }
        )
    return out


def _write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download GSM8K into local JSONL.")
    parser.add_argument("--train-out", default="data/gsm8k_train.jsonl")
    parser.add_argument("--eval-out", default="data/gsm8k_eval.jsonl")
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ds = _load_gsm8k()
    train_rows = list(ds["train"])
    test_rows = list(ds["test"])

    import random

    random.Random(args.seed).shuffle(train_rows)
    random.Random(args.seed + 1).shuffle(test_rows)

    train_out = build_records(train_rows, "gsm8k_train", args.train_size)
    eval_out = build_records(test_rows, "gsm8k_eval", args.eval_size)
    _write_jsonl(args.train_out, train_out)
    _write_jsonl(args.eval_out, eval_out)
    print("Wrote %d train rows -> %s" % (len(train_out), args.train_out))
    print("Wrote %d eval rows -> %s" % (len(eval_out), args.eval_out))


if __name__ == "__main__":
    main()
