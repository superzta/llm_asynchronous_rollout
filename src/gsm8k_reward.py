"""
GSM8K reward: robust extraction of the final numeric answer from a model
response, compared against a gold numeric answer.

Reward is in {0.0, 1.0}; partial credit is not well defined for GSM8K.
"""
import re
from typing import Any, Dict, Optional


_HASH_ANSWER_RE = re.compile(r"####\s*([\-+]?\d[\d,]*\.?\d*)")
_BOXED_ANSWER_RE = re.compile(r"\\boxed\{\s*([\-+]?\d[\d,]*\.?\d*)\s*\}")
_ANY_NUMBER_RE = re.compile(r"[\-+]?\d[\d,]*\.?\d*")


def _normalize_number(raw):
    # type: (Optional[str]) -> Optional[float]
    if raw is None:
        return None
    s = str(raw).strip()
    s = s.replace(",", "").replace("$", "").rstrip(".")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_final_number(text):
    # type: (str) -> Optional[float]
    if not text:
        return None
    m = _HASH_ANSWER_RE.search(text)
    if m:
        val = _normalize_number(m.group(1))
        if val is not None:
            return val
    m = _BOXED_ANSWER_RE.search(text)
    if m:
        val = _normalize_number(m.group(1))
        if val is not None:
            return val
    nums = _ANY_NUMBER_RE.findall(text)
    if nums:
        return _normalize_number(nums[-1])
    return None


def extract_gold_number(raw):
    # type: (Any) -> Optional[float]
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw)
    m = _HASH_ANSWER_RE.search(s)
    if m:
        return _normalize_number(m.group(1))
    return _normalize_number(s)


def evaluate_gsm8k_response(response_text, task, tolerance=1e-4):
    # type: (str, Any, float) -> Dict[str, Any]
    gold_raw = getattr(task, "gold_answer", None)
    if gold_raw is None and hasattr(task, "to_dict"):
        gold_raw = task.to_dict().get("gold_answer")
    gold = extract_gold_number(gold_raw)
    pred = extract_final_number(response_text or "")
    if gold is None:
        return {
            "reward": 0.0,
            "pass": False,
            "num_passed": 0,
            "total_tests": 1,
            "error_type": "missing_gold",
            "error_message": "gold_answer missing from task",
            "extracted_code": "",
            "extracted_answer": pred,
            "gold_answer": None,
        }
    if pred is None:
        return {
            "reward": 0.0,
            "pass": False,
            "num_passed": 0,
            "total_tests": 1,
            "error_type": "no_number",
            "error_message": "Could not extract a number from response.",
            "extracted_code": "",
            "extracted_answer": None,
            "gold_answer": gold,
        }
    correct = abs(pred - gold) <= tolerance
    return {
        "reward": 1.0 if correct else 0.0,
        "pass": bool(correct),
        "num_passed": 1 if correct else 0,
        "total_tests": 1,
        "error_type": None if correct else "wrong_answer",
        "error_message": "",
        "extracted_code": "",
        "extracted_answer": pred,
        "gold_answer": gold,
    }
