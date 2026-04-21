import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from src.coding_task import CodingTask
from src.gsm8k_reward import evaluate_gsm8k_response


PYTHON_FENCE_RE = re.compile(r"```python\s*(.*?)```", re.IGNORECASE | re.DOTALL)
GENERIC_FENCE_RE = re.compile(r"```\s*(.*?)```", re.DOTALL)


def extract_python_code(response_text):
    # type: (str) -> str
    if not response_text or not response_text.strip():
        return ""
    python_blocks = PYTHON_FENCE_RE.findall(response_text)
    if python_blocks:
        return python_blocks[0].strip()
    generic_blocks = GENERIC_FENCE_RE.findall(response_text)
    if generic_blocks:
        return generic_blocks[0].strip()
    return response_text.strip()


def _make_runner_script(candidate_code, reference_tests):
    # type: (str, List[str]) -> str
    encoded_candidate = json.dumps(candidate_code)
    encoded_tests = json.dumps(reference_tests)
    return f"""
import json

candidate_code = {encoded_candidate}
tests = {encoded_tests}
namespace = {{}}
total = len(tests)
passed = 0

try:
    exec(candidate_code, namespace, namespace)
except SyntaxError as exc:
    print(json.dumps({{
        "reward": 0.0,
        "pass": False,
        "num_passed": 0,
        "total_tests": total,
        "error_type": "syntax_error",
        "error_message": str(exc),
    }}))
    raise SystemExit(0)
except Exception as exc:
    print(json.dumps({{
        "reward": 0.0,
        "pass": False,
        "num_passed": 0,
        "total_tests": total,
        "error_type": "runtime_error",
        "error_message": str(exc),
    }}))
    raise SystemExit(0)

for test_stmt in tests:
    try:
        exec(test_stmt, namespace, namespace)
        passed += 1
    except AssertionError:
        pass
    except Exception as exc:
        print(json.dumps({{
            "reward": passed / total if total else 0.0,
            "pass": False,
            "num_passed": passed,
            "total_tests": total,
            "error_type": "runtime_error",
            "error_message": str(exc),
        }}))
        raise SystemExit(0)

reward = passed / total if total else 0.0
print(json.dumps({{
    "reward": reward,
    "pass": passed == total,
    "num_passed": passed,
    "total_tests": total,
    "error_type": None,
    "error_message": "",
}}))
"""


def _coerce_task(task_like):
    # type: (Union[CodingTask, Dict[str, Any]]) -> CodingTask
    if isinstance(task_like, CodingTask):
        return task_like
    return CodingTask.from_dict(task_like)


def evaluate_response(
    response_text,
    task,
    timeout_sec=2.0,
):
    # type: (str, Union[CodingTask, Dict[str, Any]], float) -> Dict[str, Any]
    task_obj = _coerce_task(task)
    if task_obj.task_type == "gsm8k":
        return evaluate_gsm8k_response(response_text=response_text, task=task_obj)
    code = extract_python_code(response_text)
    if not code:
        return {
            "reward": 0.0,
            "pass": False,
            "num_passed": 0,
            "total_tests": len(task_obj.reference_tests),
            "error_type": "empty_code",
            "error_message": "No code extracted from response.",
            "extracted_code": "",
        }

    with tempfile.TemporaryDirectory(prefix="tiny_coding_reward_") as tmpdir:
        script_path = Path(tmpdir) / "runner.py"
        script_path.write_text(_make_runner_script(code, task_obj.reference_tests), encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "reward": 0.0,
                "pass": False,
                "num_passed": 0,
                "total_tests": len(task_obj.reference_tests),
                "error_type": "timeout",
                "error_message": f"Execution exceeded {timeout_sec:.2f}s timeout.",
                "extracted_code": code,
            }

    stdout = proc.stdout.strip()
    if not stdout:
        return {
            "reward": 0.0,
            "pass": False,
            "num_passed": 0,
            "total_tests": len(task_obj.reference_tests),
            "error_type": "subprocess_error",
            "error_message": proc.stderr.strip() or "No output from subprocess.",
            "extracted_code": code,
        }
    last_line = stdout.splitlines()[-1].strip()
    try:
        result = json.loads(last_line)
    except json.JSONDecodeError:
        result = {
            "reward": 0.0,
            "pass": False,
            "num_passed": 0,
            "total_tests": len(task_obj.reference_tests),
            "error_type": "parse_error",
            "error_message": f"Could not parse subprocess output: {last_line[:200]}",
        }

    result["extracted_code"] = code
    return result


def _load_cli_input(path):
    # type: (str) -> Tuple[str, Dict[str, Any]]
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "task" in raw and "response" in raw:
        return raw["response"], raw["task"]
    if "response" in raw and {"task_id", "prompt", "reference_tests"}.issubset(set(raw.keys())):
        task = {
            "task_id": raw["task_id"],
            "prompt": raw["prompt"],
            "reference_tests": raw["reference_tests"],
            "starter_code": raw.get("starter_code"),
        }
        return raw["response"], task
    raise ValueError("Input JSON must contain either {response, task} or {response, task fields}.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate coding response with tiny test harness.")
    parser.add_argument("--input", required=True, help="Path to JSON input file.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Execution timeout seconds.")
    args = parser.parse_args()
    response_text, task_obj = _load_cli_input(args.input)
    result = evaluate_response(response_text=response_text, task=task_obj, timeout_sec=args.timeout)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
