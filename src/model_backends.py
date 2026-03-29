import time
from typing import Any, Optional

from src.coding_task import CodingTask


class GenerationResult:
    def __init__(self, text, latency_sec, num_tokens, tokens_per_sec):
        self.text = text
        self.latency_sec = latency_sec
        self.num_tokens = num_tokens
        self.tokens_per_sec = tokens_per_sec


class BaseModelBackend:
    def generate(self, prompt, task=None):
        # type: (str, Optional[CodingTask]) -> GenerationResult
        raise NotImplementedError


_DUMMY_SOLUTIONS = {
    "add_one": "def add_one(x: int) -> int:\n    return x + 1\n",
    "square": "def square(x: int) -> int:\n    return x * x\n",
    "reverse_string": "def reverse_string(s: str) -> str:\n    return s[::-1]\n",
    "is_even": "def is_even(n: int) -> bool:\n    return n % 2 == 0\n",
    "list_sum": "def list_sum(items):\n    return sum(items)\n",
    "max_of_two": "def max_of_two(a: int, b: int) -> int:\n    return a if a >= b else b\n",
    "factorial_small": (
        "def factorial_small(n: int) -> int:\n"
        "    out = 1\n"
        "    for i in range(2, n + 1):\n"
        "        out *= i\n"
        "    return out\n"
    ),
    "first_char_or_empty": "def first_char_or_empty(s: str) -> str:\n    return s[0] if s else ''\n",
    "count_vowels": (
        "def count_vowels(s: str) -> int:\n"
        "    return sum(1 for ch in s if ch in {'a', 'e', 'i', 'o', 'u'})\n"
    ),
    "clamp_0_10": "def clamp_0_10(x: int) -> int:\n    return max(0, min(10, x))\n",
    "fib_n": (
        "def fib_n(n: int) -> int:\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    a, b = 0, 1\n"
        "    for _ in range(2, n + 1):\n"
        "        a, b = b, a + b\n"
        "    return b\n"
    ),
    "remove_spaces": "def remove_spaces(s: str) -> str:\n    return s.replace(' ', '')\n",
    "add_one_eval": "def add_one_eval(x: int) -> int:\n    return x + 1\n",
    "string_len": "def string_len(s: str) -> int:\n    return len(s)\n",
    "min_of_three": "def min_of_three(a: int, b: int, c: int) -> int:\n    return min(a, b, c)\n",
    "ends_with_period": "def ends_with_period(s: str) -> bool:\n    return s.endswith('.')\n",
    "double_list": "def double_list(xs):\n    return [2 * x for x in xs]\n",
    "is_palindrome": "def is_palindrome(s: str) -> bool:\n    return s == s[::-1]\n",
}


class DummyModelBackend(BaseModelBackend):
    def __init__(self, sleep_sec=0.01, use_markdown_fence=True):
        self.sleep_sec = sleep_sec
        self.use_markdown_fence = use_markdown_fence

    def generate(self, prompt, task=None):
        # type: (str, Optional[CodingTask]) -> GenerationResult
        start = time.time()
        if self.sleep_sec > 0:
            time.sleep(self.sleep_sec)
        task_id = task.task_id if task else ""
        solution = _DUMMY_SOLUTIONS.get(task_id, "def solution():\n    return None\n")
        text = f"```python\n{solution}```" if self.use_markdown_fence else solution
        latency_sec = max(time.time() - start, 1e-6)
        token_count = max(len(text.split()), 1)
        return GenerationResult(
            text=text,
            latency_sec=latency_sec,
            num_tokens=token_count,
            tokens_per_sec=token_count / latency_sec,
        )


class HuggingFaceBackend(BaseModelBackend):
    """
    Lightweight stub backend for future extension.
    """

    def __init__(self, model_id, max_new_tokens=128):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

    def _lazy_init(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed. Install it to use --backend hf."
            ) from exc
        self._pipeline = pipeline("text-generation", model=self.model_id)

    def generate(self, prompt, task=None):
        # type: (str, Optional[CodingTask]) -> GenerationResult
        self._lazy_init()
        start = time.time()
        output = self._pipeline(prompt, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output[0]["generated_text"]
        latency_sec = max(time.time() - start, 1e-6)
        token_count = max(len(generated.split()), 1)
        return GenerationResult(
            text=generated,
            latency_sec=latency_sec,
            num_tokens=token_count,
            tokens_per_sec=token_count / latency_sec,
        )


def build_backend(args):
    # type: (Any) -> BaseModelBackend
    if args.backend == "dummy":
        return DummyModelBackend(sleep_sec=args.dummy_sleep_sec)
    if args.backend == "hf":
        if not args.hf_model_id:
            raise ValueError("--hf-model-id is required when --backend hf")
        return HuggingFaceBackend(model_id=args.hf_model_id, max_new_tokens=args.max_new_tokens)
    raise ValueError(f"Unsupported backend: {args.backend}")
