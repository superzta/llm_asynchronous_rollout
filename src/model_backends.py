import math
import random
import threading
import time
from typing import Any, Dict, List, Optional

from src.coding_task import CodingTask


class GenerationResult:
    def __init__(
        self,
        text,
        latency_sec,
        num_tokens,
        tokens_per_sec,
        token_logprobs=None,
        logprob_sum=None,
        metadata=None,
    ):
        self.text = text
        self.latency_sec = latency_sec
        self.num_tokens = num_tokens
        self.tokens_per_sec = tokens_per_sec
        self.token_logprobs = token_logprobs or []
        self.logprob_sum = float(logprob_sum) if logprob_sum is not None else 0.0
        self.metadata = metadata or {}


class BaseModelBackend:
    def generate(self, prompt, task=None):
        # type: (str, Optional[CodingTask]) -> GenerationResult
        raise NotImplementedError

    def supports_learning(self):
        # type: () -> bool
        return False

    def policy_gradient_update(self, samples):
        # type: (List[Dict[str, Any]]) -> Dict[str, Any]
        _ = samples
        return {"updated": False, "loss": 0.0, "avg_reward": 0.0}


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
            token_logprobs=[0.0] * token_count,
            logprob_sum=0.0,
            metadata={"task_id": task_id, "action_idx": 0},
        )


class TinyPolicyBackend(BaseModelBackend):
    """
    Lightweight trainable categorical policy over small candidate programs.
    This backend is pure-Python and supports real policy-gradient updates.
    """

    def __init__(self, lr=0.1, use_markdown_fence=True, seed=0):
        self.lr = lr
        self.use_markdown_fence = use_markdown_fence
        self.random = random.Random(seed)
        self.lock = threading.Lock()
        self.task_logits = {}  # type: Dict[str, List[float]]

    def _get_candidates(self, task):
        # type: (CodingTask) -> List[str]
        correct = _DUMMY_SOLUTIONS.get(task.task_id)
        if correct is None:
            if task.starter_code:
                correct = task.starter_code
            else:
                correct = "def solution(x):\n    return x\n"
        wrong_1 = task.starter_code or "def solution(x):\n    return None\n"
        wrong_2 = correct.replace("return", "return 0  # intentionally weak baseline", 1)
        return [correct, wrong_1, wrong_2]

    def _softmax(self, logits):
        # type: (List[float]) -> List[float]
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        z = sum(exps)
        return [e / z for e in exps]

    def _sample_action(self, probs):
        # type: (List[float]) -> int
        r = self.random.random()
        c = 0.0
        for idx, p in enumerate(probs):
            c += p
            if r <= c:
                return idx
        return len(probs) - 1

    def generate(self, prompt, task=None):
        # type: (str, Optional[CodingTask]) -> GenerationResult
        if task is None:
            raise ValueError("TinyPolicyBackend requires task metadata for generation.")
        start = time.time()
        with self.lock:
            candidates = self._get_candidates(task)
            logits = self.task_logits.get(task.task_id)
            if logits is None:
                logits = [0.0] * len(candidates)
                self.task_logits[task.task_id] = logits
            probs = self._softmax(logits)
            action_idx = self._sample_action(probs)
            chosen_prob = max(probs[action_idx], 1e-8)
        chosen_code = candidates[action_idx]
        text = "```python\n%s```" % chosen_code if self.use_markdown_fence else chosen_code
        latency_sec = max(time.time() - start, 1e-6)
        token_count = max(len(text.split()), 1)
        return GenerationResult(
            text=text,
            latency_sec=latency_sec,
            num_tokens=token_count,
            tokens_per_sec=token_count / latency_sec,
            token_logprobs=[math.log(chosen_prob)],
            logprob_sum=math.log(chosen_prob),
            metadata={"task_id": task.task_id, "action_idx": action_idx, "action_prob": chosen_prob},
        )

    def supports_learning(self):
        # type: () -> bool
        return True

    def policy_gradient_update(self, samples):
        # type: (List[Dict[str, Any]]) -> Dict[str, Any]
        if not samples:
            return {"updated": False, "loss": 0.0, "avg_reward": 0.0}
        rewards = [float(s["reward"]) for s in samples]
        baseline = sum(rewards) / max(len(rewards), 1)
        total_loss = 0.0
        with self.lock:
            for sample in samples:
                task_id = sample["task_id"]
                action_idx = int(sample["metadata"]["action_idx"])
                reward = float(sample["reward"])
                advantage = reward - baseline
                logits = self.task_logits.get(task_id)
                if logits is None:
                    continue
                probs = self._softmax(logits)
                # Gradient ascent on expected reward.
                for j in range(len(logits)):
                    grad = ((1.0 if j == action_idx else 0.0) - probs[j]) * advantage
                    logits[j] += self.lr * grad
                prob = max(probs[action_idx], 1e-8)
                total_loss += -advantage * math.log(prob)
        return {
            "updated": True,
            "loss": total_loss / max(len(samples), 1),
            "avg_reward": baseline,
            "batch_size": len(samples),
        }


class TrainableHFCausalLMBackend(BaseModelBackend):
    """
    Tiny trainable HF causal LM backend with:
    - generation
    - generated-token logprob extraction
    - policy-gradient update hook
    """

    def __init__(
        self,
        model_id,
        lr=1e-5,
        max_new_tokens=64,
        temperature=1.0,
        top_p=1.0,
        device="auto",
        seed=0,
    ):
        self.model_id = model_id
        self.lr = lr
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self.seed = seed
        self.lock = threading.Lock()
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._optimizer = None
        self._lazy_init()

    def _lazy_init(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Trainable HF backend requires torch and transformers. "
                "Install them or use --backend tiny_policy."
            ) from exc
        self._torch = torch
        torch.manual_seed(self.seed)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        if self.device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved = self.device
        self.device = resolved
        model = model.to(self.device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        self._tokenizer = tokenizer
        self._model = model
        self._optimizer = optimizer

    def supports_learning(self):
        # type: () -> bool
        return True

    def generate(self, prompt, task=None):
        # type: (str, Optional[CodingTask]) -> GenerationResult
        self._lazy_init()
        torch = self._torch
        start = time.time()
        with self.lock:
            encoded = self._tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            with torch.no_grad():
                out = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            seq = out.sequences[0]
            prompt_len = input_ids.shape[1]
            gen_ids = seq[prompt_len:]
            text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
            token_logprobs = []
            for i, scores_t in enumerate(out.scores):
                tok_id = int(gen_ids[i].item())
                log_probs = torch.log_softmax(scores_t[0], dim=-1)
                token_logprobs.append(float(log_probs[tok_id].item()))
        latency_sec = max(time.time() - start, 1e-6)
        tok_count = max(len(gen_ids), 1)
        return GenerationResult(
            text=text,
            latency_sec=latency_sec,
            num_tokens=tok_count,
            tokens_per_sec=tok_count / latency_sec,
            token_logprobs=token_logprobs,
            logprob_sum=sum(token_logprobs),
            metadata={"task_id": task.task_id if task else "", "action_idx": -1},
        )

    def _response_logprob_sum(self, prompt, response):
        # type: (str, str) -> Any
        torch = self._torch
        prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self._tokenizer.encode(response, add_special_tokens=False)
        if not response_ids:
            return None
        full_ids = prompt_ids + response_ids
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        outputs = self._model(input_ids=input_ids)
        logits = outputs.logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        start_idx = len(prompt_ids) - 1
        seq_logprob_sum = torch.tensor(0.0, device=self.device)
        for i, target_id in enumerate(response_ids):
            pos = start_idx + i
            if pos < 0 or pos >= log_probs.shape[0]:
                continue
            seq_logprob_sum = seq_logprob_sum + log_probs[pos, target_id]
        return seq_logprob_sum

    def policy_gradient_update(self, samples):
        # type: (List[Dict[str, Any]]) -> Dict[str, Any]
        self._lazy_init()
        if not samples:
            return {"updated": False, "loss": 0.0, "avg_reward": 0.0}
        torch = self._torch
        rewards = [float(s["reward"]) for s in samples]
        avg_reward = sum(rewards) / max(len(rewards), 1)
        with self.lock:
            self._model.train()
            self._optimizer.zero_grad()
            losses = []
            for sample in samples:
                prompt = sample["prompt"]
                response = sample["response"]
                reward = float(sample["reward"])
                advantage = reward - avg_reward
                seq_lp = self._response_logprob_sum(prompt, response)
                if seq_lp is None:
                    continue
                losses.append(-advantage * seq_lp)
            if not losses:
                return {"updated": False, "loss": 0.0, "avg_reward": avg_reward}
            loss = torch.stack(losses).mean()
            loss.backward()
            self._optimizer.step()
        return {
            "updated": True,
            "loss": float(loss.item()),
            "avg_reward": avg_reward,
            "batch_size": len(samples),
        }


def build_backend(args):
    # type: (Any) -> BaseModelBackend
    if args.backend == "dummy":
        return DummyModelBackend(sleep_sec=args.dummy_sleep_sec)
    if args.backend == "tiny_policy":
        return TinyPolicyBackend(lr=args.lr, seed=args.seed)
    if args.backend == "hf_trainable":
        if not args.hf_model_id:
            raise ValueError("--hf-model-id is required when --backend hf_trainable")
        return TrainableHFCausalLMBackend(
            model_id=args.hf_model_id,
            lr=args.lr,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
            seed=args.seed,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")
