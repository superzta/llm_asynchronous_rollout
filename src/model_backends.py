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

    def get_trainable_state(self):
        # type: () -> Dict[str, Any]
        return {}

    def load_trainable_state(self, state):
        # type: (Dict[str, Any]) -> None
        _ = state
        return

    def clone_for_device(self, device):
        # type: (str) -> "BaseModelBackend"
        _ = device
        return self

    def maybe_generate_chunk(self, prompt, task, generation_state, chunk_size):
        # type: (str, Optional[CodingTask], Optional[Dict[str, Any]], int) -> Dict[str, Any]
        # Default one-shot fallback.
        result = self.generate(prompt=prompt, task=task)
        return {
            "done": True,
            "generation_state": None,
            "result": result,
            "intermediate_text": result.text,
        }


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

    def clone_for_device(self, device):
        # type: (str) -> BaseModelBackend
        _ = device
        return DummyModelBackend(
            sleep_sec=self.sleep_sec,
            use_markdown_fence=self.use_markdown_fence,
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

    def get_trainable_state(self):
        # type: () -> Dict[str, Any]
        state = {"task_logits": {}, "lr": self.lr}
        with self.lock:
            for key, value in self.task_logits.items():
                state["task_logits"][key] = [float(x) for x in value]
        return state

    def load_trainable_state(self, state):
        # type: (Dict[str, Any]) -> None
        if not state:
            return
        with self.lock:
            raw = state.get("task_logits", {})
            self.task_logits = {}
            for key, value in raw.items():
                self.task_logits[key] = [float(x) for x in value]
            if "lr" in state:
                self.lr = float(state["lr"])

    def clone_for_device(self, device):
        # type: (str) -> BaseModelBackend
        _ = device
        clone = TinyPolicyBackend(
            lr=self.lr,
            use_markdown_fence=self.use_markdown_fence,
            seed=0,
        )
        clone.load_trainable_state(self.get_trainable_state())
        return clone

    def maybe_generate_chunk(self, prompt, task, generation_state, chunk_size):
        # type: (str, Optional[CodingTask], Optional[Dict[str, Any]], int) -> Dict[str, Any]
        if generation_state is None:
            generation_state = {"remaining_chunks": max(int(chunk_size), 1)}
        generation_state["remaining_chunks"] -= 1
        if generation_state["remaining_chunks"] > 0:
            return {
                "done": False,
                "generation_state": generation_state,
                "result": None,
                "intermediate_text": "",
            }
        result = self.generate(prompt=prompt, task=task)
        return {
            "done": True,
            "generation_state": None,
            "result": result,
            "intermediate_text": result.text,
        }


class TrainableHFCausalLMBackend(BaseModelBackend):
    """
    V100-friendly trainable HF causal LM backend with a REAL GRPO/PPO update.

    Design notes:
    -  generation stores the exact `response_ids` and per-token `old_logprobs`
       captured at sampling time. The update uses those tensors directly, so
       the PPO ratio is computed from matching token sequences.
    -  update = PPO-clip * group-normalized advantage + beta * KL(new || old)
       using per-token logprobs. This is the GRPO loss (without a value head).
    -  no flash-attn; default attn_impl="sdpa" which works on V100.
    -  default dtype="float16" (V100 has no bf16).
    -  optional chat template application inside generate() for Qwen-Instruct.
    """

    def __init__(
        self,
        model_id,
        lr=1e-6,
        max_new_tokens=64,
        temperature=1.0,
        top_p=1.0,
        device="auto",
        seed=0,
        dtype="float32",
        attn_impl="eager",
        use_chat_template=True,
        grpo_epsilon=0.2,
        grpo_kl_coef=0.02,
        grpo_group_size=0,  # 0 => batch-level group normalization
        grad_clip=1.0,
        weight_decay=0.0,
        decoupled_objective=True,
    ):
        self.model_id = model_id
        self.lr = lr
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self.seed = seed
        self.dtype_name = dtype
        self.attn_impl = attn_impl
        self.use_chat_template = use_chat_template
        self.grpo_epsilon = float(grpo_epsilon)
        self.grpo_kl_coef = float(grpo_kl_coef)
        self.grpo_group_size = int(grpo_group_size)
        self.grad_clip = float(grad_clip)
        self.weight_decay = float(weight_decay)
        self.decoupled_objective = bool(decoupled_objective)
        self.lock = threading.Lock()
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._optimizer = None
        self._lazy_init()

    def _resolve_dtype(self):
        torch = self._torch
        name = str(self.dtype_name).lower()
        if name in ("fp16", "float16", "half"):
            return torch.float16
        if name in ("bf16", "bfloat16"):
            return torch.bfloat16
        return torch.float32

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
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = self._resolve_dtype()
        # transformers >=4.55 prefers `dtype=`; older expects `torch_dtype=`.
        base_kwargs = {"trust_remote_code": True}
        dtype_keys_to_try = ["dtype", "torch_dtype"]
        attn_keys_to_try = [self.attn_impl, None]
        model = None
        last_err = None
        for dkey in dtype_keys_to_try:
            for attn in attn_keys_to_try:
                kwargs = dict(base_kwargs)
                kwargs[dkey] = dtype
                if attn is not None:
                    kwargs["attn_implementation"] = attn
                try:
                    model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
                    break
                except (TypeError, ValueError) as exc:
                    last_err = exc
                    continue
            if model is not None:
                break
        if model is None:
            raise last_err if last_err is not None else RuntimeError(
                "Failed to load model %s" % self.model_id
            )

        if self.device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved = self.device
        self.device = resolved
        model = model.to(self.device)
        model.gradient_checkpointing_disable()
        model.train()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self._tokenizer = tokenizer
        self._model = model
        self._optimizer = optimizer

    def supports_learning(self):
        # type: () -> bool
        return True

    def _format_prompt(self, prompt):
        """Apply chat template if requested and supported."""
        tokenizer = self._tokenizer
        if not self.use_chat_template:
            return prompt
        if not hasattr(tokenizer, "apply_chat_template"):
            return prompt
        if getattr(tokenizer, "chat_template", None) is None:
            return prompt
        try:
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return prompt

    def generate(self, prompt, task=None):
        # type: (str, Optional[CodingTask]) -> GenerationResult
        self._lazy_init()
        torch = self._torch
        start = time.time()
        formatted = self._format_prompt(prompt)

        from transformers import LogitsProcessor, LogitsProcessorList

        class _NanInfGuard(LogitsProcessor):
            """Replace NaN / +-Inf with a safe large-negative value so softmax is valid."""

            def __call__(self, input_ids, scores):
                import torch as _t
                safe_neg = _t.full_like(scores, -1e4)
                scores = _t.where(_t.isnan(scores) | _t.isinf(scores), safe_neg, scores)
                return scores

        with self.lock:
            encoded = self._tokenizer(formatted, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            self._model.eval()
            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=50,
                return_dict_in_generate=True,
                output_scores=True,
                renormalize_logits=True,
                logits_processor=LogitsProcessorList([_NanInfGuard()]),
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
            with torch.no_grad():
                out = self._model.generate(**gen_kwargs)
            self._model.train()
            seq = out.sequences[0]
            prompt_len = input_ids.shape[1]
            gen_ids_t = seq[prompt_len:]
            text = self._tokenizer.decode(gen_ids_t, skip_special_tokens=True)
            token_logprobs = []
            for i, scores_t in enumerate(out.scores):
                tok_id = int(gen_ids_t[i].item())
                # scores_t is pre-softmax logits with temperature/top-p masking
                log_probs = torch.log_softmax(scores_t[0].float(), dim=-1)
                token_logprobs.append(float(log_probs[tok_id].item()))
            prompt_ids_list = [int(x) for x in input_ids[0].tolist()]
            response_ids_list = [int(x) for x in gen_ids_t.tolist()]
        latency_sec = max(time.time() - start, 1e-6)
        tok_count = max(len(response_ids_list), 1)
        metadata = {
            "task_id": task.task_id if task else "",
            "action_idx": -1,
            "prompt_ids": prompt_ids_list,
            "response_ids": response_ids_list,
            "old_token_logprobs": list(token_logprobs),
            "formatted_prompt": formatted,
        }
        return GenerationResult(
            text=text,
            latency_sec=latency_sec,
            num_tokens=tok_count,
            tokens_per_sec=tok_count / latency_sec,
            token_logprobs=token_logprobs,
            logprob_sum=sum(token_logprobs),
            metadata=metadata,
        )

    def _recompute_new_logprobs(self, prompt_ids, response_ids):
        """Recompute current-model per-token logprobs for given response tokens."""
        torch = self._torch
        if not response_ids:
            return None
        full = list(prompt_ids) + list(response_ids)
        t_full = torch.tensor([full], dtype=torch.long, device=self.device)
        outputs = self._model(input_ids=t_full)
        logits = outputs.logits[0]  # [seq_len, vocab]
        # For predicting token at position i, use logits at i-1. Response tokens
        # begin at index len(prompt_ids).
        p_len = len(prompt_ids)
        resp_tensor = torch.tensor(response_ids, dtype=torch.long, device=self.device)
        gather_idx = torch.arange(p_len - 1, p_len - 1 + len(response_ids), device=self.device)
        resp_logits = logits[gather_idx]  # [resp_len, vocab]
        log_probs = torch.log_softmax(resp_logits.float(), dim=-1)
        new_token_logprobs = log_probs.gather(1, resp_tensor.unsqueeze(1)).squeeze(1)
        return new_token_logprobs  # [resp_len]

    def policy_gradient_update(self, samples):
        """
        Real GRPO-style PPO update using per-token old logprobs captured at
        rollout time. This is a proper clipped-surrogate objective, not a
        REINFORCE approximation.
        """
        self._lazy_init()
        if not samples:
            return {"updated": False, "loss": 0.0, "avg_reward": 0.0}
        torch = self._torch
        rewards = [float(s.get("reward", 0.0)) for s in samples]
        avg_reward = sum(rewards) / max(len(rewards), 1)

        # Group-normalized advantages. If grpo_group_size > 0, group by task_id.
        if self.grpo_group_size > 0:
            groups = {}
            for idx, s in enumerate(samples):
                key = s.get("task_id", "_")
                groups.setdefault(key, []).append(idx)
            advantages = [0.0] * len(samples)
            for _, idxs in groups.items():
                group_rewards = [rewards[i] for i in idxs]
                if len(group_rewards) == 0:
                    continue
                mean_r = sum(group_rewards) / len(group_rewards)
                var_r = sum((r - mean_r) ** 2 for r in group_rewards) / max(len(group_rewards), 1)
                std_r = max(var_r ** 0.5, 1e-6)
                for i in idxs:
                    advantages[i] = (rewards[i] - mean_r) / std_r
        else:
            mean_r = avg_reward
            var_r = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)
            std_r = max(var_r ** 0.5, 1e-6)
            advantages = [(r - mean_r) / std_r for r in rewards]

        eps = self.grpo_epsilon
        kl_coef = self.grpo_kl_coef

        policy_loss_terms = []
        kl_terms = []
        clip_fraction_terms = []
        num_tokens_total = 0
        num_samples_used = 0

        with self.lock:
            self._model.train()
            self._optimizer.zero_grad()
            for idx, sample in enumerate(samples):
                meta = sample.get("metadata", {}) or {}
                prompt_ids = meta.get("prompt_ids")
                response_ids = meta.get("response_ids")
                old_logps = meta.get("old_token_logprobs")
                if not prompt_ids or not response_ids or not old_logps:
                    continue
                n = min(len(response_ids), len(old_logps))
                if n == 0:
                    continue
                response_ids = response_ids[:n]
                old_logps = old_logps[:n]
                new_token_logprobs = self._recompute_new_logprobs(prompt_ids, response_ids)
                if new_token_logprobs is None or new_token_logprobs.numel() == 0:
                    continue
                adv = float(advantages[idx])
                adv_t = torch.full_like(new_token_logprobs, adv)
                if self.decoupled_objective:
                    # Full PPO-clip with stored old logprobs captured at rollout.
                    old_t = torch.tensor(
                        old_logps, dtype=new_token_logprobs.dtype, device=self.device
                    )
                    log_ratio = new_token_logprobs - old_t
                    ratio = torch.exp(log_ratio)
                    unclipped = ratio * adv_t
                    clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv_t
                    token_loss = -torch.min(unclipped, clipped)
                    kl_token = (old_t - new_token_logprobs).clamp(min=-20.0, max=20.0)
                    with torch.no_grad():
                        clip_frac = (
                            (ratio < 1.0 - eps) | (ratio > 1.0 + eps)
                        ).float().mean().item()
                else:
                    # Naive PPO a la Figure 5a: ignore behavior logprobs, no ratio
                    # correction, no clip. Effectively REINFORCE with group-
                    # normalized advantage. KL term degenerates to 0.
                    token_loss = -adv_t * new_token_logprobs
                    kl_token = torch.zeros_like(new_token_logprobs)
                    clip_frac = 0.0
                policy_loss_terms.append(token_loss.sum())
                kl_terms.append(kl_token.sum())
                clip_fraction_terms.append(float(clip_frac))
                num_tokens_total += int(new_token_logprobs.numel())
                num_samples_used += 1

            if num_samples_used == 0 or num_tokens_total == 0:
                return {
                    "updated": False,
                    "loss": 0.0,
                    "avg_reward": avg_reward,
                    "batch_size": len(samples),
                }

            total_policy = torch.stack(policy_loss_terms).sum() / float(num_tokens_total)
            total_kl = torch.stack(kl_terms).sum() / float(num_tokens_total)
            loss = total_policy + kl_coef * total_kl
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.grad_clip)
            self._optimizer.step()

        return {
            "updated": True,
            "loss": float(loss.item()),
            "policy_loss": float(total_policy.item()),
            "kl_penalty": float(total_kl.item()),
            "avg_clip_fraction": (
                sum(clip_fraction_terms) / max(len(clip_fraction_terms), 1)
            ),
            "avg_reward": avg_reward,
            "batch_size": len(samples),
            "num_samples_used": int(num_samples_used),
            "num_tokens_total": int(num_tokens_total),
        }

    def get_trainable_state(self):
        # type: () -> Dict[str, Any]
        self._lazy_init()
        with self.lock:
            state_dict = self._model.state_dict()
            cpu_state = {}
            for key, value in state_dict.items():
                cpu_state[key] = value.detach().to("cpu").clone()
        return {"state_dict": cpu_state}

    def load_trainable_state(self, state):
        # type: (Dict[str, Any]) -> None
        self._lazy_init()
        if not state:
            return
        raw_state = state.get("state_dict")
        if not raw_state:
            return
        with self.lock:
            self._model.load_state_dict(raw_state, strict=False)

    def clone_for_device(self, device):
        # type: (str) -> BaseModelBackend
        clone = TrainableHFCausalLMBackend(
            model_id=self.model_id,
            lr=self.lr,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            device=device,
            seed=self.seed,
            dtype=self.dtype_name,
            attn_impl=self.attn_impl,
            use_chat_template=self.use_chat_template,
            grpo_epsilon=self.grpo_epsilon,
            grpo_kl_coef=self.grpo_kl_coef,
            grpo_group_size=self.grpo_group_size,
            grad_clip=self.grad_clip,
            weight_decay=self.weight_decay,
            decoupled_objective=self.decoupled_objective,
        )
        clone.load_trainable_state(self.get_trainable_state())
        return clone

    def maybe_generate_chunk(self, prompt, task, generation_state, chunk_size):
        # type: (str, Optional[CodingTask], Optional[Dict[str, Any]], int) -> Dict[str, Any]
        # Chunk-level fallback: one generation call per chunk request; the
        # safe-boundary interruption happens between whole sample generations.
        _ = chunk_size
        result = self.generate(prompt=prompt, task=task)
        return {
            "done": True,
            "generation_state": None,
            "result": result,
            "intermediate_text": result.text,
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
            dtype=getattr(args, "hf_dtype", "float16"),
            attn_impl=getattr(args, "hf_attn_impl", "sdpa"),
            use_chat_template=bool(getattr(args, "hf_chat_template", True)),
            grpo_epsilon=float(getattr(args, "grpo_epsilon", 0.2)),
            grpo_kl_coef=float(getattr(args, "grpo_kl_coef", 0.02)),
            grpo_group_size=int(getattr(args, "grpo_group_size", 0)),
            grad_clip=float(getattr(args, "grad_clip", 1.0)),
            weight_decay=float(getattr(args, "weight_decay", 0.0)),
            decoupled_objective=bool(int(getattr(args, "decoupled_objective", 1))),
        )
    raise ValueError("Unsupported backend: %s" % args.backend)
