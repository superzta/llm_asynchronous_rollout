import json
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Union


class CodingTask:
    """
    Generic task container.

    Despite the name (kept for backward compatibility), this class can carry
    any task type whose reward is a scalar in [0, 1]. Supported task types:

      - "coding": evaluated by running reference_tests against generated code.
      - "gsm8k":  evaluated by extracting a numeric answer from the response
                  and comparing to gold_answer.
    """

    def __init__(
        self,
        task_id,
        prompt,
        reference_tests=None,
        starter_code=None,
        task_type="coding",
        gold_answer=None,
    ):
        self.task_id = task_id
        self.prompt = prompt
        self.reference_tests = list(reference_tests) if reference_tests else []
        self.starter_code = starter_code
        self.task_type = str(task_type)
        self.gold_answer = gold_answer

    @classmethod
    def from_dict(cls, raw):
        return cls(
            task_id=raw["task_id"],
            prompt=raw["prompt"],
            reference_tests=list(raw.get("reference_tests", [])),
            starter_code=raw.get("starter_code"),
            task_type=raw.get("task_type", "coding"),
            gold_answer=raw.get("gold_answer"),
        )

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "reference_tests": self.reference_tests,
            "starter_code": self.starter_code,
            "task_type": self.task_type,
            "gold_answer": self.gold_answer,
        }


def load_tasks(path):
    # type: (Union[str, Path]) -> List[CodingTask]
    task_path = Path(path)
    tasks = []  # type: List[CodingTask]
    with task_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(CodingTask.from_dict(json.loads(line)))
    return tasks


def iter_tasks(path):
    # type: (Union[str, Path]) -> Iterator[CodingTask]
    yield from load_tasks(path)


def _build_coding_prompt(task, system_instruction=None):
    instruction = system_instruction or (
        "Write only valid Python code for the requested function."
    )
    parts = [instruction, "", task.prompt]
    if task.starter_code:
        parts.extend(
            [
                "",
                "Starter code:",
                "```python",
                task.starter_code.rstrip(),
                "```",
                "",
                "Return the completed function implementation.",
            ]
        )
    return "\n".join(parts).strip() + "\n"


def _build_gsm8k_prompt(task, system_instruction=None):
    instruction = system_instruction or (
        "Solve the following grade-school math problem. Think step by step. "
        "End your answer on its own line with the format: #### <final_number>."
    )
    parts = [
        instruction,
        "",
        "Question: " + task.prompt.strip(),
        "",
        "Answer:",
    ]
    return "\n".join(parts).strip() + "\n"


def build_model_prompt(task, system_instruction=None):
    # type: (CodingTask, Optional[str]) -> str
    if task.task_type == "gsm8k":
        return _build_gsm8k_prompt(task, system_instruction=system_instruction)
    return _build_coding_prompt(task, system_instruction=system_instruction)


def repeat_tasks(tasks, epochs):
    # type: (Iterable[CodingTask], int) -> List[CodingTask]
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    source = list(tasks)
    output = []  # type: List[CodingTask]
    for _ in range(epochs):
        output.extend(source)
    return output
