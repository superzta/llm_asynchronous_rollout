import json
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Union


class CodingTask:
    def __init__(self, task_id, prompt, reference_tests, starter_code=None):
        self.task_id = task_id
        self.prompt = prompt
        self.reference_tests = list(reference_tests)
        self.starter_code = starter_code

    @classmethod
    def from_dict(cls, raw):
        return cls(
            task_id=raw["task_id"],
            prompt=raw["prompt"],
            reference_tests=list(raw["reference_tests"]),
            starter_code=raw.get("starter_code"),
        )

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "reference_tests": self.reference_tests,
            "starter_code": self.starter_code,
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


def build_model_prompt(task, system_instruction=None):
    # type: (CodingTask, Optional[str]) -> str
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


def repeat_tasks(tasks, epochs):
    # type: (Iterable[CodingTask], int) -> List[CodingTask]
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    source = list(tasks)
    output = []  # type: List[CodingTask]
    for _ in range(epochs):
        output.extend(source)
    return output
