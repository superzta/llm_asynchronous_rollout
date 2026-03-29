import unittest

from src.coding_reward import evaluate_response, extract_python_code


class CodingRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = {
            "task_id": "tiny",
            "prompt": "Write f(x) = x + 1",
            "reference_tests": ["assert f(1) == 2", "assert f(9) == 10"],
            "starter_code": "def f(x):\n    pass\n",
        }

    def test_extract_from_markdown_fence(self) -> None:
        response = "Here is code:\n```python\ndef f(x):\n    return x + 1\n```"
        code = extract_python_code(response)
        self.assertIn("def f", code)
        self.assertNotIn("```", code)

    def test_timeout_handling(self) -> None:
        response = "def f(x):\n    while True:\n        pass\n"
        result = evaluate_response(response_text=response, task=self.task, timeout_sec=0.2)
        self.assertEqual(result["error_type"], "timeout")
        self.assertEqual(result["reward"], 0.0)

    def test_syntax_error_handling(self) -> None:
        response = "def f(x)\n    return x + 1\n"
        result = evaluate_response(response_text=response, task=self.task, timeout_sec=1.0)
        self.assertEqual(result["error_type"], "syntax_error")
        self.assertEqual(result["reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
