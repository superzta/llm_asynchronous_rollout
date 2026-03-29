CODING_TASKS = [
    {
        "task_id": 1,
        "prompt": "Write a Python function add(a, b) that returns the sum of a and b.",
        "entry_point": "add",
        "tests": [
            ((1, 2), 3),
            ((5, 7), 12),
            ((-1, 1), 0),
        ],
    },
    {
        "task_id": 2,
        "prompt": "Write a Python function is_even(n) that returns True if n is even and False otherwise.",
        "entry_point": "is_even",
        "tests": [
            ((2,), True),
            ((3,), False),
            ((10,), True),
            ((11,), False),
        ],
    },
    {
        "task_id": 3,
        "prompt": "Write a Python function square(x) that returns x squared.",
        "entry_point": "square",
        "tests": [
            ((2,), 4),
            ((-3,), 9),
            ((0,), 0),
        ],
    },
    {
        "task_id": 4,
        "prompt": "Write a Python function reverse_string(s) that returns the reversed string.",
        "entry_point": "reverse_string",
        "tests": [
            (("hello",), "olleh"),
            (("abc",), "cba"),
            (("",), ""),
        ],
    },
    {
        "task_id": 5,
        "prompt": "Write a Python function max_of_two(a, b) that returns the larger of the two numbers.",
        "entry_point": "max_of_two",
        "tests": [
            ((3, 5), 5),
            ((10, 2), 10),
            ((-1, -5), -1),
        ],
    },
    {
        "task_id": 6,
        "prompt": "Write a Python function factorial(n) that returns the factorial of n. Assume n is a nonnegative integer.",
        "entry_point": "factorial",
        "tests": [
            ((0,), 1),
            ((1,), 1),
            ((5,), 120),
        ],
    },
    {
        "task_id": 7,
        "prompt": "Write a Python function count_vowels(s) that returns the number of vowels in the string.",
        "entry_point": "count_vowels",
        "tests": [
            (("hello",), 2),
            (("AEIOU",), 5),
            (("xyz",), 0),
        ],
    },
    {
        "task_id": 8,
        "prompt": "Write a Python function fibonacci(n) that returns the nth Fibonacci number, where fibonacci(0)=0 and fibonacci(1)=1.",
        "entry_point": "fibonacci",
        "tests": [
            ((0,), 0),
            ((1,), 1),
            ((6,), 8),
        ],
    },
    {
        "task_id": 9,
        "prompt": "Write a function that checks if a string is a palindrome.",
        "entry_point": "is_palindrome",
        "tests": [
            (("racecar",), True),
            (("hello",), False),
            (("a",), True),
        ],
    },
    {
        "task_id": 10,
        "prompt": "Write a Python function longest_word(sentence) that returns the longest word in a space-separated string. If there is a tie, return the first longest word.",
        "entry_point": "longest_word",
        "tests": [
            (("the quick brown fox",), "quick"),
            (("hi there friend",), "friend"),
            (("one two six",), "one"),
        ],
    },
    {
        "task_id": 11,
        "prompt": "Write a Python function merge_sorted_lists(a, b) that takes two sorted lists of integers and returns one merged sorted list.",
        "entry_point": "merge_sorted_lists",
        "tests": [
            (([1, 3, 5], [2, 4, 6]), [1, 2, 3, 4, 5, 6]),
            (([], [1, 2]), [1, 2]),
            (([1, 2, 2], [2, 3]), [1, 2, 2, 2, 3]),
        ],
    },
]