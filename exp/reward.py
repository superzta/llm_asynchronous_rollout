def extract_function_code(generated_text):
    """
    Very naive extraction.
    Later you can improve this with code block parsing.
    """
    return generated_text

def compute_code_reward(generated_code, entry_point, tests):
    local_env = {}

    try:
        exec(generated_code, {}, local_env)
    except Exception:
        return 0.0, 0, len(tests)

    if entry_point not in local_env:
        return 0.0, 0, len(tests)

    fn = local_env[entry_point]
    passed = 0

    for args, expected in tests:
        try:
            result = fn(*args)
            if result == expected:
                passed += 1
        except Exception:
            continue

    reward = passed / len(tests)
    return reward, passed, len(tests)