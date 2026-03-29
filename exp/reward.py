def extract_function_code(generated_text: str):
    """
    Very naive extraction.
    Later you can improve this with code block parsing.
    """
    return generated_text


def compute_code_reward(generated_code: str, entry_point: str, tests):
    local_env = {}
    # compute the reward and store it for the rollout 
    # first execute the generated code 
    try:
        exec(generated_code, {}, local_env)
    except Exception:
        # if the generated code is wrong caputer and return 
        return 0.0, 0, len(tests)

    if entry_point not in local_env:
        # check the model actually defined the correct functino 
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

    total = len(tests)
    reward = passed / total if total > 0 else 0.0
    return reward, passed, total