import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def load_model(device: str):
    # here load the model qwen2.5 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,dtype=torch.float16,).to(device)
    model.eval()
    return tokenizer, model

# here generate code based on teh model for rollout computation 
def generate_code(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 128):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def get_dummy_logprobs(text: str):
    # Placeholder for PPO-style old logprobs.
    # Later you can replace this with actual token logprobs from the policy.
    return [0.0] * len(text.split())

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def load_model(device: str):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to(device)

    model.eval()
    print("Model loaded successfully!")
    return tokenizer, model


def generate_code(model, tokenizer, prompt: str, device: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("Generating output...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        tokenizer, model = load_model(device)

        test_prompt = "Write a Python function to compute factorial:"
        output = generate_code(model, tokenizer, test_prompt, device)

        print("\n=== MODEL OUTPUT ===")
        print(output)

    except Exception as e:
        print("Error during model loading or inference:")
        print(e)