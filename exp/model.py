import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

def load_model(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    return tokenizer, model

def generate_code(model, tokenizer, prompt, device, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def get_dummy_logprobs(text):
    return [0.0] * len(text.split())