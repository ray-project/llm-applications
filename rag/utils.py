import os
import subprocess

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F


def get_num_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def trim(text, max_context_length):
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(enc.encode(text)[:max_context_length])


def get_credentials(llm):
    if llm.startswith("gpt"):
        return os.environ["OPENAI_API_BASE"], os.environ["OPENAI_API_KEY"]
    else:
        return os.environ["ANYSCALE_API_BASE"], os.environ["ANYSCALE_API_KEY"]


def execute_bash(command):
    results = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return results


def predict(inputs, preprocess_fnc, tokenizer, model, label_encoder, device="cpu", threshold=0.0):
    # Get probabilities
    model.eval()
    inputs = [preprocess_fnc(item) for item in inputs]
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    y_probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

    # Assign labels based on the threshold
    labels = []
    for prob in y_probs:
        max_prob = np.max(prob)
        if max_prob < threshold:
            labels.append("other")
        else:
            labels.append(label_encoder.inverse_transform([prob.argmax()])[0])
    return labels, y_probs
