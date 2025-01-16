import os


# Force the cache location before any HF imports
os.environ["TRANSFORMERS_CACHE"] = "/workspace/huggingface_cache"
os.environ["HF_HOME"] = "/workspace/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/workspace/huggingface/datasets"
os.environ["HF_HUB_CACHE"] = "/workspace/huggingface/hub"

# Create directories
for d in [
    "/workspace/huggingface_cache",
    "/workspace/huggingface",
    "/workspace/huggingface/datasets",
    "/workspace/huggingface/hub",
]:
    os.makedirs(d, exist_ok=True)

import argparse
import time

import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import login

from transformers import AutoModelForCausalLM, AutoTokenizer


# Access the token from the environment and login
access_token = os.environ.get("HF_ACCESS_TOKEN")
login(token=access_token)

parser = argparse.ArgumentParser()

parser.add_argument("--range_start", default=25, type=int)
parser.add_argument("--range_end", default=26, type=int)
args = parser.parse_args()

# Load models and tokenizers
target_checkpoint = "meta-llama/Llama-3.1-8B-Instruct"
qwen_checkpoint = "Qwen/Qwen2.5-0.5B-Instruct"
llama_assistant_checkpoint = "meta-llama/Llama-3.2-1B-Instruct"

target_model = AutoModelForCausalLM.from_pretrained(target_checkpoint, device_map="auto", torch_dtype=torch.float16)
qwen_model = AutoModelForCausalLM.from_pretrained(qwen_checkpoint, device_map="auto", torch_dtype=torch.float16)
llama_assistant_model = AutoModelForCausalLM.from_pretrained(
    llama_assistant_checkpoint, device_map="auto", torch_dtype=torch.float16
)
llama_3b_assistant_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct", device_map="auto", torch_dtype=torch.float16
)

target_tokenizer = AutoTokenizer.from_pretrained(target_checkpoint)
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_checkpoint)

# Load the tau/scrolls dataset
dataset = load_dataset("tau/scrolls", "qasper", split="test", trust_remote_code=True)
dataset_sample = dataset.select(range(args.range_start, args.range_end))

# Evaluation storage
results = []


# Function for baseline generation
def generate_baseline(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True)
    duration = time.time() - start_time
    prefill_time = None
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    time_per_token = duration / (len(outputs[0]) - len(inputs[0]))
    return generated_text, time_per_token, prefill_time, len(inputs[0]), len(outputs[0])


# Function for assisted generation
def generate_assisted(
    prompt, target_model, assistant_model, target_tokenizer, assistant_tokenizer=None, do_sample=False
):
    inputs = target_tokenizer(prompt, return_tensors="pt").to(target_model.device)
    generate_kwargs = {
        "max_new_tokens": 512,
        "do_sample": do_sample,
        "assistant_model": assistant_model,
        "assistant_tokenizer": assistant_tokenizer,
        "tokenizer": target_tokenizer,
    }
    start_time = time.time()
    outputs = target_model.generate(**inputs, **generate_kwargs)
    duration = time.time() - start_time
    prefill_time = None
    generated_text = target_tokenizer.decode(outputs[0], skip_special_tokens=True)
    time_per_token = duration / (len(outputs[0]) - len(inputs[0]))
    return generated_text, time_per_token, prefill_time, len(inputs[0]), len(outputs[0])


# Run generation for each example in the dataset
# for i, example in enumerate(dataset.select(range(20))):
baseline_time_list, baseline_tokens_list = [], []
assisted_time_list, assisted_tokens_list = [], []
for i, example in enumerate(dataset_sample):
    prompt = example["input"]  # Ensure this field corresponds to the prompt in the dataset
    print(f"Running input prompt {i}")
    # Baseline generation
    baseline_text, baseline_time_per_token, baseline_prefill_time, baseline_inp_len, baseline_out_len = (
        generate_baseline(prompt, target_model, target_tokenizer)
    )
    # Assisted generation
    qwen_text, qwen_time_per_token, qwen_prefill_time, qwen_inp_len, qwen_out_len = generate_assisted(
        prompt, target_model, qwen_model, target_tokenizer, qwen_tokenizer, do_sample=True
    )

    qwen_uag_text, qwen_uag_time_per_token, qwen_uag_prefill_time, qwen_uag_inp_len, qwen_uag_out_len = (
        generate_assisted(prompt, target_model, qwen_model, target_tokenizer, qwen_tokenizer, do_sample=False)
    )
    # Assisted generation
    (
        llama_assisted_text,
        llama_assisted_time_per_token,
        llama_assisted_prefill_time,
        llama_assisted_inp_len,
        llama_assisted_out_len,
    ) = generate_assisted(prompt, target_model, llama_assistant_model, target_tokenizer, None)
    (
        llama_3b_assisted_text,
        llama_3b_assisted_time_per_token,
        llama3b__assisted_prefill_time,
        llama_3b_assisted_inp_len,
        llama_3b_assisted_out_len,
    ) = generate_assisted(prompt, target_model, llama_3b_assistant_model, target_tokenizer, None)
    # Store results
    results.append(
        {
            "Baseline Time Per Token ": baseline_time_per_token,
            "Baseline Prefill Time": baseline_prefill_time,
            "Baseline Len Inp": baseline_inp_len,
            "Baseline Len Out": baseline_out_len,
            "Qwen USD Time Per Token ": qwen_time_per_token,
            "Qwen USD Prefill Time": qwen_prefill_time,
            "Qwen USD Len Inp": qwen_inp_len,
            "Qwen USD Len Out": qwen_out_len,
            "Qwen UAG Time Per Token ": qwen_uag_time_per_token,
            "Qwen UAG Prefill Time": qwen_uag_prefill_time,
            "Qwen UAG Len Inp": qwen_uag_inp_len,
            "Qwen UAG Len Out": qwen_uag_out_len,
            "Llama 1B Time Per Token ": llama_assisted_time_per_token,
            "Llama 1B Prefill Time": llama_assisted_prefill_time,
            "Llama 1B Len Inp": llama_assisted_inp_len,
            "Llama 1B Len Out": llama_assisted_out_len,
            "Llama 3B Time Per Token ": llama_3b_assisted_time_per_token,
            "Llama 3B Prefill Time": llama3b__assisted_prefill_time,
            "Llama 3B Len Inp": llama_3b_assisted_inp_len,
            "Llama 3B Len Out": llama_3b_assisted_out_len,
        }
    )
    print(f"Results {results[-1]}")

# Convert results to a DataFrame
df_results = pd.DataFrame(results)
# print(df_results)

# Optionally save results to a CSV
df_results.to_csv("generation_comparison_scrolls.csv", index=False)

print("Cache locations:")
print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")
print(f"HF_HOME: {os.environ.get('HF_HOME')}")
print(f"HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE')}")
