import os
from dotenv import load_dotenv

load_dotenv()

import argparse
import time
import pandas as pd
import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login


# ------------------------------------------------------------------------------
# Environment & Setup
# ------------------------------------------------------------------------------

def set_hf_cache_env():
    """
    Set environment variables for Hugging Face caching and create the corresponding directories.
    """
    print("Cache location:")
    for d in [
        os.environ['HF_HOME'],
    ]:
        os.makedirs(d, exist_ok=True)
        print(d)

def login_to_hf(token_env_var: str = "HF_ACCESS_TOKEN"):
    """
    Login to Hugging Face using an access token from the environment.
    """
    access_token = os.environ.get(token_env_var)
    if not access_token:
        raise ValueError(f"Environment variable {token_env_var} not found.")
    login(token=access_token)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generation Script")
    parser.add_argument("--num_of_examples", default=50, type=int, help="The number of examples from the dataset to run.")
    return parser.parse_args()

# ------------------------------------------------------------------------------
# Model Handling
# ------------------------------------------------------------------------------

class HFModel:
    """
    Lightweight class to wrap a Hugging Face model and tokenizer for convenience.
    """
    def __init__(self, model_name: str, device_map: str = 'auto', torch_dtype=torch.float16):
        """
        Load a model and tokenizer from the Hugging Face Hub.
        """
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        **kwargs
    ):
        """
        Generate text from the underlying model.
        Returns:
            generated_text (str), duration (float), prefill_time (None), input_len (int), output_len (int)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        start_time = time.time()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs
        )
        duration = time.time() - start_time
        
        # We currently do not measure prefill time separately, so it's None.
        prefill_time = None
        # Decode output (take the first item in the batch).
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Basic time-per-token: total duration / number of tokens generated
        input_len = len(inputs["input_ids"][0])
        output_len = len(outputs[0])
        time_per_token = duration / (output_len - input_len) if (output_len - input_len) else float('inf')

        return generated_text, time_per_token, prefill_time, input_len, output_len

# ------------------------------------------------------------------------------
# Generation Logic
# ------------------------------------------------------------------------------

def generate_baseline(prompt: str, model_obj: HFModel):
    """
    Baseline generation using a single HFModel.
    """
    return model_obj.generate_text(prompt, do_sample=True)

def generate_assisted(
    prompt: str,
    target_model_obj: HFModel,
    assistant_model_obj: HFModel = None,
    do_sample: bool = False
):
    """
    Demonstrates an assisted generation approach:
    Optionally pass an assistant model or additional arguments if there's
    custom logic that merges two models. By default, standard Transformers
    doesn't accept a second 'assistant_model', so adjust as needed.
    """
    # If you have custom logic that merges the assistant model, it would go here.
    # For demonstration, we just pass along some custom kwargs if needed.
    generate_kwargs = {}
    if assistant_model_obj is not None:
        # Possibly you'd do something with assistant_model_obj.model or .tokenizer
        # For now, assume you pass them to custom generate if your codebase supports it.
        generate_kwargs["assistant_model"] = assistant_model_obj.model
        generate_kwargs["assistant_tokenizer"] = assistant_model_obj.tokenizer

    return target_model_obj.generate_text(prompt, do_sample=do_sample, **generate_kwargs)

# ------------------------------------------------------------------------------
# Main Script
# ------------------------------------------------------------------------------

def main():
    # 1. Environment setup
    set_hf_cache_env()

    # 2. Login
    login_to_hf()

    # 3. Parse arguments
    args = parse_args()

    # 4. Load models
    target_checkpoint = "meta-llama/Llama-3.1-8B-Instruct"
    qwen_checkpoint = "Qwen/Qwen2.5-0.5B-Instruct"
    llama_assistant_checkpoint = "meta-llama/Llama-3.2-1B-Instruct"
    llama_3b_assistant_checkpoint = "meta-llama/Llama-3.2-3B-Instruct"

    target_model_obj = HFModel(target_checkpoint)
    qwen_model_obj = HFModel(qwen_checkpoint)
    llama_assistant_model_obj = HFModel(llama_assistant_checkpoint)
    llama_3b_assistant_model_obj = HFModel(llama_3b_assistant_checkpoint)

    # 5. Load dataset
    dataset = load_dataset("tau/scrolls", "qasper", split="test", trust_remote_code=True)
    dataset_sample = dataset.select(range(args.num_of_examples))

    # 6. Generation loop
    results = []
    for i, example in enumerate(dataset_sample):
        prompt = example["input"]  # Adjust if actual prompt field differs

        print(f"Running input prompt {i}...")

        # Baseline
        baseline_text, baseline_tpt, baseline_prefill, baseline_inp_len, baseline_out_len = generate_baseline(
            prompt, target_model_obj
        )

        # Qwen assisted: sample = True
        qwen_text, qwen_tpt, qwen_prefill, qwen_inp_len, qwen_out_len = generate_assisted(
            prompt, target_model_obj, qwen_model_obj, do_sample=True
        )

        # Qwen assisted: sample = False
        qwen_uag_text, qwen_uag_tpt, qwen_uag_prefill, qwen_uag_inp_len, qwen_uag_out_len = generate_assisted(
            prompt, target_model_obj, qwen_model_obj, do_sample=False
        )

        # Llama 1B assisted
        llama_assisted_text, llama_assisted_tpt, llama_assisted_prefill, llama_assisted_inp_len, llama_assisted_out_len = generate_assisted(
            prompt, target_model_obj, llama_assistant_model_obj
        )

        # Llama 3B assisted
        llama_3b_assisted_text, llama_3b_assisted_tpt, llama_3b_assisted_prefill, llama_3b_assisted_inp_len, llama_3b_assisted_out_len = generate_assisted(
            prompt, target_model_obj, llama_3b_assistant_model_obj
        )

        # Collect results
        results.append({
            "Baseline Time Per Token": baseline_tpt,
            "Baseline Prefill Time": baseline_prefill,
            "Baseline Len Inp": baseline_inp_len,
            "Baseline Len Out": baseline_out_len,

            "Qwen USD Time Per Token": qwen_tpt,
            "Qwen USD Prefill Time": qwen_prefill,
            "Qwen USD Len Inp": qwen_inp_len,
            "Qwen USD Len Out": qwen_out_len,

            "Qwen UAG Time Per Token": qwen_uag_tpt,
            "Qwen UAG Prefill Time": qwen_uag_prefill,
            "Qwen UAG Len Inp": qwen_uag_inp_len,
            "Qwen UAG Len Out": qwen_uag_out_len,

            "Llama 1B Time Per Token": llama_assisted_tpt,
            "Llama 1B Prefill Time": llama_assisted_prefill,
            "Llama 1B Len Inp": llama_assisted_inp_len,
            "Llama 1B Len Out": llama_assisted_out_len,

            "Llama 3B Time Per Token": llama_3b_assisted_tpt,
            "Llama 3B Prefill Time": llama_3b_assisted_prefill,
            "Llama 3B Len Inp": llama_3b_assisted_inp_len,
            "Llama 3B Len Out": llama_3b_assisted_out_len,
        })

        print(f"Results for prompt {i}: {results[-1]}")

    # 7. Convert to DataFrame & save
    df_results = pd.DataFrame(results)
    df_results.to_csv("generation_comparison_scrolls.csv", index=False)
    print("Results saved to generation_comparison_scrolls.csv")

if __name__ == "__main__":
    main()
