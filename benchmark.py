import os
from dotenv import load_dotenv

# Setting up the `HF_HOME` cache directory and `HF_ACCESS_TOKEN` token
load_dotenv()

import argparse
import time
import pandas as pd
import torch

from typing import Optional
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login


# ------------------------------------------------------------------------------
# Environment & Setup
# ------------------------------------------------------------------------------

def set_hf_cache_env():
    """
    Sets the environment variables for Hugging Face caching 
    and creates the corresponding directories.
    """
    print("Cache location:")
    hf_home = os.environ["HF_HOME"]  # Store in variable for clarity
    os.makedirs(hf_home, exist_ok=True)
    print(hf_home)

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
        do_sample: bool,
        max_new_tokens: int = 512,
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
        
        # Compute basic time-per-token: total duration / number of tokens generated.
        input_len = len(inputs["input_ids"][0])
        output_len = len(outputs[0])
        num_generated_tokens = output_len - input_len  # Store calculation in descriptive variable
        time_per_token = duration / num_generated_tokens if num_generated_tokens > 0 else float("inf")

        return generated_text, time_per_token, prefill_time, input_len, output_len


def tokenizers_are_identical(t1, t2) -> bool:
    """
    Return True if t1 and t2 are effectively the same tokenizer, i.e.,
    produce identical results for any input text. This is a thorough approach
    that checks object identity, init_kwargs, vocab, merges, and special tokens.
    Adjust or omit checks if your definition of "sameness" is simpler.
    """

    # 1. Same Python object?
    if t1 is t2:
        return True

    # 2. Same class?
    if type(t1) != type(t2):
        return False

    # 3. Compare essential init_kwargs
    #    Omit path-based keys from the comparison to allow local-vs-remote differences.
    def _filter_init_kwargs(iw):
        return {k: v for k, v in iw.items() if k not in ["name_or_path", "tokenizer_file"]}

    if _filter_init_kwargs(t1.init_kwargs) != _filter_init_kwargs(t2.init_kwargs):
        return False

    # 4. Compare vocabulary (simple approach)
    #    Vocab might be stored differently depending on the tokenizer, adapt as needed.
    #    Usually a safe approach is t1.get_vocab() vs t2.get_vocab()
    vocab1 = t1.get_vocab()  # token -> id
    vocab2 = t2.get_vocab()
    if len(vocab1) != len(vocab2):
        return False

    # Optional but more thorough: check each token's ID
    for token, idx in vocab1.items():
        if token not in vocab2 or vocab2[token] != idx:
            return False

    # Also verify that t2 doesn't have extra tokens that t1 lacks
    for token, idx in vocab2.items():
        if token not in vocab1 or vocab1[token] != idx:
            return False

    # 5. Compare merges if it’s a BPE-based tokenizer
    #    Not all tokenizers have `.merges`. Some have `tokenizer.sp_model`, etc.
    merges_t1 = getattr(t1, "merges", None)
    merges_t2 = getattr(t2, "merges", None)
    if merges_t1 != merges_t2:
        return False

    # 6. Compare special tokens
    if t1.special_tokens_map != t2.special_tokens_map:
        return False

    return True

# ------------------------------------------------------------------------------
# Generation Logic
# ------------------------------------------------------------------------------

def generate_baseline(prompt: str, model_obj: HFModel):
    """
    Baseline generation using a single HFModel.
    """
    return model_obj.generate_text(prompt=prompt, do_sample=True)

def generate_assisted(
    prompt: str,
    target_model_obj: HFModel,
    assistant_model_obj: Optional[HFModel],
    are_tokenizers_identical: bool,
    do_sample: bool
):
    """
    Demonstrates an assisted generation approach:
    Optionally pass an assistant model or additional arguments if there's
    custom logic that merges two models. By default, standard Transformers
    doesn't accept a second 'assistant_model', so adjust as needed.
    """
    generate_kwargs = {}
    if assistant_model_obj is not None:
        generate_kwargs["assistant_model"] = assistant_model_obj.model
        if not are_tokenizers_identical:
            generate_kwargs["assistant_tokenizer"] = assistant_model_obj.tokenizer
            generate_kwargs["tokenizer"] = target_model_obj.tokenizer
    return target_model_obj.generate_text(
        prompt=prompt,
        do_sample=do_sample,
        **generate_kwargs
    )

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
    dataset_name = "tau/scrolls"
    dataset = load_dataset(dataset_name, "qasper", split="test", trust_remote_code=True)
    dataset_sample = dataset.select(range(args.num_of_examples))

    # 6. Generation loop
    results = []
    for i, example in enumerate(dataset_sample):
        prompt = example["input"]  # Adjust if the actual prompt field is different

        print(f"Running input prompt {i}...")

        print("Running Baseline...")
        baseline_text, baseline_tpt, baseline_prefill, baseline_inp_len, baseline_out_len = \
            generate_baseline(prompt, target_model_obj)

        print("Running Qwen assisted with `do_sample=True`...")
        qwen_text, qwen_tpt, qwen_prefill, qwen_inp_len, qwen_out_len = generate_assisted(
            prompt=prompt,
            target_model_obj=target_model_obj,
            assistant_model_obj=qwen_model_obj,
            are_tokenizers_identical=False,
            do_sample=True
        )

        print("Running Qwen assisted with `do_sample=False`...")
        qwen_uag_text, qwen_uag_tpt, qwen_uag_prefill, qwen_uag_inp_len, qwen_uag_out_len = generate_assisted(
            prompt=prompt,
            target_model_obj=target_model_obj,
            assistant_model_obj=qwen_model_obj,
            are_tokenizers_identical=False,
            do_sample=False
        )

        print("Running Llama 1B assisted...")
        llama_assisted_text, llama_assisted_tpt, llama_assisted_prefill, llama_assisted_inp_len, llama_assisted_out_len = generate_assisted(
            prompt=prompt,
            target_model_obj=target_model_obj,
            assistant_model_obj=llama_assistant_model_obj,
            are_tokenizers_identical=True,
            do_sample=True
        )

        print("Running Llama 3B assisted...")
        llama_3b_assisted_text, llama_3b_assisted_tpt, llama_3b_assisted_prefill, llama_3b_assisted_inp_len, llama_3b_assisted_out_len = generate_assisted(
            prompt=prompt,
            target_model_obj=target_model_obj,
            assistant_model_obj=llama_3b_assistant_model_obj,
            are_tokenizers_identical=True,
            do_sample=True
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
    filename_results = f"latency_benchmark_on_{dataset_name}_{args.num_of_examples}_examples.csv"
    df_results.to_csv(filename_results, index=False)
    print(f"Results saved to {filename_results}")

if __name__ == "__main__":
    main()
