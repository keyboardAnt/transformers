from datetime import datetime
import os
import subprocess

from dotenv import load_dotenv


# Setting up the `HF_HOME` cache directory and `HF_ACCESS_TOKEN` token
load_dotenv()

import argparse
import gc
import time
from dataclasses import dataclass
from pprint import pprint
from threading import Thread
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import login

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import OffloadedStaticCache
from transformers.generation.streamers import BaseStreamer


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
    parser.add_argument(
        "--num_of_examples", default=50, type=int, help="The number of examples from the dataset to run."
    )
    return parser.parse_args()


def clear_memory():
    """
    Clears Python and GPU memory to ensure a fresh start for experiments.
    Includes additional cleanup steps for more thorough memory management.
    """
    # Run garbage collection multiple times to handle circular references
    for _ in range(3):
        gc.collect()

    # Clear CUDA memory if available
    if torch.cuda.is_available():
        # Explicitly empty CUDA cache
        torch.cuda.empty_cache()

        # Force synchronization of CUDA threads
        torch.cuda.synchronize()

        # Collect inter-process CUDA memory
        torch.cuda.ipc_collect()

        # Print memory stats for debugging (optional)
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f}GB")
            print(f"GPU {i} memory cached: {torch.cuda.memory_reserved(i) / 1e9:.2f}GB")

    # Reset the PyTorch CPU memory allocator
    if hasattr(torch, "cuda"):
        torch.cuda.empty_cache()

    print("Memory cleared: Python memory garbage collected and GPU cache emptied.")


# ------------------------------------------------------------------------------
# Streaming
# ------------------------------------------------------------------------------


class IdsIteratorStreamer(BaseStreamer):
    """
    A custom streamer that yields token IDs instead of decoded text.
    Skips the first `prompt_len` tokens, so you don't stream the prompt.
    """

    def __init__(self, prompt_len: int = 0):
        super().__init__()
        self.prompt_len = prompt_len
        self.tokens_seen = 0
        self.buffer = []
        self.is_finished = False

    def put(self, token_ids: Optional[torch.Tensor]):
        """
        Called by the generate() method whenever new tokens become available.

        Args:
            token_ids (Optional[torch.Tensor]): A tensor containing newly generated token IDs.
                If None, it signals that generation has ended.
        """
        if token_ids is None:
            # End of generation
            self.is_finished = True
        else:
            # If token_ids has shape (1, N), flatten it to shape (N,)
            if token_ids.dim() == 2 and token_ids.shape[0] == 1:
                token_ids = token_ids.squeeze(0)
            for tid in token_ids:
                # Skip the first `prompt_len` tokens
                if self.tokens_seen < self.prompt_len:
                    self.tokens_seen += 1
                else:
                    self.buffer.append(tid)
                    self.tokens_seen += 1

    def end(self):
        """Signals that generation is complete."""
        self.is_finished = True

    def __iter__(self):
        """
        Yields token IDs as they become available.
        """
        while not self.is_finished or self.buffer:
            if self.buffer:
                yield self.buffer.pop(0)
            else:
                # Avoid busy waiting
                time.sleep(0.01)


# ------------------------------------------------------------------------------
# Model Handling
# ------------------------------------------------------------------------------


@dataclass
class Result:
    """
    A class to store the results of a generation experiment.
    """

    tok_ids_prompt: List[int]
    tok_ids_new: List[int]
    prompt_text: str
    new_text: str
    total_gen_time_s: float
    ttft_s: float
    tpot_s: float


class HFModel:
    """
    Lightweight class to wrap a Hugging Face model and tokenizer for convenience.
    """

    def __init__(self, model_name: str, device_map: str = "auto", torch_dtype=torch.float16):
        """
        Load a model and tokenizer from the Hugging Face Hub.
        """
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, torch_dtype=torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_text(self, prompt: str, do_sample: bool, max_new_tokens: int = 512, **kwargs):
        """
        Generate text from the underlying model, measuring detailed latency metrics.

        Parameters:
            prompt (str): The input text to generate from.
            do_sample (bool): Whether to sample or use greedy decoding.
            max_new_tokens (int): Maximum number of tokens to generate.
            **kwargs: Additional arguments for the generation method.
        """
        # Clear any cached memory before starting
        clear_memory()

        # Tokenize the input prompt and move it to the model's device
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Create a streamer for raw token IDs (instead of TextIteratorStreamer)
        streamer = IdsIteratorStreamer(prompt_len=prompt_len)

        # Handle the attention mask to ensure valid memory alignment
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs["input_ids"], dtype=self.model.dtype, device=self.model.device)

        generation_kwargs = dict(
            inputs=inputs["input_ids"],
            attention_mask=attention_mask,
            cache_implementation="offloaded",
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            streamer=streamer,
            return_dict_in_generate=False,  # Return only the generated sequences
            output_scores=False,
            output_hidden_states=False,
            output_attentions=False,
            **kwargs,
        )

        # Warmup
        for _ in range(2):
            self.model.generate(**generation_kwargs)

        # Reset the streamer
        streamer = IdsIteratorStreamer(prompt_len=prompt_len)
        generation_kwargs["streamer"] = streamer

        # Create thread with daemon=True to ensure it's cleaned up
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)
        start_time = time.time()
        thread.start()

        new_token_ids_tensors = []
        time_to_first_token = None

        for chunk_of_ids in streamer:
            # Record TTFT if it's the very first token(s)
            if time_to_first_token is None:
                time_to_first_token = time.time() - start_time

            # chunk_of_ids might be shape=() (a single scalar) or shape=(n,) (n tokens)
            # -> force it to be at least shape=(1,):
            if chunk_of_ids.dim() == 0:
                chunk_of_ids = chunk_of_ids.unsqueeze(0)

            new_token_ids_tensors.append(chunk_of_ids)

        # Stop the timer here
        total_gen_time = time.time() - start_time

        # Now flatten all the chunks into a single 1-D tensor:
        if new_token_ids_tensors:
            # E.g. [tensor([101, 102]), tensor([103]), tensor([104, 105])]
            new_token_ids = torch.cat(new_token_ids_tensors, dim=0)  # shape=(N,)
        else:
            new_token_ids = torch.empty(0, dtype=torch.long, device=self.model.device)

        # Move to model device if necessary
        new_token_ids = new_token_ids.to(self.model.device)
        generated_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        print("=" * 50)
        print("Generated text:\n", generated_text)
        print("=" * 50)

        tpot: float = float("inf")
        if len(new_token_ids) > 1:
            tpot = (total_gen_time - time_to_first_token) / (len(new_token_ids) - 1)

        # Make sure to set a timeout for join to prevent hanging
        thread.join(timeout=300)  # 5 minute timeout
        if thread.is_alive():
            print("Warning: Generation thread did not complete within timeout")

        return Result(
            tok_ids_prompt=inputs["input_ids"][0].tolist(),
            tok_ids_new=new_token_ids,
            prompt_text=prompt,
            new_text=generated_text,
            total_gen_time_s=total_gen_time,
            ttft_s=time_to_first_token,
            tpot_s=tpot,
        )


def tokenizers_are_identical(t1, t2) -> bool:
    """
    Return True if t1 and t2 are effectively the same tokenizer, i.e.,
    produce identical results for any input text.
    """
    # 1. Same Python object?
    if t1 is t2:
        print("✓ Tokenizers are the same Python object")
        return True

    # 2. Same class?
    if type(t1) != type(t2):
        print(f"✗ Different tokenizer classes: {type(t1)} vs {type(t2)}")
        return False

    # 3. Compare vocabulary
    vocab1 = t1.get_vocab()
    vocab2 = t2.get_vocab()
    if len(vocab1) != len(vocab2):
        print(f"✗ Different vocabulary sizes: {len(vocab1)} vs {len(vocab2)}")
        return False

    # Check each token's ID
    for token, idx in vocab1.items():
        if token not in vocab2 or vocab2[token] != idx:
            print(f"✗ Token mismatch: '{token}' has different IDs ({idx} vs {vocab2.get(token, 'missing')})")
            return False

    # Check for extra tokens in t2
    for token, idx in vocab2.items():
        if token not in vocab1 or vocab1[token] != idx:
            print(f"✗ Extra token in t2: '{token}' with ID {idx}")
            return False

    # 4. Compare merges
    merges_t1 = getattr(t1, "merges", None)
    merges_t2 = getattr(t2, "merges", None)
    if merges_t1 != merges_t2:
        print("✗ Different merges rules")
        return False

    # 5. Compare special tokens
    if t1.special_tokens_map != t2.special_tokens_map:
        print("✗ Different special tokens maps:")
        print(f"  T1: {t1.special_tokens_map}")
        print(f"  T2: {t2.special_tokens_map}")
        return False

    print("✓ Tokenizers are identical")
    return True


# ------------------------------------------------------------------------------
# Generation Logic
# ------------------------------------------------------------------------------


def generate_assisted(
    prompt: str, target_model_obj: HFModel, do_sample: bool, assistant_model_obj: Optional[HFModel] = None
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
        are_tokenizers_identical: bool = tokenizers_are_identical(
            target_model_obj.tokenizer, assistant_model_obj.tokenizer
        )
        print("Tokenizers are identical:", are_tokenizers_identical)
        if not are_tokenizers_identical:
            generate_kwargs["assistant_tokenizer"] = assistant_model_obj.tokenizer
            generate_kwargs["tokenizer"] = target_model_obj.tokenizer
    return target_model_obj.generate_text(prompt=prompt, do_sample=do_sample, **generate_kwargs)


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
    # target_checkpoint = "meta-llama/Llama-3.3-70B-Instruct"
    target_checkpoint = "meta-llama/Llama-3.1-8B-Instruct"
    qwen_checkpoint = "Qwen/Qwen2.5-0.5B-Instruct"
    llama_assistant_checkpoint = "meta-llama/Llama-3.2-1B-Instruct"
    llama_3b_assistant_checkpoint = "meta-llama/Llama-3.2-3B-Instruct"

    target_model_obj = HFModel(target_checkpoint)
    qwen_model_obj = HFModel(qwen_checkpoint)
    llama_assistant_model_obj = HFModel(llama_assistant_checkpoint)
    llama_3b_assistant_model_obj = HFModel(llama_3b_assistant_checkpoint)

    # 5. Load dataset
    dataset_path = "tau/scrolls"
    dataset_name = "qasper"
    dataset_split = "test"
    print("Loading dataset:")
    print(f"Dataset path: {dataset_path}")
    print(f"Dataset name: {dataset_name}")
    print(f"Dataset split: {dataset_split}")
    dataset = load_dataset(path=dataset_path, name=dataset_name, split=dataset_split, trust_remote_code=True)
    dataset_sample = dataset.select(range(args.num_of_examples))

    # 6. Generation loop
    results: List[Dict[str, float]] = []
    for i, example in enumerate(dataset_sample):
        prompt = example["input"]  # Adjust if the actual prompt field is different

        print("=" * 100)
        print(f"Running input prompt {i}...")
        print("Prompt:\n", prompt)
        print("=" * 100)

        print("Running Baseline with `do_sample=False`...")
        baseline_do_sample_false_result = generate_assisted(
            prompt=prompt, do_sample=False, target_model_obj=target_model_obj
        )

        print("Running Baseline with `do_sample=True`...")
        baseline_do_sample_true_result = generate_assisted(
            prompt=prompt, do_sample=True, target_model_obj=target_model_obj
        )

        print("Running Qwen assisted with `do_sample=False`...")
        qwen_uag_result = generate_assisted(
            prompt=prompt,
            target_model_obj=target_model_obj,
            do_sample=False,
            assistant_model_obj=qwen_model_obj,
        )

        print("Running Qwen assisted with `do_sample=True`...")
        qwen_result = generate_assisted(
            prompt=prompt,
            target_model_obj=target_model_obj,
            do_sample=True,
            assistant_model_obj=qwen_model_obj,
        )

        print("Running Llama 1B assisted...")
        llama_assisted_result = generate_assisted(
            prompt=prompt,
            target_model_obj=target_model_obj,
            do_sample=True,
            assistant_model_obj=llama_assistant_model_obj,
        )

        print("Running Llama 3B assisted...")
        llama_3b_assisted_result = generate_assisted(
            prompt=prompt,
            target_model_obj=target_model_obj,
            do_sample=True,
            assistant_model_obj=llama_3b_assistant_model_obj,
        )

        # Collect results
        results.append(
            {
                "Baseline `do_sample=False` TPOT": baseline_do_sample_false_result.tpot_s,
                "Baseline `do_sample=False` TTFT": baseline_do_sample_false_result.ttft_s,
                "Baseline `do_sample=False` Len Inp": len(baseline_do_sample_false_result.tok_ids_prompt),
                "Baseline `do_sample=False` New Toks": len(baseline_do_sample_false_result.tok_ids_new),
                "Baseline `do_sample=True` TPOT": baseline_do_sample_true_result.tpot_s,
                "Baseline `do_sample=True` TTFT": baseline_do_sample_true_result.ttft_s,
                "Baseline `do_sample=True` Len Inp": len(baseline_do_sample_true_result.tok_ids_prompt),
                "Baseline `do_sample=True` New Toks": len(baseline_do_sample_true_result.tok_ids_new),
                "Qwen USD TPOT": qwen_result.tpot_s,
                "Qwen USD TTFT": qwen_result.ttft_s,
                "Qwen USD Len Inp": len(qwen_result.tok_ids_prompt),
                "Qwen USD New Toks": len(qwen_result.tok_ids_new),
                "Qwen UAG TPOT": qwen_uag_result.tpot_s,
                "Qwen UAG TTFT": qwen_uag_result.ttft_s,
                "Qwen UAG Len Inp": len(qwen_uag_result.tok_ids_prompt),
                "Qwen UAG New Toks": len(qwen_uag_result.tok_ids_new),
                "Llama 1B TPOT": llama_assisted_result.tpot_s,
                "Llama 1B TTFT": llama_assisted_result.ttft_s,
                "Llama 1B Len Inp": len(llama_assisted_result.tok_ids_prompt),
                "Llama 1B New Toks": len(llama_assisted_result.tok_ids_new),
                "Llama 3B TPOT": llama_3b_assisted_result.tpot_s,
                "Llama 3B TTFT": llama_3b_assisted_result.ttft_s,
                "Llama 3B Len Inp": len(llama_3b_assisted_result.tok_ids_prompt),
                "Llama 3B New Toks": len(llama_3b_assisted_result.tok_ids_new),
            }
        )

        print(f"Results for prompt {i}:")
        pprint(results[-1])

        print("TTFT stats:")
        ttft_values = [v for k, v in results[-1].items() if "TTFT" in k]
        ttft_mean = np.mean(ttft_values)
        ttft_std = np.std(ttft_values)
        print("Mean: ", ttft_mean)
        print("Std: ", ttft_std)
        print("Their ratio (std/mean): ", ttft_std / ttft_mean)
        print("Min: ", np.min(ttft_values))
        print("Max: ", np.max(ttft_values))
        print("Median: ", np.median(ttft_values))

        print("TPOTs small to large:")
        tpots = {k: v for k, v in results[-1].items() if "TPOT" in k}
        sorted_tpots = sorted(tpots.items(), key=lambda x: x[1])
        pprint(sorted_tpots)

    # 7. Convert to DataFrame & save
    df_results = pd.DataFrame(results)
    
    # Create output directory if it doesn't exist
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
    dirpath = f"benchmark_results/{timestamp}_{commit_hash}"
    os.makedirs(dirpath, exist_ok=True)

    # Save to the benchmark_results directory
    dataset_info = f"{dataset_path}_{dataset_name}_{dataset_split}".replace("/", "-")
    filename_results = os.path.join(dirpath, f"latency_benchmark_on_{dataset_info}_{args.num_of_examples}_examples.csv")
    df_results.to_csv(filename_results, index=False)
    print(f"Results saved to {filename_results}")


if __name__ == "__main__":
    main()
