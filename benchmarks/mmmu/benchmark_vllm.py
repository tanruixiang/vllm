# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import json
import os
import random
import time
from typing import List, Dict, Optional

import numpy as np
from datasets import load_dataset

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser
from data_utils import construct_prompt, load_yaml, process_single_sample, load_mmmu_dataset
from eval_utils import (parse_multi_choice_response, parse_open_response, evaluate, 
                       run_benchmark, load_benchmark_dataset, load_benchmark_config)

def vllm_generate_func(llm: LLM):
    """Create a generation function for vLLM that matches the common interface"""
    def generate(prompts: List[str], generation_params) -> List[str]:
        """Generate responses using vLLM"""
        # Use the generation_params which is the sampling_params object
        outputs = llm.generate(prompts, generation_params)
        responses = []
        for output in outputs:
            response = output.outputs[0].text.strip()
            responses.append(response)
        return responses
    return generate


def setup_vllm_generation_params(args):
    """Setup vLLM sampling parameters from args"""
    # This will be called with the LLM instance to get default sampling params
    # We need to modify this to work with the actual LLM instance
    return args  # For now, return args and handle in the main function


def main(args: dict):
    # Pop sampling arguments
    max_tokens = args.pop("max_tokens", None)
    temperature = args.pop("temperature", None)
    top_p = args.pop("top_p", None)
    top_k = args.pop("top_k", None)
    
    # Pop benchmark specific arguments
    split = args.pop("split", "validation")
    subject = args.pop("subject", None)
    max_samples = args.pop("max_samples", -1)
    output_path = args.pop("output_path", "benchmark_results_vllm.json")
    config_path = args.pop("config_path", "eval_config.yaml")
    seed = args.pop("seed", 42)
    batch_size = args.pop("batch_size", 1)
    
    # Create an LLM with remaining args
    print(f"Loading vLLM model...")
    llm = LLM(**args)
    
    # Create sampling params using the LLM instance
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k
    if seed is not None:
        sampling_params.seed = seed
    
    # Store args for common benchmark function
    class Args:
        def __init__(self):
            self.seed = seed
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.top_p = top_p
            self.top_k = top_k
    
    benchmark_args = Args()
    
    # Load evaluation config
    config = load_benchmark_config(config_path)
    
    # Load dataset
    samples = load_benchmark_dataset(split=split, subject=subject, max_samples=max_samples)
    
    # Create generation function
    generate_func = vllm_generate_func(llm)
    
    # Model info for saving
    model_info = {
        'model': args.get('model', 'unknown'),
        'split': split,
        'subject': subject,
        'max_samples': max_samples,
        'batch_size': batch_size
    }
    
    # Use the common benchmark function, but pass sampling_params directly as generation_params
    def generate_with_params(prompts: List[str], generation_params) -> List[str]:
        # Ignore generation_params and use our pre-configured sampling_params
        outputs = llm.generate(prompts, sampling_params)
        responses = []
        for output in outputs:
            response = output.outputs[0].text.strip()
            responses.append(response)
        return responses
    
    # Run benchmark
    results = run_benchmark(
        samples=samples,
        config=config,
        args=benchmark_args,
        generate_func=generate_with_params,
        setup_generation_params_func=None,  # We handle params ourselves
        batch_size=batch_size,
        subject=subject,
        output_path=output_path,
        model_info=model_info
    )
    
    return results


def create_parser():
    parser = FlexibleArgumentParser(
        description="Benchmark vLLM models on MMMU dataset using offline inference"
    )
    
    # Add engine args (this includes model, tensor_parallel_size, etc.)
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")
    
    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int, default=512,
                               help="Maximum number of tokens to generate")
    sampling_group.add_argument("--temperature", type=float, default=0.0,
                               help="Temperature for sampling (0.0 = deterministic)")
    sampling_group.add_argument("--top-p", type=float, default=1.0,
                               help="Top-p (nucleus) sampling parameter")
    sampling_group.add_argument("--top-k", type=int, default=None,
                               help="Top-k sampling parameter")
    
    # Add benchmark specific params
    benchmark_group = parser.add_argument_group("Benchmark parameters")
    benchmark_group.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test", "dev"],
        help="Dataset split to use"
    )
    benchmark_group.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Specific subject to evaluate (e.g., 'Art', 'Biology'). If None, evaluates all subjects"
    )
    benchmark_group.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process (-1 for all)"
    )
    benchmark_group.add_argument(
        "--output-path",
        type=str,
        default="benchmark_results_vllm.json",
        help="Path to save the results"
    )
    benchmark_group.add_argument(
        "--config-path",
        type=str,
        default="eval_config.yaml",
        help="Path to evaluation config file"
    )
    benchmark_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    benchmark_group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference"
    )
    
    return parser


def invoke_main() -> None:
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
