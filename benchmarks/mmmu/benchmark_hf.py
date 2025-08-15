# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import json
import os
import random
from typing import List, Dict

import numpy as np
from transformers import AutoTokenizer, AutoModel, set_seed
import torch
from datasets import load_dataset
from typing import Optional
from data_utils import construct_prompt, load_yaml, process_single_sample, CAT_SHORT2LONG, load_mmmu_dataset
from eval_utils import (parse_multi_choice_response, parse_open_response, evaluate,
                       run_benchmark, load_benchmark_dataset, load_benchmark_config)
from vllm.utils import FlexibleArgumentParser

def load_model_and_tokenizer(model_path: str):
    """Load HuggingFace model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True
    )
    model = model.eval().cuda()
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512, 
                     temperature: float = 0.7, top_p: float = 0.9, 
                     do_sample: bool = True, seed: int = 42) -> str:
    """Generate response using HuggingFace model"""
    # Set seed for reproducibility
    set_seed(seed)
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            seed=seed if hasattr(model.generation_config, 'seed') else None
        )
    
    # Decode only the new tokens
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()

def hf_generate_func(model, tokenizer):
    """Create a generation function for HuggingFace models that matches the common interface"""
    def generate(prompts: List[str], generation_params) -> List[str]:
        """Generate responses using HuggingFace model"""
        responses = []
        for prompt in prompts:
            response = generate_response(
                model, tokenizer, prompt,
                max_new_tokens=generation_params.max_new_tokens,
                temperature=generation_params.temperature,
                top_p=generation_params.top_p,
                do_sample=generation_params.do_sample,
                seed=generation_params.seed
            )
            responses.append(response)
        return responses
    return generate


def main(args):
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Load evaluation config
    config = load_benchmark_config(args.config_path if hasattr(args, 'config_path') else 'eval_config.yaml')
    
    # Load dataset
    samples = load_benchmark_dataset(split=args.split, subject=args.subject, max_samples=args.max_samples)
    
    # Create generation function
    generate_func = hf_generate_func(model, tokenizer)
    
    # Model info for saving
    model_info = {
        'model_path': args.model_path,
        'split': args.split,
        'subject': args.subject,
        'max_samples': args.max_samples
    }
    
    # Run benchmark using common logic
    results = run_benchmark(
        samples=samples,
        config=config,
        args=args,
        generate_func=generate_func,
        setup_generation_params_func=None,  # We pass args directly
        batch_size=1,  # HF processes one at a time
        subject=args.subject,
        output_path=args.output_path,
        model_info=model_info
    )
    
    return results

def invoke_main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark HuggingFace models on MMMU dataset from HuggingFace Hub"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,  
        help="Path to the HuggingFace model",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test", "dev"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Specific subject to evaluate (e.g., 'Art', 'Biology'). If None, evaluates all subjects"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process (-1 for all)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="benchmark_results.json",
        help="Path to save the results"
    )
    # 新增参数：固化随机种子和采样参数
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (0.0 = deterministic)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        default=True,
        help="Whether to use sampling (vs greedy decoding)"
    )
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    invoke_main()  # pragma: no cover
