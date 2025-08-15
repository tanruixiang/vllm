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
from eval_utils import parse_multi_choice_response, parse_open_response, evaluate





def process_samples(llm: LLM, samples: List[Dict], config: Dict, args) -> List[Dict]:
    """Process samples and generate predictions using vLLM"""
    results = []
    
    # Set fixed seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create sampling params
    sampling_params = llm.get_default_sampling_params()
    if args.max_tokens is not None:
        sampling_params.max_tokens = args.max_tokens
    if args.temperature is not None:
        sampling_params.temperature = args.temperature
    if args.top_p is not None:
        sampling_params.top_p = args.top_p
    if args.top_k is not None:
        sampling_params.top_k = args.top_k
    if hasattr(args, 'seed') and args.seed is not None:
        sampling_params.seed = args.seed
    
    # Batch process samples for efficiency
    batch_size = getattr(args, 'batch_size', 1)
    
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i+batch_size]
        batch_prompts = []
        
        # Prepare batch prompts
        for sample in batch_samples:
            prompt_data = construct_prompt(sample, config)
            prompt = prompt_data['final_input_prompt']
            batch_prompts.append(prompt)
            
            # Store prompt data for later use
            sample['_prompt_data'] = prompt_data
            sample['_prompt'] = prompt
        
        print(f"Processing batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size} "
              f"(samples {i+1}-{min(i+batch_size, len(samples))}/{len(samples)})")
        
        # Generate responses
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Process outputs
        for j, output in enumerate(outputs):
            sample = batch_samples[j]
            response = output.outputs[0].text.strip()
            prompt_data = sample['_prompt_data']
            
            # Parse response based on question type
            if sample['question_type'] == 'multiple-choice':
                parsed_pred = parse_multi_choice_response(
                    response, 
                    prompt_data['all_choices'], 
                    prompt_data['index2ans']
                )
            else:
                parsed_pred = parse_open_response(response)
            
            # Store results
            result = {
                'id': sample['id'],
                'question': sample['question'],
                'answer': sample['answer'],
                'question_type': sample['question_type'],
                'response': response,
                'parsed_pred': parsed_pred,
                'prompt': sample['_prompt'],
                'subject': sample.get('subject', 'unknown')
            }
            results.append(result)
        
        # Clean up memory periodically
        if i % (batch_size * 10) == 0:
            gc.collect()
    
    print(f"Average generation time: {time_collector.get_average_time():.3f}s")
    return results


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
    
    # Store sampling parameters for later use
    class Args:
        def __init__(self):
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.top_p = top_p
            self.top_k = top_k
            self.seed = seed
            self.batch_size = batch_size
    
    inference_args = Args()
    
    # Load evaluation config
    if os.path.exists(config_path):
        config = load_yaml(config_path)
    else:
        # Default config
        config = {
            'multi_choice_example_format': 'Question: {}\nOptions:\n{}\nAnswer:',
            'short_ans_example_format': 'Question: {}\nAnswer:',
            'task_instructions': 'Please answer the following question based on the given information.'
        }
    
    # Load MMMU dataset from HuggingFace
    print(f"Loading MMMU dataset from HuggingFace Hub...")
    print(f"Split: {split}, Subject: {subject}")
    
    dataset = load_mmmu_dataset(subset=split, subject=subject)
    
    # Convert dataset samples to our format
    samples = []
    for sample in dataset:
        samples.append(process_single_sample(sample))
    
    # Limit number of samples if specified
    if max_samples > 0:
        samples = samples[:max_samples]
    
    print(f"Processing {len(samples)} samples...")
    
    # Process samples
    results = process_samples(llm, samples, config, inference_args)
    
    # Evaluate results
    judge_dict, metrics = evaluate(results)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['acc']:.4f}")
    
    # Group results by subject if multiple subjects
    if subject is None:
        subject_results: Dict[str, List[Dict]] = {}
        for result in results:
            subj = result.get('subject', 'unknown')
            if subj not in subject_results:
                subject_results[subj] = []
            subject_results[subj].append(result)
        
        print("\nResults by Subject:")
        for subj, subject_samples in subject_results.items():
            subject_judge_dict, subject_metrics = evaluate(subject_samples)
            print(f"{subj}: {subject_metrics['acc']:.4f} ({len(subject_samples)} samples)")
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump({
            'results': results,
            'metrics': metrics,
            'judge_dict': judge_dict,
            'args': {
                'model': args.get('model', 'unknown'),
                'split': split,
                'subject': subject,
                'max_samples': max_samples,
                'seed': seed,
                'batch_size': batch_size
            }
        }, f, indent=2)
    
    print(f"Results saved to {output_path}")


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
