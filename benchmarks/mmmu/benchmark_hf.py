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
from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.utils import FlexibleArgumentParser
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from data_utils import construct_prompt, load_yaml, process_single_sample, CAT_SHORT2LONG, load_mmmu_dataset
from eval_utils import parse_multi_choice_response, parse_open_response, evaluate


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

def process_samples(model, tokenizer, samples: List[Dict], config: Dict, args) -> List[Dict]:
    """Process samples and generate predictions"""
    results = []
    
    # Set fixed seed for reproducibility
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    for i, sample in enumerate(samples):
        print(f"Processing sample {i+1}/{len(samples)}: {sample['id']}")
        
        # Construct prompt
        prompt_data = construct_prompt(sample, config)
        prompt = prompt_data['final_input_prompt']
        
        # Generate response
        response = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            seed=args.seed
        )
        
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
            'prompt': prompt,
            'subject': sample.get('subject', 'unknown')
        }
        results.append(result)

    return results

def main(args):
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Load evaluation config
    config_path = args.config_path if hasattr(args, 'config_path') else 'eval_config.yaml'
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
    print(f"Split: {args.split}, Subject: {args.subject}")
    
    dataset = load_mmmu_dataset(subset=args.split, subject=args.subject)
    
    # Convert dataset samples to our format
    samples = []
    for sample in dataset:
        samples.append(process_single_sample(sample))
    
    # Limit number of samples if specified
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    
    print(f"Processing {len(samples)} samples...")
    
    # Process samples
    results = process_samples(model, tokenizer, samples, config, args)
    
    # Evaluate results
    judge_dict, metrics = evaluate(results)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['acc']:.4f}")
    
    # Group results by subject if multiple subjects
    if args.subject is None:
        subject_results: Dict[str, List[Dict]] = {}
        for result in results:
            subject = result.get('subject', 'unknown')
            if subject not in subject_results:
                subject_results[subject] = []
            subject_results[subject].append(result)
        
        print("\nResults by Subject:")
        for subject, subject_samples in subject_results.items():
            subject_judge_dict, subject_metrics = evaluate(subject_samples)
            print(f"{subject}: {subject_metrics['acc']:.4f} ({len(subject_samples)} samples)")
    
    # Save results
    output_path = args.output_path if hasattr(args, 'output_path') else 'benchmark_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'results': results,
            'metrics': metrics,
            'judge_dict': judge_dict,
            'args': {
                'model_path': args.model_path,
                'split': args.split,
                'subject': args.subject,
                'max_samples': args.max_samples
            }
        }, f, indent=2)
    
    print(f"Results saved to {output_path}")

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
