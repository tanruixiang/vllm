# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch
from eval_utils import (
    add_common_benchmark_args,
    load_benchmark_config,
    load_benchmark_dataset,
    run_benchmark,
)
from transformers import AutoModel, AutoTokenizer, set_seed

from vllm.utils import FlexibleArgumentParser


def load_model_and_tokenizer(model: str):
    """Load HuggingFace model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(
        model, torch_dtype="auto", trust_remote_code=True
    )
    model = model.eval().cuda()

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    do_sample: bool,
    seed: int,
) -> str:
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
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
            seed=seed if hasattr(model.generation_config, "seed") else None,
        )

    # Decode only the new tokens
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()


def hf_generate_func(model, tokenizer, generation_params):
    """Create a generation function for HuggingFace models
        that matches the common interface"""

    def generate(prompts: list[str]) -> list[str]:
        """Generate responses using HuggingFace model"""
        responses = []
        for prompt in prompts:
            response = generate_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=generation_params.max_new_tokens,
                temperature=generation_params.temperature,
                top_p=generation_params.top_p,
                top_k=generation_params.top_k,
                do_sample=generation_params.do_sample,
                seed=generation_params.seed,
            )
            responses.append(response)
        return responses

    return generate


def main(args):
    # Load model and tokenizer
    print(f"Loading model from {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Load evaluation config
    config = load_benchmark_config(
        args.config_path if hasattr(args, "config_path") else "eval_config.yaml"
    )

    # Load dataset
    samples = load_benchmark_dataset(
        split=args.split, subject=args.subject, max_samples=args.max_samples
    )

    # Create generation function
    generate_func = hf_generate_func(model, tokenizer, args)

    # Model info for saving
    model_info = {
        "model": args.model,
        "split": args.split,
        "subject": args.subject,
        "max_samples": args.max_samples,
    }

    # Run benchmark using common logic
    results = run_benchmark(
        samples=samples,
        config=config,
        args=args,
        generate_func=generate_func,
        batch_size=1,  # HF processes one at a time
        subject=args.subject,
        output_path=args.output_path,
        model_info=model_info,
    )

    return results


def invoke_main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark HuggingFace models on MMMU dataset from HuggingFace Hub"
    )

    # Add common benchmark arguments
    parser = add_common_benchmark_args(parser, framework="hf")

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
