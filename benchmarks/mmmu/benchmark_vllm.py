# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from eval_utils import (
    add_common_benchmark_args,
    load_benchmark_config,
    load_benchmark_dataset,
    run_benchmark,
)

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


def main(args: dict):
    # Pop sampling arguments
    max_tokens = args.pop("max_tokens", None)
    temperature = args.pop("temperature", None)
    top_p = args.pop("top_p", None)
    top_k = args.pop("top_k", None)

    # Pop benchmark specific arguments
    split = args.pop("split")
    subject = args.pop("subject")
    max_samples = args.pop("max_samples")
    output_path = args.pop("output_path")
    config_path = args.pop("config_path")
    seed = args.pop("seed")
    batch_size = args.pop("batch_size")

    # Create an LLM with remaining args
    print("Loading vLLM model...")
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
    samples = load_benchmark_dataset(
        split=split, subject=subject, max_samples=max_samples
    )

    # Model info for saving
    model_info = {
        "model": args.get("model", "unknown"),
        "split": split,
        "subject": subject,
        "max_samples": max_samples,
        "batch_size": batch_size,
    }

    # Use the common benchmark function, but pass sampling_params
    # directly as generation_params
    def generate_with_params(prompts: list[str]) -> list[str]:
        # Use our pre-configured sampling_params
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
        batch_size=batch_size,
        subject=subject,
        output_path=output_path,
        model_info=model_info,
    )

    return results


def create_parser():
    parser = FlexibleArgumentParser(
        description="Benchmark vLLM models on MMMU dataset using offline inference"
    )

    # Add engine args (this includes model, tensor_parallel_size, etc.)
    EngineArgs.add_cli_args(parser)

    # Add common benchmark arguments
    parser = add_common_benchmark_args(parser, framework="vllm")

    return parser


def invoke_main() -> None:
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
