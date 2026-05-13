#!/usr/bin/env python3
"""
Utility script to run SkillNav expert routing with multiple vision-language models.

Example:
    python maps_nav_src/scripts/test_vlm_models.py \
        --input_json /path/to/router_inputs.json \
        --gpu 0
"""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from moe.vLLM_API import (  # noqa: E402
    DEFAULT_VLM_DEPLOYMENT_MODELS,
    deploy_vlm_models_for_expert_indices,
)


def _load_batch_inputs(input_json):
    if input_json is None:
        return [
            {
                "instr_id": "demo_0",
                "scan": "DEMO_SCAN_A",
                "full_instruction": "Exit the current room, walk to the hallway, and stop near the stairs.",
                "previous_viewpoint_list": [],
                "previous_sub_instruction_list": [],
            },
            {
                "instr_id": "demo_1",
                "scan": "DEMO_SCAN_B",
                "full_instruction": "Turn left, follow the corridor past the couch, and wait by the plant.",
                "previous_viewpoint_list": [],
                "previous_sub_instruction_list": [
                    "Walk forward to the end of the corridor."
                ],
            },
        ]

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of router inputs.")
    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test multiple VLM checkpoints for expert routing."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default=None,
        help="Optional JSON file providing batch_inputs_instruction_localization.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Override the default checkpoints. Provide HuggingFace repo ids or paths.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device id that each model should use sequentially.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override max_num_seqs passed to vLLM (defaults to len(batch_inputs)).",
    )
    parser.add_argument(
        "--limit_mm_per_prompt",
        type=int,
        default=20,
        help="Image cap forwarded to load_vLLM_model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed forwarded to load_vLLM_model.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallelism per deployment.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization hint.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save deployment results as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    batch_inputs = _load_batch_inputs(args.input_json)

    deployment_results = deploy_vlm_models_for_expert_indices(
        batch_inputs_instruction_localization=batch_inputs,
        model_ckpts=args.models or DEFAULT_VLM_DEPLOYMENT_MODELS,
        logger=None,
        gpu_id=args.gpu,
        batch_size=args.batch_size,
        limit_mm_per_prompt=args.limit_mm_per_prompt,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    for model_name, output in deployment_results.items():
        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"Expert indices: {output['expert_indices']}")
        print(f"Resolved sub-instructions: {output['sub_instructions']}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(deployment_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved detailed results to {args.output_json}")


if __name__ == "__main__":
    main()
