#!/usr/bin/env python3
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
import re

# Paths to the router output logs we want to merge and analyze.
LOG_FILES = [
    # Non-residential basic
    Path(
        "/localscratch2/tianyi/ScaleVLN/VLN-DUET/datasets/GSA-R2R/exprs_map/"
        "multi-models/vlms/GPT4odagger-clip.b16-seed.0-aug.mp3d.prevalent.moe-"
        "top1-routing-add_prev_sub_instructions/logs/"
        "test_non_residential_basic_mp3d_filtered_part_1_router_outputs.log"
    ),
    Path(
        "/localscratch2/tianyi/ScaleVLN/VLN-DUET/datasets/GSA-R2R/exprs_map/"
        "multi-models/vlms/GPT4odagger-clip.b16-seed.0-aug.mp3d.prevalent.moe-"
        "top1-routing-add_prev_sub_instructions/logs/"
        "test_non_residential_basic_mp3d_filtered_part_2_router_outputs.log"
    ),
    Path(
        "/localscratch2/tianyi/ScaleVLN/VLN-DUET/datasets/GSA-R2R/exprs_map/"
        "multi-models/vlms/GPT4odagger-clip.b16-seed.0-aug.mp3d.prevalent.moe-"
        "top1-routing-add_prev_sub_instructions/logs/"
        "test_non_residential_basic_mp3d_filtered_part_3_router_outputs.log"
    ),
    # Non-residential scene
    Path(
        "/localscratch2/tianyi/ScaleVLN/VLN-DUET/datasets/GSA-R2R/exprs_map/"
        "multi-models/vlms/GPT4odagger-clip.b16-seed.0-aug.mp3d.prevalent.moe-"
        "top1-routing-add_prev_sub_instructions/logs/"
        "test_non_residential_scene_mp3d_filtered_part_1_router_outputs.log"
    ),
    Path(
        "/localscratch2/tianyi/ScaleVLN/VLN-DUET/datasets/GSA-R2R/exprs_map/"
        "multi-models/vlms/GPT4odagger-clip.b16-seed.0-aug.mp3d.prevalent.moe-"
        "top1-routing-add_prev_sub_instructions/logs/"
        "test_non_residential_scene_mp3d_filtered_part_2_router_outputs.log"
    ),
    Path(
        "/localscratch2/tianyi/ScaleVLN/VLN-DUET/datasets/GSA-R2R/exprs_map/"
        "multi-models/vlms/GPT4odagger-clip.b16-seed.0-aug.mp3d.prevalent.moe-"
        "top1-routing-add_prev_sub_instructions/logs/"
        "test_non_residential_scene_mp3d_filtered_part_3_router_outputs.log"
    ),
    # Residential basic
    Path(
        "/localscratch2/tianyi/ScaleVLN/VLN-DUET/datasets/GSA-R2R/exprs_map/"
        "multi-models/vlms/GPT4odagger-clip.b16-seed.0-aug.mp3d.prevalent.moe-"
        "top1-routing-add_prev_sub_instructions/logs/"
        "test_residential_basic_mp3d_filtered_part_1_router_outputs.log"
    ),
    Path(
        "/localscratch2/tianyi/ScaleVLN/VLN-DUET/datasets/GSA-R2R/exprs_map/"
        "multi-models/vlms/GPT4odagger-clip.b16-seed.0-aug.mp3d.prevalent.moe-"
        "top1-routing-add_prev_sub_instructions/logs/"
        "test_residential_basic_mp3d_filtered_part_2_router_outputs.log"
    ),
    Path(
        "/localscratch2/tianyi/ScaleVLN/VLN-DUET/datasets/GSA-R2R/exprs_map/"
        "multi-models/vlms/GPT4odagger-clip.b16-seed.0-aug.mp3d.prevalent.moe-"
        "top1-routing-add_prev_sub_instructions/logs/"
        "test_residential_basic_mp3d_filtered_part_3_router_outputs.log"
    ),
    Path(
        "/localscratch2/tianyi/ScaleVLN/VLN-DUET/datasets/GSA-R2R/exprs_map/"
        "multi-models/vlms/GPT4odagger-clip.b16-seed.0-aug.mp3d.prevalent.moe-"
        "top1-routing-add_prev_sub_instructions/logs/"
        "test_residential_basic_mp3d_filtered_part_4_router_outputs.log"
    ),
]

# Canonical skills we care about; use fuzzy matching so slight name variants still count.
CANONICAL_SKILLS = {
    "directional adjustment": "Directional Adjustment",
    "vertical movement": "Vertical Movement",
    "stop and pause": "Stop and Pause",
    "landmark detection": "Landmark Detection",
    "area and region identification": "Area and Region Identification",
}

PRED_RE = re.compile(r'"predicted_skill":\s*"([^"]+)"')


def best_skill_match(name: str, threshold: float = 0.55):
    """Return the best-matching canonical skill (or None if too far)."""
    name_lower = name.lower()
    best_name = None
    best_score = 0.0
    for canonical_lower, canonical_title in CANONICAL_SKILLS.items():
        score = SequenceMatcher(None, name_lower, canonical_lower).ratio()
        if score > best_score:
            best_score = score
            best_name = canonical_title
    return best_name if best_score >= threshold else None


def iter_predicted_skills(paths):
    """Yield predicted_skill strings in order across all logs."""
    for path in paths:
        with path.open() as handle:
            for line in handle:
                match = PRED_RE.search(line)
                if match:
                    yield match.group(1).strip()


def main():
    # Merge skills across logs in listed order.
    merged_skills = list(iter_predicted_skills(LOG_FILES))

    counts = Counter()
    stop_streak = 0
    total_kept = 0

    for raw_skill in merged_skills:
        skill = best_skill_match(raw_skill)
        if skill is None:
            continue  # Skip anything that fails fuzzy matching.

        if skill == "Stop and Pause":
            if stop_streak >= 4:
                continue  # Ignore stop repeated over 3 times in a row.
            stop_streak += 1
        else:
            stop_streak = 0

        counts[skill] += 1
        total_kept += 1

    print(f"Processed skills (after filtering): {total_kept}")
    for canonical in CANONICAL_SKILLS.values():
        count = counts.get(canonical, 0)
        pct = (count / total_kept * 100.0) if total_kept else 0.0
        print(f"{canonical:32s} {count:6d} {pct:7.2f}%")


if __name__ == "__main__":
    main()
