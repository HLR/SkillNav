import argparse
import asyncio
import base64
import json
import logging
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DISABLE_TQDM"] = "1"
import sys
import time
from collections import Counter
from io import BytesIO
from typing import List
from pathlib import Path

sys.path.append("..")
sys.path.append(".")
sys.path.append("map_nav_src")
sys.path.append("map_nav_src/moe")

from PIL import Image
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info
from utils.get_images import load_vp_lookup, convert_path2img, extract_cand_img

import torch
import gc
import re

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    _FASTAPI_AVAILABLE = True
    _FASTAPI_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only triggered when FastAPI missing
    FastAPI = None
    HTTPException = Exception
    BaseModel = object
    uvicorn = None
    _FASTAPI_AVAILABLE = False
    _FASTAPI_IMPORT_ERROR = exc


SKILL_INDEX = {
    "Directional Adjustment": 0,
    "Vertical Movement": 1,
    "Stop and Pause": 2,
    "Landmark Detection": 3,
    "Area and Region Identification": 4,   
}

DEFAULT_VLM_DEPLOYMENT_MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "THUDM/GLM-4.1V-9B-Thinking",
    "OpenGVLab/InternVL3_5-8B",
]

ROUTER_SERVER_LOGGER_NAME = "moe.router_server"
ROUTER_SERVER_DEFAULT_LOG = os.path.join(os.path.dirname(__file__), "router_server.log")

_MODULE_DIR = Path(__file__).resolve().parent
_MAP_NAV_SRC_DIR = _MODULE_DIR.parent
_PROMPTS_ENV = os.environ.get("SCALEVLN_PROMPTS_DIR")
if _PROMPTS_ENV:
    _PROMPTS_DIR = Path(_PROMPTS_ENV).expanduser()
    if not _PROMPTS_DIR.is_absolute():
        _PROMPTS_DIR = (_MAP_NAV_SRC_DIR / _PROMPTS_DIR).resolve()
else:
    _PROMPTS_DIR = (_MAP_NAV_SRC_DIR / "prompts").resolve()

LOCALIZATION_PROMPT_TEMPLATE = _PROMPTS_DIR / "localization_template_add_prev_subinstruction.txt"
SKILL_ROUTING_PROMPT_TEMPLATE = _PROMPTS_DIR / "skill_routing_template_strong_format.txt"


if _FASTAPI_AVAILABLE:

    class RouterRequest(BaseModel):
        scan: str
        instr_id: str
        full_instruction: str
        previous_viewpoint_list: List[str]
        previous_sub_instruction_list: List[str] = []

    class RouterResponse(BaseModel):
        expert_indices: List[int]
        sub_instructions: List[str]
        latency_seconds: float
        log_entries: List[dict]

else:  # pragma: no cover - FastAPI missing

    class RouterRequest:  # type: ignore
        pass

    class RouterResponse:  # type: ignore
        pass


def _set_cuda_visible_devices(device_id):
    """Keep CUDA device selection consistent across helpers."""
    if device_id is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


def load_prompt_template(path):
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(
            f"Prompt template not found at {path_obj}. "
            "Set SCALEVLN_PROMPTS_DIR to the directory containing the prompts if they reside elsewhere."
        )
    with path_obj.open('r', encoding='utf-8') as f:
        return f.read()


# Utility to encode image to base64
def encode_image(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    base64_bytes = base64.b64encode(img_bytes)
    return base64_bytes.decode('utf-8')


def extract_first_sub_instruction(instruction):
    import re

    # Normalize spacing
    instruction = instruction.strip()

    # First split by period (.)
    sentences = re.split(r'\.\s*', instruction)
    if not sentences:
        return instruction

    first_sentence = sentences[0]

    # Now split by conjunctions like "and", "then", "after that", etc., but keep commas
    # Look for common second-action triggers
    split_clauses = re.split(r',\s*(and|then|after that|afterwards)\b', first_sentence, flags=re.IGNORECASE)

    # Reconstruct the first sub-instruction: take the first clause + any connecting comma
    if len(split_clauses) > 1:
        return split_clauses[0].strip() + ','  # Ensure it ends with a comma like original
    else:
        return first_sentence.strip()


def parse_qwen_response(response):
    """
    Parse and clean the Qwen model response to extract valid JSON.
    
    Args:
        response (str): Raw response from Qwen model
        
    Returns:
        dict: Parsed JSON object or None if parsing fails
    """
    import re
    import json
    
    if not response:
        return None
    
    def try_parse(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def fix_unclosed_braces(text):
        # Count unmatched braces and attempt to close them
        open_braces = text.count('{')
        close_braces = text.count('}')
        if open_braces > close_braces:
            text += '}' * (open_braces - close_braces)
        return text

    def clean_trailing_commas(text):
        # Remove trailing commas before } or ]
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return text

    # Try raw JSON
    result = try_parse(response)
    if result is not None:
        return result

    # Try extracting code block content
    code_block_pattern = re.compile(r'```(?:json)?\s*(.*?)\s*```', re.DOTALL)
    match = code_block_pattern.search(response)
    if match:
        result = try_parse(match.group(1))
        if result is not None:
            return result

    # Try extracting from anywhere in text
    brace_start = response.find('{')
    brace_end = response.rfind('}')
    if brace_start != -1:
        candidate = response[brace_start:(brace_end + 1 if brace_end > brace_start else None)]
        candidate = clean_trailing_commas(candidate)
        candidate = fix_unclosed_braces(candidate)
        result = try_parse(candidate)
        if result is not None:
            return result
        
    # If all fails, return a structured error
    print(f"Failed to parse response as JSON: {response[:100]}...")
    return {
        "error": "Failed to parse response",
        "raw_response": response
    }


def clean_llm_json_response(raw_response):
    cleaned = raw_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def load_vLLM_model(model_ckpt, limit_mm_per_prompt=None, seed=42, tensor_parallel_size=1, gpu_memory_utilization=0.9, max_num_seqs=1):
        
    if "VL" in model_ckpt or 'llava' in model_ckpt:
        llm = LLM(
            model=model_ckpt, # "Qwen/Qwen2.5-VL-7B-Instruct"
            # chat_template_format="chatml",  
            trust_remote_code=True,
            limit_mm_per_prompt={"image": limit_mm_per_prompt},
            seed=seed,
            tensor_parallel_size=tensor_parallel_size,
            # half_precision=half_precision,
            gpu_memory_utilization=gpu_memory_utilization,  # lower to avoid OOM (adjust as needed)
            max_num_seqs=max_num_seqs,
        )
        return llm
    
        # processor = AutoProcessor.from_pretrained(
        #     model_ckpt,
        #     trust_remote_code=True,
        #     use_fast=True,
        # )
        # return llm, processor
        
    else:
        llm = LLM(
            model=model_ckpt, 
            seed=seed,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,  # lower to avoid OOM (adjust as needed)
            dtype="bfloat16",
            
        )
        return llm


def load_two_vllms(batch_size,
                   localizer_model_ckpt="Qwen/Qwen2.5-VL-7B-Instruct",
                   skill_model_ckpt="Qwen/Qwen2.5-VL-7B-Instruct",
                   localizer_gpu_id=0,
                   skill_gpu_id=1):
    # -----------------------------
    _set_cuda_visible_devices(localizer_gpu_id)

    localizer_llm = load_vLLM_model(
        model_ckpt=localizer_model_ckpt,
        limit_mm_per_prompt=20,  # Limit in MB per prompt
        seed=0,
        gpu_memory_utilization=0.9,  # GPU memory utilization
        tensor_parallel_size=1,  # Number of GPUs to use
        max_num_seqs=batch_size,  # Maximum number of sequences to generate in parallel
    )
    print("✅ Loaded Qwen2.5-VL-7B on CUDA:0")

    # -----------------------------
 
    reuse_localizer = (
        skill_model_ckpt == localizer_model_ckpt
        and skill_gpu_id == localizer_gpu_id
    )
    if reuse_localizer:
        skill_llm = localizer_llm
        print(f"✅ Reusing {localizer_model_ckpt} for Skill routing on CUDA:{localizer_gpu_id}")
    else:
        _set_cuda_visible_devices(skill_gpu_id)
        skill_llm = load_vLLM_model(
            model_ckpt=skill_model_ckpt,
            limit_mm_per_prompt=20,  # Limit in MB per prompt
            seed=0,
            gpu_memory_utilization=0.9,  # GPU memory utilization
            tensor_parallel_size=1,  # Number of GPUs to use
            max_num_seqs=batch_size  # Maximum number of sequences to generate in parallel
        )
        print(f"✅ Loaded {skill_model_ckpt} on CUDA:{skill_gpu_id}")


    return localizer_llm, skill_llm



def load_two_vllms_with_id(batch_size, 
                   localizer_model_ckpt="Qwen/Qwen2.5-VL-7B-Instruct", 
                   skill_model_ckpt="Qwen/Qwen2.5-VL-7B-Instruct", 
                   gpu_memory_utilization=0.9,
                   localizer_gpu_id=0, 
                   skill_gpu_id=1):
    # -----------------------------
    # Load Localizer model on localizer_gpu_id
    # -----------------------------
    _set_cuda_visible_devices(localizer_gpu_id)

    localizer_llm = load_vLLM_model(
        model_ckpt=localizer_model_ckpt,
        limit_mm_per_prompt=20,
        seed=0,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
        max_num_seqs=batch_size,
    )
    print(f"✅ Loaded {localizer_model_ckpt} on CUDA:{localizer_gpu_id}")

    # -----------------------------
    # Load Skill model on skill_gpu_id
    # -----------------------------
    reuse_localizer = (
        skill_model_ckpt == localizer_model_ckpt
        and skill_gpu_id == localizer_gpu_id
    )
    if reuse_localizer:
        skill_llm = localizer_llm
        print(f"✅ Reusing {localizer_model_ckpt} for Skill routing on CUDA:{localizer_gpu_id}")
    else:
        _set_cuda_visible_devices(skill_gpu_id)
        skill_llm = load_vLLM_model(
            model_ckpt=skill_model_ckpt,
            limit_mm_per_prompt=20,
            seed=0,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            max_num_seqs=batch_size,
        )
        print(f"✅ Loaded {skill_model_ckpt} on CUDA:{skill_gpu_id}")

    return localizer_llm, skill_llm

def infer_with_vllm_instruction_localization(llm, prompt_template_path, batch_inputs, max_tokens=3000, limit_mm_per_prompt=20):

    prompt_template = load_prompt_template(prompt_template_path)

    messages = []
    results = []
    
    for item in batch_inputs:
        full_instruction = item["full_instruction"]
        previous_viewpoint_list = item["previous_viewpoint_list"]
        previous_sub_instruction_list = item.get("previous_sub_instruction_list", [])
        # Convert list of previously completed sub-instructions into a single string
        previous_sub_instructions = " ".join(previous_sub_instruction_list) if previous_sub_instruction_list else ""
        
        if not previous_sub_instruction_list:
            results.append({
                "Sub-instruction to be executed": extract_first_sub_instruction(full_instruction),
                "Reasoning": "There are no previous sub-instruction executed, so the agent has not started yet. The next step is the first sub-instruction."
            })
            continue
        
        
        # Format the prompt using the template        
        prompt = prompt_template.replace("{instruction}", full_instruction).replace("{previous_sub_instructions}", previous_sub_instructions)
    
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ]
            }

        # # Ensure we do not exceed the image limit per message
        # img_stat_index = 0 if len(previous_viewpoint_list)<limit_mm_per_prompt else (len(previous_viewpoint_list) - limit_mm_per_prompt) 
        
        if limit_mm_per_prompt and len(previous_viewpoint_list) > limit_mm_per_prompt:
            selected_imgs = previous_viewpoint_list[-limit_mm_per_prompt:]
        else:
            selected_imgs = previous_viewpoint_list
    
        for img_path in selected_imgs:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = Image.open(img_path).convert("RGB")
            base64_image = encode_image(img)
            new_image = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            message["content"].append(new_image)
        
        messages.append(message)

    
    sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            stop=["\\n"]
        )
    outputs = llm.chat([[message] for message in messages], sampling_params, use_tqdm=False)

    torch.cuda.empty_cache()
    gc.collect()
    
    for output in outputs:
        raw_response = output.outputs[0].text.strip()
        # print("-"*10)
        # print(f"[Raw Response] {raw_response}")
        # print("-"*10)
        try:
            cleaned_response = clean_llm_json_response(raw_response)
            parsed_response = parse_qwen_response(cleaned_response)
            key = "Sub-instruction to be executed"
            if key not in parsed_response:
                print(f"[Warning] Expected key not found. Available keys: {list(parsed_response.keys())}")
                results.append({
                    "Sub-instruction to be executed": "",
                    "Reasoning": f"Expected key not found. Raw output: {raw_response}"
                })
            else:
                results.append(parsed_response)
        except json.JSONDecodeError:
            print(f"[Warning] Failed to parse JSON. Raw output: {raw_response}")
            results.append({
                "Sub-instruction to be executed": "",
                "Reasoning": f"Parsing failed. Raw output: {raw_response}"
            })

    return results


def infer_with_vllm_instruction_localization_chatlm(llm, prompt_system_template_path, prompt_user_template_path, batch_inputs, max_tokens=3000, limit_mm_per_prompt=20):

    system_prompt = load_prompt_template(prompt_system_template_path)
    user_prompt_template = load_prompt_template(prompt_user_template_path)

    messages = []
    results = []
    
    for item in batch_inputs:
        
        message = []
        
        # System message
        message.append({
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt}
            ]
        })
        
        full_instruction = item["full_instruction"]
        previous_viewpoint_list = item["previous_viewpoint_list"]
        
        
        # Format the prompt using the template        
        user_prompt = user_prompt_template.replace("{instruction}", full_instruction)

        # User message (text + image placeholders)
        user_content = [{"type": "text", "text": user_prompt}]
        
        # # Ensure we do not exceed the image limit per message
        # img_stat_index = 0 if len(previous_viewpoint_list)<limit_mm_per_prompt else (len(previous_viewpoint_list) - limit_mm_per_prompt) 
        
        if limit_mm_per_prompt and len(previous_viewpoint_list) > limit_mm_per_prompt:
            selected_imgs = previous_viewpoint_list[-limit_mm_per_prompt:]
        else:
            selected_imgs = previous_viewpoint_list
    
        for img_path in selected_imgs:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = Image.open(img_path).convert("RGB")
            base64_image = encode_image(img)
            new_image = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            user_content.append(new_image)
        
        # Add user message
        message.append({
            "role": "user",
            "content": user_content
        })
        
        messages.append(message)
    
    sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            # stop=["\\n"]
            # stop=["\n", "Okay", "So,", "So the", "First", "The user", "So the user"]
            stop=["\\n"]
        )
    outputs = llm.chat([[message] for message in messages], sampling_params, use_tqdm=False)

    torch.cuda.empty_cache()
    gc.collect()
    
    for output in outputs:
        raw_response = output.outputs[0].text.strip()
        # print("-"*10)
        # print(f"[Raw Response] {raw_response}")
        # print("-"*10)
        try:
            cleaned_response = clean_llm_json_response(raw_response)
            parsed_response = parse_qwen_response(cleaned_response)
            key = "Sub-instruction to be executed"
            if key not in parsed_response:
                print(f"[Warning] Expected key not found. Available keys: {list(parsed_response.keys())}")
                results.append({
                    "Sub-instruction to be executed": "",
                    "Reasoning": f"Expected key not found. Raw output: {raw_response}"
                })
            else:
                results.append(parsed_response)
        except json.JSONDecodeError:
            print(f"[Warning] Failed to parse JSON. Raw output: {raw_response}")
            results.append({
                "Sub-instruction to be executed": "",
                "Reasoning": f"Parsing failed. Raw output: {raw_response}"
            })

    return results


def infer_with_vllm_skill_routing(llm, prompt_template_path, batch_inputs, params_type="SamplingParams", max_tokens=20):

    prompt_template = load_prompt_template(prompt_template_path)

    prompts = []
    for item in batch_inputs:
        full_instruction = item["full_instruction"]
        sub_instruction = item["sub_instruction"]
        reasoning = item["reasoning"]
        
        # Format the prompt using the template
        prompt = prompt_template.format(
            full_instruction=full_instruction,
            sub_instruction=sub_instruction,
            reasoning=reasoning
        )
        
        if params_type == 'BeamSearchParams':
            prompts.append({"prompt": prompt.strip()})
        elif params_type == 'SamplingParams':
            prompts.append(prompt.strip())
    
    results = []
    
    if params_type == 'BeamSearchParams':
        # Use Beam Search for diversity
        params = BeamSearchParams(beam_width=5, max_tokens=max_tokens)
        outputs = llm.beam_search(prompts, params, use_tqdm=False)
        
        for output, prompt in zip(outputs, prompts):
            beams = []
            for seq in output.sequences:
                text = seq.text
                
                prompt_str = prompt["prompt"] if isinstance(prompt, dict) else prompt
                if text.startswith(prompt_str):
                    text = text[len(prompt_str):]
                    
                beams.append(text.strip())
            results.append(beams)
        
        
    elif params_type == 'SamplingParams':
        # Use sampling to reduce memory vs beam search
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            # stop=["\\n"]
            # stop=["\n", "Okay", "So,", "So the", "First", "The user", "So the user"]
            stop=["*"]
        )
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
        
    torch.cuda.empty_cache()
    gc.collect()

    return results


def extract_majority_skill(text_list):
    """
    Extracts Skill_Name enclosed by ***** from a list of generated texts
    and returns the most frequently occurring skill as the final skill.
    If multiple skills have the same maximum count, returns the first one encountered.
    """
    pattern = r"\*{5}(.*?)\*{5}"  # matches *****Skill_Name*****
    all_skills = []

    for text in text_list:
        matches = re.findall(pattern, text)
        all_skills.extend(match.strip() for match in matches if match.strip())

    if not all_skills:
        return None  # or return "" based on your system needs

    skill_counts = Counter(all_skills)
    majority_skill, _ = skill_counts.most_common(1)[0]
    return majority_skill


def get_expert_indices(localizer_llm, skill_llm, batch_inputs_instruction_localization, logger=None):

    
    loc_results = infer_with_vllm_instruction_localization(
        llm=localizer_llm,
        prompt_template_path=LOCALIZATION_PROMPT_TEMPLATE,
        batch_inputs=batch_inputs_instruction_localization,
        max_tokens=3000
    )
    
    batch_inputs_skill_routing = []
    prev_sub_instruction_list = []
    for item, loc_result in zip(batch_inputs_instruction_localization, loc_results):
        sub_instruction = loc_result.get("Sub-instruction to be executed", "").strip()
        reasoning = loc_result.get("Reasoning", "").strip()
        if not sub_instruction:
            continue  # skip invalid results

        batch_inputs_skill_routing.append({
            "intru_id": item['instr_id'],
            "scan": item['scan'],
            "full_instruction": item['full_instruction'],
            "previous_viewpoint_list": item.get('previous_viewpoint_list', []),
            "previous_sub_instruction_list": item.get('previous_sub_instruction_list', []),
            "sub_instruction": sub_instruction,
            "reasoning": reasoning,
        })
        prev_sub_instruction_list.append(sub_instruction)
        
    skill_results = infer_with_vllm_skill_routing(
        llm=skill_llm,
        prompt_template_path=SKILL_ROUTING_PROMPT_TEMPLATE,
        batch_inputs=batch_inputs_skill_routing,
        params_type="BeamSearchParams"
    )
    
    expert_indices = []
    skills = []
    log_records = []
    for item, result in zip(batch_inputs_skill_routing, skill_results):
        skill = extract_majority_skill(result)
        skills.append(skill)
        skill_index = SKILL_INDEX.get(skill, 1) # Take Landmark as default
        expert_indices.append(skill_index)

    for item, skill in zip(batch_inputs_skill_routing, skills):
        log_data = {
            "instr_id": item['intru_id'],
            "scan": item['scan'],
            "full_instruction": item['full_instruction'],
            "previous_viewpoints": item.get('previous_viewpoint_list', []),
            "previous_sub_instructions": item.get('previous_sub_instruction_list', []),
            "sub_instruction": item['sub_instruction'],
            "reasoning": item['reasoning'],
            "predicted_skill": skill,
            "predicted_skill_index": SKILL_INDEX.get(skill, 1)
        }
        log_records.append(log_data)
        if logger:
            logger.info(json.dumps(log_data, ensure_ascii=False, indent=2))
            
    return expert_indices, prev_sub_instruction_list, log_records


def deploy_vlm_models_for_expert_indices(
    batch_inputs_instruction_localization,
    model_ckpts=None,
    logger=None,
    gpu_id=0,
    batch_size=None,
    limit_mm_per_prompt=20,
    seed=0,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
):
    """
    Sequentially deploy a list of VLM checkpoints and run get_expert_indices with a shared LLM.

    Args:
        batch_inputs_instruction_localization (list): Router batches expected by get_expert_indices.
        model_ckpts (list[str], optional): Checkpoints to evaluate. Defaults to DEFAULT_VLM_DEPLOYMENT_MODELS.
        logger (logging.Logger, optional): Logger to record routing decisions.
        gpu_id (int): CUDA device id used for each deployment.
        batch_size (int, optional): Overrides max_num_seqs passed to vLLM. Defaults to len(batch_inputs).
        limit_mm_per_prompt (int): Passed to load_vLLM_model for controlling images per prompt.
        seed (int): Random seed forwarded to load_vLLM_model.
        tensor_parallel_size (int): Number of GPUs per deployment.
        gpu_memory_utilization (float): Utilization hint for vLLM.

    Returns:
        dict: Mapping from checkpoint name to its expert indices and resolved sub instructions.
    """
    if not batch_inputs_instruction_localization:
        raise ValueError("batch_inputs_instruction_localization must not be empty.")

    model_ckpts = model_ckpts or DEFAULT_VLM_DEPLOYMENT_MODELS
    deployment_results = {}
    batch_size = batch_size or len(batch_inputs_instruction_localization)

    for model_ckpt in model_ckpts:
        _set_cuda_visible_devices(gpu_id)
        try:
            llm = load_vLLM_model(
                model_ckpt=model_ckpt,
                limit_mm_per_prompt=limit_mm_per_prompt,
                seed=seed,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_seqs=batch_size,
            )
        except ValueError as exc:
            if "No available memory for the cache blocks" in str(exc):
                raise RuntimeError(
                    f"vLLM could not reserve KV cache for {model_ckpt}. "
                    "Increase --gpu_memory_utilization or lower --batch_size/limit_mm_per_prompt."
                ) from exc
            raise

        if logger:
            logger.info(f"Running expert routing with shared LLM: {model_ckpt}")
        else:
            print(f"[vLLM_API] Running expert routing with shared LLM: {model_ckpt}")

        try:
            expert_indices, prev_sub_instruction_list, _ = get_expert_indices(
                localizer_llm=llm,
                skill_llm=llm,
                batch_inputs_instruction_localization=batch_inputs_instruction_localization,
                logger=logger,
            )
        finally:
            del llm
            torch.cuda.empty_cache()
            gc.collect()

        deployment_results[model_ckpt] = {
            "expert_indices": expert_indices,
            "sub_instructions": prev_sub_instruction_list,
        }

    return deployment_results


def _require_fastapi():
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI and uvicorn are required to run the router server. "
            f"Original import error: {_FASTAPI_IMPORT_ERROR}"
        )


def _setup_router_server_logger(log_file=None):
    log_path = log_file or ROUTER_SERVER_DEFAULT_LOG
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(ROUTER_SERVER_LOGGER_NAME)
    if logger.handlers:
        return logger

    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def create_router_server_app(
    batch_size,
    localizer_model_ckpt="Qwen/Qwen2.5-VL-7B-Instruct",
    skill_model_ckpt="Qwen/Qwen3-8B",
    localizer_gpu_id=0,
    skill_gpu_id=1,
    gpu_memory_utilization=0.9,
    log_file=None,
):
    """
    Initialize a FastAPI application that hosts the router models.
    """
    _require_fastapi()
    router_logger = _setup_router_server_logger(log_file)
    router_logger.info(
        "Loading router models localizer=%s skill=%s batch_size=%s",
        localizer_model_ckpt,
        skill_model_ckpt,
        batch_size,
    )

    localizer_llm, skill_llm = load_two_vllms_with_id(
        batch_size=batch_size,
        localizer_model_ckpt=localizer_model_ckpt,
        skill_model_ckpt=skill_model_ckpt,
        gpu_memory_utilization=gpu_memory_utilization,
        localizer_gpu_id=localizer_gpu_id,
        skill_gpu_id=skill_gpu_id,
    )
    router_logger.info("Router models ready.")

    app = FastAPI(title="ScaleVLN Router Server")
    request_lock = asyncio.Lock()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/route", response_model=RouterResponse)
    async def route(batch: List[RouterRequest]):
        if not batch:
            raise HTTPException(status_code=400, detail="Batch inputs cannot be empty.")

        inputs = [
            item.model_dump() if hasattr(item, "model_dump") else item.dict()
            for item in batch
        ]
        start_time = time.perf_counter()

        async with request_lock:
            expert_indices, sub_instructions, log_entries = get_expert_indices(
                localizer_llm=localizer_llm,
                skill_llm=skill_llm,
                batch_inputs_instruction_localization=inputs,
                logger=router_logger,
            )

        latency = time.perf_counter() - start_time
        router_logger.info(
            "Processed batch_size=%d latency=%.3fs", len(inputs), latency
        )
        return RouterResponse(
            expert_indices=expert_indices,
            sub_instructions=sub_instructions,
            latency_seconds=latency,
            log_entries=log_entries,
        )

    return app


def _parse_router_server_args():
    parser = argparse.ArgumentParser(description="Run the ScaleVLN router server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host.")
    parser.add_argument("--port", type=int, default=8010, help="Server port.")
    parser.add_argument("--batch_size", type=int, default=4, help="Router batch size.")
    parser.add_argument(
        "--localizer_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct"
    )
    parser.add_argument("--skill_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--localizer_gpu_id", type=int, default=0)
    parser.add_argument("--skill_gpu_id", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument(
        "--log_file",
        type=str,
        default=ROUTER_SERVER_DEFAULT_LOG,
        help="Router server log file path.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        help="Uvicorn log level (e.g., info, debug).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_router_server_args()
    app = create_router_server_app(
        batch_size=args.batch_size,
        localizer_model_ckpt=args.localizer_model,
        skill_model_ckpt=args.skill_model,
        localizer_gpu_id=args.localizer_gpu_id,
        skill_gpu_id=args.skill_gpu_id,
        gpu_memory_utilization=args.gpu_memory_utilization,
        log_file=args.log_file,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
