import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DISABLE_TQDM"] = "1"

import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append("map_nav_src")
sys.path.append("map_nav_src/moe")

from pprint import pprint
import json
import base64
from PIL import Image
from io import BytesIO
from vllm import LLM, SamplingParams
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.get_images import load_vp_lookup, convert_path2img, extract_cand_img

import torch
import gc
import re
from collections import Counter

from vllm import LLM
from transformers import AutoProcessor

from vllm.sampling_params import BeamSearchParams

from utils.get_images import load_vp_lookup, convert_path2img, extract_cand_img


SKILL_INDEX = {
    "Directional Adjustment": 0,
    "Vertical Movement": 1,
    "Stop and Pause": 2,
    "Landmark Detection": 3,
    "Area and Region Identification": 4,   
}


def load_prompt_template(path):
    with open(path, 'r') as f:
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


def load_vLLM_model(model_ckpt, limit_mm_per_prompt=None, seed=42, tensor_parallel_size=1, gpu_memory_utilization=0.7, max_num_seqs=1):
        
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


def load_two_vllms(batch_size, localizer_model_ckpt="Qwen/Qwen2.5-VL-7B-Instruct", skill_model_ckpt="Qwen/Qwen2.5-VL-7B-Instruct"):
    # -----------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    localizer_llm = load_vLLM_model(
        model_ckpt=localizer_model_ckpt,
        limit_mm_per_prompt=20,  # Limit in MB per prompt
        seed=0,
        gpu_memory_utilization=0.7,  # GPU memory utilization
        tensor_parallel_size=1,  # Number of GPUs to use
        max_num_seqs=batch_size,  # Maximum number of sequences to generate in parallel
    )
    print("✅ Loaded Qwen2.5-VL-7B on CUDA:0")

    # -----------------------------
 
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    if skill_model_ckpt == localizer_model_ckpt:
        skill_llm = localizer_llm
        print(f"✅ Loaded Qwen2.5-VL-7B as Skill model on CUDA 1")
    else:
        skill_llm = load_vLLM_model(
            model_ckpt=skill_model_ckpt,
            limit_mm_per_prompt=20,  # Limit in MB per prompt
            seed=0,
            gpu_memory_utilization=0.7,  # GPU memory utilization
            tensor_parallel_size=1,  # Number of GPUs to use
            max_num_seqs=batch_size  # Maximum number of sequences to generate in parallel
        )


    return localizer_llm, skill_llm



def load_two_vllms_with_id(batch_size, 
                   localizer_model_ckpt="Qwen/Qwen2.5-VL-7B-Instruct", 
                   skill_model_ckpt="Qwen/Qwen2.5-VL-7B-Instruct", 
                   gpu_memory_utilization=0.7,
                   localizer_gpu_id=0, 
                   skill_gpu_id=1):
    # -----------------------------
    # Load Localizer model on localizer_gpu_id
    # -----------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(localizer_gpu_id)

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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(skill_gpu_id)

    if skill_model_ckpt == localizer_model_ckpt:
        skill_llm = localizer_llm
        print(f"✅ Loaded Qwen2.5-VL-7B as Skill model on CUDA 1")
    else:
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
        prompt_template_path="/home/matiany3/ScaleVLN/VLN-DUET/map_nav_src/prompts/localization_template_add_prev_subinstruction.txt",
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
        prompt_template_path="prompts/skill_routing_template_strong_format.txt",
        batch_inputs=batch_inputs_skill_routing,
        params_type="BeamSearchParams"
    )
    
    expert_indices = []
    skills = []
    for item, result in zip(batch_inputs_skill_routing, skill_results):
        skill = extract_majority_skill(result)
        skills.append(skill)
        skill_index = SKILL_INDEX.get(skill, 1) # Take Landmark as default
        expert_indices.append(skill_index)

    
    if logger:
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
            logger.info(json.dumps(log_data, ensure_ascii=False, indent=2))
            
    return expert_indices, prev_sub_instruction_list


if __name__ == "__main__" :
    
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    
    from moe.vLLM_load_demo import load_two_vllms
    
    llm_vl, llm_skill = load_two_vllms()