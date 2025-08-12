import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import base64
import json
import time
import torch
import gc
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append("../map_nav_src")

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.get_images import load_vp_lookup, convert_path2img, extract_cand_img

import warnings
warnings.filterwarnings("ignore", message="`do_sample` is set to `False`. However, `temperature` is set to .*")

import re
import gc

# Constant Prompt
AVAILABLE_SKILLS = [
    "Directional Adjustment",
    "Vertical Movement",
    "Halt and Pause",
    "Landmark Detection",
    "Area and Region Identification",
    # "Temporal Order Planning"
]

# Mapping of skill indices to names
SKILL_INDEX = {
    0: "Directional Adjustment",
    1: "Vertical Movement",
    2: "Halt and Pause",
    3: "Landmark Detection",
    4: "Area and Region Identification",
    # 5: "Temporal Order Planning"     
}

# Description of each skill
SKILL_DESCRIPTION = """
Navigation Skills:
- Directional Adjustment:
Involves turning or changing heading. Look for instructions like "turn left", "go back", or "face the hallway". Used when the agent needs to rotate or reorient without necessarily changing position.
- Vertical Movement:
Involves moving across floors or elevation changes. Triggered by terms like "go upstairs", "down the stairs", or "take the elevator". Watch for floor changes in visuals or references to vertical navigation.
- Halt and Pause:
Involves stopping at a specific location. Triggered by instructions like "stop", "wait", or "stand in front of". Used when the endpoint or a mid-action pause is important.
- Landmark Detection:
Requires identifying and responding to specific objects or features in the environment. Triggered by mentions of visible items like "lamp", "chair", "red sofa", "painting". Used when object recognition is necessary to proceed or confirm position.
- Area and Region Identification:
Involves recognizing or transitioning between distinct spaces or rooms. Triggered by mentions like "enter the kitchen", "in the bedroom", "exit hallway". Requires understanding of semantic regions based on context or appearance.
- Temporal Order Planning:
Involves following a multi-step or sequential plan. Triggered by words like "after", "then", "before", or instructions that chain together several actions. Requires tracking sequence and dependencies.
"""

ANALYZE_INPUT_PROMPT = """
You will be given:
- The original full navigation instruction.
- The sub-instruction that should be executed next, based on reasoning.
- A reasoning explanation derived from the visual history and instruction.
- A list of candidate viewpoint images representing potential next observations.
"""

ANALYZE_TASK_PROMPT = """
Your task:
1. Read and understand the sub-instruction to be executed.
2. Use the reasoning explanation and candidate images to infer what skills are likely required to carry out that sub-instruction.
3. Assign confidence percentages (as strings, e.g., "30%") to each relevant skill.
"""

ANALYZE_OUTPUT_PROMPT = """
Return your result as a **valid JSON object** with two keys:
- `"instruction"`: the exact sub-instruction to be executed.
- `"skill-routing"`: a dictionary mapping skill names to confidence percentages.

IMPORTANT:
- Use only the sub-instruction and reasoning provided — do not decompose the instruction your
- The output must be raw JSON — no markdown, no explanations, no extra text.
"""

ANALYZE_EXAMPLE_INPUT = """
Example:

Original Whole Instruction: "At the bottom of the stairs, go through the nearest archway to your left. Head straight until you enter the room with a pool table. Step slightly to the left to get out of the way."

Sub-instruction to be executed for next step: "Head straight until you enter the room with a pool table."

Reasoning based on previous viewpoints path and original instruction: The agent appears to be just outside the archway. The next step is likely to involve entering the archway and preparing to head straight.

Candidate Viewpoints: Images 0–4

Expected Output:
{
"instruction": "Head straight until you enter the room with a pool table.",
"skill-routing": {
    "Landmark Detection": "70%",
    "Directional Adjustment": "30%"
}
}
"""

IDENTIFY_REASONING_GUIDE = """
Step-by-Step Reasoning Instructions:
1.	Decompose the instruction into sub-instructions.
- Break the full instruction into smaller steps. Each sentence or clause typically represents one step.
- Example:
    - Original: “At the bottom of the stairs, go through the nearest archway to your left. Head straight until you enter the room with a pool table. Step slightly to the left to get out of the way.”
    - Decomposed:
    - “At the bottom of the stairs, go through the nearest archway to your left.”
    - “Head straight until you enter the room with a pool table.”
    - “Step slightly to the left to get out of the way.”
2.	Analyze the sequence of previous viewpoint images.
These represent what the agent has already seen or done.
- If there are no previous images, treat this as the start of the navigation and return the first sub-instruction, exactly as written.
- If there are previous images, use visual clues (e.g., stairs, archway, pool table) to infer completed actions.

3.	Evaluate each sub-instruction for completion.
Use spatial and object-based cues to check whether the step has been executed.
- If the current image shows the agent inside the goal location (e.g., the room with a pool table) and a final positioning instruction exists (e.g., “Step slightly to the left”), that positioning step should be returned next.
- Do not return earlier steps (e.g., “Head straight…”) once the goal location has been reached.
- Do not skip to “no further instruction” unless every clause has been addressed.

4.	When the goal location has been reached, and only a final positional instruction remains (e.g., “Wait”, “Step left”), return that as the next sub-instruction.

5.	If not all steps are completed, return the next incomplete sub-instruction, using the exact wording from the original instruction.
- Never paraphrase or alter the wording.

6.	Avoid repeating sub-instructions that have already been completed based on the visual evidence from previous images.

7.	Output the result in the following JSON format:
{
"Sub-instruction to be executed": "<exact next instruction clause>",
"Reasoning": "<why this is the next step based on image sequence>"
}

CHECKPOINT:
If the agent is already inside the goal location and the remaining instruction is positional (e.g., “Step to the left”, “Wait”), then that is the correct next step to return.
"""

# System prompt for skill analysis
ANALYZE_SYSTEM_PROMPT = f"""
You are analyzing a natural language navigation instruction used in an indoor, vision-and-language task. 
Your goal is to identify which of the six defined navigation skills ({AVAILABLE_SKILLS}) are involved in each part of the instruction, and to assign confidence percentages to reflect each skill's importance in executing that part.
Confidence Assignment Principles:
- Use lexical cues, visual scene dependencies, and task relevance to assign confidence values.
- Reflect how central each skill is to executing that instruction, not just whether it's mentioned.
- Confidence scores for all included skills should sum to 100%.
"""

IDENTIFY_SYSTEM_PROMPT = (
    "You are a visual reasoning assistant for indoor navigation. "
    "Your task is to analyze a list of previously observed images and a natural language instruction. "
    "Determine which parts of the instruction have already been completed, and return the next step to be executed.\n\n"
    "Your response must:\n"
    "- Return the next action using *exact phrasing* from the original instruction (no paraphrasing).\n"
    "- Match the sub-instruction to the visual context from previous images.\n"
    "- If the goal (e.g., pool table) has clearly been reached, return the final sub-instruction.\n"
    "- If *all* sub-instructions have been completed based on the visual path, do not return anything further. Stop reasoning.\n"
    # "- After completing all movement sub-instructions, if the final instruction is a waiting or positioning action (e.g., 'wait there', 'step slightly to the left'), you must return it as the last sub-instruction — but only once all movement has been completed."
    "- If the final destination has been reached and the last step is a positional or waiting action (e.g., 'wait there', 'step to the left'), return that as the next step.\n"
    "- You must reason about whether the agent is already at the destination."
    "If the current image shows the goal destination (e.g. inside the room with the pool table, or inside the open doorway), and the instruction contains a final step like 'wait' or 'adjust your position', that is the next sub-instruction."
    # "IMPORTANT: If no previous images are provided, you must always return the FIRST sub-instruction from the instruction — without exception."
)

FORMAT_NOTE = """   
IMPORTANT: Your response must be a valid JSON object without any surrounding text, code blocks, or explanations.
Do not include markdown formatting like ```json or ```. Just provide the raw JSON object.
"""

def get_qwen_model(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    # global model, processor
        
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    return model, processor


def qwen_infer(model, processor, system_prompt, user_prompt, previous_viewpoint_list=None, candidate_viewpoint_list=None, max_tokens=3000):
    """
    Run inference with Qwen2.5-VL-7B-Instruct model.
    
    Args:
        system (str): System prompt
        text (str): Input text prompt
        previous_viewpoint_list (list): List of previous viewpoint image paths
        candidate_viewpoint_list (list): List of candidate viewpoint image paths
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        tuple: (generated_text, token_info)
    """
    if not model or not processor:
        model, processor = get_qwen_model()
        
    previous_viewpoint_list = previous_viewpoint_list or []
    candidate_viewpoint_list = candidate_viewpoint_list or []
    
    user_content = []
    
    # Label and add previous images
    for i, image_path in enumerate(previous_viewpoint_list):
        if image_path is not None and os.path.exists(image_path):
            user_content.append({"type": "text", "text": f"Previous Viewpoint Image {i}:"})
            user_content.append({"type": "image", "image": image_path})

    # Label and add candidate images (offset indices)
    for i, image_path in enumerate(candidate_viewpoint_list):
        if image_path is not None and os.path.exists(image_path):
            idx = i + len(previous_viewpoint_list)
            user_content.append({"type": "text", "text": f"Candidate Viewpoint Image {idx}:"})
            user_content.append({"type": "image", "image": image_path})
    
    # Add text prompt
    user_content.append({
        "type": "text", 
        "text": user_prompt
    })

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt
                }
            ]
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    
    start_time = time.time()
    
    # Preparation for inference
    text_inputs = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process images
    image_inputs, _ = process_vision_info(messages)  # Ignore video inputs
    
    inputs = processor(
        text=[text_inputs],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    )
    # inputs = inputs.to("cuda")
    inputs = inputs.to(model.device)
    # inputs = {k: v.to(model.device) for k, v in inputs.items()}  # 🔧 Safe multi-GPU device mapping
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            do_sample=False,  # Deterministic generation
            # do_sample=True,
            # temperature=0.7
        )
        
        # Extract only the newly generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode the generated text
        answer = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    
    # Create token usage information
    elapsed = time.time() - start_time
    token_info = {
        "completion_tokens": len(generated_ids_trimmed[0]),
        "prompt_tokens": len(inputs.input_ids[0]),
        "total_tokens": len(inputs.input_ids[0]) + len(generated_ids_trimmed[0]),
        "elapsed_time": elapsed
    }
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return answer, token_info
    

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
            
            
def identify_instructions_with_qwen(model, processor, instruction, previous_viewpoint_list=None, max_tokens=3000):
    """
    Run inference with Qwen2.5-VL-7B-Instruct model to identify the next-step sub-instruction using chain-of-thought reasoning.
    
    Args:
        system (str): System prompt
        text (str): Full instruction for the navigation task
        previous_viewpoint_list (list): List of previous viewpoint image paths
        candidate_viewpoint_list (list): List of candidate viewpoint image paths
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        tuple: (generated_text, token_info)
    """
    
    if not model or not processor:
        model, processor = get_qwen_model()

    previous_viewpoint_list = previous_viewpoint_list or []
    
    if not previous_viewpoint_list:
        return {
            "Sub-instruction to be executed": extract_first_sub_instruction(instruction),
            "Reasoning": "There are no previous viewpoints, so the agent has not started yet. The next step is the first sub-instruction."
        }
    
    full_prompt = (
        f"Instruction: {instruction}\n\n"
        "Previous viewpoint images represent what the agent has already seen.\n"
        "Use the following reasoning strategy to determine what to do next:\n\n"
        + IDENTIFY_REASONING_GUIDE # Insert the chain-of-thought reformatted above
        + "\n\nNow, using the instruction and the visual history, identify the next step."
        + FORMAT_NOTE  # Add the format note
    )
    
    # Run inference
    response, _ = qwen_infer(
        model=model,
        processor=processor,
        system_prompt=IDENTIFY_SYSTEM_PROMPT,
        user_prompt=full_prompt,
        previous_viewpoint_list=previous_viewpoint_list,
        max_tokens=max_tokens
    )
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Try to parse the response as JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Return raw response if not valid JSON
        # return response
        print(f"[Warning] Failed to parse JSON: {response}")
        return {"Sub-instruction to be executed": "", "Reasoning": f"Parsing failed. Raw output: {response}"}, None


def analyze_with_qwen(model, processor, instruction, sub_instruction, reasoning, candidate_viewpoint_list=None, max_tokens=3000):
    """
    Analyze navigation instruction using Qwen model.
    
    Args:
        instruction (str): The navigation instruction
        # previous_viewpoint_list (list): List of previous viewpoint image paths
        candidate_viewpoint_list (list): List of candidate viewpoint image paths
        
    Returns:
        str: Analysis results (may need to be parsed into JSON)
    """
    
    if not model or not processor:
        model, processor = get_qwen_model()
        
    # previous_viewpoint_list = previous_viewpoint_list or []
    candidate_viewpoint_list = candidate_viewpoint_list or []
    
    full_prompt = f"""{SKILL_DESCRIPTION} 
    
{ANALYZE_INPUT_PROMPT}

{ANALYZE_TASK_PROMPT}

{ANALYZE_OUTPUT_PROMPT}

{ANALYZE_EXAMPLE_INPUT}

Now analyze the following:

Original Whole Instruction: "{instruction}"

Sub-instruction to be executed for next step: "{sub_instruction}"

Reasoning based on previous viewpoints path and original instruction: {reasoning}

Candidate Viewpoints: Images 0–{len(candidate_viewpoint_list) - 1}

IMPORTANT: Output must be a valid JSON object, no surrounding text or formatting.

{FORMAT_NOTE}
"""
    
    # Run inference
    response, _ = qwen_infer(
        model=model,
        processor=processor,
        system_prompt=ANALYZE_SYSTEM_PROMPT,
        user_prompt=full_prompt,
        # previous_viewpoint_list=previous_viewpoint_list,
        candidate_viewpoint_list=candidate_viewpoint_list,
        max_tokens=max_tokens
    )
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Try to parse the response as JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Return raw response if not valid JSON

        return response
    
    
def parse_qwen_response(response):
    """
    Parse and clean the Qwen model response to extract valid JSON.
    
    Args:
        response (str): Raw response from Qwen model
        
    Returns:
        dict: Parsed JSON object or None if parsing fails
    """
    import re
    
    '''
    if not response:
        return None
        
    # Strip markdown code blocks and explanations
    try:
        # First try: check if it's already valid JSON
        return json.loads(response)
    except json.JSONDecodeError:
        # Second try: extract JSON from markdown code blocks
        json_pattern = r'```(?:json)?\s*(.*?)\s*```'
        import re
        matches = re.search(json_pattern, response, re.DOTALL)
        
        if matches:
            try:
                return json.loads(matches.group(1))
            except json.JSONDecodeError:
                pass
                
        # Third try: Look for JSON-like structure with curly braces
        try:
            json_pattern = r'(\{.*\})'
            matches = re.search(json_pattern, response, re.DOTALL)
            if matches:
                return json.loads(matches.group(1))
        except (json.JSONDecodeError, AttributeError):
            pass
            
        # Fourth try: Remove common explanations and non-JSON text
        lines = response.split('\n')
        json_lines = []
        capture = False
        
        for line in lines:
            if '{' in line:
                capture = True
            
            if capture:
                json_lines.append(line)
            
            if '}' in line and capture:
                break
                
        if json_lines:
            try:
                return json.loads(''.join(json_lines))
            except json.JSONDecodeError:
                pass
        '''
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

        

def extract_from_response(response):
        if not isinstance(response, dict):
            response = parse_qwen_response(response)

        # Assume model order is aligned with SKILL_INDEX key order (0-5)
        skill_routing = response.get("skill-routing", {})
        
        # Initialize all weights to 0.0
        weights = [0.0] * len(SKILL_INDEX)
        
        skill_names = []
        skill_index_list = []
        for skill_name, confidence_str in skill_routing.items():
            skill_names.append(skill_name)
            
            confidence_value = float(confidence_str.strip('%'))
            # Get the index for the skill name
            skill_index = next((i for i, name in SKILL_INDEX.items() if name == skill_name), None)
            skill_index_list.append(skill_index)
            
            if skill_index is not None:
                weights[skill_index] = confidence_value / 100.0  # Normalize to 0-1 scale

        return skill_names, skill_index_list, weights  # Aligned with r
    
    
if __name__ == "__main__":
    
    vp_lookup = load_vp_lookup()
    
    model, processor = get_qwen_model()
    
    data = [
        {
        "distance": 10.86,
        "scan": "8194nk5LbLH",
        "path_id": 4332,
        "instr_id": "4332_0",
        "path": [
            "c9e8dc09263e4d0da77d16de0ecddd39",
            "f33c718aaf2c41469389a87944442c62",
            "ae91518ed77047b3bdeeca864cd04029",
            "6776097c17ed4b93aee61704eb32f06c"
        ],
        "heading": 4.055,
        "instruction": "Walk to the other end of the lobby and wait near the exit.",
        }
        
    ]
    
    start_time = time.time()
    
    for index, item in enumerate(data):
        # if index != 1: continue
        
        instruction = item['instruction']
        paths = item['path']
        original_image_list = convert_path2img(paths, item['scan'], vp_lookup)
        print(f"Original Image List from Path: {original_image_list}")
        
        
        for idx, viewpoint in enumerate(paths):
            if idx == 0:
                continue
            previous_viewpoint_list = convert_path2img(paths[:idx+1], item['scan'], vp_lookup)
            candidate_viewpoint_list = extract_cand_img(paths[idx], item['scan'], vp_lookup)
            
            print('-'*20)
            print(f"\nOriginal Instruction: {instruction}")
            
            print(f"\nPrevious Viewpoint List: {previous_viewpoint_list}\n")
            # print(f"Candidate Viewpoint List: {candidate_viewpoint_list}")
            
            result = identify_instructions_with_qwen(model, processor, instruction, previous_viewpoint_list)
            # result = identify_instructions_with_gpt4o(instruction, previous_viewpoint_list)
            print(json.dumps(result, indent=4))
            
            sub_instruction = result.get("Sub-instruction to be executed", "")
            reasoning = result.get("Reasoning", "")
            
            ### Openai
                        
            skill_routing = analyze_with_qwen(model, processor, instruction, sub_instruction, reasoning, candidate_viewpoint_list)
            # skill_routing = analyze_with_gpt4o(instruction, sub_instruction, reasoning, candidate_viewpoint_list)
            print(json.dumps(skill_routing, indent=4))
            
            skill_names, skill_index_list, weights = extract_from_response(skill_routing)
            print(f"Skill Names: {skill_names}")
            print(f"Skill Index List: {skill_index_list}")
            print(f"Weights: {weights}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")