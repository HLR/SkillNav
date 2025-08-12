from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from openai import OpenAI

import base64
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

import os
import time

# InternVL
import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig

import httpx
import json

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

MAX_RETRIES = 3
NUM_THREADS = 8  # adjust based on CPU capacity

# OPENAI_KEY
generation_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=generation_key,
)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def _get_qwen_model(model="Qwen2.5-VL-7B-Instruct"):
    global _qwen_model, _qwen_processor
    # if _qwen_model is None or _qwen_processor is None:
    if model == "Qwen2.5-VL-7B-Instruct":
        _qwen_model_dir = "/home/matiany3/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/5b5eecc7efc2c3e86839993f2689bbbdf06bd8d4"
        print("Loading Qwen2.5-VL-7B-Instruct model...")
    elif model == "Qwen2.5-VL-32B-Instruct":
        _qwen_model_dir = "/home/matiany3/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/6bcf1c9155874e6961bcf82792681b4f4421d2f7"
        # os.environ["FLASH_ATTENTION_2_ENABLED"] = "0"
        print("Loading Qwen2.5-VL-32B-Instruct model...")
    elif model == "Qwen/Qwen2.5-7B-Instruct":
            _qwen_model_dir = "/home/matiany3/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
            print("Loading Qwen/Qwen2.5-7B-Instruct model...")
    
    _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(_qwen_model_dir, 
                                                                    torch_dtype="auto", 
                                                                    # torch_dtype=torch.float16,
                                                                    device_map="auto"
                                                                    )
    _qwen_processor = AutoProcessor.from_pretrained(_qwen_model_dir, 
                                                    use_fast=True)
    print(f"{model} model loaded successfully")
    print("The model is loaded on: ",next(_qwen_model.parameters()).device)  # Check the device of the model parameters
    print("-"*25)
    return _qwen_model, _qwen_processor


def qwen_infer(system_prompt, user_prompt, image_list, model, processor, max_tokens=1000, response_format=None):
    """
    Run inference with Qwen2.5-VL-7B-Instruct model.
    
    Args:
        system: System prompt
        text: Input text prompt
        image_list: List of image paths
        model: Model name (ignored, always uses Qwen2.5-VL-7B-Instruct)
        max_tokens: Maximum number of tokens to generate
        response_format: Response format specification (ignored for Qwen)
        
    Returns:
        Tuple of (generated_text, token_info)
    """
    
    user_content = []
    
    # Add images
    for i, image_path in enumerate(image_list):
        if image_path is not None and os.path.exists(image_path):
            user_content.append(
                {
                    "type": "text",
                    "text": f"Image {i}:"
                }
            )
            user_content.append(
                {
                    "type": "image",
                    "image": image_path
                }
            )
        else:
            print(f"Warning: Image {i} not found at path: {image_path}")
    
    
    # Add text prompt
    user_content.append({
        "type": "text", 
        "text": user_prompt
    })
    
    # print("User content for Qwen inference:", json.dumps(user_content, indent=4))  # For debugging
    
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

    # # Display the messages for debugging
    # print(json.dumps(messages, indent=4)) # Convert to JSON format if needed
    
    # Track token usage (approximate)
    start_time = time.time()
    
    # Preparation for inference
    text_inputs = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # image_inputs, video_inputs = process_vision_info(messages)
    image_inputs, _ = process_vision_info(messages)  # Ignore video inputs for now
    
    # print("-"*25)
    # print("Processed text inputs:", text_inputs)  # For debugging
    # print("Processed image inputs:", image_inputs)  # For debugging
    # print("-"*25)
    
    inputs = processor(
        text=[text_inputs],
        images=image_inputs,
        # videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda")
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            do_sample=False  # Deterministic generation (temperature=0)
        )
        
        # Extract only the newly generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode the generated text
        answer = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    
    # Create a simple token usage object (format similar to OpenAI for compatibility)
    elapsed = time.time() - start_time
    token_info = {
        "completion_tokens": len(generated_ids_trimmed[0]),
        "prompt_tokens": len(inputs.input_ids[0]),
        "total_tokens": len(inputs.input_ids[0]) + len(generated_ids_trimmed[0]),
        "elapsed_time": elapsed
    }
    
    return answer, token_info


def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def get_intern_model(model_name="OpenGVLab/InternVL3-8B"):
    if model_name == "OpenGVLab/InternVL3-8B":
        model_dir = "/home/matiany3/.cache/huggingface/hub/models--OpenGVLab--InternVL3-8B/snapshots/26dcadeb50cde4369918519c760feb2946814e10"
    else:
        raise ValueError(f"Model {model_name} not supported.")

    # Check number of GPUs
    gpu_count = torch.cuda.device_count()

    if gpu_count == 1:
        device_map = "auto"
    elif gpu_count > 1:
        device_map = split_model(model_dir)
    else:
        raise RuntimeError("No GPU available.")
    
    _intern_model = AutoModel.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16, # 16-bit (bf16 / fp16)
        # load_in_8bit=True, # 8-bit quantization
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map
        ).eval()
    
    
    _intern_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)

    return _intern_model, _intern_tokenizer


def generate_temporal_instructions(instructions, max_retries=3, max_tokens=2000, temperature=0, model="gpt-4o", num_threads=4):
    """
    Generate temporally ordered instructions from a list of natural instructions using GPT-4o.

    Args:
        client: OpenAI client instance
        instructions: List of raw navigation instructions
        max_retries: Retry attempts for API failures
        max_tokens: Max tokens in response
        temperature: Sampling temperature
        model: OpenAI model name
        num_threads: Max threads for parallel generation

    Returns:
        List of structured temporal instructions (same length, failed entries set as "")
    """

    few_shots_examples = """
**Example 1:**
Instruction: "Turn around and walk down the stairs. Stop once you get down them."
Output:
Turn around. Walk down the stairs. Stop at the bottom of the stairs.

**Example 2:**
Instruction: "Walk toward the dining room but turn left before entering it and go into the open area."
Output:
Walk toward the dining room. Stop at the entrance. Turn left. Enter the open area.

**Example 3:**
Instruction: "After you leave the laundry room, make a left in the hallway, and go to the bedroom straight ahead. When you are in the doorway of the room go to the doorway of the closet on the left and wait."
Output:
Exit the laundry room. Turn left in the hallway. Walk to the bedroom straight ahead. Enter the doorway of the bedroom. Go to the doorway of the closet on the left. Wait there.

**Example 4:**
Instruction: "Start moving forward down the corridor. You will pass offices on your left and right. Keep going down the hallway until you get to an exit sign on your right and what looks like some lockers in front of you. There will also be a brown door with an exit sign above it in front of you."
Output:
Start moving forward down the corridor. Pass the offices on your left and right. Continue walking down the hallway. Reach the exit sign on your right and the lockers in front of you. Stop in front of the brown door with the exit sign above it.
"""

    system_prompt = """
You are an expert at converting natural language navigation instructions into detailed, logically ordered sub-instructions for agents.

Your task is to:
- Break down instructions into a sequence of minimal, goal-directed steps.
- Make all implicit temporal or spatial relationships explicit.
- Preserve execution order by reconstructing intermediate actions that are implied, not directly stated.

Use the following logic:
- (A) --> [after / then / once / as soon as] --> (B): Do A fully, then B.
- (B) --> [before] --> (A): Move toward A, then perform B at a point prior.
- (A) --> [until] --> (B): Continue A until B is reached.
- Avoid "then", "before", "until", "once" etc. in the output.

Format:
- Single sentence, steps separated by periods.
- Each step must be minimal, concrete, and goal-focused.
"""

    def call_openai(instr):
        user_prompt = f"{few_shots_examples}\nInstruction: \"{instr}\"\n\nOutput:"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        for attempt in range(max_retries):
            try:
                res = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return res.choices[0].message.content.strip()
            except Exception as e:
                print(f"[Retry {attempt + 1}] Failed for: {instr[:60]}... Error: {e}")
                time.sleep(1)
        return ""

    results = ["" for _ in instructions]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_idx = {
            executor.submit(call_openai, instr): idx
            for idx, instr in enumerate(instructions)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                print(f"Failed to generate for index {idx}: {e}")
                results[idx] = ""

    return results

if __name__ == "__main__":
    # _model, _tokenizer = get_intern_model()
    # print("InternVL model loaded successfully")
    
    # _qwen_model, _qwen_processor = _get_qwen_model()
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    # "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    # )
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # print("Qwen model loaded successfully")
    
    
    instructions = ['Wall through the room to the next set of doors. You will see a gray plaque saying "SALA XVI" on the floor to your left. If you don\'t, you went the wrong way. Turn around and go the correct way. ', 'Make a left at the doorway leading to the stairs. Walk around the right side of the rail. Make a left into the room with the French doors. ', 'Exit the bedroom, and walk forward. Turn left at the corner, and make another left inside the room that has the big black alphabet frame on the wall in the outside. Enter the room and stop there. ', 'Turn to your left and exit the room out of the door beside the wooden drawers. Once out of the room walk across the small area and through the next entry way on the left. Stop inside the room before you get to the doors leading outside. ', 'Walk past the stairs into the sitting area. Wait on the other side of the sitting area near the exterior doors. ', 'Face sink and turn right.  Exit the room through the door. Turn right  and walk between chairs and couch. Stop near round table. ', 'continue down stairs, walk straight into lobby stop at tree. ', 'Go through the door on the right, then turn left and walk down the hallway. Walk just past the table and then stop. ']
    instructions_plans = generate_temporal_instructions(instructions, max_retries=3, max_tokens=2000, temperature=0, model="gpt-4o", num_threads=4)
    print("Generated temporal instructions:")
    for instr, plan in zip(instructions, instructions_plans):
        print('-'*20)
        print(f"Instruction: {instr}\nPlan: {plan}\n")