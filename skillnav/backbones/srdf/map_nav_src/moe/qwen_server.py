import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append("../map_nav_src")
sys.path.append("../map_nav_src/moe")

from fastapi import FastAPI, UploadFile, File, Form
from typing import List

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import logging
logging.basicConfig(level=logging.INFO)


from router_qwen import qwen_infer
from router_qwen import identify_instructions_with_qwen, analyze_with_qwen
import importlib
# Reload the batch module first
router_qwen_batch = importlib.import_module("router_qwen_batch")
importlib.reload(router_qwen_batch)
# THEN import the functions from the reloaded module
from router_qwen_batch import (
    identify_instructions_with_qwen_infer_batch,
    analyze_with_qwen_infer_batch
)

from pydantic import BaseModel
from typing import List

from asyncio import gather, to_thread


class IdentifyRequest(BaseModel):
    instructions: List[str]
    previous_viewpoint_lists: List[List[str]]
    max_tokens: int = 10000  # default max tokens for inference

class AnalyzeRequest(BaseModel):
    instructions: List[str]
    sub_instructions: List[str]
    reasonings: List[str]
    candidate_viewpoint_lists: List[List[str]]  # list of lists of strings
    max_tokens: int = 10000


app = FastAPI()

# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    # "Qwen/Qwen2.5-VL-32B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
qwen_processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    # "Qwen/Qwen2.5-VL-32B-Instruct",
    trust_remote_code=True
)
        
def get_qwen_model():
    global model, processor
    if model is None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )
    return model, processor


@app.post("/generate-vl")
async def generate_vl(
    user_prompt: str = Form(...),
    image_list: List[str] = Form(...),  # list of image paths as strings
    system_prompt: str = Form("You are a helpful assistant.")
):
    global qwen_model, qwen_processor  # <-- 🔥 Add this line
    
    # model, processor = get_qwen_model()
    print("Received image_list:", image_list)
    print("Type:", type(image_list), "Length:", len(image_list))
    try:
        # Call your inference function with image paths
        answer, token_info = qwen_infer(
            model=qwen_model,
            processor=qwen_processor,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_list=image_list,  # list of paths like ["img1.jpg", "img2.jpg"]
            max_tokens=2000
        )
    except Exception as e:
        return {
            "error": str(e)
        }

    return {
        "response": answer,
        "token_info": token_info
    }
    
    
@app.post("/generate-vl-batch")
async def generate_vl_batch(
    user_prompts: List[str] = Form(...),
    image_lists: List[str] = Form(...),  # comma-separated strings of paths
    system_prompt: str = Form("You are a helpful assistant.")
):
    """
    Batch-inference endpoint: multiple prompts and corresponding image path lists.
    `image_lists` must match `user_prompts` in length.
    Each image list is a comma-separated string (e.g., "img1.jpg,img2.jpg")
    """
    global qwen_model, qwen_processor  # <-- 🔥 Add this line
    
    if len(user_prompts) != len(image_lists):
        return {"error": "user_prompts and image_lists must have the same length."}

    results = []
    try:
        for prompt, image_str in zip(user_prompts, image_lists):
            image_paths = [s.strip() for s in image_str.split(",") if s.strip()]
            answer, token_info = qwen_infer(
                model=qwen_model,
                processor=qwen_processor,
                system_prompt=system_prompt,
                user_prompt=prompt,
                image_list=image_paths,
                max_tokens=3000
            )
            results.append({"prompt": prompt, "response": answer, "token_info": token_info})
    except Exception as e:
        return {"error": str(e)}

    return {"results": results}


@app.post("/identify-instruction-batch")
async def identify_instructions_with_qwen_batch(request: IdentifyRequest):
    global qwen_model, qwen_processor  # <-- 🔥 Add this line

    instructions = request.instructions
    previous_viewpoint_lists = request.previous_viewpoint_lists
    max_tokens = request.max_tokens

    '''
    results = identify_instructions_with_qwen_infer_batch(
        model=qwen_model,
        processor=qwen_processor,
        instructions=instructions,
        previous_viewpoint_lists=previous_viewpoint_lists,
        max_tokens=max_tokens
    )
    
    return {"results": results}
    '''
    
    # '''
    results = []
    for instr, prev_imgs in zip(instructions, previous_viewpoint_lists):
        # Ensure image paths are real and accessible
        for img_path in prev_imgs:
            assert os.path.exists(img_path), f"Missing image: {img_path}"
        
        # print('-'*20)
        # print("Processing instruction:", instr)
        # print("Previous viewpoint image list:", prev_imgs)        
        result = identify_instructions_with_qwen(
            model=qwen_model,
            processor=qwen_processor,
            instruction=instr,
            previous_viewpoint_list=prev_imgs,
            max_tokens=max_tokens
        )
        # print("Identify Result:", result)
        results.append(result)
    
    return {"results": results}
    # '''
    

@app.post("/analyze-skill-batch")
async def analyze_with_qwen_batch(request: AnalyzeRequest):
    global qwen_model, qwen_processor  # <-- 🔥 Add this line

    instructions = request.instructions
    sub_instructions = request.sub_instructions
    reasonings = request.reasonings
    candidate_viewpoint_lists = request.candidate_viewpoint_lists
    max_tokens = request.max_tokens

    if len(sub_instructions) != len(candidate_viewpoint_lists):
        return {"error": "sub_instructions and candidate_viewpoint_lists must have the same length."}

    '''
    try:
        results = analyze_with_qwen_infer_batch(
            model=qwen_model,
            processor=qwen_processor,
            instructions=instructions,
            sub_instructions=sub_instructions,
            reasonings=reasonings,
            candidate_viewpoint_lists=candidate_viewpoint_lists,
            max_tokens=max_tokens
        )
    except Exception as e:
        return {"error": str(e)}
    
    return {"results": results}
    '''
    
    # '''
    results = []

    try:
        for instruction, sub_instruction, reasoning, candidate_viewpoints in zip(
            instructions, sub_instructions, reasonings, candidate_viewpoint_lists
        ):
            
            # Clean candidate viewpoint strings
            candidate_viewpoints = [s.strip() for s in candidate_viewpoints if s.strip()]
            
            # Ensure image paths are real and accessible
            for img_path in candidate_viewpoints:
                assert os.path.exists(img_path), f"Missing image: {img_path}"

            # Run inference
            answer = analyze_with_qwen(
                model=qwen_model,
                processor=qwen_processor,
                instruction=instruction,
                sub_instruction=sub_instruction,
                reasoning=reasoning,
                candidate_viewpoint_list=candidate_viewpoints,
                max_tokens=max_tokens
            )

            results.append(answer)

    except Exception as e:
        return {"error": str(e)}

    return {"results": results}
    # '''
    
