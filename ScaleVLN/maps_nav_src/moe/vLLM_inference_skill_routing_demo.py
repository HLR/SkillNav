import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append("map_nav_src")
sys.path.append("map_nav_src/moe")

from moe.vLLM_API import load_vLLM_model, infer_with_vllm_skill_routing, extract_majority_skill


SKILL_INDEX = {
    "Directional Adjustment": 0,
    "Vertical Movement": 1,
    "Stop and Pause": 2,
    "Landmark Detection": 3,
    "Area and Region Identification": 4,   
}


if __name__ == "__main__":
    
    llm = load_vLLM_model(
        model_ckpt="THUDM/GLM-4.1V-9B-Thinking",
        gpu_memory_utilization=0.6,
        tensor_parallel_size=1,
        seed=42
    )
        
    batch_inputs_skill_routing = [
        {   
            "intru_id": "6753_0",
            "scan": "oLBMNvg9in8",
            "full_instruction": "Walk up all the stairs to the top, and stop at the second stair from the top.",
            "sub_instruction": "Walk up all the stairs to the top.",
            "reasoning": "The agent is currently at the bottom of the stairs, and the next logical step is to ascend the stairs.",
            "skill-routing": {
                "Vertical Movement": "100%",
                "Halt and Pause": "0%"
            }
        },
        {
            "intru_id": "4469_0",
            "scan": "QUCTc6BB5sX",
            "full_instruction": "Turn right, and walk a bit, and turn left. Walk towards the black chairs, and turn right. Stop in front of the refrigerator.",
            "sub_instruction": "Turn right.",
            "reasoning": "The agent has already walked a short distance and turned left, reaching the area with black chairs. The next logical step is to turn right to proceed towards the refrigerator.",
            'skill-routing': {
                'Directional Adjustment': '100%'
            }
        },
        {
            "intru_id": "4728_0",
            "scan": "Z6MFQCViBuw",
            "full_instruction": "Turn left and stop just passed the doorway straight ahead.",
            "sub_instruction": " Stop just past the doorway straight ahead.",
            "reasoning": "The agent has already walked through the doorway and is now positioned just past it, indicating the completion of the navigation.",
            'skill-routing':  {
                'Halt and Pause': '100%'
                }
        }
        
    ]
    
    # prompt_template_path_skill_routing = "prompts/skill_routing_template.txt"
    prompt_template_path_skill_routing = "prompts/skill_routing_template_strong_format.txt"
    # prompt_template_path_skill_routing = "prompts/skill_routing_template_glm.txt"
    # params_type = "BeamSearchParams"  # or "SamplingParams"
    results = infer_with_vllm_skill_routing(
        llm, 
        prompt_template_path_skill_routing, 
        batch_inputs_skill_routing, 
        params_type="BeamSearchParams",
        # params_type="SamplingParams",
        max_tokens=20)



    
    for output in results:
        # prompt = output.prompt
        # generated_text = output.outputs[0].text
        # skill = generated_text.replace("*", "").strip()
        print('-'*20)
        # pprint(f"Prompt: {prompt!r}")
        # print('-'*10)
        print(f"Generated text: {output!r}")
        print('-'*10)
        
        skill = extract_majority_skill(output)
        skill_index = SKILL_INDEX.get(skill, -1)
        print(skill, "->", skill_index)
        print('-'*20)
    