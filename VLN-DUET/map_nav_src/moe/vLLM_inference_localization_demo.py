import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append("/home/matiany3/ScaleVLN/VLN-DUET/map_nav_src")
sys.path.append("/home/matiany3/ScaleVLN/VLN-DUET/map_nav_src/moe")

from moe.vLLM_API import load_vLLM_model, infer_with_vllm_instruction_localization
from utils.get_images import load_vp_lookup, convert_path2img, extract_cand_img

if __name__ == "__main__":

    vp_lookup = load_vp_lookup()
    
    llm_vl = load_vLLM_model(
        # model_ckpt="Qwen/Qwen2.5-VL-7B-Instruct",
        # model_ckpt="llava-hf/llava-1.5-7b-hf",
        # model_ckpt="OpenGVLab/InternVL2_5-4B",
        model_ckpt="Qwen/Qwen2-VL-7B-Instruct",
        # model_ckpt="deepseek-ai/deepseek-vl2-tiny",
        # model_ckpt="THUDM/GLM-4.1V-9B-Thinking",
        gpu_memory_utilization=0.6,
        limit_mm_per_prompt=15,  # Limit in MB per prompt
        seed=0,
        tensor_parallel_size=1,  # Number of GPUs to use
        max_num_seqs=4
    )
    
    data = [
            {
                "scan": "QUCTc6BB5sX",
                "instr_id": "6440_2",
                "instruction": "With the storage lockers to your right, exit the garage via the door that's ahead of you and against the right hand wall. After exiting the garage, turn right ninety degrees and move forward until arched doorway is to your immediate right. ",
                "traj_gt": [
                "caf7c76c38a94f02943720cde114cc4f",
                "414195b76186408b81db63defa6b8d83",
                "d30cfd0e67de459bb27053a7682c0104",
                "e049cbd79d51410e91b09ab7fd2074ec",
                "3cdccfc437a14153939cf51089407b43"
                ],
                "traj_fail": [
                "caf7c76c38a94f02943720cde114cc4f",
                "414195b76186408b81db63defa6b8d83",
                "d30cfd0e67de459bb27053a7682c0104",
                "e049cbd79d51410e91b09ab7fd2074ec",
                "d30cfd0e67de459bb27053a7682c0104",
                "e9003c50993947ab886bd2a6d7b30989"
                ],
                "traj_success": [
                "caf7c76c38a94f02943720cde114cc4f",
                "414195b76186408b81db63defa6b8d83",
                "d30cfd0e67de459bb27053a7682c0104",
                "e049cbd79d51410e91b09ab7fd2074ec",
                "3cdccfc437a14153939cf51089407b43"
                ]
            },
            {
                "scan": "2azQ1b91cZZ",
                "instr_id": "2665_2",
                "instruction": "At the bottom of the stairs, go through the nearest archway to your left. Head straight until you enter the room with a pool table. Step slightly to the left to get out of the way. ",
                "traj_gt": [
                "9f0079fa767e402cb515c7751a13e265",
                "a868c39ea01143fcbaca1f255f9e1178",
                "c56e92a10dda45a0a27fe34224c8294e",
                "e25cb07854e64017ba4282afbebd4d53",
                "d6fcbe8ab9bb402d857f3a0022ec8a07",
                "97eb119eb9b94677bea3af3079620966",
                "3fb5a48d8a71413aacaa51f6bc569e59"
                ],
                "traj_fail": [
                "9f0079fa767e402cb515c7751a13e265",
                "f672e020e2a043b2ac119e8d09e7df89",
                "bd0fed0d97ec441ea72b98bbc1ca0a79",
                "c9f0045f40984820ac2bef8de7ce9dfc",
                "49b3c844c10c4f1f8e36342aedb7bf94",
                "6e95a11ced8b44b39fc3d795b7b32721",
                "a9d9d5e3d5e44d5080b643057d25fc27",
                "23e06a377e3a4c87ab4c2c2a47b227ce",
                "a79f7b47f6c047fc991840810e24e4e4",
                "a8037d63a223407e89b8a7c9bb560c3f",
                "a79f7b47f6c047fc991840810e24e4e4",
                "c3f77143c7ea412fa56c31817b0732b2",
                "73a5096ab14842e9b2091fc1ad4f43cb",
                "73a5096ab14842e9b2091fc1ad4f43cb",
                "c3f77143c7ea412fa56c31817b0732b2",
                "6e95a11ced8b44b39fc3d795b7b32721"
                ],
                "traj_success": [
                "9f0079fa767e402cb515c7751a13e265",
                "a868c39ea01143fcbaca1f255f9e1178",
                "c56e92a10dda45a0a27fe34224c8294e",
                "e25cb07854e64017ba4282afbebd4d53",
                "d6fcbe8ab9bb402d857f3a0022ec8a07",
                "97eb119eb9b94677bea3af3079620966",
                "3fb5a48d8a71413aacaa51f6bc569e59"
                ]
            },
            {
                "scan": "X7HyMhZNoso",
                "instr_id": "1404_1",
                "instruction": "Walk onto the grass and up the stairs. Enter the house. ",
                "traj_gt": [
                "fc8b960dcf5243eab12f5b4e4b2c0160",
                "5b8db88286b44218bb317abdfab54f8f",
                "e24dec06f43b4a36abe526aafe9e3709",
                "5445d1e47e204f598d836d7940013231",
                "997ec56720304a069672a8a0fe2b80e6",
                "0990cc040127481d97727123df0c9e56",
                "2739968bfacb412cb7997d6d59f461c2"
                ],
                "traj_fail": [
                "fc8b960dcf5243eab12f5b4e4b2c0160",
                "62bed01b816d45f8b42229e495b1faa1",
                "c7f4af49f3ce490a977e282f2a479266",
                "62bed01b816d45f8b42229e495b1faa1",
                "5445d1e47e204f598d836d7940013231",
                "c369774dbf94451388cbd59a7d9341ea",
                "997ec56720304a069672a8a0fe2b80e6",
                "0b124e1ec3bf4e6fb2ec42f179cc9ff0",
                "59076fa091d1423e893a1122e42e93d2",
                "0b124e1ec3bf4e6fb2ec42f179cc9ff0",
                "2739968bfacb412cb7997d6d59f461c2",
                "ecdb8d01949647b38464f99e427f53ea",
                "0990cc040127481d97727123df0c9e56",
                "997ec56720304a069672a8a0fe2b80e6"
                ],
                "traj_success": [
                "fc8b960dcf5243eab12f5b4e4b2c0160",
                "5b8db88286b44218bb317abdfab54f8f",
                "fc8b960dcf5243eab12f5b4e4b2c0160",
                "c7f4af49f3ce490a977e282f2a479266",
                "62bed01b816d45f8b42229e495b1faa1",
                "997ec56720304a069672a8a0fe2b80e6",
                "0990cc040127481d97727123df0c9e56",
                "0b124e1ec3bf4e6fb2ec42f179cc9ff0"
                ]
            },
            {
                "scan": "2azQ1b91cZZ",
                "instr_id": "878_2",
                "instruction": "Go forward through an archway towards the console table, and veer right to an open doorway, and go through it. Wait there. ",
                "traj_gt": [
                "2d9efbd449f54a8ca2d563f9fab3e9bc",
                "64c00dea4b1a41a98bd439d56b753283",
                "6a512589ab024c6b899b73af496b0019",
                "ffc16df830484bdf97716d0858568965",
                "51e50a1c1dda46db856161cdb3fded5e",
                "d5d55adb422942ee92a5f92bbbf5bb03"
                ],
                "traj_fail": [
                "2d9efbd449f54a8ca2d563f9fab3e9bc",
                "29d286b3af0a4a49a162b1481b5c8127",
                "43b53aa5b25a42a692edfa432d7cae80",
                "55bb9a9d764e41b98f7f1c1843885baa",
                "69d18f7069e44875bc6459bb25804499",
                "a313bf96ab8b4693b32d558920f4b4cb",
                "fe326a17d5f44104befb9c5a8da24127",
                "f113a975447c4f9dbac6564c10ed67d1",
                "0ae9a10c4c974a6f94b251899e1c3322"
                ],
                "traj_success": [
                "2d9efbd449f54a8ca2d563f9fab3e9bc",
                "64c00dea4b1a41a98bd439d56b753283",
                "6a512589ab024c6b899b73af496b0019",
                "ffc16df830484bdf97716d0858568965",
                "51e50a1c1dda46db856161cdb3fded5e"
                ]
            },
        ]
    
    # # --- Manully
    # batch_inputs_instruction_localization = []
    # for item in data:
    #     new_item = item.copy()
        
    #     instruction = item['instruction']
    #     new_item['full_instruction'] = instruction
    #     new_item['previous_sub_instruction_list'] = []
    #     del new_item['instruction']
        
    #     paths = item['traj_gt']
    #     # original_image_list = convert_path2img(paths, item['scan'], vp_lookup)
    #     # new_item['previous_viewpoint_list'] = convert_path2img(paths[:1], item['scan'], vp_lookup)
    #     new_item['previous_viewpoint_list'] = []
        
    #     batch_inputs_instruction_localization.append(new_item)
    
    # # Step 1
    # results = infer_with_vllm_instruction_localization(
    #     llm=llm_vl,
    #     # prompt_template_path="/home/matiany3/ScaleVLN/VLN-DUET/map_nav_src/prompts/localization_template.txt",
    #     prompt_template_path="/home/matiany3/ScaleVLN/VLN-DUET/map_nav_src/prompts/localization_template_add_prev_subinstruction.txt",
    #     batch_inputs=batch_inputs_instruction_localization,
    #     max_tokens=3000
    # )
    
    # for idx, result in enumerate(results):
        
    #     print('-'*20)   
    #     print(batch_inputs_instruction_localization[idx]['full_instruction'])
    #     print(batch_inputs_instruction_localization[idx]['previous_sub_instruction_list'])
    #     print(batch_inputs_instruction_localization[idx]['previous_viewpoint_list'])
    #     print(f"ob {idx}:\n{result}")
        
    #     sub_instruction = result['Sub-instruction to be executed']
    #     batch_inputs_instruction_localization[idx]['previous_sub_instruction_list'].append(sub_instruction)
    #     batch_inputs_instruction_localization[idx]['previous_viewpoint_list']=convert_path2img(paths[:1], item['scan'], vp_lookup)
    
    # # Step 2 
    # results = infer_with_vllm_instruction_localization(
    #     llm=llm_vl,
    #     # prompt_template_path="/home/matiany3/ScaleVLN/VLN-DUET/map_nav_src/prompts/localization_template.txt",
    #     prompt_template_path="/home/matiany3/ScaleVLN/VLN-DUET/map_nav_src/prompts/localization_template_add_prev_subinstruction.txt",
    #     batch_inputs=batch_inputs_instruction_localization,
    #     max_tokens=3000
    # )
    
    # for idx, result in enumerate(results):
    #     print('-'*20)
           
    #     print(batch_inputs_instruction_localization[idx]['full_instruction'])
    #     print(batch_inputs_instruction_localization[idx]['previous_sub_instruction_list'])
    #     print(batch_inputs_instruction_localization[idx]['previous_viewpoint_list'])

    #     print(f"ob {idx}:\n{result}")
        
    #     sub_instruction = result['Sub-instruction to be executed']
    #     batch_inputs_instruction_localization[idx]['previous_sub_instruction_list'].append(sub_instruction)
    #     batch_inputs_instruction_localization[idx]['previous_viewpoint_list']=convert_path2img(paths[:2], item['scan'], vp_lookup)
    
    # # Step 3
    # results = infer_with_vllm_instruction_localization(
    #     llm=llm_vl,
    #     # prompt_template_path="/home/matiany3/ScaleVLN/VLN-DUET/map_nav_src/prompts/localization_template.txt",
    #     prompt_template_path="/home/matiany3/ScaleVLN/VLN-DUET/map_nav_src/prompts/localization_template_add_prev_subinstruction.txt",
    #     batch_inputs=batch_inputs_instruction_localization,
    #     max_tokens=3000
    # )
    
    # for idx, result in enumerate(results):
    #     print('-'*20)
           
    #     print(batch_inputs_instruction_localization[idx]['full_instruction'])
    #     print(batch_inputs_instruction_localization[idx]['previous_sub_instruction_list'])
    #     print(batch_inputs_instruction_localization[idx]['previous_viewpoint_list'])

    #     print(f"ob {idx}:\n{result}")
    
    # --- Loop
    batch_inputs_instruction_localization = []

# Initialize input batch
for item in data:
    new_item = item.copy()
    instruction = item['instruction']
    new_item['full_instruction'] = instruction
    new_item['previous_sub_instruction_list'] = []
    new_item['previous_viewpoint_list'] = []
    del new_item['instruction']
    batch_inputs_instruction_localization.append(new_item)

# Determine maximum number of steps to loop through (e.g., max trajectory length)
max_steps = max(len(item['traj_gt']) for item in data)

# Main reasoning loop
for step in range(max_steps):
    print('-' * 50)
    print(f"Step {step + 1}/{max_steps}")
    results = infer_with_vllm_instruction_localization(
        llm=llm_vl,
        prompt_template_path="/home/matiany3/ScaleVLN/VLN-DUET/map_nav_src/prompts/localization_template_add_prev_subinstruction.txt",
        batch_inputs=batch_inputs_instruction_localization,
        max_tokens=3000
    )

    for idx, result in enumerate(results):
        item = data[idx]
        paths = item['traj_gt']

        print('-' * 20)
        print(f"ob {idx} Input:\n")
        print(batch_inputs_instruction_localization[idx]['full_instruction'])
        print(batch_inputs_instruction_localization[idx]['previous_sub_instruction_list'])
        print(batch_inputs_instruction_localization[idx]['previous_viewpoint_list'])
        print(f"ob {idx}:\n{result}")

        # Stop if no new sub-instruction is returned
        sub_instruction = result.get('Sub-instruction to be executed')
        if not sub_instruction:
            continue

        # Append the new sub-instruction
        batch_inputs_instruction_localization[idx]['previous_sub_instruction_list'].append(sub_instruction)

        # Update viewpoint list based on the current step
        if step + 1 <= len(paths):
            batch_inputs_instruction_localization[idx]['previous_viewpoint_list'] = convert_path2img(paths[:step + 1], item['scan'], vp_lookup)