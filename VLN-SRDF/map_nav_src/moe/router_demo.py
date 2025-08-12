import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append("../map_nav_src")
sys.path.append("../map_nav_src/moe")

import httpx
import json
import asyncio

from utils.get_images import load_vp_lookup, convert_path2img, extract_cand_img
from router_qwen import extract_from_response
from llm_utils import generate_temporal_instructions
import time

async def identify_instructions_with_qwen_batch(
    instructions,
    previous_viewpoint_lists,
    server_url="http://localhost:8001/identify-instruction-batch",
):
    payload = {
        "instructions": instructions,
        "previous_viewpoint_lists": previous_viewpoint_lists
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(server_url, json=payload)
        response.raise_for_status()
        return response.json()["results"]
    

async def analyze_with_qwen_batch(
    instructions,
    sub_instructions,
    reasonings,
    candidate_viewpoint_lists,
    server_url="http://localhost:8001/analyze-skill-batch",
):
    payload = {
        "instructions": instructions,
        "sub_instructions": sub_instructions,
        "reasonings": reasonings,
        "candidate_viewpoint_lists": candidate_viewpoint_lists
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(server_url, json=payload)
        response.raise_for_status()
        return response.json()["results"]
        # return response.json() # if there is 'error' instead of 'results'


async def process_batch_navigation(instructions, previous_viewpoint_lists, candidate_viewpoint_lists):
    loaded_weights = []
    print('-'*20)
    try:
        identify_results_raw = await identify_instructions_with_qwen_batch(
            instructions, previous_viewpoint_lists
        )
        print(type(identify_results_raw), identify_results_raw)
        
    except Exception as e:
        print(f"Failed to identify instructions: {e}")
        return [], []


    # Ensure each result is a parsed dict
    identify_results = []
    for i, r in enumerate(identify_results_raw):
        # print(type(r),r)
        
        if isinstance(r, str):
            try:
                r = json.loads(r)
            except json.JSONDecodeError:
                print(f"Failed to decode identify_result at index {i}: {r}")
                r = {}
        identify_results.append(r)

    sub_instrs = [r.get("Sub-instruction to be executed", "") for r in identify_results]
    reasonings = [r.get("Reasoning", "") for r in identify_results]
    
    print('-'*20)
    print('Instructions:', instructions)
    print("Sub-instructions:", sub_instrs)
    print("Reasonings:", reasonings)
    print("Candidate Viewpoint Lists:", candidate_viewpoint_lists)

    skill_routings = await analyze_with_qwen_batch(
        instructions, sub_instrs, reasonings, candidate_viewpoint_lists
    )
    
    print('-'*20)
    print("Skill Routings:", skill_routings)

    for i in range(len(instructions)):
        _, _, weights =extract_from_response(skill_routings[i])

        total = sum(weights)
        normalized_weights = [w / total if total > 0 else 1.0 / len(weights) for w in weights]
            
        loaded_weights.append(normalized_weights)
        
    return sub_instrs, loaded_weights


if __name__ == "__main__":
    start_time = time.time()
    vp_lookup = load_vp_lookup()

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
        
    instructions = []
    previous_viewpoint_lists = []
    candidate_viewpoint_lists = []

    for item in data:
        instruction = item['instruction']
        instructions.append(instruction)
        
        paths = item['traj_gt']
        original_image_list = convert_path2img(paths, item['scan'], vp_lookup)
        
        # print('-'*20)
        # print(f"Original Image List from Path: {original_image_list}")
        
        # for idx, viewpoint in enumerate(paths):
        #     if idx == 0:
        #         continue
        
        previous_viewpoint_list = convert_path2img(paths[:2], item['scan'], vp_lookup)
        candidate_viewpoint_list = extract_cand_img(paths[2], item['scan'], vp_lookup)
        previous_viewpoint_lists.append(previous_viewpoint_list)
        candidate_viewpoint_lists.append(candidate_viewpoint_list)

    # print('-'*20)        
    # print(instructions)
    # print(f"Previous viewpoint lists: {previous_viewpoint_lists}")
    # print('-'*20)
    # print(f"Candidate viewpoint lists: {candidate_viewpoint_lists}")

    instructions_plans = generate_temporal_instructions(instructions, max_retries=3, max_tokens=2000, temperature=0, model="gpt-4o", num_threads=4)
    
    sub_instructions, loaded_weights = asyncio.run(
                process_batch_navigation(
                    instructions_plans, previous_viewpoint_lists, candidate_viewpoint_lists
                )
            )

    print('-'*20)
    print("Sub-instructions:", sub_instructions)
    print("Loaded Weights:", loaded_weights)

    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")


    '''
    data = {
        'user_prompt': 'What are the difference among these images?',
        'system_prompt': 'You are a helpful assistant.',
        'image_list': [
            '/home/matiany3/MapGPT/datasets/R2R/generated_images/1LXtFkjw3qL/0b22fa63d0f54a529c525afbf2e8bb25/12.jpg',
        '/home/matiany3/MapGPT/datasets/R2R/generated_images/1LXtFkjw3qL/0b22fa63d0f54a529c525afbf2e8bb25/17.jpg'
    ]
    }

    # Set a higher timeout (e.g., 60 seconds)
    timeout = httpx.Timeout(60.0, connect=10.0)

    res = httpx.post("http://localhost:8001/generate-vl", 
                    data=data,
                    timeout=timeout)
    print(res.status_code)
    print(res.json())
    '''

