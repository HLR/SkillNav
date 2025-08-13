import os 
import json


### Save lookup
def load_vp_lookup():
    
    save_path = "/home/matiany3/ScaleVLN/VLN-DUET/datasets/R2R/candidates/R2R_vp_lookup.json"
    
    if not os.path.exists(save_path):
        env_list = ["train", "val_seen", "val_unseen", "test"]

        vp_lookup = {}
        for env in env_list:
            candidate_path = f'/home/matiany3/ScaleVLN/VLN-DUET/datasets/R2R/candidates/{env}_candidates_labeled.json'
            with open(candidate_path, 'r') as f:
                candidates = json.load(f)

            scan_items = list(candidates.items())

            for scan_id, viewpoints in scan_items:
                    # Build lookup for fast access
                    vp_lookup[scan_id] = {}
                    for vp_entry in viewpoints:
                        for vp_id, vp_data in vp_entry.items():
                            vp_lookup[scan_id][vp_id] = vp_data

        ### Save the lookup to a JSON file
        with open(save_path, 'w') as f:
            json.dump(vp_lookup, f, indent=4)
        print(f"Saved vp_lookup to {save_path}")
        
    else:
        with open(save_path, 'r') as f:
            vp_lookup = json.load(f)

    return vp_lookup


def convert_path2img(path, scan, vp_lookup):
    image_list = []
    
    for index, viewpoint in enumerate(path):
        if index == 0:
            pass
            # start_image_path = vp_lookup[scan][path[index]]['image_path_start']
            # image_list.append(start_image_path)
        else:
            for cand in vp_lookup[scan][path[index-1]]['candidates']:
                # print(cand['viewpointId'])
                if cand['viewpointId'] == viewpoint:
                    image_list.append(cand['image_path_view'])
                continue
    
    # new_image_list = convert_image_path(image_list)
    return image_list


def extract_cand_img(viewpoint, scan, vp_lookup):
    # Extract the candidates for a given viewpoint
    # candidates_viewpoint_list = []
    candidates_image_list = []
    if scan in vp_lookup and viewpoint in vp_lookup[scan]:
        for cand in vp_lookup[scan][viewpoint]['candidates']:
            # candidates_viewpoint_list.append(cand['viewpointId'])
            candidates_image_list.append(cand['image_path_view'])
        return candidates_image_list
    else:
        print(f"Viewpoint {viewpoint} not found in scan {scan}.")
        return None
    
def extract_cand_heading_elevation(viewpoint, next_viewpoint, scan, vp_lookup):
    # Extract the heading and elevation for a given viewpoint
    if scan in vp_lookup and viewpoint in vp_lookup[scan]:
        for cand in vp_lookup[scan][viewpoint]['candidates']:
            if cand == next_viewpoint:
                heading = cand['heading']
                elevation = cand['elevation']
            else:
                continue
            
        return heading, elevation
    else:
        print(f"Viewpoint {viewpoint} not found in scan {scan}.")
        return None, None
    

if __name__ == "__main__":
    # Example usage
    vp_lookup = load_vp_lookup()
    
    scan = "jtcxE69GiFV"
    print(vp_lookup[scan])
    

#     data = {
#     "scan": "2azQ1b91cZZ",
#     "instr_id": "5389_2",
#     "instruction": "Exit living area, make a left walk forward slightly and wait. ",
#     "traj_gt": [
#       "be8f9fbb02d6432e99ad51bbf570c795",
#       "a32965abf4d149fb8f9a4ebe5706ac04",
#       "8c913cf4718c4a0991fafeb6528417e8",
#       "a65f29102bfd4d2fa38239959b0098d9",
#       "2a47f095717143d989ac571609ffaf0a"
#     ],
#     "traj_fail": [
#       "be8f9fbb02d6432e99ad51bbf570c795",
#       "a32965abf4d149fb8f9a4ebe5706ac04",
#       "8c913cf4718c4a0991fafeb6528417e8",
#       "a65f29102bfd4d2fa38239959b0098d9",
#       "6e6a55cea2ea4235bdeab2545d9af45c",
#       "f5a6564a68744895b25176ee9af53c57",
#       "8b00915324764301b56bc873b06b1b1d"
#     ],
#     "traj_success": [
#       "be8f9fbb02d6432e99ad51bbf570c795",
#       "a32965abf4d149fb8f9a4ebe5706ac04",
#       "8c913cf4718c4a0991fafeb6528417e8",
#       "a65f29102bfd4d2fa38239959b0098d9",
#       "6e6a55cea2ea4235bdeab2545d9af45c"
#     ]
#   }
#     # Example
#     traj_gt = data["traj_gt"]
#     scan = data["scan"]
#     instr_id = data["instr_id"]
    
#     image_list = convert_path2img(traj_gt, scan, vp_lookup)
#     print(image_list)
    
    
#     current_viewpoints = data['traj_success']
#     for current_viewpoint in current_viewpoints:
#         image_list = extract_cand_img(current_viewpoint, scan, vp_lookup)
#         print(f"Candidate Image list for {current_viewpoint}: {image_list}")
    


