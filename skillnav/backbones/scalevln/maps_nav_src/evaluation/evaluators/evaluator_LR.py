import json
import os
import numpy as np
from .evaluator_base import Evaluator
from utils.math_utils import euclidean_distance, is_projection_inside_segment
from collections import defaultdict

class Evaluator_LR(Evaluator):
    def __init__(self, graphs, paths, distances, annt_file, submission_file) -> None:
        super().__init__(graphs, paths, distances, annt_file, submission_file)
        self.id = "Landmark Recognition"

    def eval(self):
        metrics = defaultdict(list)
        for instr_id, pred_path in self.preds.items():
            traj_id = instr_id.split('_')[0]

            datum_annt = self.meta[traj_id]
            traj_scores = self._eval_item(datum_annt, pred_path)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            
            metrics['instr_id'].append(traj_id)
        
        avg_metrics = {
            'sr': np.mean(metrics['success']) * 100,
            'success_towards': np.mean(metrics['success_towards']) * 100,
            'success_past': np.mean(metrics['success_past']) * 100,
            'num_paths': len(metrics['success'])
        }
        return avg_metrics

    def _eval_item(self, datum_annt, pred_path):
        scores = {}
        scan = datum_annt['scan']
        graph = self.graphs[scan]
        instruction = datum_annt['instructions'][0]

        path = sum(pred_path, [])

        deduplicate_vps = []
        deduplicate_vps.append(path[0])
        for i in range(1, len(path)):
            if path[i] == deduplicate_vps[-1]:
                continue
            deduplicate_vps.append(path[i])

        start_vp = datum_annt['path'][0]
        start_pos = np.array(graph.nodes[start_vp]['position'])
        end_vp = deduplicate_vps[-1]
        end_pos = np.array(graph.nodes[end_vp]['position'])
        obj_center = datum_annt['obj_center']
        sr = 0
        if "walk towards" in instruction:
            dist1 = euclidean_distance(start_pos[:2], obj_center[:2])
            dist2 = euclidean_distance(end_pos[:2], obj_center[:2])
            if dist2 < dist1:
                sr = 1
            scores['success'] = sr
            scores['success_towards'] = sr
    
        elif 'walk past' in instruction:
            # angle measure
            if is_projection_inside_segment(start_pos[:2], end_pos[:2], obj_center[:2]): #  likely checks if the object is “between” the start and end positions of the trajectory, i.e., if the trajectory potentially intersects with the object in the 2D plane.
                # distance measure
                if euclidean_distance(end_pos[:2], obj_center[:2]) < 3:
                    sr = 1
            scores['success'] = sr
            scores['success_past'] = sr
        else:
            raise ValueError(f"Valid action (walk towards, walk past) for {self.id} not in the instruction")
        return scores
