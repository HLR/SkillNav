import json
import os
import numpy as np
from .evaluator_base import Evaluator
from utils.math_utils import ndtw
from collections import defaultdict

ERROR_MARGIN = 3.0
def get_locations(G, view_pts):
    locations = []
    locations.append(np.array(G.nodes[view_pts[0]]["position"]))
    for i in range(1, len(view_pts)):
        if view_pts[i-1] == view_pts[i]:
            continue
        locations.append(np.array(G.nodes[view_pts[i]]["position"]))
    return locations

class Evaluator_NU(Evaluator):
    def __init__(self, graphs, paths, distances, annt_file, submission_file) -> None:
        super().__init__(graphs, paths, distances, annt_file, submission_file)
        self.id = "Numerical Directional Region"

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
            'path_SR': np.mean(metrics['path_SR']) * 100,
            'nDTW': np.mean(metrics['nDTW']) * 100,
            'num_paths': len(metrics['path_SR'])
        }
        return avg_metrics

    def _eval_item(self, datum_annt, pred_path):
        scores = {}
        scan = datum_annt['scan'] 
        graph = self.graphs[scan]
        sim_distances = self.distances[scan]

        parts = datum_annt['path_id'].split('-')
        path_key = f"{parts[-2]}-{parts[-1]}"

        path = sum(pred_path, []) # pred_path is a list of lists of predicted view points. The sum(pred_path, []) flattens it into a single list.
        # removes consecutive duplicate view points in the predicted path to ensure no redundant positions are considered. 
        deduplicate_vps = []
        deduplicate_vps.append(path[0])
        for i in range(1, len(path)):
            if path[i] == deduplicate_vps[-1]:
                continue
            deduplicate_vps.append(path[i])
        
        distances = []
        for i, gt_vp in enumerate(datum_annt['set_paths'][path_key]): 
            distances.append(sim_distances[gt_vp[-1]][deduplicate_vps[-1]])
        distance = np.min(distances)

        gt_idx = distances.index(distance)
        ndtw_to_path_curr = ndtw(get_locations(graph, deduplicate_vps), get_locations(graph, datum_annt['set_paths'][path_key][gt_idx]))
        ndtw_negs = []
        for k in datum_annt['set_paths'].keys():
            if k != path_key:
                for neg_vp in datum_annt['set_paths'][k]:
                    ndtw_negs.append(ndtw(get_locations(graph, deduplicate_vps), get_locations(graph, neg_vp)))
        ndtw_to_path_neg = np.max(ndtw_negs)

        goal_success = int(float(distance) <= ERROR_MARGIN)
        ndtw_success = int(ndtw_to_path_curr > ndtw_to_path_neg)

        scores['goal_SR'] = goal_success
        scores['path_SR'] = goal_success * ndtw_success
        scores['nDTW'] = float(ndtw_to_path_curr)
        return scores
