import numpy as np
from .evaluator_base import Evaluator
from utils.math_utils import check_angle_between_vectors
from collections import defaultdict

ERROR_MARGIN = 3.0
class Evaluator_DC(Evaluator):
    def __init__(self, graphs, paths, distances, annt_file, submission_file) -> None:
        super().__init__(graphs, paths, distances, annt_file, submission_file)
        self.id = "Direction Change"

    def eval(self):
        metrics = defaultdict(list)
        for instr_id, pred_path in self.preds.items():
            traj_id = instr_id.split('_')[0]

            datum_annt = self.meta[traj_id]
            traj_scores = self._eval_item(datum_annt, pred_path)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            
            metrics['instr_id'].append(traj_id)

        # Paired success rate for left/right at same start orientation
        paired_success = defaultdict(list) 
        skip_idx = set()
        for i, instr_id in enumerate(metrics['instr_id']):
            if i in skip_idx:
                continue
            pair_id, _, path_id = instr_id.rsplit('-',2)
            if path_id == '1': # turn left
                inv_path_id = '0'
            elif path_id == '0': # turn right
                inv_path_id = '1'
            elif path_id == '2': # turn around
                continue
            else:
                raise ValueError("Invalid instr_id for direction change category")
            
            query_key = "-".join([pair_id, "path", inv_path_id])
            idx = metrics['instr_id'].index(query_key)
            skip_idx.add(idx)
            skip_idx.add(i)

            paired_success[pair_id].append(metrics['success'][idx]) 
            paired_success[pair_id].append(metrics['success'][i]) 

        metrics['pair_success'] = []
        for k,v in paired_success.items():
            if len(v) == 2:
                if np.all(v == [1., 1.]):
                    metrics['pair_success'].append(1.)
                else:
                    metrics['pair_success'].append(0.)
            else:
                raise Exception("Some items are not evaluated")

        print(len(metrics['success_left']), len(metrics['success_right']), len(metrics['success_around']))
        avg_metrics = {
            'sr': np.mean(metrics['success']) * 100,
            'sr_left': np.mean(metrics['success_left']) * 100,
            'sr_right': np.mean(metrics['success_right']) * 100,
            'sr_around': np.mean(metrics['success_around']) * 100,
            'pair_sr': np.mean(metrics['pair_success']) * 100,
            'num_paths': len(metrics['success'])
        }
        return avg_metrics

    def _eval_item(self, datum_annt, pred_path):
        scores = {}
        scan = datum_annt['scan'] 
        graph = self.graphs[scan]
        data = datum_annt

        instruction = data['instruction_template']
        path = sum(pred_path, [])

        deduplicate_vps = []
        deduplicate_vps.append(path[0])
        for i in range(1, len(path)):
            if path[i] == deduplicate_vps[-1]:
                continue
            deduplicate_vps.append(path[i])
        path = deduplicate_vps

        if len(path) == 1 or path[0] == path[1]:
            if "turn left" in instruction:
                scores['success'] = 0.
                scores['success_left'] = 0.
            elif "turn right" in instruction:
                scores['success'] = 0.
                scores['success_right'] = 0.
            elif "turn around" in instruction:
                scores['success'] = 0.
                scores['success_around'] = 0.
            else:
                raise ValueError()
            return scores
        
        pos0 = graph.nodes[data['gt_vp'][0]]['position']
        if 'heading_vp' in data.keys():
            heading_vec = graph.nodes[data['heading_vp']]['position'] - pos0
        else:
            pos_11 = graph.nodes[data['gt_vp'][1]]['position']
            pos_12 = graph.nodes[data['vp_neg'][1]]['position']
            heading_vec =  (pos_11 + pos_12)/2 - pos0

        v1 = graph.nodes[path[1]]['position'] - graph.nodes[path[0]]['position']

        is_left = False
        if ((heading_vec[0]) * (v1[1]) - (heading_vec[1]) * (v1[0])) > 0:
            is_left = True

        degrees = check_angle_between_vectors(heading_vec, v1)

        is_around = False
        if degrees > 120: # definition of turn around
            is_around = True

        if "turn left" in instruction:
            if is_left and not is_around:
                scores['success'] = 1.
                scores['success_left'] = 1.
            else:
                scores['success'] = 0.
                scores['success_left'] = 0.
        elif "turn right" in instruction:
            if not is_left and not is_around:
                scores['success'] = 1.
                scores['success_right'] = 1.
            else:
                scores['success'] = 0.
                scores['success_right'] = 0.
        elif "turn around" in instruction:
            if is_around:
                scores['success'] = 1.
                scores['success_around'] = 1.
            else:
                scores['success'] = 0.
                scores['success_around'] = 0.
        else:
            raise ValueError(f"Using incorrect evaluator, using direction change for text {instruction}")
        
        return scores
