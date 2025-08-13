import numpy as np
from .evaluator_base import Evaluator
from collections import defaultdict

class Evaluator_RR(Evaluator):
    def __init__(self, graphs, paths, distances, annt_file, submission_file) -> None:
        super().__init__(graphs, paths, distances, annt_file, submission_file)
        self.id = "Room Recognition"

    def eval(self):
        metrics = defaultdict(list)
        for instr_id, pred_path in self.preds.items():
            traj_id = instr_id.split('_')[0]
            if traj_id not in self.meta.keys():
                continue
            datum_annt = self.meta[traj_id]
            traj_scores = self._eval_item(datum_annt, pred_path)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            
            metrics['instr_id'].append(traj_id)
        
        avg_metrics = {
                'sr': np.mean(metrics['success']) * 100,
                'oracle_success': np.mean(metrics['oracle_success']) * 100,
                'success_into': np.mean(metrics['success_into']) * 100,
                'oracle_success_into': np.mean(metrics['oracle_success_into']) * 100,
                'success_exit': np.mean(metrics['success_exit']) * 100,
                'oracle_success_exit': np.mean(metrics['oracle_success_exit']) * 100,
                'num_paths': len(metrics['success'])
        }
        return avg_metrics

    def _eval_item(self, datum_annt, pred_path):
        scores = {}
        instruction = datum_annt['instructions'][0]

        path = sum(pred_path, [])

        deduplicate_vps = []
        deduplicate_vps.append(path[0])
        for i in range(1, len(path)):
            if path[i] == deduplicate_vps[-1]:
                continue
            deduplicate_vps.append(path[i])

        if 'go into' in instruction:
            oracle_sr = 0
            sr = 0
            for i, vp in enumerate(deduplicate_vps):
                if vp in datum_annt['valid_ends']:
                    oracle_sr = 1
                if i == len(deduplicate_vps)-1:
                    if vp in datum_annt['valid_ends']:
                        sr = 1

            scores['success'] = sr
            scores['oracle_success'] = oracle_sr
            scores['success_into'] = sr
            scores['oracle_success_into'] = oracle_sr
    
        elif 'exit' in instruction:
            oracle_sr = 0
            sr = 0
            for i, vp in enumerate(deduplicate_vps):
                if vp not in datum_annt['region_starts']:
                    oracle_sr = 1
                if i == len(deduplicate_vps)-1:
                    if vp not in datum_annt['region_starts']:
                        sr = 1
            scores['success'] = sr
            scores['oracle_success'] = oracle_sr
            scores['success_exit'] = sr
            scores['oracle_success_exit'] = oracle_sr
        else:
            raise ValueError(f"Valid action (go into, exit) for {self.id} not in the instruction")
        return scores
