import json
import os
import numpy as np
from .evaluator_base import Evaluator
from utils.math_utils import euclidean_distance, is_projection_inside_segment, check_angle_between_vectors
from collections import defaultdict

ERROR_MARGIN = 3.0
class Evaluator_VM(Evaluator):
    def __init__(self, graphs, paths, distances, annt_file, submission_file) -> None:
        super().__init__(graphs, paths, distances, annt_file, submission_file)
        self.id = "Vertical Movement"

    def eval(self):
        # NOTE: Full set
        metrics = defaultdict(list)
        for instr_id, pred_path in self.preds.items():
            traj_id = instr_id.split('_')[0]

            datum_annt = self.meta[traj_id]
            traj_scores = self._eval_item(datum_annt, pred_path)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            
            metrics['instr_id'].append(traj_id)
        
        # NOTE: filtered set
        for instr_id, pred_path in self.preds.items():
            traj_id = instr_id.split('_')[0]

            datum_annt = self.meta[traj_id]
            if not datum_annt['has_both_dir']:
                continue
            traj_scores = self._eval_item(datum_annt, pred_path)
            for k, v in traj_scores.items():
                metrics[k+"_double_dir"].append(v)
            
            metrics['instr_id'].append(traj_id)
        
        avg_metrics = {
                'sr': np.mean(metrics['success']) * 100,
                'oracle_sr': np.mean(metrics['oracle_success']) * 100,
                'spl': np.mean(metrics['spl']) * 100,
                'nDTW': np.mean(metrics['nDTW']) * 100,

                'sr_double_dir': np.mean(metrics['success_double_dir']) * 100,
                'oracle_sr_double_dir': np.mean(metrics['oracle_success_double_dir']) * 100,
                'spl_double_dir': np.mean(metrics['spl_double_dir']) * 100,
                'nDTW_double_dir': np.mean(metrics['nDTW_double_dir']) * 100,

                'num_paths_double': len(metrics['success_double_dir']),
                'num_paths': len(metrics['success']),
                
            }
        return avg_metrics
