import json
import os
from collections import defaultdict
import numpy as np
ERROR_MARGIN=3.0

def cal_dtw(shortest_distances, prediction, reference, success=None, threshold=3.0):
    dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
    dtw_matrix[0][0] = 0
    for i in range(1, len(prediction)+1):
        for j in range(1, len(reference)+1):
            best_previous_cost = min(
                dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
            cost = shortest_distances[prediction[i-1]][reference[j-1]]
            dtw_matrix[i][j] = cost + best_previous_cost

    dtw = dtw_matrix[len(prediction)][len(reference)]
    ndtw = np.exp(-dtw/(threshold * len(reference)))
    if success is None:
        success = float(shortest_distances[prediction[-1]][reference[-1]] < threshold)
    sdtw = success * ndtw

    return {
        'DTW': dtw, # measure the faithfulness of instruction following and the fidelity between the predicted and the ground-truth trajectory
        'nDTW': ndtw, # which penalizes deviations from the ground-truth trajectories
        'SDTW': sdtw # which penalizes deviations from the ground-truth trajectories and also considers the success rate.
    }

def cal_cls(shortest_distances, prediction, reference, threshold=3.0):
    def length(nodes):
      return np.sum([
          shortest_distances[a][b]
          for a, b in zip(nodes[:-1], nodes[1:])
      ])

    coverage = np.mean([
        np.exp(-np.min([  # pylint: disable=g-complex-comprehension
            shortest_distances[u][v] for v in prediction
        ]) / threshold) for u in reference
    ])
    expected = coverage * length(reference)
    score = expected / (expected + np.abs(expected - length(prediction)))
    return coverage * score

class Evaluator(object):
    def __init__(self, graphs, paths, distances, annt_file, submission_file) -> None:
        self.id = "Standard"
        self.graphs = graphs
        self.paths = paths
        self.distances = distances

        self.meta = {str(item['path_id']):item for item in json.load(open(annt_file))}
        self.preds = dict()
        print(len(json.load(open(submission_file))))
        for item in json.load(open(submission_file)):
            traj = []
            for element in item['trajectory']:
                if len(element) == 0:
                    continue
                for datum in element:
                    if isinstance(datum, str):
                        traj.append([datum])
            self.preds[str(item['instr_id'])] = traj

    def eval(self):
        metrics = defaultdict(list)
        for instr_id, pred_path in self.preds.items():
            traj_id = instr_id.split('_')[0]

            datum_annt = self.meta[traj_id]
            traj_scores = self._eval_item(datum_annt, pred_path)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            
            metrics['instr_id'].append(traj_id)

        if os.environ.get('DEBUG', False):
            print("Error case ides: ")
            for id, success in zip(metrics['instr_id'], metrics['success']):
                if success < 1:
                    print(id)

        avg_metrics = {
                'action_steps': np.mean(metrics['action_steps']),
                'steps': np.mean(metrics['trajectory_steps']),
                'lengths': np.mean(metrics['trajectory_lengths']),
                'nav_error': np.mean(metrics['nav_error']),
                'oracle_error': np.mean(metrics['oracle_error']),
                'sr': np.mean(metrics['success']) * 100,
                'oracle_sr': np.mean(metrics['oracle_success']) * 100,
                'spl': np.mean(metrics['spl']) * 100,
                'nDTW': np.mean(metrics['nDTW']) * 100,
                'SDTW': np.mean(metrics['SDTW']) * 100,
                'CLS': np.mean(metrics['CLS']) * 100,
                'num_paths': len(metrics['success'])
            }
        return avg_metrics
    
    def _get_nearest(self, shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

    def _eval_item(self, datum_annt, pred_path):
        scores = {}
        scan = datum_annt['scan']
        gt_path = datum_annt['path']
        shortest_distances = self.distances[scan]

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        nearest_position = self._get_nearest(shortest_distances, gt_path[-1], path)

        scores['nav_error'] = shortest_distances[path[-1]][gt_path[-1]]
        scores['oracle_error'] = shortest_distances[nearest_position][gt_path[-1]]

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])
        
        scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN)

        scores.update(
            cal_dtw(shortest_distances, path, gt_path, scores['success'], ERROR_MARGIN)
        )
        scores['CLS'] = cal_cls(shortest_distances, path, gt_path, ERROR_MARGIN)

        return scores
