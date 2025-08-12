''' Batched REVERIE navigation environment '''

import json
import os
import numpy as np
import math
import random
import networkx as nx
from collections import defaultdict
import copy

import sys
sys.path.append('/home/matiany3/Matterport3DSimulator/build')
import MatterSim


from utils.data import load_nav_graphs, new_simulator
from utils.data import angle_feature, get_all_point_angle_feature

from r2r.eval_utils import cal_dtw, cal_cls
from typing import Any, List, Union
from collections import defaultdict
from numpy import ndarray
def euclidean_distance(
    pos_a: Union[List[float], ndarray], pos_b: Union[List[float], ndarray]
) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)

def dot_product(v1, v2):
    return np.dot(v1, v2)

def norm(v):
    return np.linalg.norm(v)

def is_projection_inside_segment(A, B, P):
    # Convert points to numpy arrays for vector operations
    A = np.array(A)
    B = np.array(B)
    P = np.array(P)
    
    # Calculate the vector AP
    AP = P - A
    
    # Calculate the vector AB
    AB = B - A
    
    # Calculate the projected point P' on line AB
    proj_scale = dot_product(AP, AB) / norm(AB)**2
    proj_point = A + proj_scale * AB
    
    # Check if the projected point lies within the segment AB
    # It lies within the segment if it's not further from A than B is, and vice versa
    return abs(norm(proj_point - A)  + norm(proj_point - B) - norm(AB)) < 1e-7

def check_angle_between_vectors(v1, v2, threshold_angle=30):
    # Convert vectors to numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Calculate the dot product of the vectors
    dot_product = np.dot(v1, v2)

    # Compute the magnitudes of the vectors
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)

    # Calculate the angle in radians and then convert to degrees
    angle = np.arccos(cos_angle)

    # Check if the angle is less than the threshold
    return  angle

ERROR_MARGIN = 3.0

class EnvBatch(object):
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, connectivity_dir, scan_data_dir=None, feat_db=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feat_db: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        self.feat_db = feat_db
        self.image_w = 640
        self.image_h = 480
        self.vfov = 60
        
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            if scan_data_dir:
                sim.setDatasetPath(scan_data_dir)
            sim.setNavGraphPath(connectivity_dir)
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.setBatchSize(1)
            sim.initialize()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode([scanId], [viewpointId], [heading], [0])

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((36, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()[0]

            feature = self.feat_db.get_image_feature(state.scanId, state.location.viewpointId)
            feature_states.append((feature, state))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction([index], [heading], [elevation])


class R2RNavBatch(object):
    ''' Implements the REVERIE navigation task, using discretized viewpoints and pretrained features '''

    def __init__(
        self, view_db, instr_data, connectivity_dir, 
        batch_size=64, angle_feat_size=4, seed=0, name=None, sel_data_idxs=None,
    ):
        self.env = EnvBatch(connectivity_dir, feat_db=view_db, batch_size=batch_size)
        self.data = instr_data
        self.scans = set([x['scan'] for x in self.data])
        self.connectivity_dir = connectivity_dir
        self.batch_size = batch_size
        self.angle_feat_size = angle_feat_size
        self.name = name

        self.gt_trajs = self._get_gt_trajs(self.data) # for evaluation

        # in validation, we would split the data
        if sel_data_idxs is not None:
            t_split, n_splits = sel_data_idxs
            ndata_per_split = len(self.data) // n_splits 
            start_idx = ndata_per_split * t_split
            if t_split == n_splits - 1:
                end_idx = None
            else:
                end_idx = start_idx + ndata_per_split
            self.data = self.data[start_idx: end_idx]

        # use different seeds in different processes to shuffle data
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self._load_nav_graphs()

        self.sim = new_simulator(self.connectivity_dir)
        self.angle_feature = get_all_point_angle_feature(self.sim, self.angle_feat_size)
        
        self.buffered_state_dict = {}
        print('%s loaded with %d instructions, using splits: %s' % (
            self.__class__.__name__, len(self.data), self.name))

    # def _get_gt_trajs(self, data):
    #     gt_trajs = {
    #         x['instr_id']: (x['scan'], x['path']) \
    #             for x in data if len(x['path']) > 1
    #     }
    #     return gt_trajs

    def _get_gt_trajs(self, data):
        gt_trajs = {}
        for x in data:
            if len(x['path']) > 1:
                # if gt_trajs.get(x['instr_id'], None) is not None:
                #     print("Warning: duplicate instr_id from this path_id %s" % x['path_id'])
                gt_trajs[x['instr_id']]= (x['scan'], x['path'])
                
            elif 'gt_vp' in x:
                gt_trajs[x['instr_id']]= (x['scan'], x['gt_vp'])
            # if 'example_path' in x:
            #     gt_trajs[x['instr_id']]= (x['scan'], x['example_path'])
            
        return gt_trajs

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.connectivity_dir, self.scans)
        self.shortest_paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.shortest_distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        batch = self.data[self.ix: self.ix+batch_size]
        if len(batch) < batch_size:
            random.shuffle(self.data)
            self.ix = batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        base_elevation = (viewId // 12 - 1) * math.radians(30)

        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    self.sim.makeAction([0], [1.0], [1.0])
                else:
                    self.sim.makeAction([0], [1.0], [0])

                state = self.sim.getState()[0]
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation - base_elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = angle_feature(loc_heading, loc_elevation, self.angle_feat_size)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            "normalized_elevation": state.elevation + loc.rel_elevation,
                            'scanId': scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1),
                            'position': (loc.x, loc.y, loc.z),
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'normalized_elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx', 'position']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                visual_feat = feature[ix]
                c_new['heading'] = c_new['normalized_heading'] - base_heading
                c_new['elevation'] = c_new['normalized_elevation'] - base_elevation
                angle_feat = angle_feature(c_new['heading'], c_new['elevation'], self.angle_feat_size)
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                c_new.pop('normalized_elevation')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex
           
            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)
            # [visual_feature, angle_feature] for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

            ob = {
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'position': (state.location.x, state.location.y, state.location.z),
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instruction' : item['instruction'],
                'instr_encoding': item['instr_encoding'],
                'gt_path' : item['path'],
                'path_id' : item['path_id']
            }
            # RL reward. The negative distance between the state and the final state
            # There are multiple gt end viewpoints on REVERIE. 
            if ob['instr_id'] in self.gt_trajs:
                ob['distance'] = self.shortest_distances[ob['scan']][ob['viewpoint']][item['path'][-1]]
            else:
                ob['distance'] = 0

            obs.append(ob)
        return obs

    def reset(self, **kwargs):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch(**kwargs)
        
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()


    ############### Nav Evaluation ###############
    def _get_nearest(self, shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

    def _eval_item(self, scan, pred_path, gt_path):
        scores = {}

        shortest_distances = self.shortest_distances[scan]

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

    # def eval_metrics(self, preds):
    #     ''' Evaluate each agent trajectory based on how close it got to the goal location 
    #     the path contains [view_id, angle, vofv]'''
    #     print('eval %d predictions' % (len(preds)))

    #     metrics = defaultdict(list)
    #     for item in preds:
    #         instr_id = item['instr_id']
    #         traj = item['trajectory']
    #         scan, gt_traj = self.gt_trajs[instr_id]
    #         traj_scores = self._eval_item(scan, traj, gt_traj)
    #         for k, v in traj_scores.items():
    #             metrics[k].append(v)
    #         metrics['instr_id'].append(instr_id)
        
    #     avg_metrics = {
    #         'action_steps': np.mean(metrics['action_steps']),
    #         'steps': np.mean(metrics['trajectory_steps']),
    #         'lengths': np.mean(metrics['trajectory_lengths']),
    #         'nav_error': np.mean(metrics['nav_error']),
    #         'oracle_error': np.mean(metrics['oracle_error']),
    #         'sr': np.mean(metrics['success']) * 100,
    #         'oracle_sr': np.mean(metrics['oracle_success']) * 100,
    #         'spl': np.mean(metrics['spl']) * 100,
    #         'nDTW': np.mean(metrics['nDTW']) * 100,
    #         'SDTW': np.mean(metrics['SDTW']) * 100,
    #         'CLS': np.mean(metrics['CLS']) * 100,
    #     }
    #     return avg_metrics, metrics
        

    def _eval_item_dc(self, scan, pred_path, gt_path, instr_id):
        scores = {}
        graph = self.graphs[scan]
        data = self.instr_id2data[instr_id]
        start_heading = data['heading']
        instruction = data['instruction']

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'
        assert pred_path[0] != pred_path[1]

        pos0 = graph.nodes[data['gt_vp'][0]]['position']
        pos_11 = graph.nodes[data['gt_vp'][1]]['position']
        pos_12 = graph.nodes[data['vp_neg'][1]]['position']
        heading_vec =  (pos_11 + pos_12)/2 - pos0

        if os.environ.get('DEBUG', False):
            v1 = graph.nodes[gt_path[1]]['position'] - graph.nodes[gt_path[0]]['position']
        else:
            v1 = graph.nodes[path[1]]['position'] - graph.nodes[path[0]]['position']

        is_left = False
        if ((heading_vec[0]) * (v1[1]) - (heading_vec[1]) * (v1[0])) > 0:
            is_left = True
        
        if "turn left" in instruction:
            if is_left:
                scores['success'] = 1.
            else:
                scores['success'] = 0.
        elif "turn right" in instruction:
            if not is_left:
                scores['success'] = 1.
            else:
                scores['success'] = 0.
        
        if scores['success'] == 0 and os.environ.get('DEBUG', False):
            print(instruction)
            print("is left", is_left)
            import ipdb;ipdb.set_trace() # breakpoint 410

        return scores

    def _eval_item_rr(self, scan, pred_path, gt_path, instr_id):
        scores = {}
        data = self.instr_id2data[instr_id]
        instruction = data['instruction']

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

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
                if vp in data['valid_ends']:
                    oracle_sr = 1
                if i == len(deduplicate_vps)-1:
                    if vp in data['valid_ends']:
                        sr = 1
            scores['success'] = sr
            scores['oracle_success'] = oracle_sr
            scores['success_into'] = sr
            scores['oracle_success_into'] = oracle_sr
    
        elif 'exit' in instruction:
            oracle_sr = 0
            sr = 0
            for i, vp in enumerate(deduplicate_vps):
                if vp not in data['region_starts']:
                    oracle_sr = 1
                if i == len(deduplicate_vps)-1:
                    if vp not in data['region_starts']:
                        sr = 1
            scores['success'] = sr
            scores['oracle_success'] = oracle_sr
            scores['success_exit'] = sr
            scores['oracle_success_exit'] = oracle_sr
        else:
            raise ValueError("For task type room-recognition, valid action (go into, exit) not in the instruction")
        return scores
    
    def _eval_item_landmark(self, scan, pred_path, gt_path, instr_id, angle_max=90):
        scores = {}
        shortest_distances = self.shortest_distances[scan]
        data = self.instr_id2data[instr_id]
        instruction = data['instruction']

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        deduplicate_vps = []
        deduplicate_vps.append(path[0])
        for i in range(1, len(path)):
            if path[i] == deduplicate_vps[-1]:
                continue
            deduplicate_vps.append(path[i])

        graph = self.graphs[scan]
        start_vp = data['path'][0]
        example_end_vp = data['path'][-1]
        start_pos = np.array(graph.nodes[start_vp]['position'])
        end_vp = deduplicate_vps[-1]
        end_pos = np.array(graph.nodes[end_vp]['position'])
        obj_center = data['obj_center']
        sr = 0
        if "walk towards" in instruction:
            dist1 = euclidean_distance(start_pos[:2], obj_center[:2])
            dist2 = euclidean_distance(end_pos[:2], obj_center[:2])
            if dist2 < dist1:
                sr = 1
            scores['success'] = sr
            scores['success_towards'] = sr
    
        elif 'walk past' in instruction:
            # angle measurement
            if shortest_distances[example_end_vp][deduplicate_vps[-1]] < 3:
                if deduplicate_vps[-1] == example_end_vp:
                    sr = 1
                elif is_projection_inside_segment(start_pos[:2], end_pos[:2], obj_center[:2]):
                    if check_angle_between_vectors(obj_center[:2]- start_pos[:2], end_pos[:2]-start_pos[:2]) < angle_max:
                        sr = 1
            scores['success'] = sr
            scores['success_past'] = sr
        else:
            raise ValueError("For task type room-recognition, valid action (go into, exit) not in the instruction")
        return scores

    def eval_metrics(self, preds, task_type='std'):
        ''' Evaluate each agent trajectory based on how close it got to the goal location 
        the path contains [view_id, angle, vofv]'''
        if len(preds) != 1:
            print('eval %d predictions' % (len(preds)))

        metrics = defaultdict(list)
        if "gpt_tagging" in self.data[0].keys():
            metrics["tagging"] = defaultdict(list)

        for i, item in enumerate(preds):
            # NOTE: instr_id is not in the item, skip this item
            if item.get('instr_id', None) is None:
                print(f"Warning: instr_id not in item, skip this item: {item}")
                continue
            
            instr_id = item['instr_id']
            traj = item['trajectory']
            scan, gt_traj = self.gt_trajs[instr_id]
            if task_type == 'dc':
                traj_scores = self._eval_item_dc(scan, traj, gt_traj, instr_id)
            elif task_type == 'rr':
                traj_scores = self._eval_item_rr(scan, traj, gt_traj, instr_id)
            elif task_type == 'landmark':
                traj_scores = self._eval_item_landmark(scan, traj, gt_traj, instr_id)
            else:
                traj_scores = self._eval_item(scan, traj, gt_traj)

            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)

            if "tagging" in metrics.keys():
                instr_tagging = self.instr_id2data[instr_id]
                num_tags = np.sum([len(v) for v in instr_tagging['gpt_tagging']])
                assert num_tags > 0
                for k, v in instr_tagging['gpt_tagging'].items():
                    tag_perc = len(v) / num_tags
                    if tag_perc <= 0.4:
                        continue

                    # NOTE: for numerical, skip text mention first
                    if k == "NU":
                        skip = False
                        for vv in v:
                            if ("second" not in vv) and ("third" not in vv) and ("fourth" not in vv) and ("fifth" not in vv):
                                skip = True
                        if skip:
                            continue
                        print(vv)

                    metrics["tagging"][k].append(traj_scores['success'])

        if task_type == 'dc':
            # NOTE: calculate pared success here
            paired_success = defaultdict(list)
            skip_idx = set()
            for i, instr_id in enumerate(metrics['instr_id']):
                if i in skip_idx:
                    continue
                pair_id, _, path_id = instr_id.rsplit('-',2)
                if path_id == '1_0':
                    inv_path_id = '0_0'
                elif path_id == '0_0':
                    inv_path_id = '1_0'
                else:
                    import ipdb;ipdb.set_trace() # breakpoint 464
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
                    import ipdb;ipdb.set_trace() # breakpoint 479

            avg_metrics = {
                'sr': np.mean(metrics['success']) * 100,
                'pair_sr': np.mean(metrics['pair_success']) * 100,
            }
        elif task_type == 'rr':
            avg_metrics = {
                'sr': np.mean(metrics['success']) * 100,
                'oracle_success': np.mean(metrics['oracle_success']) * 100,
                'success_into': np.mean(metrics['success_into']) * 100,
                'oracle_success_into': np.mean(metrics['oracle_success_into']) * 100,
                'success_exit': np.mean(metrics['success_exit']) * 100,
                'oracle_success_exit': np.mean(metrics['oracle_success_exit']) * 100,
            }
        elif task_type == 'landmark':
            avg_metrics = {
                'sr': np.mean(metrics['success']) * 100,
                'success_towards': np.mean(metrics['success_towards']) * 100,
                'success_past': np.mean(metrics['success_past']) * 100,
            }
        else:   
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
            }
        

        if "tagging" in metrics.keys():
            tagging_metric = {}
            for k,v in metrics['tagging'].items():
                if len(v) == 0:
                    continue
                tagging_metric[k] = (np.mean(v) * 100, np.round(len(v) / len(metrics['nav_error']), decimals=4))
            avg_metrics.update({"tagging_SR": tagging_metric})

        return avg_metrics, metrics