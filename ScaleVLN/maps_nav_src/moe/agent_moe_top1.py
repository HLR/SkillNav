import json
import os
import sys
import numpy as np
# import random
from random import random, seed as set_seed, randint
import math
import time
from collections import defaultdict

import line_profiler
import logging

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.distributed import is_default_gpu
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from moe.agent_base_moe import Seq2SeqAgent
# from r2r.agent_base import Seq2SeqAgent
from r2r.eval_utils import cal_dtw

from models.graph_utils import GraphMap
from models.model import VLNBert, Critic
from models.ops import pad_tensors_wgrad

from moe.router import NavigationSkillAnalyzer
# import utils.get_images
# from importlib import reload
# reload(utils.get_images)
from utils.get_images import load_vp_lookup, convert_path2img, extract_cand_img

from router_qwen import extract_from_response
from llm_utils import generate_temporal_instructions

import moe.vLLM_API 
from importlib import reload
reload(moe.vLLM_API)
from moe.vLLM_API import (
    load_vLLM_model, 
    load_two_vllms,
    load_two_vllms_with_id,
    infer_with_vllm_instruction_localization, 
    infer_with_vllm_skill_routing, 
    extract_majority_skill,
    get_expert_indices
)

# from multiprocessing import Pipe, Process
# from llm_worker_localizer import localizer_worker
# from llm_worker_skill import skill_worker

import httpx
import asyncio
timeout = httpx.Timeout(120.0, connect=10.0)


class GMapNavAgents(Seq2SeqAgent):
    
    def __init__(self, args, env, rank):
        super().__init__(args, env, rank)
        self.args = args

        # Create log directory if it doesn't exist
        os.makedirs(self.args.log_dir, exist_ok=True)

        # Set up logger
        if self.args.debug:
            log_file_path = os.path.join(self.args.log_dir, self.env.name + "_debug_outputs.log")
        else:
            log_file_path = os.path.join(self.args.log_dir, self.env.name + "_router_outputs.log")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Prevent duplicate handlers if multiple instances are created
        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_file_path, mode='w')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # if self.args.routing_mode == 'moe' or self.args.routing_mode == 'shared':
        if self.args.routing_mode != 'fixed':
        #     # self.analyzer = NavigationSkillAnalyzer()
        #     ### If using FastAPI, you do not need to load the model here
        #     # self.analyzer.initialize_qwen_model(self.args.router_model)
            self.vp_lookup = load_vp_lookup()
        
        if self.args.routing_mode == 'top1' and not self.args.debug and not self.args.localizer_gpu_id:
            # Load vLLM model for routing
            self.localizer_llm, self.skill_llm = load_two_vllms(batch_size = self.args.batch_size, localizer_model_ckpt="Qwen/Qwen2.5-VL-7B-Instruct", skill_model_ckpt="Qwen/Qwen3-8B")
        
        if self.args.routing_mode == 'top1' and self.args.localizer_gpu_id and self.args.skill_gpu_id:

            self.localizer_llm, self.skill_llm = load_two_vllms_with_id(
                batch_size = self.args.batch_size,
                localizer_model_ckpt=self.args.localizer_model, 
                # skill_model_ckpt="Qwen/Qwen3-8B",
                skill_model_ckpt=self.args.skill_model, 
                gpu_memory_utilization=self.args.gpu_memory_utilization,
                localizer_gpu_id=self.args.localizer_gpu_id, 
                skill_gpu_id=self.args.skill_gpu_id
            )

    def update_logger(self, env_name):
        log_file_name = f"{env_name}_debug_outputs.log" if self.args.debug else f"{env_name}_router_outputs.log"
        log_file_path = os.path.join(self.args.log_dir, log_file_name)

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        file_handler = logging.FileHandler(log_file_path, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    
    def _build_model(self):
        self.vln_berts = nn.ModuleList([])
        self.critics = nn.ModuleList([])  # Store multiple critic models if needed
        self.vln_bert_optimizers = [] # Store optimizers for multiple VLN-BERTs
        self.critic_optimizers = []  # Store optimizers for multiple critics

        if isinstance(self.args.resume_files, list):
            self.load_resumes(self.args.resume_files) # Load all resume files into self.resumes
            for model_idx, resume_file in enumerate(self.args.resume_files):
                if self.args.resume_weights[model_idx] == 0: continue # Skip if weight is 0
                
                self.vln_bert = VLNBert(self.args).cuda()
                self.critic = Critic(self.args).cuda()

                # Initialize optimizers for each model
                if self.args.optim == 'rms':
                    optimizer = torch.optim.RMSprop
                elif self.args.optim == 'adam':
                    optimizer = torch.optim.Adam
                elif self.args.optim == 'adamW':
                    optimizer = torch.optim.AdamW
                elif self.args.optim == 'sgd':
                    optimizer = torch.optim.SGD
                else:
                    assert False

                self.vln_bert_optimizer = optimizer(self.vln_bert.parameters(), lr=self.args.lr)
                self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.args.lr)

                if resume_file in self.resumes:
                    self._load_model_weights(resume_file, self.vln_bert, self.vln_bert_optimizer, self.critic, self.critic_optimizer)

                    self.vln_berts.append(self.vln_bert)
                    self.critics.append(self.critic)
                    self.vln_bert_optimizers.append(self.vln_bert_optimizer)
                    self.critic_optimizers.append(self.critic_optimizer)
                # print('-'*20)
                # print(f"Corresponding model weight: {self.args.resume_weights[model_idx]} - {resume_file}")
        else:
            self.vln_bert = VLNBert(self.args).cuda()
            self.critic = Critic(self.args).cuda()
            # Initialize optimizers for the single models
            if self.args.optim == 'rms':
                optimizer = torch.optim.RMSprop
            elif self.args.optim == 'adam':
                optimizer = torch.optim.Adam
            elif self.args.optim == 'adamW':
                optimizer = torch.optim.AdamW
            elif self.args.optim == 'sgd':
                optimizer = torch.optim.SGD
            else:
                assert False
            self.vln_bert_optimizer = optimizer(self.vln_bert.parameters(), lr=self.args.lr)
            self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.args.lr)

        self.scanvp_cands = {}

    def _load_model_weights(self, resume_file, vln_bert, vln_bert_optimizer, critic, critic_optimizer):
        """Loads model weights and optionally optimizer state from a resume file."""
        
        # print("-"*20)
        # print("Function `_load_model_weights` is called!")
        
        states = self.resumes.get(resume_file)
        
        if states:
            def recover_state(name, model, optimizer):
                if name in states:
                    state = model.state_dict()
                    model_keys = set(state.keys())
                    load_keys = set(states[name]['state_dict'].keys())
                    state_dict = states[name]['state_dict']

                    if model_keys != load_keys:
                        print(f"NOTICE: DIFFERENT KEYS IN {name} for {resume_file}")
                        if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
                            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                        if list(model_keys)[0].startswith('module.') and (not list(load_keys)[0].startswith('module.')):
                            state_dict = {'module.' + k: v for k, v in state_dict.items()}
                        same_state_dict = {}
                        extra_keys = []
                        
                        for k, v in state_dict.items():
                            if k in model_keys:
                                same_state_dict[k] = v
                            else:
                                extra_keys.append(k)
                        state_dict = same_state_dict
                        print(f'Extra keys in {name} state_dict for {resume_file}: {", ".join(extra_keys)}')

                    state.update(state_dict)
                    model.load_state_dict(state)

                    if self.args.resume_optimizer and 'optimizer' in states[name]:
                        optimizer.load_state_dict(states[name]['optimizer'])
                else:
                    print(f"WARNING: Weights for {name} not found in {resume_file}")

            recover_state("vln_bert", vln_bert, vln_bert_optimizer)
            recover_state("critic", critic, critic_optimizer)
        else:
            print(f"WARNING: No state data found for {resume_file}")
                        
    def _language_variable(self, obs):
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]
        
        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
        # mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool_)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
        mask = torch.from_numpy(mask).cuda()
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask
        }

    def _panorama_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens, batch_cand_vpids = [], []
        
        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32) # 
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)
            
            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts, 
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens, 
            'cand_vpids': batch_cand_vpids,
        }

    def _nav_gmap_variable(self, obs, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(obs)
        
        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []                
            for k in gmap.node_positions.keys():
                if self.args.act_visited_nodes:
                    if k == obs[i]['viewpoint']:
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
                else:
                    if gmap.graph.visited(k):
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )   # cuda

            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i+1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds, 
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks, 
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
        }

    def _nav_vp_variable(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, nav_types):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i], 
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp], 
                obs[i]['heading'], obs[i]['elevation']
            )                    
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': gen_seq_masks(view_lens+1),
            'vp_nav_masks': vp_nav_masks,
            'vp_cand_vpids': [[None]+x for x in cand_vpids],
        }

    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0    # Stop if arrived 
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                            dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                    + self.env.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % (scan))

        return torch.from_numpy(a).cuda()

    def _teacher_action_r4r(self, obs, vpids, ended, visited_masks=None, imitation_learning=False, t=None, traj=None):
        """R4R is not the shortest path. The goal location can be visited nodes.
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if imitation_learning:
                    assert ob['viewpoint'] == ob['gt_path'][t]
                    if t == len(ob['gt_path']) - 1:
                        a[i] = 0    # stop
                    else:
                        goal_vp = ob['gt_path'][t + 1]
                        for j, vpid in enumerate(vpids[i]):
                            if goal_vp == vpid:
                                a[i] = j
                                break
                else:
                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = 0    # Stop if arrived 
                    else:
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        for j, vpid in enumerate(vpids[i]):
                            if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                                if self.args.expert_policy == 'ndtw':
                                    dist = - cal_dtw(
                                        self.env.shortest_distances[scan], 
                                        sum(traj[i]['path'], []) + self.env.shortest_paths[scan][ob['viewpoint']][vpid][1:], 
                                        ob['gt_path'], 
                                        threshold=3.0
                                    )['nDTW']
                                elif self.args.expert_policy == 'spl':
                                    # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                                    dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                            + self.env.shortest_distances[scan][cur_vp][vpid]
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = j
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            
            curr_vp = action
            prev_vp = ob['viewpoint']
            
            if action is not None:            # None is the <stop> action
                # traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                # traj[i]['path_heading'].append(ob['heading'])
                # traj[i]['path_elevation'].append(ob['elevation'])
                
                mini_path = gmaps[i].graph.path(ob['viewpoint'], action)
                traj[i]['path'].append(mini_path)
                for vp in mini_path:
                    
                    key = f"{ob['scan']}_{prev_vp}"
                    if key in self.scanvp_cands and vp in self.scanvp_cands[key]:
                        inside_viewidx = self.scanvp_cands[key][vp]
                        inside_heading =  (inside_viewidx % 12) * math.radians(30)
                        inside_elevation = (inside_viewidx // 12 - 1) * math.radians(30)
                    else:
                        # Fallback: compute heading/elevation geometrically
                        # src_pos = gmaps[i].graph.nodes[prev_vp]['position']
                        # tgt_pos = gmaps[i].graph.nodes[vp]['position']
                        # inside_heading, inside_elevation = compute_heading_elevation(src_pos, tgt_pos)
                        inside_heading = ob['heading']
                        inside_elevation = ob['elevation']
                    
                    traj[i]['path_heading'].append(inside_heading)     # Egocentric at time of transition
                    traj[i]['path_elevation'].append(inside_elevation)
                    traj[i]['trajectory'].append(
                        [vp, inside_heading, inside_elevation]
                    )

                    prev_vp = vp
                
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                # elif len(traj[i]['path'][-1]) >= 2:
                    prev_vp = traj[i]['path'][-1][-2]
                # else:
                    # prev_vp = traj[i]['path'][-1][-1] # TODO
                    
                viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                self.env.env.sims[i].newEpisode([ob['scan']], [action], [heading], [elevation])

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    # ! get input embeddings for the current step for batch_size observations
    def _get_input_embeds(self, obs, language_inputs, pano_inputs, gmaps, ended, expert_indices, t):
        # --- Randomly select a VLN-BERT model for this step ---

        '''            
        ### --- Solution 1
        if self.args.routing_mode == 'random':
            set_seed(self.args.seed + t)
            vln_bert_idx = int(random() * len(self.vln_berts))  # Ensure the index is within bounds
            self.vln_bert = self.vln_berts[vln_bert_idx]
                
        for i, gmap in enumerate(gmaps):
            if not ended[i]:
                gmap.node_step_ids[obs[i]['viewpoint']] = t + 1
                
        txt_embeds = self.vln_bert('language', language_inputs)
                            
        # graph representation
        pano_inputs = self._panorama_feature_variable(obs)
        
        pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs) # pano_embeds: (bs, n_panos, dim)
        avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                          torch.sum(pano_masks, 1, keepdim=True)
        
        if self.args.debug:
            self.logger.info(f"Step {t}: txt_embeds shape: {txt_embeds.shape}, pano_embeds shape: {pano_embeds.shape}, avg_pano_embeds shape: {avg_pano_embeds.shape}")
            self.logger.info(f"Step {t}: pano_inputs['view_lens'] shape: {pano_inputs['view_lens'].shape}, pano_inputs['cand_vpids'] shape: {len(pano_inputs['cand_vpids'])}, pano_inputs['nav_types'] shape: {pano_inputs['nav_types'].shape}")
        ### --- End of Solution 1
        '''
        
        ### --- Solution 2: Pre-allocate lists for collecting per-ob embeddings            
        
        batch_size = len(obs)  
        
        if expert_indices is None:
            set_seed(self.args.seed + t)
            if len(self.vln_berts) > 1:
                expert_indices = [randint(0, len(self.vln_berts) - 1) for _ in range(batch_size)]
            else:
                expert_indices = [0] * batch_size
        else:
            if len(expert_indices) < batch_size:
                # Pad with default expert index (e.g., 1)
                expert_indices += [1] * (batch_size - len(expert_indices))
            elif len(expert_indices) > batch_size:
                # Trim to match batch size
                expert_indices = expert_indices[:batch_size]
        
        txt_embeds_list, pano_embeds_list, avg_pano_embeds_list, pano_lengths = [], [], [], [] # pano_lengths: track per-ob n_panos for masking if needed
        
        self.logger.info(f"Step {t}: Using expert indices: {expert_indices} for obs: {[ob['instr_id'] for ob in obs]}")
        
        for i, (ob, expert_idx) in enumerate(zip(obs, expert_indices)):
            
            self.vln_bert = self.vln_berts[expert_idx]  # Select the VLN-BERT model for this observation
            
            # Prepare single-ob language_inputs
            single_language_inputs = {k: v[i:i+1] for k, v in language_inputs.items()}
            txt_embed = self.vln_bert('language', single_language_inputs)  # (1, seq_len, dim)
            txt_embeds_list.append(txt_embed)

            # Mark gmap
            if not ended[i]:
                gmaps[i].node_step_ids[obs[i]['viewpoint']] = t + 1

            # Prepare single-ob pano_inputs
            single_pano_inputs = self._panorama_feature_variable([obs[i]])

            # Forward pano
            pano_embed, pano_mask = self.vln_bert('panorama', single_pano_inputs)  # (1, n_panos, dim), (1, n_panos)

            avg_pano_embed = torch.sum(pano_embed * pano_mask.unsqueeze(2), 1) / \
                            torch.sum(pano_mask, 1, keepdim=True)  # (1, dim)

            pano_embeds_list.append(pano_embed.squeeze(0))  # (n_panos, dim)
            try:
                avg_pano_embeds_list.append(avg_pano_embed)
            except Exception as e:
                self.logger.warning(f"Embed failed for idx {i}, instr_id={ob['instr_id']}: {e}")
                continue
            pano_lengths.append(pano_embed.shape[1])

        # Reconstruct unified tensors
        txt_embeds = torch.cat(txt_embeds_list, dim=0)            # (bs, seq_len, dim)
        # pano_embeds = torch.cat(pano_embeds_list, dim=0)          # (bs, n_panos, dim)
        pano_embeds = pad_tensors(pano_embeds_list).cuda()  # (bs, max_n_panos, dim)
        avg_pano_embeds = torch.cat(avg_pano_embeds_list, dim=0)  # (bs, dim)
        
        
        return txt_embeds, pano_embeds, avg_pano_embeds, gmaps
    
        
    # @profile
    def rollout(self, train_ml=None, train_rl=False, reset=True):

        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        self._update_scanvp_cands(obs)

        batch_size = len(obs)
        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'path_heading': [ob['heading']],
            'path_elevation': [ob['elevation']],
            'trajectory': [[ob['viewpoint'], ob['heading'], ob['elevation']]],
            'details': {},
        } for ob in obs]
        
        self.prev_sub_instruction_list = [[] for _ in obs]  # One list per agent
        # self.prev_viewpoints = [[] for _ in obs]
        
        # instructions = [ob['instruction'] for ob in obs]
        # # --- If no reordered instructions, use original instructions
        # reordered_instructions = instructions   
        
        # --- Using LLM to reorder Original instructions into clear sub-instructions
        instructions = [ob['instruction'] for ob in obs]
        
        if self.args.instruction_reorder:
            reordered_instructions = generate_temporal_instructions(instructions, max_retries=3, max_tokens=2000, temperature=0, model="gpt-4o", num_threads=batch_size)
        else:
            reordered_instructions = instructions
            
        if self.args.debug:
            self.logger.info('-'*20)
            for i, ob in enumerate(obs):
                self.logger.info(f"Obs {i}: instr_id={ob['instr_id']}, instruction={ob['instruction']}, reordered_instruction={reordered_instructions[i]}")
        
        # Language input: txt_ids, txt_masks
        language_inputs = self._language_variable(obs)
        # with torch.no_grad():
        #     txt_embeds_start = self.vln_berts[0]('language', language_inputs) # (batch_size, seq_len, dim)
       
        # if self.args.debug:
            # self.logger.info(f"language_inputs: {language_inputs}")
            # self.logger.info(f"Initial instructions: {instructions}")
            # self.logger.info(f"Reordered instructions: {reordered_instruction}")
        
        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # Init the logs
        masks = []
        entropys = []
        ml_loss = 0.     

        for t in range(self.args.max_action_len):
            
            if self.args.routing_mode == 'top1':
                # original_paths = [traj[i]['path'] for i, ob in enumerate(obs)] 
                # paths = [[vp for step in traj_steps for vp in step] for traj_steps in original_paths]
                    
                # previous_viewpoint_lists = [convert_path2img(paths[i], ob['scan'], self.vp_lookup) for i, ob in enumerate(obs)]
                # # candidate_viewpoint_lists = [extract_cand_img(paths[i][-1], ob['scan'], self.vp_lookup) for i, ob in enumerate(obs)]

                batch_inputs_router = []
                
                for i, ob in enumerate(obs):
                    original_path = traj[i]['path']
                    path = [vp for step in original_path for vp in step]  # Flatten the path
                    previous_viewpoint_list = convert_path2img(path, ob['scan'], self.vp_lookup)
                    
                    if self.args.debug:
                        self.logger.info(f"Step {t}, obs[{i}]: {ob['instr_id']}, len of previous_viewpoint_list: {len(previous_viewpoint_list)},previous_viewpoint_list: {previous_viewpoint_list}")
                    
                    new_item = {
                        'scan': ob['scan'],
                        'instr_id': ob['instr_id'],
                        'full_instruction': reordered_instructions[i], # Using reordered instructions
                        'previous_viewpoint_list': previous_viewpoint_list,
                        'previous_sub_instruction_list': self.prev_sub_instruction_list[i],
                    }
                    batch_inputs_router.append(new_item)
                
                expert_indices, sub_instructions = get_expert_indices(self.localizer_llm, self.skill_llm, batch_inputs_router, self.logger)
                
                # --- Update the batch_inputs
                for i, sub_instr in enumerate(sub_instructions):
                    if sub_instr:  # Only append valid sub-instructions
                        self.prev_sub_instruction_list[i].append(sub_instr)
                                    
                # if not self.args.debug:
                #     expert_indices = get_expert_indices(self.localizer_llm, self.skill_llm, batch_inputs_router, self.logger)
                # else:
                #     expert_indices = None
            
            ### --- Solution 2: Pre-allocate lists for collecting per-ob embeddings            
            pano_inputs = self._panorama_feature_variable(obs)
            
            # --- Random
            if self.args.routing_mode == 'random':
                txt_embeds, pano_embeds, avg_pano_embeds, gmaps = self._get_input_embeds(
                    obs, language_inputs, pano_inputs, gmaps, ended, None, t
                )
            
            # ---Routing
            if self.args.routing_mode == 'top1':
                txt_embeds, pano_embeds, avg_pano_embeds, gmaps = self._get_input_embeds(
                    obs, language_inputs, pano_inputs, gmaps, ended, expert_indices, t
                )
            
            # Update graph map with pano embeddings
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]): # traverse all candidates
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

            # navigation policy
            nav_inputs = self._nav_gmap_variable(obs, gmaps)
            nav_inputs.update(
                self._nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_inputs['cand_vpids'], 
                    pano_inputs['view_lens'], pano_inputs['nav_types'],
                )
            )
            nav_inputs.update({
                'txt_embeds': txt_embeds,
                'txt_masks': language_inputs['txt_masks'],
            })
    
            nav_outs = self.vln_bert('navigation', nav_inputs) # there are some inf in nav_outs
            
            if self.args.fusion == 'local':
                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['vp_cand_vpids']
            elif self.args.fusion == 'global':
                nav_logits = nav_outs['global_logits']
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_logits = nav_outs['fused_logits'] # Dynamic Fused
                nav_vpids = nav_inputs['gmap_vpids']
                        
            nav_probs = torch.softmax(nav_logits, 1)
            
            # update graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    gmap.node_stop_scores[i_vp] = {
                       'stop': nav_probs[i, 0].data.item()
                    }
                                        
            if train_ml is not None:
                # Supervised training
                if self.args.dataset == 'r2r':
                    # nav_targets = self._teacher_action(
                    #     obs, nav_vpids, ended, 
                    #     visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None
                    # )
                    nav_targets = self._teacher_action_r4r(
                        obs, nav_vpids, ended, 
                        visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                        imitation_learning=(self.feedback=='teacher'), t=t, traj=traj
                    )
                elif self.args.dataset == 'r4r':
                    nav_targets = self._teacher_action_r4r(
                        obs, nav_vpids, ended, 
                        visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                        imitation_learning=(self.feedback=='teacher'), t=t, traj=traj
                    )
                # print(t, nav_logits, nav_targets)
                ml_loss += self.criterion(nav_logits, nav_targets)
                # print(t, 'ml_loss', ml_loss.item(), self.criterion(nav_logits, nav_targets).item())
                                                 
            # Determinate the next navigation viewpoint
            if self.feedback == 'teacher':
                a_t = nav_targets                 # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = nav_logits.max(1)        # student forcing - argmax
                a_t = a_t.detach() 
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach() 
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # hyper-param
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs['gmap_visited_masks'].logical_not()).data.numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Determine stop actions
            if self.feedback == 'teacher' or self.feedback == 'sample': # in training
                # a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs] #	In training, the agent stops when it reaches the last ground-truth viewpoint.
            else:
                a_t_stop = a_t == 0 # In inference, it stops when a_t == 0.

            # Prepare environment action
            cpu_a_t = []  
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len):
                # if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1): # determines if an action should be set to None (which means no movement will happen)
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])   

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf')}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node: # If stop_node is found and it is not the current viewpoint, the function adds a path to that stop node.
                        # traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))
                        # traj[i]['path_heading'].append(obs[i]['heading'])
                        # traj[i]['path_elevation'].append(obs[i]['elevation'])
                        # traj[i]['trajectory'].append(
                        #     [gmaps[i].graph.path(obs[i]['viewpoint'], stop_node), obs[i]['heading'], obs[i]['elevation']]
                        # )
                        
                        curr_vp = stop_node
                        prev_vp = obs[i]['viewpoint']
                        
                        mini_path = gmaps[i].graph.path(prev_vp, curr_vp)
                        traj[i]['path'].append(mini_path)
                        for vp in mini_path:
                            
                            key = f"{obs[i]['scan']}_{prev_vp}"
                            if key in self.scanvp_cands and vp in self.scanvp_cands[key]:
                                inside_viewidx = self.scanvp_cands[key][vp]
                                inside_heading =  (inside_viewidx % 12) * math.radians(30)
                                inside_elevation = (inside_viewidx // 12 - 1) * math.radians(30)
                            else:

                                inside_heading = obs[i]['heading']
                                inside_elevation = obs[i]['elevation']
                                
                            traj[i]['path_heading'].append(inside_heading)     # Egocentric at time of transition
                            traj[i]['path_elevation'].append(inside_elevation)
                            traj[i]['trajectory'].append(
                                [vp, inside_heading, inside_elevation]
                            )

                            prev_vp = vp
                        
                    if self.args.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                            }

            # new observation and update graph
            obs = self.env._get_obs()
            self._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break
        
        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
            self.loss += ml_loss
            self.logs['IL_loss'].append(ml_loss.item())

        return traj
