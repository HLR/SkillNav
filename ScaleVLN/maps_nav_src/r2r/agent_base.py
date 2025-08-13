import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.distributed import is_default_gpu
from utils.logger import print_progress

from tqdm import tqdm

class BaseAgent(object):
    ''' Base class for an REVERIE agent to generate and save trajectories. '''

    def __init__(self, env):
        self.env = env
        self.results = {}

    def load_resumes(self, path):
        self.resumes = {}
        try:
            start = time.time()
            states = torch.load(path, map_location=lambda storage, loc: storage)
            print(f"Checkpoint loaded in {time.time() - start:.2f} seconds")
            
            self.resumes[path] = states
            print(f"Loaded resume data from: {path}")
        
        except FileNotFoundError:
            print(f"Warning: Resume file not found: {path}")
        except Exception as e:
            print(f"Error loading resume file {path}: {e}")
            
    
    def get_results(self, detailed_output=False):
        output = []
        for k, v in self.results.items():
            
            # if self.env.name in ['test', 'val_unseen_part_1']: 
            #     output.append({'instr_id': k, 'trajectory': v['trajectory']})
            # else:
            #     output.append({'instr_id': k, 'trajectory': v['path']})
            
            trajectory = v['trajectory'] if self.env.name in ['test', 'val_unseen_part_1'] else v['path']
            if isinstance(trajectory, torch.Tensor):
                trajectory = trajectory.tolist()
            output.append({'instr_id': k, 'trajectory': trajectory})
                
            # output.append({'scan': v.get('scan', ''), 'instr_id': k, 'trajectory': v['path'], 'gt_traj': v.get('gt_traj', ''), 'evaluation': v.get('evaluation','')})
            # output.append({'instr_id': k, 'trajectory': v['path']})
        

            if detailed_output:
                output[-1]['details'] = v['details']
                output[-1]['a_t'] = v.get('a_t', None)
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj                     
                    
        else:   # Do a full round
            total_instructions = len(self.env.data)  # You may need to confirm this
            with tqdm(total=total_instructions, desc="Testing full dataset") as pbar:
                while True:
                    for traj in self.rollout(**kwargs):
                        if traj['instr_id'] in self.results:
                            looped = True
                        else:
                            self.loss = 0
                            self.results[traj['instr_id']] = traj                        
                            pbar.update(1)
                    if looped:
                        break
                
                

    def test_viz(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj
        else:   # Do a full round
            while True:
                for traj in self.rollout_viz(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj
                if looped:
                    break

class Seq2SeqAgent(BaseAgent):
    env_actions = {
      'left': (0, -1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0, -1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, rank=0):
        super().__init__(env)
        self.args = args

        self.default_gpu = is_default_gpu(self.args)
        self.rank = rank
        
        self.feedback = args.feedback

        # Models
        # if self.args.resume_file is not None:
        self._build_model()

        if self.args.world_size > 1:
            self.vln_bert = DDP(self.vln_bert, device_ids=[self.rank], find_unused_parameters=True)
            self.critic = DDP(self.critic, device_ids=[self.rank], find_unused_parameters=True)

        self.models = (self.vln_bert, self.critic)
        self.device = torch.device('cuda:%d'%self.rank) 

        # Optimizers
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
        if self.default_gpu:
            print('Optimizer: %s' % self.args.optim)

        self.vln_bert_optimizer = optimizer(self.vln_bert.parameters(), lr=self.args.lr)
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.args.lr)
        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

        # Evaluations
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, reduction='sum')

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _build_model(self):
        raise NotImplementedError('child class should implement _build_model: self.vln_bert & self.critic')

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None, viz=False):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        if viz:
            super().test_viz(iters=iters)
        else:
            super().test(iters=iters)

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.vln_bert.train()
        self.critic.train()

        self.losses = []
        for iter in range(1, n_iters + 1):

            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            self.loss = 0

            if self.args.train_alg == 'imitation':
                self.feedback = 'teacher'
                self.rollout(
                    train_ml=1., train_rl=False, **kwargs
                )
            elif self.args.train_alg == 'dagger': 
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(
                        train_ml=self.args.ml_weight, train_rl=False, **kwargs
                    )
                self.feedback = 'expl_sample' if self.args.expl_sample else 'sample'
                self.rollout(train_ml=1, train_rl=False, **kwargs)
            else:
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(
                        train_ml=self.args.ml_weight, train_rl=False, **kwargs
                    )
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)

            #print(self.rank, iter, self.loss)
            self.loss.backward()

            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()

            if self.args.aug is None:
                print_progress(iter, n_iters+1, prefix='Progress:', suffix='Complete', bar_length=50)

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        start = time.time()
        states = torch.load(path, map_location=lambda storage, loc: storage)
        print(f"Checkpoint loaded in {time.time() - start:.2f} seconds")

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
         
            # Regular weight loading logic
            if name not in states:
                print(f"WARNING: {name} weights not found in checkpoint")
                return
                
            load_keys = set(states[name]['state_dict'].keys())
            state_dict = states[name]['state_dict']
            
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
                if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                if list(model_keys)[0].startswith('module.') and (not list(load_keys)[0].startswith('module.')):
                    state_dict = {'module.'+k: v for k, v in state_dict.items()}
                same_state_dict = {}
                extra_keys = []
                for k, v in state_dict.items():
                    if k in model_keys:
                        same_state_dict[k] = v
                    else:
                        extra_keys.append(k)
                state_dict = same_state_dict
                print('Extra keys in state_dict: %s' % (', '.join(extra_keys)))
                
            state.update(state_dict)
            model.load_state_dict(state)
            
            ### Display model info
            if self.args.debug:
                print(f"✔️ Loaded model '{name}' from {path}")
                print(f"Model summary for {name}:")
                print(f"  - Total parameters: {sum(p.numel() for p in model.parameters())}")
                print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
                # print(f"  - First few keys in state_dict: {list(state_dict.keys())[:5]}")
                print(f"\nSample weights from '{name}':")
                printed = 0
                for k, v in model.state_dict().items():
                    print(f" - {k}: shape={tuple(v.shape)}")
                    print(f"   values: {v.view(-1)[:5].tolist()}")  # Print first 5 values
                    printed += 1
                    if printed >= 5:
                        break  # Limit output to avoid clutter
                print('-'*20)
                        
            if self.args.resume_optimizer:
                optimizer.load_state_dict(states[name]['optimizer'])
                
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        
        for param in all_tuple:
            recover_state(*param)
            
        return states['vln_bert']['epoch'] - 1
        
        # # Return the appropriate epoch/iteration value
        # if 'vln_bert' in states and 'epoch' in states['vln_bert']:
        #     return states['vln_bert']['epoch'] - 1
        # elif 'iteration' in states:
        #     return states['iteration']
        # else:
        #     return 0    
        
        


