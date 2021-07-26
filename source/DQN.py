#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import vizdoom as vzd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import random
import math

from vizdoom import Mode
from time import sleep
from collections import deque
from collections import namedtuple

from base import Agent, Model

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device('cpu')


class DuelQNet(Model):
    """
    This is Duel DQN architecture.
    see https://arxiv.org/abs/1511.06581 for more information.
    """
    def __init__(self, available_actions_count) -> None:
        #self.convultions, self.softmax, self.logsoftmax = 
        super().__init__(available_actions_count)
        
        self.state_fc = nn.Sequential(
            nn.Linear(3944, 256), # 30 x 45 = 96, 64
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_fc = nn.Sequential(
            nn.Linear(3944, 256), # 30 x 45 = 96, 64
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, available_actions_count)
        )
    
    def initialize_weights(self, layer):
        return super().initialize_weights(layer)
    
    def feature_size(self):
        return super().feature_size()

    def forward(self, x):

        x, size = super().forward(x)
        size = int(size//2)

        x1 = x[:, :size]  # input for the net to calculate the state value # 30x45 = 96
        x2 = x[:, size:]  # relative advantage of actions in the state
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        x = state_value + (advantage_values - advantage_values.mean(dim=1).reshape(-1, 1))

        return x

class DQNAgent(Agent):

    def __init__(self, options, action_size, epsilon=1, epsilon_decay=0.9996, epsilon_min=0.1, scheduler=None) -> None:
        
        super().__init__(options, action_size)
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_net = DuelQNet(action_size)
        self.target_net = DuelQNet(action_size)
        
        for model in [self.q_net, self.target_net]:
            
            if self.opt.weights_dir != '' and self.opt.load_model:
                print("Loading model from: ", self.opt.weights_dir)
                model.load_state_dict(torch.load(self.opt.weights_dir))
                self.epsilon = self.epsilon_min
                
            else:
                # Applies a inital weight to Conv and Linear Layers
                # Currently Xavier Uniform. Orthogonal is also implemented
                model.apply(model.initialize_weights)
            
            model.to(DEVICE)
            
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.opt.learning_rate)
        # self.optimizer = optim.SGD(self.q_net.parameters(), lr=self.opt.learning_rate)

        self.scheduler = self.get_scheduler(scheduler)

    def get_scheduler(self, scheduler):
        return super().get_scheduler(scheduler)
    
    def act(self, state):
        
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size)), None, None
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(DEVICE)
            action = torch.argmax(self.q_net(state)).item()
            return action, None, None

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        
        states, actions, rewards, next_states, dones, not_dones, row_idx = super().train()

        # value of the next states with double q learning
        # see https://arxiv.org/abs/1509.06461 for more information on double q learning
        with torch.no_grad():
            next_states = torch.from_numpy(next_states).float().to(DEVICE)
            idx = row_idx, np.argmax(self.q_net(next_states).cpu().data.numpy(), 1)
            next_state_values = self.target_net(next_states).cpu().data.numpy()[idx]
            next_state_values = next_state_values[not_dones]

        # this defines y = r + discount * max_a q(s', a)
        q_targets = rewards.copy()
        q_targets[not_dones] += self.opt.discount_factor * next_state_values
        q_targets = torch.from_numpy(q_targets).float().to(DEVICE)

        # this selects only the q values of the actions taken
        idx = row_idx, actions
        states = torch.from_numpy(states).float().to(DEVICE)
        action_values = self.q_net(states)[idx].float().to(DEVICE)

        self.optimizer.zero_grad()
        td_error = self.criterion(q_targets, action_values)
        td_error.backward()
        self.optimizer.step()

        if self.scheduler:
            if self.scheduler.__module__ == optim.lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                self.scheduler.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
      

# https://d2l.ai/chapter_optimization/lr-scheduler.html
# A Cosinus Scheduler to gradually decrease the learning rate
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0,
                 warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                    math.pi *
                    (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr