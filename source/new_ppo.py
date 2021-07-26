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
import itertools as it
import skimage.transform

from vizdoom import Mode
from time import sleep, time
from collections import deque
from tqdm import trange
from collections import namedtuple

from base import Agent, Model

# A named tuple to save information about steps that have been performed by the net and what came back from the net
Experience = namedtuple('Experience', ('state', 'action', 'action_log_prob', 'value', 'reward', 'mask'))
   
class ActorCritc(Model):
    
    def __init__(self, available_actions_count, _type) -> None:
        super().__init__(available_actions_count)
        
        #critic
        self.critic = nn.Sequential(
            nn.Linear(self.feature_size(), 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        #actor
        self.actor = nn.Sequential(
            nn.Linear(self.feature_size(), 128),
            nn.ReLU(),
            nn.Linear(128, available_actions_count)
        )
        
        self.type = _type
        
    def initialize_weights(self, layer):
        return super().initialize_weights(layer)
        
    def forward(self, x):
        """forward pass

        Args:
            x (tensor): the state input
            type (string): "actor" or "critic"
        """
        # gets the convulted and flattend input back
        x = super().forward(x)
        
        if self.type=='actor':
            x = self.actor(x)
            x = self.softmax(x)
            dist = Categorical(x)
            return dist
            
        elif self.type=='critic':
            x = self.critic(x)
            return x
            
        else:
            raise(NotImplementedError, "You forgot to give the type 'actor' or 'critic' for the forward pass")
    
    def feature_size(self):
        return super().feature_size()

class PPOAgent(Agent):
    
    def __init__(self, options, action_size) -> None:
        super().__init__(options, action_size)
        
        self.actor_model = ActorCritc(action_size, "actor")
        self.critic_model = ActorCritc(action_size, "critic")
        
        for model in [self.actor_model, self.critic_model]:
            
            if self.opt.weights_dir != '':
                print("Loading model from: ", self.opt.weights_dir)
                model.load_state_dict(torch.load(self.opt.weights_dir))
            
            else:
                # Applies a inital weight to Conv and Linear Layers
                # Currently Xavier Uniform. Orthogonal is also implemented
                model.apply(model.initialize_weights)
            
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.opt.learning_rate)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.opt.learning_rate)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=self.opt.learning_rate)
        
    def step(self, states, actions, action_log_probs, values, rewards, masks, next_state):
        self.memory.append((states, actions, action_log_probs, values, rewards, masks))
        
        self.steps_taken = (self.steps_taken + 1) % self.opt.buffer_update_freq
        
        if self.steps_taken == 0:
            self.optimize(next_state)
            self.memory = []
    
    def act(self, state):
        """Returns action, log_prob, value for given state as per current policy."""
        
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        action_probs = self.actor_model(state)
        value = self.critic_model(state)

        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)

        return action.item(), log_prob, value
    
    def optimize(self, next_state):
        """optimize the weights and biases in the network
        
        """
        # next_state
        # next_state = torch.from_numpy(next_state).float().to(DEVICE) # Propably not needed
        next_value = self.critic_model(next_state)
        
        # Process batch from memory
        memory = Experience(*zip(*self.memory))
        
        batch = {
            'state': torch.stack(memory.state).detach(),
            'action': torch.stack(memory.action).detach(),
            'reward': torch.tensor(memory.reward).detach(),
            'mask': torch.stack(memory.mask).detach(),
            'action_log_prob': torch.stack(memory.action_log_prob).detach(),
            'value': torch.stack(memory.value).detach()
        }
    
    def random_iterator(self, returns, advantage):
        
        # TODO: anders an state.size rankommen
        memory_size = self.states.size(0)
        for _ in range(memory_size // self.batch_size):
            rand_ids = np.random.randint(0, memory_size, self.batch_size)
            yield self.states[rand_ids, :], self.actions[rand_ids], self.log_probs[rand_ids], returns[rand_ids, :], advantage[rand_ids, :]
        
    def get_advantages(values, masks, rewards):
        returns = []
        gae = 0
        gamma = self.opt.discount_factor
        
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + gamma * 0.95 * masks[i] * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    
    def train():
        pass