#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import vizdoom as vzd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import itertools as it
import skimage.transform

from vizdoom import Mode
from time import sleep, time
from collections import deque
from tqdm import trange

from base import Agent, Model

# Other parameters
frame_repeat = 12
resolution = (30, 45)

# A named tuple to save information about steps that have been performed by the net and what came back from the net
Experience = namedtuple('Experience', ('state', 'action', 'action_log_prob', 'value', 'reward', 'mask'))

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device('cpu')
    
def preprocess(img):
    """Down samples image to resolution"""
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    return img

def test(game, agent):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            game.make_action(actions[best_action_index], frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print("Results: mean: %.1f +/- %.1f," % (
        test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
          "max: %.1f" % test_scores.max())
    
class ActorCritc(Model):
    
    def __init__(self, available_actions_count) -> None:
        super().__init__(available_actions_count)
    
    def forward(self, x):
        return super().forward(x)
    
    def feature_size(self):
        return super().feature_size()

class PPOAgent(Agent):
    
    def __init__(self, options, action_size) -> None:
        super().__init__(options, action_size)
        
        self.net = ActorCritc()
        
        if self.opt.weights_dir != '':
            print("Loading model from: ", self.opt.weights_dir)
            self.net.load_state_dict(torch.load(self.opt.weights_dir))
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.opt.learning_rate)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=self.opt.learning_rate)
        
    def step(self, states, actions, action_log_probs, values, rewards, masks):
        self.memory.append((states, actions, action_log_probs, values, rewards, masks))
    
    def act(self, state):
        return super().act(state, self.net)
    
    def optimize(self):
        """optimize the weights and biases in the network
        
        old ppo line 386
        """
        
    