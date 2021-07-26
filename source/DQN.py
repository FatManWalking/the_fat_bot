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
            nn.Linear(3944, 64), # 30 x 45 = 96, 64
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_fc = nn.Sequential(
            nn.Linear(3944, 64), # 30 x 45 = 96, 64
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

class DQNAgent:
    def __init__(self, options, action_size, epsilon=1, epsilon_decay=0.9996, epsilon_min=0.1, scheduler=None):
        
        print(options)
        self.action_size = action_size
        self.opt = options
        self.memory = deque(maxlen=options.memory_size)
        self.criterion = nn.MSELoss()
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        if options.load_model:
            print("Try loading model from: ", options.weights_dir)
            #try: 
            self.q_net = torch.load(options.weights_dir)
            self.target_net = torch.load(options.weights_dir)
            self.epsilon = self.epsilon_min
            #except:
            #    raise FileNotFoundError(f"There was no file with name {options.weights_dir}")

        else:
            print("Initializing new model")
            self.q_net = DuelQNet(action_size).to(DEVICE)
            self.target_net = DuelQNet(action_size).to(DEVICE)

        self.optim = optim.SGD(self.q_net.parameters(), lr=options.learning_rate)

        if scheduler:
            # self.scheduler = CosineScheduler(30, warmup_steps=5, base_lr=self.lr, final_lr=self.lr*0.0001)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, 30, eta_min=0, last_epoch=-1, verbose=False)
        else:
            self.scheduler = None

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(DEVICE)
            action = torch.argmax(self.q_net(state)).item()
            return action

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
                    
        batch = random.sample(self.memory, self.opt.batch_size)
        batch = np.array(batch, dtype=object)

        states = np.stack(batch[:, 0]).astype(float)
        actions = batch[:, 1].astype(int)
        rewards = batch[:, 2].astype(float)
        next_states = np.stack(batch[:, 3]).astype(float)
        dones = batch[:, 4].astype(bool)
        not_dones = ~dones

        row_idx = np.arange(self.opt.batch_size)  # used for indexing the batch

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

        self.optim.zero_grad()
        td_error = self.criterion(q_targets, action_values)
        self.optim.step()

        if self.scheduler:
            if self.scheduler.__module__ == optim.lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                self.scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        

# https://d2l.ai/chapter_optimization/lr-scheduler.html
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

"""
if __name__ == '__main__':
    # Initialize game and actions
    game = create_simple_game()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Initialize our agent with the set parameters
    agent = DQNAgent(len(actions), lr=learning_rate, batch_size=batch_size,
                     memory_size=replay_memory_size, discount_factor=discount_factor,
                     load_model=load_model, scheduler=True)

    # Run the training for the set number of epochs
    if not skip_learning:
        agent, game = run(game, agent, actions, num_epochs=train_epochs, frame_repeat=frame_repeat,
                          steps_per_epoch=learning_steps_per_epoch)

        print("======================================")
        print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.close()
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
"""