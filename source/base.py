from abc import ABC, abstractmethod

from time import sleep, time
from collections import deque
import random
from tqdm import trange

import vizdoom as vzd
from vizdoom import Mode

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np

class Agent(ABC):
    
    @abstractmethod
    def __init__(self, options, action_size) -> None:
        """Initalize an Agent for PPO

        Args:
            options (dict): Contains all terminal defined options full list below        
            action_size (int): number of actions that can be taken each step
            
        Options (full list):
            --scene:                the .cfg and .wad
            --mode:                 "train" or "eval"
            --model_name:           name of the model file to be saved
            --weights_dir:          name of the model file to be loaded
            --n_train_iterations:   number of epochs
            --learning_rate:        learning rate (both actor and critic, same for simplicity)
            --replay_memory_size:   batch size
            --discount_factor:      discount / gamma factor
            --n_workers:            number of actor critic workers
            --buffer_update_freq    refresh buffer after every x actions
            --entropy_coeff         entropy regularization weight
            --value_loss_coeff      value loss regularization weight
            --max_grad_norm         norm bound for clipping gradients
            --grad_clip             magnitude bound for clipping gradients
            --log_frequency         number of batches between each tensorboard log
            --save_frequency        number of batches between each model save
        """
        super().__init__()
        self.opt = options
        self.action_size = action_size
        self.memory = deque(maxlen=self.opt.replay_memory_size)
        
        self.criterion = nn.MSELoss()
        
        """ Generic steps that need the actual Model class object
        self.net = Model()
        
        if self.opt.weights_dir != '':
            print("Loading model from: ", self.opt.weights_dir)
            self.net.load_state_dict(torch.load(self.opt.weights_dir))
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.opt.learning_rate)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.opt.learning_rate)
        """
              
    @abstractmethod
    def step(self):
        """appends everything related to one step to the memory
        """
        pass
    
    @abstractmethod
    def act(self, state, actor):
        """
        Returns action, log_prob, value for given state as per current policy.
        """
        
        # Forward pass
        values, action_logits = actor.forward(state)
        probs = actor.softmax(action_logits)
        log_probs = actor.logsoftmax(action_logits)

        # Choose action stochastically
        actions = probs.multinomial(1)
        
        # Evaluate action
        action_log_probs = log_probs.gather(1, actions)
        
        return values, actions, action_log_probs
    
    @abstractmethod
    def optimize(self):
        pass
    
    @abstractmethod
    def train(self):
        pass

class Model(ABC, nn.Module):
    
    @abstractmethod
    def __init__(self, available_actions_count) -> None:
        super().__init__()
        
        
        self.convultion = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            # Conv Layer 2
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            # Conv Layer 3
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            # Conv Layer 4
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()     
        )

        #critic
        self.critic = nn.Sequential(
            nn.Linear(self.feature_size(), 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        #actor
        self.actor = nn.Sequential(
            nn.Linear(self.feature_size(), 64),
            nn.ReLU(),
            nn.Linear(64, available_actions_count)
        )

        self.softmax = nn.Softmax()

        self.logsoftmax = nn.LogSoftmax()
    
    @abstractmethod
    def forward(self, x):
        """
        forward pass through the net producing the state value and the action logits

        input: 
            x (tensor): batch of input states to be processed

        return:
            state_value = forward of critic
            action_logits = forward of actor
        """
        x = self.convultion(x)
        size = x.size(1)*x.size(2)*x.size(3)
        x = x.view(-1, size)
        
        #TODO: Check if correct
        x1 = x[:, :int(size//2)]  # input for the net to calculate the state value
        x2 = x[:, int(size//2):]  # relative advantage of actions in the state
        state_value = self.critic(x1).reshape(-1, 1)
        action_logits = self.actor(x2)
        
        return state_value, action_logits
    
    @abstractmethod
    def feature_size(self):
        return self.convultion(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    
    