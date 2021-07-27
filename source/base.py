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

"""
The Goal of this file is a baseline overview over Reinforcement-Learning algorithms

Agent and Model class are both written in a way that DQN as well as PPO can inherit from them by only calling super().
Our Goal was to only have the method specific things in the according files to mark out their differences. This helps to
keep the code clean and easy to debug, as well as giving as a basic understanding what seperates them in practise 
oppose to only reading theorically about them.
"""

class Agent(ABC):
    
    @abstractmethod
    def __init__(self, options, action_size) -> None:
        """Initalize an Agent for Reinforcement Learning

        Args:
            options (dict): Contains all terminal defined options list below or in main.py for full list      
            action_size (int): number of actions that can be taken each step
            
        Options (full list):

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
            --save_frequency        number of batches between each model save
        """
        super().__init__()
        
        # Set the configured options and the given action_size
        self.opt = options
        self.action_size = action_size
        
        # The agent has a memory where he stores states, the action he took when being in that state and things like he got for that
        # This determines how many samples there are in the memory to randomly batch from
        self.memory = deque(maxlen=self.opt.memory_size)
        
        # Mean Squared Error Loss Function
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
    def get_scheduler(self, scheduler):
        """gradually schedule the learning rate

        tried different schedulers, saw no significant change between those with otherwise same parameters
        
        Args:
            scheduler (bool): True or False, if u want to use a scheduler or not

        Returns:
            torch lr_scheduler or None: scheduler if true, None if False
        """
        # return CosineScheduler(30, warmup_steps=5, base_lr=self.lr, final_lr=self.lr*0.0001)
        if scheduler:
            #from SGDR: Stochastic Gradient Descent with Warm Restarts by Ilya Loshchilov and Frank Hutter but without restarts
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 30, eta_min=0, last_epoch=-1, verbose=False)
    
    @abstractmethod
    def append_memory(self):
        """appends everything related to one step to the memory
        """
        pass
    
    @abstractmethod
    def act(self, state, actor):
        """
        Returns the action to be taken by the agent.
        """
        pass
    
    @abstractmethod
    def train(self):
        """trains the agent and optimizes the network

        takes a batch out of the stored states and values (memory)
        base class returns the batch output after some processing
        for methode specific training
        
        Returns:
            states, actions, rewards, next_states, dones, not_dones, row_idx
        """
        batch = random.sample(self.memory, self.opt.batch_size)
        batch = np.array(batch, dtype=object)

        states = np.stack(batch[:, 0]).astype(float)
        actions = batch[:, 1].astype(int)
        rewards = batch[:, 2].astype(float)
        next_states = np.stack(batch[:, 3]).astype(float)
        dones = batch[:, 4].astype(bool)
        not_dones = ~dones
        
        row_idx = np.arange(self.opt.batch_size)  # used for indexing the batch
        
        return states, actions, rewards, next_states, dones, not_dones, row_idx

class Model(ABC, nn.Module):
    
    @abstractmethod
<<<<<<< HEAD
    def __init__(self, available_actions_count) -> None:
        """defining the CNN architecture for the Agent to decide with

        Args:
            available_actions_count (int): numbers of actions the agent can take in a single step
        """
=======
    def __init__(self, available_actions_count, input_shape) -> None:
>>>>>>> 8e322f8440b6505c0afbd2844fca9cacbd35b015
        super().__init__()
        
        self.input_shape = (1, ) + input_shape
        self.convultions = nn.Sequential(
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

        self.softmax = nn.Softmax()

        self.logsoftmax = nn.LogSoftmax()

        # return convultions, softmax, logsoftmax
    
    @abstractmethod 
    def initialize_weights(self, layer):
        """initalizes the weights and biases

        initalizes the weights and biases in case there was no state_dict provided to load them from
        model_apply() iterativly goes through every layer and calls this methode
        
        Args:
            layer (nn.Layer): the layer to be initalized
        """
        # gain = nn.init.calculate_gain(self.cfg.nonlinearity)
        init_weight = 'xavier_uniform'
        gain = 1

        if init_weight == 'orthogonal':
            if type(layer) == nn.Conv2d:
                nn.init.orthogonal_(layer.weight.data, gain=gain)
            elif type(layer) == nn.Linear:
                nn.init.orthogonal_(layer.weight.data, gain=gain)
                layer.bias.data.fill_(0)
            else:
                pass
            
        elif init_weight == 'xavier_uniform':
            if type(layer) == nn.Conv2d:
                nn.init.xavier_uniform_(layer.weight.data, gain=gain)

            elif type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight.data, gain=gain)
                layer.bias.data.fill_(0)

            else:
                pass
            
    @abstractmethod
    def forward(self, x):
        """
        forward pass through the net producing the state value and the action logits

        input: 
            x (tensor): batch of input states to be processed

        return:
            x (tensor): tensor after convultion and flatten
        """
        x = self.convultions(x)
        
        size = x.size(1)*x.size(2)*x.size(3)
        x = x.view(-1, size)
        
        return x, size
        
    @abstractmethod
    def feature_size(self):
        """dynamically sets the dimension of the first linear Layer
        
        by calling this a forward pass through the layers definied in features is performed
        and the resulting output_size can be used in the first linear Layer. Effectivly making
        the net independend of the chosen resultion

        Returns:
            int: size of the flattend ccn output before linear layers
        """
        
        return self.convultions(autograd.Variable(torch.zeros(1, * self.input_shape))).view(1, -1).size(1)
    
    
    