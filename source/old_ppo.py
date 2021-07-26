#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# E. Culurciello, L. Mueller, Z. Boztoprak
# December 2020

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


def run(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch=2000):
    """
    Run num epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """

    start_time = time()

    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0
        print("\nEpoch #" + str(epoch + 1))

        for _ in trange(steps_per_epoch, leave=False):
            state = preprocess(game.get_state().screen_buffer)
            action = agent.get_action(state)
            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            if not done:
                next_state = preprocess(game.get_state().screen_buffer)
            else:
                next_state = np.zeros((1, 30, 45)).astype(np.float32)

            agent.append_memory(state, action, reward, next_state, done)

            if global_step > agent.batch_size:
                agent.train()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1

        agent.update_target_net()
        train_scores = np.array(train_scores)

        print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()),
              "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

        test(game, agent)
        if save_model:
            print("Saving the network weights to:", model_savefile)
            torch.save(agent.q_net, model_savefile)
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    game.close()
    return agent, game

class ActorCritic(nn.Module):
    
    def __init__(self, available_actions_count):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        #critic
        self.critic = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        #actor
        self.actor = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, available_actions_count)
        )

        self.softmax = nn.Softmax()

        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        """
        forward pass through the net producing the state value and the action logits

        input: 
            x (tensor): batch of input states to be processed

        return:
            state_value (type) = [descripton]
            action_logits (type) = [descripton]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x1 = x[:, :96]  # input for the net to calculate the state value
        x2 = x[:, 96:]  # relative advantage of actions in the state
        state_value = self.critic(x1).reshape(-1, 1)
        action_logits = self.actor(x2)
        
        return state_value, action_logits
    
    def act(self, x):
        """
        TODO: add docstring

        Input:
            x
        """
        # Forward pass
        values, action_logits = self.forward(x)
        probs = self.softmax(action_logits)
        log_probs = self.logsoftmax(action_logits)

        # Choose action stochastically
        actions = probs.multinomial(1)

        # Evaluate action
        action_log_probs = log_probs.gather(1, actions)
        return values, actions, action_log_probs

    def evaluate_action(self, x, action):
        """
        TODO: Add docstring
        """
        # Forward pass 
        value, action_logits = self.forward(x)
        probs = self.softmax(action_logits)
        log_probs = self.logsoftmax(action_logits)

        # Evaluate actions
        action_log_probs = log_probs.gather(1, actions)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return value, action_log_probs, dist_entropy

class PPOAgent:

    def __init__(self, options, action_size):
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
        
        Missing:
            --input_shape (resultion)
            --tau value
        """
        
        self.opt = options
        self.action_size = action_size
        self.memory = deque(maxlen=self.opt.replay_memory_size)
        self.criterion = nn.MSELoss()

        self.ppo = ActorCritic(action_size)

        if self.opt.weights_dir != '':
            print("Loading model from: ", self.opt.weights_dir)
            self.ppo.load_state_dict(torch.load(self.opt.weights_dir))
            
        self.optimizer = optim.Adam(self.ppo.parameters(), lr=self.opt.learning_rate)

        self.ppo.to(DEVICE)
    
    def append_memory(self, states, actions, action_log_probs, values, rewards, masks):
        self.memory.append((states, actions, action_log_probs, values, rewards, masks))
    
    # TODO: not fitted for DOOM yet. Last change marked
    def train(self):
        """
        TODO: add docstring
    
        """
        # Episode length: How many episode are played by each worker
        episode_lengths = np.zeros(self.opt.n_workers)

        # Get a inital state of the game without performing any actions
        initial_actions = np.zeros(self.opt.n_workers) # TODO: fit this to DOOM
        states, _, _ = self.env_step(None, initial_actions) # Agent does a step in the enviroment

        # Start an episode
        for eps_num in range(1, self.opt.n_train_iterations):

            # Perform a forward pass
            values, actions, action_log_probs = self.ppo.act(states)

            # Perform the retrieved action that came back from the forward pass
            next_state, rewards, dones = self.env_step(states, actions)
            masks = torch.FloatTensor([[0.0] if done else [1.0] for done in dones])

            # Save experience to buffer
            self.append_memory(
                states.data, actions.data, action_log_probs.data, values.data, rewards, masks
            )

            # Perform optimization #TODO: Check alternative for optimize_model
            if eps_num % self.opt.buffer_update_freq == 0:
                loss, value_loss, action_loss, entropy_loss = self.optimize_model()
                # Reset memory
                self.memory = []

            # Log episode length
            for worker in range(self.opt.n_workers):
                if not dones[worker]:
                    episode_lengths[worker] += 1
                else:
                    self.writer.add_scalar('episode_length/' + str(worker), episode_lengths[worker], eps_num)
                    print(worker, episode_lengths[worker])
                    episode_lengths[worker] = 0

            # Save network
            if eps_num % self.opt.save_frequency == 0:
                if not os.path.exists(self.opt.exp_name):
                    os.mkdir(self.opt.exp_name)
                torch.save(self.net.state_dict(), f'{self.opt.exp_name}/{str(i).zfill(7)}.pt')

            # Write results to log
            if eps_num % self.opt.log_frequency == 0:
                self.writer.add_scalar('loss/total', loss, eps_num)
                self.writer.add_scalar('loss/action', action_loss, eps_num)
                self.writer.add_scalar('loss/value', value_loss, eps_num)
                self.writer.add_scalar('loss/entropy', entropy_loss, eps_num)

            # Move on to next state
            states = next_states

        """
        -----------------------------------------

        -----------------------------------------

        -----------------------------------------

        -----------------------------------------
        From here: Old training
        """
        batch = random.sample(self.memory, self.batch_size)
        batch = np.array(batch, dtype=object)

        states = np.stack(batch[:, 0]).astype(float)
        """
        with open('Batch_Normal.txt', 'w') as f:
            f.write(str(batch[:, 3]))
        print(batch[:, 3].shape)

        raise Exception('')
        """
        actions = batch[:, 1].astype(int)
        rewards = batch[:, 2].astype(float)
        next_states = np.stack(batch[:, 3]).astype(float)
        dones = batch[:, 4].astype(bool)
        not_dones = ~dones

        row_idx = np.arange(self.batch_size)  # used for indexing the batch

        # value of the next states with double q learning
        # see https://arxiv.org/abs/1509.06461 for more information on double q learning
        with torch.no_grad():
            next_states = torch.from_numpy(next_states).float().to(DEVICE)
            idx = row_idx, np.argmax(self.q_net(next_states).cpu().data.numpy(), 1)
            next_state_values = self.target_net(next_states).cpu().data.numpy()[idx]
            next_state_values = next_state_values[not_dones]

        # this defines y = r + discount * max_a q(s', a)
        q_targets = rewards.copy()
        q_targets[not_dones] += self.discount * next_state_values
        q_targets = torch.from_numpy(q_targets).float().to(DEVICE)

        # this selects only the q values of the actions taken
        idx = row_idx, actions
        states = torch.from_numpy(states).float().to(DEVICE)
        action_values = self.q_net(states)[idx].float().to(DEVICE)

        self.optimizer.zero_grad()
        td_error = self.criterion(q_targets, action_values)
        td_error.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    # Not fitted for Doom [chage via try-error implementation (run, bugs, print, fix, repeat)]
    def optimize_model(self):
        """
        Performs a single step of optimization.

        Arguments:
            next_state (tensor): next frame of the game
            done (bool): True if next_state is a terminal state, else False

        Returns:
            loss (float)
        """
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
        # TODO: Check if these dimensions are still valid
        state_shape = batch['state'].size()[2:]
        action_shape = batch['action'].size()[-1]

        # Compute returns
        returns = torch.zeros(self.para['buffer'] + 1, 8, 1)
        for i in reversed(range(self.para['buffer'])):
            returns[i] = returns[i+1] * 0.99 * batch['mask'][i] + batch['reward'][i]
        returns = returns[:-1]
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Process batch
        values, action_log_probs, dist_entropy = self.model.evaluate_actions(batch['state'].view(-1, *state_shape), batch['action'].view(-1, action_shape)) ### HERE
        values = values.view(self.para['buffer'], 8, 1)
        action_log_probs = action_log_probs.view(self.para['buffer'], 8, 1)

        # Compute advantages
        advantages = returns - values.detach()

        # Action loss
        ratio = torch.exp(action_log_probs - batch['action_log_prob'].detach())
        surr1 = ratio * advantages 
        surr2 = torch.clamp(ratio, 1-self.opt.grad_clip, 1+self.opt.grad_clip) * advantages
        action_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = (returns - values).pow(2).mean()
        value_loss = self.opt.value_loss_coeff * value_loss

        # Total loss
        loss = value_loss + action_loss - dist_entropy * self.opt.entropy_coeff

        # Optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.net.parameters(), self.opt.max_grad_norm)
        self.optimizer.step()

        return loss, value_loss * self.opt.value_loss_coeff, action_loss, - dist_entropy * self.opt.entropy_coeff
    
    #TODO: Unchanged starting here
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


