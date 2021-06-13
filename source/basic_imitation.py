# File for imitation learning on the new basic.wad to compare it to basic reinforcement learning

from __future__ import print_function

from time import sleep
import vizdoom as vzd
from argparse import ArgumentParser

DEFAULT_CONFIG = "lib/ViZDoom/scenarios/basic.cfg"

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

# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
train_epochs = 5
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# Tell the agent to get actions in a supervised manner
supervised = True

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 10#0

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 10

model_savefile = "./model-doom.pth"
save_model = True
load_model = False
skip_learning = False

# Configuration file path
# config_file_path = "../../scenarios/simpler_basic.cfg"
# config_file_path = "../../scenarios/rocket_basic.cfg"
config_file_path = DEFAULT_CONFIG

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

def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)

    # Enables freelook in engine
    # game.add_game_args("+freelook 1")

    # Enables spectator mode, so you can play. Sounds strange but it is the agent who is supposed to watch not you.
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    game.init()
    print("Doom initialized.")

    return game

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

            # One episode is 1 action performed in the current state and the reward given
            state = preprocess(game.get_state().screen_buffer)

            # Get the supervised action here
            #TODO: Make sure every state gets a corresponding action somehow as transmitted in actions
            game.advance_action()
            last_action = game.get_last_action()
            reward = game.get_last_reward()

            map_supervised_action(last_action, actions)

            print("Action:", last_action)
            print("Reward:", reward)
            print("=====================")

            #reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            if not done:
                next_state = preprocess(game.get_state().screen_buffer)
            else:
                next_state = np.zeros((1, 30, 45)).astype(np.float32)

            agent.append_memory(state, last_action, reward, next_state, done)

            # performed actions are given as training examples to the agent after one batch is finished
            if global_step > agent.batch_size:
                #TODO: Bugfix
                agent.train()

            # finished the episode by completing the target
            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1
        
        print("Episode finished!")
        print("Total reward:", game.get_total_reward())
        print("************************")
        sleep(2.0)

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

def map_supervised_action(last_action, actions):
    # This function takes the recorded action and maps it to a actions combination for the DuelQ Agent
    pass

# old if __name__==__main__ of learning_pytorch.py
def result():    

    # Reinitialize the game with window visible
    game.close()
    # game.set_window_visible(True)
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

class DuelQNet(nn.Module):
    """
    This is Duel DQN architecture.
    see https://arxiv.org/abs/1511.06581 for more information.
    """

    def __init__(self, available_actions_count):
        super(DuelQNet, self).__init__()
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

        self.state_fc = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_fc = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, available_actions_count)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 192)
        x1 = x[:, :96]  # input for the net to calculate the state value
        x2 = x[:, 96:]  # relative advantage of actions in the state
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        x = state_value + (advantage_values - advantage_values.mean(dim=1).reshape(-1, 1))

        return x

class DQNAgent:
    def __init__(self, action_size, memory_size, batch_size, discount_factor, 
                 lr, load_model, epsilon=1, epsilon_decay=0.9996, epsilon_min=0.1):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount = discount_factor
        self.lr = lr
        self.memory = deque(maxlen=memory_size)
        self.criterion = nn.MSELoss()

        if load_model:
            print("Loading model from: ", model_savefile)
            self.q_net = torch.load(model_savefile)
            self.target_net = torch.load(model_savefile)
            self.epsilon = self.epsilon_min

        else:
            print("Initializing new model")
            self.q_net = DuelQNet(action_size).to(DEVICE)
            self.target_net = DuelQNet(action_size).to(DEVICE)

        self.opt = optim.SGD(self.q_net.parameters(), lr=self.lr)

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
        batch = random.sample(self.memory, self.batch_size)
        batch = np.array(batch, dtype=object)

        states = np.stack(batch[:, 0]).astype(float)

        #TODO: Bugfix
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

        self.opt.zero_grad()
        td_error = self.criterion(q_targets, action_values)
        td_error.backward()
        self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


if __name__ == "__main__":
    parser = ArgumentParser("ViZDoom example showing how to use SPECTATOR mode.")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")
    args = parser.parse_args()
    #game = vzd.DoomGame()

    # Initialize game and actions
    game = create_simple_game()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    print(actions)

    # Initialize our agent with the set parameters
    agent = DQNAgent(len(actions), lr=learning_rate, batch_size=batch_size,
                     memory_size=replay_memory_size, discount_factor=discount_factor,
                     load_model=load_model)

    episodes = 10

    # Run the training for the set number of epochs
    if not skip_learning:
        agent, game = run(game, agent, actions, num_epochs=train_epochs, frame_repeat=frame_repeat,
                          steps_per_epoch=learning_steps_per_epoch)

        print("======================================")
        print("Training finished. It's time to watch!")

    game.close()