"""
This is our version of a PPO trained doom agent.
This main file contains a argparse for all important hyperparameters to be tuned
including the .cfg and .wad of the example to be loaded.

This structur and the PPO agent are based on the Implementation of PPO by the OpenAI team.
Reference:
    "Proximal Policy Optimization Algorithms" Schulman et al.

And the Github https://github.com/amanda-lambda/drl-experiments/blob/master/ppo.py
--> A PPO implementation for Flappy Bird
"""

from PPO import *
from DQN import *
import torch
import skimage.transform

from vizdoom import Mode
import vizdoom as vzd

import argparse
import numpy as np
import itertools as it
from time import sleep, time
from tqdm import trange

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device('cpu')

overall_train_score = []

parser = argparse.ArgumentParser(description="options")

# Genric Options
parser.add_argument("--scene",
                    type=str,
                    help="run a specific .cfg and .wad",
                    default="../wads/New.cfg")
parser.add_argument("--frame_repeat",
                    type=int,
                    help="repeat frame n times",
                    default=12)

# DIRECTORY options
parser.add_argument("--model_name",
                    type=str,
                    help="name of experiment, to be used as save_dir",
                    default="../models/DuelQ_from_basic.pth")
parser.add_argument("--weights_dir",
                    type=str,
                    help="name of model to load",
                    default='../models/DuelQ_from_basic.pth')
parser.add_argument("--text_file",
                    type=str,
                    help="name of model to load",
                    default='../rewards/DuelQ_from_basic.txt')

# TRAIN options
parser.add_argument("--model",
                    type=str,
                    help="the model architecture",
                    default='DQN',
                    choices=["DQN", "PPO"])
parser.add_argument("--epochs",
                    type=int,
                    help="number of iterations to train network",
                    default=1_000)
parser.add_argument("--steps",
                    type=int,
                    help="number of steps per episode",
                    default=1000)
parser.add_argument("--batch_size",
                    type=int,
                    help="number of states per batch",
                    default=64)
parser.add_argument("--learning_rate",
                    type=float,
                    help="learning rate",
                    default=1e-5) # PPO 1e-5
parser.add_argument("--memory_size",
                    type=int,
                    help="number of states to keep in memory for batching",
                    default=1_000)
parser.add_argument("--discount_factor",
                    type=float,
                    help="discount factor used for discounting return",
                    default=0.99)
parser.add_argument("--resultion",
                    type=tuple,
                    help="resultion used for training",
                    default=(90, 135))

# PPO specific parameters
parser.add_argument("--n_workers",
                    type=int,
                    help="number of actor critic workers",
                    default=1)
parser.add_argument("--buffer_update_freq",
                    type=int,
                    help="refresh buffer after every x actions",
                    default=20)
parser.add_argument("--entropy_coeff",
                    type=float,
                    help="entropy regularization weight",
                    default=0.01)
parser.add_argument("--value_loss_coeff",
                    type=float,
                    help="value loss regularization weight",
                    default=0.5)
parser.add_argument("--max_grad_norm",
                    type=int,
                    help="norm bound for clipping gradients",
                    default=40)
parser.add_argument("--grad_clip",
                    type=float,
                    help="magnitude bound for clipping gradients",
                    default=0.1)

# LOGGING options
parser.add_argument("--log_frequency",
                    type=int,
                    help="number of batches between each tensorboard log",
                    default=100)
parser.add_argument("--save_frequency",
                    type=int,
                    help="number of batches between each model save",
                    default=1_000)

# Training, Loading, Testing
parser.add_argument("--save_model",
                    type=bool,
                    help="save model after training",
                    default=True)
parser.add_argument("--load_model",
                    type=bool,
                    help="load model before training",
                    default=True)
parser.add_argument("--skip_training",
                    type=bool,
                    help="skip training",
                    default=False)
parser.add_argument("--test_episodes",
                    type=int,
                    help="number of episode to test",
                    default=10)
parser.add_argument("--watch_episodes",
                    type=int,
                    help="number of episode to watch",
                    default=10)
parser.add_argument("--save_freq",
                    type=int,
                    help="number of episode to train before saving",
                    default=10)

# Source: Vizdoom 
def create_simple_game(config_file_path):
    """
    creates a simple run of the game to train the agent(s)

    Hides the window and sets the color to a normalized grayscale to save resources
    This mode is really only used for training and is tuned to save resources elsewhere
    """
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")

    return game

# Source: Vizdoom 
def preprocess(img, resultion):
    """Down samples image to resolution"""
    img = skimage.transform.resize(img, resultion)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

# Source: Vizdoom 
def test(game, agent, options):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for _ in trange(options.test_episodes, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            # gets the picture of what you would see in the game
            state = preprocess(game.get_state().screen_buffer, options.resultion)
            # the image goes through the CNN and returns an index that correlates to a action in the game
            best_action_index, _, _ = agent.act(state)
            # actually performing that action
            game.make_action(actions[best_action_index], options.frame_repeat)
        
        # get the reward, changes of the action had a good/bad effect or a living penalty is in place
        reward = game.get_total_reward()
        # log the score
        test_scores.append(reward)

    test_scores = np.array(test_scores)
    print("Results: mean: %.1f +/- %.1f," % (
        test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
          "max: %.1f" % test_scores.max())

# Removed hardcoded parameters and replaced them with dynamic solultions
def run(game, agent, actions, options):
    """
    Run num epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """

    start_time = time()

    for epoch in range(options.epochs):
        game.new_episode()
        train_scores = []
        global_step = 0
        print("\nEpoch #" + str(epoch + 1))

        for _ in trange(options.steps, leave=False):
            state = preprocess(game.get_state().screen_buffer, options.resultion)
            action, _, _ = agent.act(state)
            reward = game.make_action(actions[action], options.frame_repeat)
            done = game.is_episode_finished()

            if not done:
                next_state = preprocess(game.get_state().screen_buffer, options.resultion)
            else:
                x = (1,) + options.resultion
                next_state = np.zeros(x).astype(np.float32)

            agent.append_memory(state, action, reward, next_state, done)

            if global_step > options.batch_size:
                agent.train()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1

        agent.update_target_net()
        overall_train_score.append(train_scores)
        with open(options.text_file,"w") as textfile:
            textfile.write(str(overall_train_score))
        train_scores = np.array(train_scores)
        

        print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()),
              "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

        test(game, agent, options)
        if options.model_name and not (epoch%options.save_freq):
            print("Saving the network weights to:", options.model_name)
            torch.save(agent.q_net.state_dict(), options.model_name)
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    game.close()
    return agent, game


if __name__ == '__main__':
    """
    Hier the gives args are laoded and the model is intialized accordingly
    """
    options = parser.parse_args()

    # Initialize a game for each worker and actions
    if options.model == 'PPO':
        game = [create_simple_game(options.scene) for i in range(options.n_workers)]
        n = game[0].get_available_buttons_size()
    
    elif options.model == 'DQN':
        game = create_simple_game(options.scene)
        n = game.get_available_buttons_size()
    else:
        raise(NotImplementedError, "The only model options available are DQN and PPO.")
    
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    # print(actions)

    #TODO: Check where the gameinstances are needed
    # Initialize our agent with the set parameters
    
    if options.model == 'PPO':
        agent = PPOAgent(options, len(actions), scheduler=True)
        
    elif options.model == 'DQN':
        agent = DQNAgent(options, len(actions), scheduler=True)

    # Run the training for the set number of epochs
    if not options.skip_training:
        
        agent, game = run(game, agent, actions, options)

        print("======================================")
        print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.close()
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(options.watch_episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer, options.resultion)
            best_action_index, _, _ = agent.act(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(options.frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)