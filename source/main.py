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

from new_ppo import PPOAgent
from HP_Tuning_Q import DQNAgent
import torch

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

parser = argparse.ArgumentParser(description="options")

# Genric Options
parser.add_argument("--scene",
                    type=str,
                    help="run a specific .cfg and .wad",
                    default="../wads/BasicAugment.cfg")
parser.add_argument("--mode",
                    type=str,
                    help="run the network in train or evaluation mode",
                    default="train",
                    choices=["train", "eval"])
parser.add_argument("--frame_repeat",
                    type=int,
                    help="repeat frame n times",
                    default=12)
# DIRECTORY options
parser.add_argument("--model_name",
                    type=str,
                    help="name of experiment, to be used as save_dir",
                    default="model_name")
parser.add_argument("--weights_dir",
                    type=str,
                    help="name of model to load",
                    default="")

# TRAIN options
parser.add_argument("--model",
                    type=str,
                    help="the model architecture",
                    default='DQN',
                    choices=["DQN", "PPO"])
parser.add_argument("--epochs",
                    type=int,
                    help="number of iterations to train network",
                    default=10)
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
parser.add_argument("--replay_memory_size",
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
                    default=True)
parser.add_argument("--test_episodes",
                    type=int,
                    help="number of episode to test",
                    default=10)
parser.add_argument("--watch_episodes",
                    type=int,
                    help="number of episode to watch",
                    default=10)


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

def preprocess(img):
    """Down samples image to resolution"""
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def test(game, agent, options):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for test_episode in trange(options.test_episodes, leave=False):
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

# TODO: implement having more than one worker
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
                next_state = np.zeros((1, 90, 135)).astype(np.float32) # 30x45 = 30x45

            agent.append_memory(state, action, reward, next_state, done)

            if global_step > agent.batch_size:
                agent.train()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1

        agent.update_target_net()
        overall_train_score.append(train_scores)
        with open(text_file,"w") as textfile:
            textfile.write(str(overall_train_score))
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


if __name__ == '__main__':
    """
    Hier the gives args are laoded and the model is intialized accordingly
    """
    options = parser.parse_args()

    # Initialize a game for each worker and actions
    if options.model == 'PPO':
        games = [create_simple_game(options.scene) for i in range(options.n_workers)]
        n = games[0].get_available_buttons_size()
    
    elif options.model == 'DQN':
        games = create_simple_game(options.scene)
        n = games.get_available_buttons_size()
    else:
        raise(NotImplementedError, "The only model options available are DQN and PPO.")
    
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    # print(actions)

    #TODO: Check where the gameinstances are needed
    # Initialize our agent with the set parameters
    
    if options.model == 'PPO':
        agent = PPOAgent(options, len(actions))
        
    elif options.model == 'DQN':
        agent = DQNAgent(options, len(actions))

    # Run the training for the set number of epochs
    if options.mode == 'train':
        
        agent, game = run(games, agent, actions, num_epochs=options.epochs, frame_repeat=options.frame_repeat,
                          steps_per_epoch=options.steps)

        print("======================================")
        print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.close()
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    games.init()

    for _ in range(options.watch_episodes):
        games.new_episode()
        while not game.is_episode_finished():
            state = preprocess(games.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(options.frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)