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

import argparse
from ppo import *

from vizdoom import Mode
import vizdoom as vzd


# Genric Options
parser.add_argument("--scene",
                    type=str,
                    help="run a specific .cfg and .wad",
                    default="../scenarios/simpler_basic.cfg")
parser.add_argument("--mode",
                    type=str,
                    help="run the network in train or evaluation mode",
                    default="train",
                    choices=["train", "eval"])

# DIRECTORY options
parser.add_argument("--exp_name",
                    type=str,
                    help="name of experiment, to be used as save_dir",
                    default="exp_model")
parser.add_argument("--weights_dir",
                    type=str,
                    help="name of model to load",
                    default="")

# TRAIN options
parser.add_argument("--n_train_iterations",
                    type=int,
                    help="number of iterations to train network",
                    default=10) 
parser.add_argument("--learning_rate",
                    type=float,
                    help="learning rate",
                    default=1e-5) # PPO 1e-5
parser.add_argument("--replay_memory_size",
                    type=int,
                    help="number of states to keep in memory for batching",
                    default=10_000)
parser.add_argument("--discount_factor",
                    type=float,
                    help="discount factor used for discounting return",
                    default=0.99)

# PPO specific parameters
parser.add_argument("--n_workers",
                    type=int,
                    help="number of actor critic workers",
                    default=5)
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


if __name__ == '__main__':
    """
    Hier the gives args are laoded and the model is intialized accordingly
    """
    options = parser.parse_args()

    # Initialize a game for each worker and actions
    games = [create_simple_game(options.scene) for i in range options.n_workers]
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    print(actions)

    #TODO: Check where the gameinstances are needed
    # Initialize our agent with the set parameters
    agent = PPOAgent(options, len(actions))

    # Run the training for the set number of epochs
    if options.mode == 'train':
        #TODO: run function in PPO.py
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