# The Fat Bot
A reinforcement trained bot for doom, created as part of a machine learning research project at DHBW Mannheim

## Clone this Repository
This project is based on VizDoom. After cloning this repository by the method of your choice set up VizDOOM.\
To run vizdoom follow this [Guide](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#linux_build) provided by VizDoom itself.
We personally recommend the setup over a Linux-Setup, which was also the only setup we developed this project in (Ubuntu 20.4).
We also provide a YAML `Doom_Yaml.yml`, which we know is a functional conda enviroment for this project.

![Scene aus VizDOOM](https://i.imgur.com/lHclExd.png)


## ⚙️ Running the code
Almost every parameter is adjustable via terminal commands when executing the code.

Examples:
```sh
# General format of commands
python main.py --model=<DQN, PPO> --epochs=<100>

# So, for example, to train DQN from the Enviroment "./wads/New.cfg":
python main.py --model=a2c --scene=./wads/New.cfg

# To watch a game using DQN:
python main.py --model=dqn --skip_training=True --weights_dir=../models/DuelQ_from_basic.pth
```
For help use 
```sh
python main.py -h

usage: main.py [-h] [--scene SCENE] [--frame_repeat FRAME_REPEAT] [--model_name MODEL_NAME] [--weights_dir WEIGHTS_DIR] [--text_file TEXT_FILE] [--model {DQN,PPO}] [--epochs EPOCHS]
               [--steps STEPS] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] [--memory_size MEMORY_SIZE] [--discount_factor DISCOUNT_FACTOR] [--resultion RESULTION]
               [--n_workers N_WORKERS] [--buffer_update_freq BUFFER_UPDATE_FREQ] [--entropy_coeff ENTROPY_COEFF] [--value_loss_coeff VALUE_LOSS_COEFF] [--max_grad_norm MAX_GRAD_NORM]
               [--grad_clip GRAD_CLIP] [--save_model SAVE_MODEL] [--load_model LOAD_MODEL] [--skip_training SKIP_TRAINING] [--test_episodes TEST_EPISODES] [--watch_episodes WATCH_EPISODES]
               [--save_freq SAVE_FREQ]

options

optional arguments:
  -h, --help            show this help message and exit
  --scene SCENE         run a specific .cfg and .wad
  --frame_repeat FRAME_REPEAT
                        repeat frame n times
  --model_name MODEL_NAME
                        name of experiment, to be used as save_dir
  --weights_dir WEIGHTS_DIR
                        name of model to load
  --text_file TEXT_FILE
                        name of model to load
  --model {DQN,PPO}     the model architecture
  --epochs EPOCHS       number of iterations to train network
  --steps STEPS         number of steps per episode
  --batch_size BATCH_SIZE
                        number of states per batch
  --learning_rate LEARNING_RATE
                        learning rate
  --memory_size MEMORY_SIZE
                        number of states to keep in memory for batching
  --discount_factor DISCOUNT_FACTOR
                        discount factor used for discounting return
  --resultion RESULTION
                        resultion used for training
  --n_workers N_WORKERS
                        number of actor critic workers
  --buffer_update_freq BUFFER_UPDATE_FREQ
                        refresh buffer after every x actions
  --entropy_coeff ENTROPY_COEFF
                        entropy regularization weight
  --value_loss_coeff VALUE_LOSS_COEFF
                        value loss regularization weight
  --max_grad_norm MAX_GRAD_NORM
                        norm bound for clipping gradients
  --grad_clip GRAD_CLIP
                        magnitude bound for clipping gradients
  --save_model SAVE_MODEL
                        save model after training
  --load_model LOAD_MODEL
                        load model before training
  --skip_training SKIP_TRAINING
                        skip training
  --test_episodes TEST_EPISODES
                        number of episode to test
  --watch_episodes WATCH_EPISODES
                        number of episode to watch
  --save_freq SAVE_FREQ
                        number of episode to train before saving
```
## How to navigate this project
### `\lib`
Contains references and directories which have been added and worked with in the devolopment of this project

### `\source`
Contains the original source code of the project
`DQN.py` and `PPO.py` are the respective algorithms with that name.\
`main.py` is the file to train and or evaluate the models and takes terminal arguments via arg_parser\
Example: `python main.py --skip_training = True` will skip the training and only load a given model\
`base.py` contains Abstract Base Classes used to implement both DQN and PPO. All generics part of those algorithms are called via those,
to be able to have a clear visual difference overview between these algorithms in the code itself. This helped use when developing the project in understanding
the algorithms and maintaining the code base easily.

### `\models`
Directory that will automatically be created, when not given and stores the weight directories of the trained models.

### `\rewards`
Directory that will automatically be created, when not given and stores the rewards earned per epoch. Was used to visualize the progress and analyzing needed changes to the enviroment, scripts and rewards.

### `\scenarios`
Containing configuration files and the according .wad file for it. Use matching files when running the code.

## Results
The full evalution of our Results can be read in the [attached paper](https://github.com/FatManWalking/the_fat_bot/blob/main/TheFatBot%20-%20DOOM%20for%20reinforcement%20learning.pdf) documenting our thought process, theortical overview, implementation and accomplished results.

(Use Light Mode on GitHub to read the axis)
In short we tried different hyperparameter combinations in different enviroments and got mixed results.
As you can see there are certain pikes in the reward based performance:
![Different Plots](https://user-images.githubusercontent.com/46927512/127165861-e5167b0c-4c64-4958-8afb-f92156f25123.png)

There also is a Google Drive provided with project related data like metrics, wads, graphs and model files [DOOM-Folder](https://drive.google.com/drive/folders/1rHizC5ppqcJWElBOVd-HAEOcjqUoW5lT?usp=sharing)
