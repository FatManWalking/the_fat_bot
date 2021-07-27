# The Fat Bot
A reinforcement trained bot for doom, created as part of a machine learning research project at DHBW Mannheim

This project is based on VizDoom. To run vizdoom follow this [Guide](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#linux_build) provided by VizDoom itself.
We personally recommend the setup over a Linux-Setup, which was also the only setup we developed this project in (Ubuntu 20.4).
We also provide a YAML `Doom_Yaml.yml`, which we know is a functional conda enviroment for this project.

## ⚙️ Running the code
Almost every parameter is adjustable via terminal commands when executing the code. For a full list of options scroll thorugh the `source/main.py`

Examples:
```sh
# General format of commands
python main.py --model=<DQN, PPO> --epochs=<100>

# So, for example, to train DQN from the Enviroment "./wads/New.cfg":
python main.py --model=a2c --scene=./wads/New.cfg

# To watch a game using DQN:
python main.py --model=dqn --skip_training=True --weights_dir=../models/DuelQ_from_basic.pth
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

### `\other`
Hier könnte ihr Ordner stehen, für nur 3.99€ im Monat. Lass sie jetzt hier ihren Ordner anzeigen.
@Dickermann 3.99 viel zu teuer du großkapitalist. Das alles sollte öffentlich kostenlos zugänglich sein


Google Drive mit allen Modellen, Wads und was es sonst so zu dem Projekt gibt: [DOOM-Ordner](https://drive.google.com/drive/folders/1rHizC5ppqcJWElBOVd-HAEOcjqUoW5lT?usp=sharing)
