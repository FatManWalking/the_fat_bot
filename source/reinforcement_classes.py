import torch
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict, Union, Optional

#import of different classes and modules for reinforcement learning from used packages tianshou amd ViZDoom frameworks
import tianshou
from tianshou.policy import imitation

class vanilla_imiation(imitation.base.ImitationPolicy):

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        **kwargs: Any,
    ) -> None:

        super().__init__(model, optim, **kwargs)
        #self.model = model
        #self.optim = optim
        assert self.action_type in ["continuous", "discrete"], \
            "Please specify action_space."


