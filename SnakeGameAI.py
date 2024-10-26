from SnakeGame import SnakeGame
from model import Lin_Qnet, DeepQTrainer

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from collections import deque

# Game Window Configuration
GRID_RESOLUTION : tuple[int, int] = (45, 80)

# Pixel Size of each Grid
GRID_PIXEL_RESOLUTION : int = 10 

# Snake Speed in moves per second
MPS : int = 20 # FPS

# Training Agent settings
MAX_MEMORY : int = 1_000
BATCH_SIZE : int = 100
# number of games until which the agent makes random decisions
RATE_RANDOM_CHOICES : int = 100 

LR = 0.001      # Learning Rate
GAMMA = 0.9     # discount rate for deep Q Learning


class SnakeGameAI(SnakeGame):

    def __init__(self, model:Lin_Qnet, trainer:DeepQTrainer) -> None :

        super(SnakeGameAI, self).__init__(GRID_RESOLUTION, GRID_PIXEL_RESOLUTION, MPS, 'SnakeAI')

        self.model = model
        self.trainer = trainer

        self.n_games : int = 0
        self.memory = deque(maxlen=MAX_MEMORY)

    def get_state(self):
        pass

    def get_action(self):
        pass

    def _remember(self, state:list[int], action:list[int], 
              nxt_state:list[int], reward:int, game_over:int):
        self.memory.append((state, action, reward, nxt_state, game_over))

    def train_short_memory(self, state:list[int], action:list[int], 
              nxt_state:list[int], reward:int, game_over:int):
        pass

    def train_long_memory(self):
        pass



if __name__ == '__main__':
    # optimizer = optim.Adam(model.parameters(), lr=self.lr)
    # criterion = nn.MSELoss()

    SnakeGameAI(None, None).run()