from SnakeGame import SnakeGame, DIRECTIONS
from model import Lin_Qnet, DeepQTrainer

import torch
import torch.nn as nn
import torch.optim as optim

import random
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

        self.dir_list = list(DIRECTIONS)

    def get_state(self) -> tuple[float, np.ndarray]:
        # ! State of the model (11 values)
        # * Danger around the head   (3 values) [UP, RIGHT, DOWN, LEFT] // removeing whichever is the snake
        # * Movement Direction       (4 values) [UP, RIGHT, DOWN, LEFT]
        # * Food Direction           (4 values) [UP, RIGHT, DOWN, LEFT]

        danger    = np.zeros((3), dtype=int)
        direction = np.zeros((4), dtype=int)
        food      = np.zeros((4), dtype=int)

        dir_idx = self.dir_list.index(self.movement_direction)

        # Danger Direction
        count = 0
        for key in self.dir_list:
            new_pos = self.snake[0] + DIRECTIONS[key]
            if not self._is_snake(new_pos):
                danger[count] = self._is_wall(new_pos)
                count += 1

        # Movement Direction
        direction[dir_idx] = 1

        # Food Direction
        distance = self.food - self.snake[0]
        food[0] = distance[0] < 0 # UP
        food[1] = distance[1] > 0 # RIGHT
        food[2] = distance[0] > 0 # DOWN
        food[3] = distance[1] < 0 # LEFT
        distance = np.sqrt(np.sum(distance**2))

        return distance, np.hstack((danger, direction, food))

    def get_action(self, state):

        # Generate random moves for exploration of the play area
        # This occurs only untill n_games == RATE_RANDOM_CHOICES
        epsilon = RATE_RANDOM_CHOICES - self.n_games
        action = np.zeros((3), dtype=int)
        
        if random.randint(0, 200) < epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            predict = self.model(state_tensor)
            move = torch.argmax(predict).item()
            action[move] = 1

        return action.copy()

    def _remember(self, state:list[int], action:list[int], 
              nxt_state:list[int], reward:int, game_over:int):
        self.memory.append((state, action, nxt_state, reward, game_over))

    def train_short_memory(self, state:list[int], action:list[int], 
              nxt_state:list[int], reward:int, game_over:int):
        
        # Save to memory
        self._remember(state, action, nxt_state, reward, game_over)

        # Training
        self.trainer.train(state, action, nxt_state, reward, game_over)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        state, action, nxt_state, reward, game_over = zip(*mini_sample)
        self.trainer.train_step(state, action, nxt_state, reward, game_over)



if __name__ == '__main__':
    # optimizer = optim.Adam(model.parameters(), lr=self.lr)
    # criterion = nn.MSELoss()

    SnakeGameAI(None, None).get_state()