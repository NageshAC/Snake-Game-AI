import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pickle
import numpy as np
from pathlib import Path
from itertools import pairwise
from collections import OrderedDict

class Lin_Qnet (nn.Module):
    def __init__ (self, layer_sizes : list | tuple | np.ndarray):
        super(Lin_Qnet, self).__init__()

        layers_dict = OrderedDict()
        for n, (i, o) in enumerate(pairwise(layer_sizes)):
            layers_dict[f'Linear{n}'] = nn.Linear(i, o)


        self.model = nn.Sequential(layers_dict)

        self.def_model_name = 'model.pth'
        self.model_folder_path = Path('./model')
        self.score_path = 'plot_score.pickle'
        self.mean_scores_path = 'plot_mean_score.pickle'

    def forward(self, x):
        return self.model(x)

    def save(self, plot_scores, plot_mean_scores, file_name=None) -> None:
        if not file_name:
            file_name = self.def_model_name
        if not self.model_folder_path.exists():
            self.model_folder_path.mkdir()

        file_name = self.model_folder_path / Path(file_name)

        torch.save(self.state_dict(), str(file_name))
        pickle.dump(plot_scores, Path(self.model_folder_path, self.score_path).open('wb'))
        pickle.dump(plot_mean_scores, Path(self.model_folder_path, self.mean_scores_path).open('wb'))

        # print("Saving Model state as: ", str(file_name))

    def load (self, file_name=None) -> list:
        if file_name is None:
            file_name = self.def_model_name

        file_name = self.model_folder_path / file_name

        if file_name.exists():
            self.load_state_dict(torch.load(str(file_name)))
            # print("Loading Model state from: ", str(file_name))
            plot_scores = pickle.load( Path(self.model_folder_path, self.score_path).open('rb') )
            plot_mean_scores = pickle.load( Path(self.model_folder_path, self.mean_scores_path).open('rb') )

            return plot_scores, plot_mean_scores
        
        # print("Saved Model doesnt exists!")
        return [], []

class DeepQTrainer:
    def __init__(self, 
                 model:nn.Module, 
                 gamma:float, 
                 optimizer:optim.Optimizer | None = None, 
                 criterion:nn.Module | None = None):

        self.gamma = gamma
        self.model = model
        self.optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=0.001)
        self.criterion = criterion if criterion else nn.MSELoss()

    def train(self, 
              state:list[int]|list[list[int]], 
              action:list[int]|list[list[int]], 
              nxt_state:list[int]|list[list[int]], 
              reward:int|list[int], 
              game_over:int|list[int]):
        
        # convert all parameters to pyTorch tensors
        state       = torch.tensor(state,       dtype=torch.float)
        nxt_state   = torch.tensor(nxt_state,   dtype=torch.float)
        reward      = torch.tensor(reward,      dtype=torch.float)
        action      = torch.tensor(action,      dtype=torch.long)

        # if the state is is one dimentional (only one state -> short training)
        # then convert it to shape (1, x)
        # else it's anyway in shape (n, x)
        if len(state.shape) == 1:
            state     = torch.unsqueeze( state    , 0)
            nxt_state = torch.unsqueeze( nxt_state, 0)
            reward    = torch.unsqueeze( reward   , 0)
            action    = torch.unsqueeze( action   , 0)
            game_over      = (game_over, )

        # get the prediction of current state
        predict = self.model(state)

        # clone this to target
        target = predict.clone()

        # update the target to new Q values using the formula:
        # Q_new = reward + gamma * max (prediction of nxt_state)
        # update this Q_new to target node where the action is maximum
        # TODO: can be done without using loop iteration
        for idx in range(len(game_over)):
            Q_new = reward[idx]

            # since there is no next state if game over
            # ignore game over state
            if not game_over[idx]:
                Q_new += (self.gamma * torch.max(self.model(nxt_state[idx])))

            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, predict)
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    Lin_Qnet([3,12,3])