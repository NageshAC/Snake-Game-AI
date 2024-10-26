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
    def __init__(self, model:nn.Module, lr:float, gamma:float, optimizer:optim.Optimizer, criterion:nn.Module):

        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, state, action, nxt_state, reward, done):
        pass




if __name__ == '__main__':
    Lin_Qnet([3,12,3])