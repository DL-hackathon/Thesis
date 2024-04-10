# 1. Multilayer Perceptron Model (MLP)

import torch
from torch import nn
import torch.nn.functional as F


window_size = 32
step = 1

class MLP(nn.Module):
    '''
    Multilayer Perceptron Model
    '''
    def __init__(self, window_size: int=window_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5 * window_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4))
    def forward(self, x):
        '''
        Forward step
        '''
        return self.layers(x)    

# 2. CNN

class CNN(nn.Module):
    '''
    1D Convolutional neural network
    '''    
    def __init__(self, batch_size=32, window_size: int=window_size, features_size=5):        
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.window_size = window_size
        self.features_size = features_size 
        self.conv1 = nn.Conv1d(features_size, features_size * 4, 2, 2, groups=5)        
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(features_size * 4, features_size * 16, 2, 2, groups=20)       
        self.pool2 = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(160, 80)
        self.fc2 = nn.Linear(80, 4)
        
    def forward(self, x):
        '''
        Forward step
        '''
        self.bach_size = x.shape[0]
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.reshape(self.bach_size, 160)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)        
        return x

# 3. GRU

class GRU(nn.Module):
    '''
    Gated recurrent unit
    '''    
    def __init__(self, num_sensors=32, num_states=4, hidden_dim=128, num_layers=2, dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(num_sensors, hidden_dim, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim*num_layers, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, num_states)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        Forward step
        '''
        h = self.gru(x)[1].permute(1, 0, 2)
        h = h.reshape(h.size(0), -1)
        linear_out = self.linear1(h)
        linear_out = F.relu(linear_out)
        linear_out = self.dropout(linear_out)
        out = self.linear2(linear_out)
        return out