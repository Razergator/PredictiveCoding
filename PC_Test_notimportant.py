import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import predictive_coding as pc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using {device}')

seed = 42
torch.manual_seed(seed)

n_train = 10000
n_val = 500
n_test = 5000
batch_size = 500

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
dataset_train = datasets.MNIST('./data', download=True, train=True, transform=transform)
dataset_eval = datasets.MNIST('./data', download=True, train=False, transform=transform)

# Randomly sample the train dataset
train_dataset = torch.utils.data.Subset(dataset_train, random.sample(range(len(dataset_train)), n_train))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Randomly sample the val dataset
val_dataset, test_dataset, not_used = torch.utils.data.random_split(dataset_eval, [n_val, n_test, dataset_eval.__len__()-n_val-n_test])

print(f'train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}')

input_size = 10        # for the 10 classes
hidden_size = 256
hidden2_size = 256
output_size = 28*28    # for the 28 by 28 mnist images

activation_fn = nn.ReLU

pc_model = nn.Sequential(
      nn.Linear(input_size, hidden_size),
      activation_fn(),
      pc.PCLayer(),
      nn.Linear(hidden_size, hidden2_size),
      activation_fn(),
      pc.PCLayer(),
      nn.Linear(hidden2_size, output_size)
  )

pc_model.train()
pc_model.to(device)

# number of neural activity updates
T = 20                              
# optimizer for activity updates
optimizer_x_fn = optim.Adam         
optimizer_x_kwargs = {'lr': 0.1}    
# optimizer for weight updates
optimizer_p_fn = optim.Adam          
optimizer_p_kwargs = {"lr": 0.001, "weight_decay":0.001} 

trainer = pc.PCTrainer(pc_model, 
    T = T,
    update_x_at = "all",
    optimizer_x_fn = optimizer_x_fn,
    optimizer_x_kwargs = optimizer_x_kwargs,
    optimizer_p_fn = optimizer_p_fn,
    optimizer_p_kwargs = optimizer_p_kwargs,
    update_p_at= "last",
    plot_progress_at=[]
)

#define PC loss function
def loss_fn(output, _target):
    return 0.5*(output - _target).pow(2).sum()

# This class contains the parameters of the prior mean \mu parameter (see figure)
class BiasLayer(nn.Module):
    def __init__(self, out_features, offset=0.):
        super(BiasLayer, self).__init__()
        self.bias = nn.Parameter(offset*torch.ones(out_features) if offset is not None
                                  else 2*np.sqrt(out_features)*torch.rand(out_features)-np.sqrt(out_features), requires_grad=True)

    def forward(self, x):
        return torch.zeros_like(x) + self.bias  # return the prior mean \mu witht the same shape as the input x to make sure the batch size is the same


def test_normal(model, dataset, batch_size=1000):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # add bias layer for inferece
    test_model = nn.Sequential(
        BiasLayer(10, offset=0.),
        pc.PCLayer(),
        model
    )
    test_model.train()
    test_model.to(device)

# make pc_trainer for test_model
    trainer_normal_test = pc.PCTrainer(test_model,
        T = 100,
        update_x_at = "all",
        optimizer_x_fn = optimizer_x_fn,
        optimizer_x_kwargs = optimizer_x_kwargs,
        update_p_at = "never",
        optimizer_p_fn = optimizer_p_fn,
        optimizer_p_kwargs = optimizer_p_kwargs,
        plot_progress_at=[]
    )