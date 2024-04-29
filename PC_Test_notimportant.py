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
      pc.PCLayer(),         # contains neural activity of layer 2                  
      activation_fn(),
      nn.Linear(hidden_size, hidden2_size),
      pc.PCLayer(),         # contains neural activity of layer 1
      activation_fn(),
      nn.Linear(hidden2_size, output_size)
  )

pc_model.train()

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
    optimizer_x_fn = optimizer_x_fn,
    optimizer_x_kwargs = optimizer_x_kwargs,
    optimizer_p_fn = optimizer_p_fn,
    optimizer_p_kwargs = optimizer_p_kwargs
)

epochs = 10

def loss_fn(output, _target):
    return 0.5*(output - _target).pow(2).sum()

for epoch in range(epochs):
    for data, label in train_loader:
        labels_one_hot = F.one_hot(label).float()
        trainer.train_on_batch(inputs=labels_one_hot, loss_fn=loss_fn, loss_fn_kwargs={'_target':data}, is_log_progress=False, is_return_results_every_t=False, is_checking_after_callback_after_t=False)