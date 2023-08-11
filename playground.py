from typing import List
import os
from functools import partial
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from utils import get_dataloaders
from datagen import kinematic_feature_names, kinematic_feature_names_jigsaws, class_names, all_class_names, state_variables



### -------------------------- DATA -----------------------------------------------------
tasks = ["Suturing"]
Features = kinematic_feature_names_jigsaws[38:] + state_variables #kinematic features + state variable features

one_hot = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
observation_window = 10
prediction_window = 10
batch_size = 4
user_left_out = 2
cast = True
include_image_features = False
normalizer = '' # ('standardization', 'min-max', 'power', '')
step = 3 # 10 Hz

train_dataloader, valid_dataloader = get_dataloaders(tasks,
                                                     user_left_out,
                                                     observation_window,
                                                     prediction_window,
                                                     batch_size,
                                                     one_hot,
                                                     class_names = all_class_names,
                                                     feature_names = Features,
                                                     include_image_features=include_image_features,
                                                     cast = cast,
                                                     normalizer = normalizer,
                                                     step=step)

print("datasets lengths: ", len(train_dataloader.dataset), len(valid_dataloader.dataset))
print("X shape: ", train_dataloader.dataset.X.shape, valid_dataloader.dataset.X.shape)
print("Y shape: ", train_dataloader.dataset.Y.shape, valid_dataloader.dataset.Y.shape)

# loader generator aragement: (src, tgt, future_gesture, future_kinematics)
print("Obs Kinematics Shape: ", train_dataloader.dataset[0][0].shape) 
print("Obs Target Shape: ", train_dataloader.dataset[0][1].shape)
print("Future Target Shape: ", train_dataloader.dataset[0][2].shape)
print("Future Kinematics Shape: ", train_dataloader.dataset[0][3].shape)
print("Train N Trials: ", train_dataloader.dataset.get_num_trials())
print("Train Max Length: ", train_dataloader.dataset.get_max_len())
print("Test N Trials: ", valid_dataloader.dataset.get_num_trials())
print("Test Max Length: ", valid_dataloader.dataset.get_max_len())
print("Features: ", train_dataloader.dataset.get_feature_names())

X, Y, Y_future, P = valid_dataloader.dataset.get_trial(0)
print('Single Trial:')
print(f'X shape: {X.shape}')
print(f'Y shape: {Y.shape}')
print(f'Y_future shape: {Y_future.shape}')
print(f'P shape: {P.shape}')


### -------------------------- Model -----------------------------------------------------
from models.recognition_direct_crf import Trainer as CRF_Trainer

args = dict(
    hidden_dim = 64, # the hidden size of the rnn or transformer-encoder
    num_layers = 1, # number of rnn or transformer-encoder layer
    encoder_type = 'gru',
    emb_dim = 64, # not used with transformer
    dropout = 0.1,
    lr = 2e-5,
    save_best_val_model = True,
    recovery = False,
    nhead = 4, # not used with rnn
    max_len = observation_window, # not used with rnn
    dim_feedforward = 512 # not used with rnn
)
epochs = 5
model_dir = 'saved_model_files'

trainer = CRF_Trainer()
trainer.train_and_evaluate(train_dataloader, valid_dataloader, epochs, model_dir, args, device)



