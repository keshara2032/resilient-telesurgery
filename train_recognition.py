import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import glob
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
import json
from typing import List
import os
from functools import partial
import torch.nn.functional as F
from timeit import default_timer as timer
from utils import get_dataloaders
from datagen import kinematic_feature_names, kinematic_feature_names_jigsaws, kinematic_feature_names_jigsaws_patient_position, class_names, all_class_names, state_variables
from tqdm import tqdm
from collections import OrderedDict
from config import *


# end of imports #


# manual seeding ensure reproducibility
torch.manual_seed(0)


# tasks and features to be included
tasks = ["Suturing"]
# Features = kinematic_feature_names_jigsaws[38:] + state_variables #all patient side kinematic features + state variable features
# Features = kinematic_feature_names_jigsaws[0:]  #all patient side kinematic features + state variable features
# Features = kinematic_feature_names_jigsaws_patient_position + state_variables #kinematic features + state variable features
Features = kinematic_feature_names_jigsaws_patient_position  #kinematic features only


train_dataloader, valid_dataloader = get_dataloaders(tasks,
                                                     dataloader_params["user_left_out"],
                                                     dataloader_params["observation_window"],
                                                     dataloader_params["prediction_window"],
                                                     dataloader_params["batch_size"],
                                                     dataloader_params["one_hot"],
                                                     class_names = class_names['Suturing'],
                                                     feature_names = Features,
                                                     include_image_features=dataloader_params["include_image_features"],
                                                     cast = dataloader_params["cast"],
                                                     normalizer = dataloader_params["normalizer"],
                                                     step=dataloader_params["step"])

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

