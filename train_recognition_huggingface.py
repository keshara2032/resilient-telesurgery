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
from datautils.utils import get_dataloaders
from datautils.datagen import kinematic_feature_names,colin_features, segmentation_features, kinematic_feature_names_jigsaws, kinematic_feature_names_jigsaws_patient_position, class_names, all_class_names, state_variables
from tqdm import tqdm
from collections import OrderedDict
from config import *
from datautils.dataloader_k import *
from genutils.utils import *

import datetime
import argparse

from huggingface_hub import hf_hub_download
import torch
from transformers import AutoformerForPrediction

torch.manual_seed(0)



# end of imports #


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="A simple command-line argument parser")

# Add arguments
parser.add_argument("--model", help="Specify which model to run", required=True)
parser.add_argument("--dataloader", help="Specify which dataloader", required=True)
# parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")

# Parse the arguments
args = parser.parse_args()

# Access the parsed arguments
model_name = args.model
dataloader = args.dataloader
# verbose_mode = args.verbose



# manual seeding ensure reproducibility
# torch.manual_seed(0)



# tasks and features to be included
task = "Suturing"

context = dataloader_params["context"]


if(context == 0): #kin only
    # Features = kinematic_feature_names_jigsaws[38:]  #all patient side kinematic features
    Features = kinematic_feature_names_jigsaws_patient_position  #kinematic features only
    # Features = kinematic_feature_names_jigsaws[0:]  #all  kinematic features 
    
elif(context == 1): #context only
    Features = state_variables

elif(context == 2): # context + kin
    Features = kinematic_feature_names_jigsaws[38:] + state_variables #all patient side kinematic features + state variable features
    # Features = kinematic_feature_names_jigsaws_patient_position + state_variables #kinematic features + state variable features

elif(context == 3): # img features only
    Features = img_features 
    
elif(context == 4): # img features + kin
    Features = img_features + kinematic_feature_names_jigsaws_patient_position
    
elif(context == 5): # img features + kin + context
    Features = img_features + kinematic_feature_names_jigsaws_patient_position + state_variables


elif(context == 6): # colin_features
    Features = colin_features

elif(context == 7): # colin+context
    Features = colin_features + state_variables
    
elif(context == 8): # colin + kinematic 14 
    Features = colin_features + kinematic_feature_names_jigsaws_patient_position
   
elif(context == 9): # colin + kinematic 14  + context
    Features = colin_features + kinematic_feature_names_jigsaws_patient_position + state_variables
   
   
elif(context == 10): # segonly
    Features = segmentation_features

   
elif(context == 11): # kin+seg
    Features = segmentation_features + kinematic_feature_names_jigsaws_patient_position 
   
   
   
elif(context == 12): # kin + seg + context
    Features = segmentation_features + kinematic_feature_names_jigsaws_patient_position + state_variables
    
    
elif(context == 13): # kin + seg + context + colin
    Features = segmentation_features + colin_features + kinematic_feature_names_jigsaws_patient_position + state_variables

elif(context == 14): # kin + seg + context + colin
    Features = segmentation_features + colin_features  + state_variables + kinematic_feature_names_jigsaws_patient_position

epochs = learning_params["epochs"]
observation_window = dataloader_params["observation_window"],


if(dataloader == "kw"):
    train_dataloader, valid_dataloader = generate_data(dataloader_params["user_left_out"],task,Features, dataloader_params["batch_size"], observation_window)
else:
    train_dataloader, valid_dataloader = get_dataloaders([task],
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


batch = next(iter(train_dataloader))
features = batch[0].shape[-1]
output_dim = batch[1].shape[-1]
input_dim = features  

print("Input Features:",input_dim, "Output Classes:",output_dim)


### DEFINE MODEL HERE ###
# model_name = 'tcn' 
# model_name = 'transformer'



### Subjects 
subjects = [2,3,4,5,6,7,8,9]
# subjects = [4]


accuracy = []

print("len dataloader:",train_dataloader.dataset.__len__())
input("Press any key to begin training...")
# Train Loop

REPEAT = 1
for i in range(REPEAT):
    for subject in (subjects):

            model = AutoformerForPrediction.from_pretrained("huggingface/autoformer-tourism-monthly")

            user_left_out = subject

            if(dataloader == "kw"):
                train_dataloader, valid_dataloader = generate_data(user_left_out,task,Features, dataloader_params["batch_size"], observation_window)
            else:
                train_dataloader, valid_dataloader = get_dataloaders([task],
                                                                user_left_out,
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
                

            
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                for bi, (src, tgt, future_gesture, future_kinematics) in enumerate(tqdm(train_dataloader)):

                    outputs = model(
                                    past_values=tgt,
                                    past_time_features=src
                                )   
                    
                    loss = outputs.loss
                    loss.backward()