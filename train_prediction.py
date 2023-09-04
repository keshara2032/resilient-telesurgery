from typing import List
import os
from functools import partial
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from utils import get_dataloaders
from datagen import kinematic_feature_names, kinematic_feature_names_jigsaws, kinematic_feature_names_jigsaws_no_rot_ps, class_names, all_class_names, state_variables
from models.utils import plot_bars

import warnings
warnings.filterwarnings('ignore')


# ------------------------------------- Functions ----------------------------------
def train_model(model, optimizer, train_dataloader):
    pass

def eval_model(model, valid_dataloader):
    pass

def save_artifacts(model, subject_id_to_exclude, train_accuracy, valid_accuracy, train_losses, valid_losses):
    # save the losses for the current model and subject
    # save the accuracy for the current model and subject
    # save the model itself
    pass

def plot_artifacts(train_losses, valid_losses, model, subject_id_to_exclude):
    # plot and save train vs. valid losses
    # plot the predictions for a sample trial from the valid set
    pass

### -------------------------- DATA -----------------------------------------------------
tasks = ["Suturing"]
Features = kinematic_feature_names_jigsaws[38:] + state_variables #kinematic features + state variable features
# Features = kinematic_feature_names_jigsaws_no_rot_ps + state_variables

one_hot = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
observation_window = 10
prediction_window = 10
batch_size = 64
subject_id_to_exclude = 2
cast = True
include_resnet_features = True
include_colin_features=True
include_segmentation_features=False
normalizer = '' # ('standardization', 'min-max', 'power', '')
step = 3 # 10 Hz

for subject_id_to_exclude in [5]:
    train_dataloader, valid_dataloader = get_dataloaders(tasks=tasks,
                                                        subject_id_to_exclude=subject_id_to_exclude,
                                                        observation_window=observation_window,
                                                        prediction_window=prediction_window,
                                                        batch_size=batch_size,
                                                        one_hot=one_hot,
                                                        class_names=class_names['Suturing'],
                                                        feature_names=Features,
                                                        include_resnet_features=include_resnet_features,
                                                        include_segmentation_features=include_segmentation_features,
                                                        include_colin_features=include_colin_features,
                                                        cast=cast,
                                                        normalizer=normalizer,
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
    print(train_dataloader.dataset.samples_per_trial)
    exit()

    #------------------------------------------Build the model and the optimizer---------------------------

    # Build the Model
    from models.transformer import TransformerEncoderDecoderModel
    args = dict(
        num_encoder_layers = 6,
        num_decoder_layers = 6,
        emb_dim = 128,
        dropout = 0.4,
        optimizer_type = 'Adam',
        weight_decay = 0.001,
        lr = 1e-3,
        nhead = 4,
        dim_feedforward = 512
    )

    model = TransformerEncoderDecoderModel(encoder_input_dim=len(train_dataloader.dataset.get_feature_names()),
                 decoder_input_dim=len(train_dataloader.dataset.get_target_names()),
                 num_encoder_layers=args['num_encoder_layers'],
                 num_decoder_layers=args['num_decoder_layers'],
                 emb_dim=args['emb_dim'],
                 nhead=args['nhead'],
                 tgt_vocab_size=len(train_dataloader.dataset.get_target_names()),
                 max_encoder_len=observation_window,
                 max_decoder_len=prediction_window,
                 decoder_embedding_dim=args[''],
                 dim_feedforward=args['dim_feedforward'],
                 dropout=args['dropout'])
    
    # Build the optimizer
    if args['optimizer_type'] == 'Adam':
        optimizer_cls = torch.optim.Adam
    elif args['optimizer_type'] == 'AdamW':
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    #----------------------------------------Training Loop-------------------------------------------------
    experiment_name = 'transformer_kin_context'
    results = {}
    epochs = 20
    train_losses, valid_losses = [], []
    for epoch in range(epochs):
        epoch_train_losses, train_accuracy, train_edit_score = train_model(model, optimizer, train_dataloader)
        epoch_valid_losses, valid_accuracy, valid_edit_score = eval_model(model, valid_dataloader)

        train_losses.append(epoch_train_losses)
        valid_losses.append(epoch_valid_losses)
    
    # save model, accuracy, edit_score, loss-plots for the current subject
    save_artifacts(model, train_accuracy, valid_accuracy, train_losses, valid_losses)
    plot_artifacts(train_losses, valid_losses)