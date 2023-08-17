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
from models.utils import plot_bars

import warnings
warnings.filterwarnings('ignore')



### -------------------------- DATA -----------------------------------------------------
tasks = ["Suturing"]
Features = kinematic_feature_names_jigsaws[38:] + state_variables #kinematic features + state variable features

one_hot = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
observation_window = 10
prediction_window = 10
batch_size = 64
user_left_out = 2
cast = True
include_image_features = False
normalizer = '' # ('standardization', 'min-max', 'power', '')
step = 6 # 5 Hz

for user_left_out in [2, 3, 4, 5, 6, 7, 8, 9]:
    train_dataloader, valid_dataloader = get_dataloaders(tasks,
                                                        user_left_out,
                                                        observation_window,
                                                        prediction_window,
                                                        batch_size,
                                                        one_hot,
                                                        class_names = class_names['Suturing'],
                                                        feature_names = Features,
                                                        include_image_features = include_image_features,
                                                        cast = cast,
                                                        normalizer = normalizer,
                                                        step=step)

    # print("datasets lengths: ", len(train_dataloader.dataset), len(valid_dataloader.dataset))
    # print("X shape: ", train_dataloader.dataset.X.shape, valid_dataloader.dataset.X.shape)
    # print("Y shape: ", train_dataloader.dataset.Y.shape, valid_dataloader.dataset.Y.shape)

    # # loader generator aragement: (src, tgt, future_gesture, future_kinematics)
    # print("Obs Kinematics Shape: ", train_dataloader.dataset[0][0].shape) 
    # print("Obs Target Shape: ", train_dataloader.dataset[0][1].shape)
    # print("Future Target Shape: ", train_dataloader.dataset[0][2].shape)
    # print("Future Kinematics Shape: ", train_dataloader.dataset[0][3].shape)
    # print("Train N Trials: ", train_dataloader.dataset.get_num_trials())
    # print("Train Max Length: ", train_dataloader.dataset.get_max_len())
    # print("Test N Trials: ", valid_dataloader.dataset.get_num_trials())
    # print("Test Max Length: ", valid_dataloader.dataset.get_max_len())
    # print("Features: ", train_dataloader.dataset.get_feature_names())
    # print(train_dataloader.dataset.samples_per_trial)





    ### -------------------------- Model -----------------------------------------------------
    from models.recognition_direct_crf import Trainer as CRF_Trainer

    # import optuna
    # import math

    # def objective(trial):

    #     print('Optuna Trial: ', trial.number)

    #     # optimization
    #     optimizer_type = trial.suggest_categorical('optimizer_type', ['Adam', 'AdamW'])
    #     lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    #     weight_decay = trial.suggest_float('weight_decay', 0.0001, 0.1, log=True)

    #     # batch size
    #     batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])

    #     # model: general
    #     hidden_dim_exp = trial.suggest_int('hidden_dim_exp', 5, 9)  # d_model = 2^d_model_exp
    #     hidden_dim = int(math.pow(2, hidden_dim_exp))

    #     emb_dim_exp = trial.suggest_int('emb_dim_exp', 5, 9)  # d_model = 2^d_model_exp
    #     emb_dim = int(math.pow(2, emb_dim_exp))

    #     dropout = trial.suggest_float('dropout', 0.1, 0.5)

    #     num_layers = trial.suggest_int('num_layers', 1, 6)
        
    #     # model: transformer only
    #     encoder_type = trial.suggest_categorical('encoder_type', ['gru', 'lstm', 'transformer'])
    #     if encoder_type == 'transformer':
    #         nhead_exp = trial.suggest_int('nhead_exp', 1, 3)
    #         nhead = int(math.pow(2, nhead_exp))
    #         dim_feedforward = trial.suggest_int('dim_feedforward', 512, 2048)
    #     else: #values won't be used
    #         nhead = 4
    #         dim_feedforward = 512
        
    #     users = list(range(2, 9 + 1)) 
    #     val_losses = list()
    #     for user_left_out in users:
    #         print("Training and Validation for user: ", user_left_out)
            
    #     # Load your data and define loss function here
    #         train_dataloader, valid_dataloader = get_dataloaders(tasks,
    #                                                         user_left_out,
    #                                                         observation_window,
    #                                                         prediction_window,
    #                                                         batch_size,
    #                                                         one_hot,
    #                                                         class_names = class_names['Suturing'],
    #                                                         feature_names = Features,
    #                                                         include_image_features = include_image_features,
    #                                                         cast = cast,
    #                                                         normalizer = normalizer,
    #                                                         step=step)
            
    #         # Perform training loop with the defined batch size and model
    #         args = dict(
    #             hidden_dim = hidden_dim, # the hidden size of the rnn or transformer-encoder
    #             num_layers = num_layers, # number of rnn or transformer-encoder layer
    #             encoder_type = encoder_type,
    #             emb_dim = emb_dim, # not used with transformer
    #             dropout = dropout,
    #             optimizer_type = optimizer_type,
    #             weight_decay = weight_decay,
    #             lr = lr,
    #             save_best_val_model = True,
    #             recovery = False,
    #             nhead = nhead, # not used with rnn
    #             max_len = observation_window, # not used with rnn
    #             dim_feedforward = dim_feedforward # not used with rnn
    #         )
    #         epochs = 5
    #         model_dir = 'saved_model_files'

    #         for key, value in trial.params.items():
    #             print(f"    {key}: {value}")

    #         trainer = CRF_Trainer()
    #         validation_loss = trainer.train_and_evaluate(train_dataloader, valid_dataloader, epochs, model_dir, args, device)
    #         val_losses.append(validation_loss)
    #         print(f"Loss for user {user_left_out}: {validation_loss}")

    #     return sum(val_losses)/len(val_losses)  # Return the loss as the objective to minimize

    # if __name__ == '__main__':
    #     study = optuna.create_study(direction='minimize')  # Objective is to minimize the validation loss
    #     study.optimize(objective, n_trials=20)  # You can adjust the number of trials
        
    #     print(f"Number of finished trials: {len(study.trials)}")
    #     print("Best trial:")
    #     trial = study.best_trial

    #     print(f"Value: {trial.value}")
    #     print("Params: ")
    #     for key, value in trial.params.items():
    #         print(f"    {key}: {value}")

    args = dict(
        hidden_dim = 512, # the hidden size of the rnn or transformer-encoder
        num_layers = 6, # number of rnn or transformer-encoder layer
        encoder_type = 'lstm',
        emb_dim = 128, # not used with transformer
        dropout = 0.4,
        optimizer_type = 'Adam',
        weight_decay = 0.001,
        lr = 1e-3,
        save_best_val_model = True,
        recovery = False,
        nhead = 4, # not used with rnn
        max_len = observation_window, # not used with rnn
        dim_feedforward = 512 # not used with rnn
    )
    epochs = 20
    model_dir = 'saved_model_files'

    trainer = CRF_Trainer()
    trainer.train_and_evaluate(train_dataloader, valid_dataloader, epochs, model_dir, args, device)



