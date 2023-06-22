from typing import List
import os
from functools import partial
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from torch.utils.data import DataLoader
from dataset import get_dataloaders

from model import RecognitionModel, ScheduledOptim, get_tgt_mask
from utils import get_classification_report, visualize_gesture_ts



# Data Params
task = "Peg_Transfer"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
observation_window = 30
prediction_window = 30
batch_size = 16
user_left_out = 1
train_dataloader, valid_dataloader = get_dataloaders(task, user_left_out, observation_window, prediction_window, batch_size)

print("datasets lengths: ", len(train_dataloader.dataset), len(valid_dataloader.dataset))
print("X shape: ", train_dataloader.dataset.X.shape, valid_dataloader.dataset.X.shape)
print("Y shape: ", train_dataloader.dataset.Y.shape, valid_dataloader.dataset.Y.shape)


# Model Params
torch.manual_seed(0)
emb_size = 32
nhead = 1
ffn_hid_dim = 512
num_encoder_layers = 1
num_decoder_layers = 1
num_features = len(train_dataloader.dataset.get_feature_names())
num_output_classes = len(train_dataloader.dataset.get_target_names())
max_len = train_dataloader.dataset.get_max_len()


# model initialization
recognition_transformer = RecognitionModel(encoder_input_dim=num_features,
                                            decoder_input_dim=num_output_classes, 
                                            num_encoder_layers=num_decoder_layers,
                                            num_decoder_layers=num_decoder_layers,
                                            emb_size=emb_size,
                                            nhead=nhead,
                                            tgt_vocab_size=num_output_classes,
                                            max_len = max_len,
                                            decoder_embedding_dim = -1,
                                            dim_feedforward=ffn_hid_dim, # don't use embeddings for decoder input labels
                                            dropout=0.1)
for p in recognition_transformer.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)
recognition_transformer = recognition_transformer.to(device)

# loss function
loss_fn = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(recognition_transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
schd_optim = ScheduledOptim(optimizer, lr_mul=1, d_model=emb_size, n_warmup_steps=2000)

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    running_loss = 0.0

    for i, (src, src_image, tgt, future_gesture, future_kinematics) in enumerate(train_dataloader):

        # transpose inputs into the correct shape [seq_len, batch_size, features/classes]
        src = src.transpose(0, 1) # the srd tensor is of shape [batch_size, sequence_length, features_dim]; we transpose it to the proper dimension for the transformer model
        tgt = tgt.transpose(0, 1)
        tgt_input = tgt[:-1, :]
        
        # get the target mask
        tgt_mask = get_tgt_mask(observation_window, device)

        # model outputs
        logits = model(src, tgt_input, tgt_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), torch.argmax(tgt_out, dim=-1).reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

        # printing statistics
        running_loss += loss.item()
        if i % 50 == 49:    # print every 50 mini-batches
            print(f'[{i + 1:5d}] loss: {running_loss / 50:.3f}')
            running_loss = 0.0

    return losses / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0
    running_loss = 0.0
    pred = []
    gt = []
    for src, src_image, tgt, future_gesture, future_kinematics in valid_dataloader:

        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        tgt_input = tgt[:-1, :]

        tgt_mask = get_tgt_mask(observation_window, device)

        logits = model(src, tgt_input, tgt_mask)

        tgt_out = tgt[1:, :]

        logits_reshaped = logits.reshape(-1, logits.shape[-1])
        tgt_reshaped = torch.argmax(tgt_out, dim=-1).reshape(-1)
        loss = loss_fn(logits_reshaped, tgt_reshaped)

        predicted_targets = torch.argmax(logits_reshaped, dim=-1).cpu().detach().numpy()
        accuracy = np.mean(predicted_targets == tgt_reshaped.cpu().numpy())
        pred.append(predicted_targets.reshape(-1))
        gt.append(tgt_reshaped.cpu().numpy().reshape(-1))

        print(f"Valid: Accuracy for frame: {accuracy}")
        losses += loss.item()

    pred, gt = np.concatenate(pred), np.concatenate(gt)
    print(get_classification_report(pred, gt, train_dataloader.dataset.get_target_names()))

    return losses / len(list(valid_dataloader))



from timeit import default_timer as timer
NUM_EPOCHS = 1

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(recognition_transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(recognition_transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, max_len, start_symbol, num_classes):

    model.eval()
    
    ys = torch.ones(1).fill_(start_symbol)
    ys = F.one_hot(ys.to(torch.int64), num_classes)
    print(src.shape, ys.shape)
    memory = model.encode(src)
    
    for _ in range(max_len-1):
        tgt_mask = (get_tgt_mask(ys.size(0), device).type(torch.bool))
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.fc_output(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        new_y = F.one_hot(torch.ones(1).type_as(src.data).fill_(next_word), num_classes)
        ys = torch.cat([ys, new_y], dim=0)
    return ys

X_trial, Y_trial = valid_dataloader.dataset.get_trial(0)
initial_symbol = torch.argmax(Y_trial[0])
pred = greedy_decode(recognition_transformer, X_trial, X_trial.shape[0], torch.argmax(Y_trial[0]), len(valid_dataloader.dataset.get_target_names()))