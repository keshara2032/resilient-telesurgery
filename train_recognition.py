from typing import List
import os
from functools import partial
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from model import RecognitionModel, DirectRecognitionModel, ScheduledOptim, get_tgt_mask
from utils import get_classification_report, visualize_gesture_ts, get_dataloaders



# Data Params
tasks = ["Needle_Passing", "Suturing", "Knot_Tying"]
class_names = {
    "Peg_Transfer": ["S1", "S2", "S3", "S4", "S5", "S6", "S7"],
    "Suturing": ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11'],
    "Knot_Tyring": ['G1', 'G11', 'G12', 'G13', 'G14', 'G15'],
    "Needle_Passing": ["G1", 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11']
}
all_class_names = ["G1", 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15']

feature_names = [ "PSML_position_x", "PSML_position_y", "PSML_position_z", \
            "PSML_velocity_x", "PSML_velocity_y", "PSML_velocity_z", \
            "PSML_orientation_x", "PSML_orientation_y", "PSML_orientation_z", "PSML_orientation_w", \
            "PSML_gripper_angle", \
            "PSMR_position_x", "PSMR_position_y", "PSMR_position_z", \
            "PSMR_velocity_x", "PSMR_velocity_y", "PSMR_velocity_z", \
            "PSMR_orientation_x", "PSMR_orientation_y", "PSMR_orientation_z", "PSMR_orientation_w", \
            "PSMR_gripper_angle"]
one_hot = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
observation_window = 30
prediction_window = 40
batch_size = 64
user_left_out = 2
train_dataloader, valid_dataloader = get_dataloaders(tasks,
                                                     user_left_out,
                                                     observation_window,
                                                     prediction_window,
                                                     batch_size,
                                                     one_hot,
                                                     class_names=all_class_names,
                                                     feature_names=feature_names)

print("datasets lengths: ", len(train_dataloader.dataset), len(valid_dataloader.dataset))
print("X shape: ", train_dataloader.dataset.X.shape, valid_dataloader.dataset.X.shape)
print("Y shape: ", train_dataloader.dataset.Y.shape, valid_dataloader.dataset.Y.shape)

# loader generator aragement: (src, src_image, tgt, future_gesture, future_kinematics)
print("Obs Kinematics Shape: ", train_dataloader.dataset[0][0].shape) 
print("Obs Target Shape: ", train_dataloader.dataset[0][2].shape)
print("Future Target Shape: ", train_dataloader.dataset[0][3].shape)
print("Future Kinematics Shape: ", train_dataloader.dataset[0][4].shape)
print("Train N Trials: ", train_dataloader.dataset.get_num_trials())
print("Train Max Length: ", train_dataloader.dataset.get_max_len())
print("Test N Trials: ", valid_dataloader.dataset.get_num_trials())
print("Test Max Length: ", valid_dataloader.dataset.get_max_len())
print(train_dataloader.dataset.get_feature_names())
exit()

# Model Params
torch.manual_seed(0)
emb_size = 64
nhead = 4
ffn_hid_dim = 512
num_encoder_layers = 1
num_decoder_layers = 1
decoder_embedding_dim = 8
num_features = len(train_dataloader.dataset.get_feature_names())
num_output_classes = len(train_dataloader.dataset.get_target_names())
max_len = observation_window


# model initialization
# recognition_transformer = RecognitionModel(encoder_input_dim=num_features,
#                                             decoder_input_dim=num_output_classes, 
#                                             num_encoder_layers=num_decoder_layers,
#                                             num_decoder_layers=num_decoder_layers,
#                                             emb_size=emb_size,
#                                             nhead=nhead,
#                                             tgt_vocab_size=num_output_classes,
#                                             max_len = max_len,
#                                             decoder_embedding_dim = decoder_embedding_dim,
#                                             dim_feedforward=ffn_hid_dim,
#                                             dropout=0.1,
#                                             activation=torch.nn.GELU())

recognition_transformer = DirectRecognitionModel(encoder_input_dim=num_features,
                                            num_encoder_layers=num_decoder_layers,
                                            emb_size=emb_size,
                                            nhead=nhead,
                                            tgt_vocab_size=num_output_classes,
                                            max_len = max_len,
                                            dim_feedforward=ffn_hid_dim,
                                            dropout=0.1)
recognition_transformer = recognition_transformer.to(device)
for p in recognition_transformer.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)

# loss function
loss_fn = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(recognition_transformer.parameters(), lr=2e-5, betas=(0.9, 0.98), eps=1e-9)
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
        # logits = model(src, tgt_input, tgt_mask)
        logits = model(src)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        if one_hot:
            tgt_comp = torch.argmax(tgt_out, dim=-1).reshape(-1)
        else:
            tgt_comp = tgt_out.reshape(-1)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_comp)
        loss.backward()

        optimizer.step()
        losses += loss.item()

        # printing statistics
        running_loss += loss.item()
        if i % 50 == 0:    # print every 50 mini-batches
            print(f'[{i + 1:5d}] loss: {running_loss / 50:.3f}')
            running_loss = 0.0

    return losses / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0
    running_loss = 0.0
    pred = []
    gt = []
    accuracy = 0
    n_batches = len(valid_dataloader)
    for src, src_image, tgt, future_gesture, future_kinematics in valid_dataloader:

        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        tgt_input = tgt[:-1, :]

        tgt_mask = get_tgt_mask(observation_window, device)

        # logits = model(src, tgt_input, tgt_mask)
        logits = model(src)
        logits_reshaped = logits.reshape(-1, logits.shape[-1])

        tgt_out = tgt[1:, :]
        if one_hot:
            tgt_comp = torch.argmax(tgt_out, dim=-1).reshape(-1)
            tgt_reshaped = torch.argmax(tgt_out, dim=-1).reshape(-1)
        else:
            tgt_comp = tgt_out.reshape(-1)
            tgt_reshaped = tgt_out.reshape(-1)
        loss = loss_fn(logits_reshaped, tgt_comp)

        predicted_targets = torch.argmax(logits_reshaped, dim=-1).cpu().detach().numpy()
        accuracy += np.mean(predicted_targets == tgt_reshaped.cpu().numpy())
        # print("predictions: ", predicted_targets)
        # print("ground truth: ", tgt_reshaped.cpu().numpy())
        pred.append(predicted_targets.reshape(-1))
        gt.append(tgt_reshaped.cpu().numpy().reshape(-1))

        # print(f"Valid: Accuracy for frame: {accuracy}")
        losses += loss.item()

    pred, gt = np.concatenate(pred), np.concatenate(gt)
    print(get_classification_report(pred, gt, train_dataloader.dataset.get_target_names()))
    print(f"Evaluation accuracy: {accuracy/n_batches}")
    return losses / len(list(valid_dataloader))



from timeit import default_timer as timer
NUM_EPOCHS = 10

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(recognition_transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(recognition_transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, max_len, start_symbol, num_classes):

    model.eval()
    with torch.no_grad():
        if not one_hot:
            ys = torch.ones(1).fill_(start_symbol).to(torch.long).to(device).view(1, 1)
        else:
            ys = F.one_hot(ys.to(torch.int64), num_classes).to(torch.float32).to(device).unsqueeze(1)
        memory = model.encode(src.unsqueeze(1)).to(device)

        for i in range(max_len-1):
            tgt_mask = get_tgt_mask(ys.shape[0], device)
            out = model.decode(ys, memory, tgt_mask)
            prob = model.fc_output(out[-1, :, :])
            next_word = torch.argmax(prob.view(-1))
            # next_word = next_word.item()
            if not one_hot:
                new_y = next_word.to(torch.long).to(device).view(1, 1)
            else:
                new_y = F.one_hot(torch.ones(1).fill_(next_word).to(torch.int64), num_classes).to(torch.float32).reshape(1, 1, -1)
            ys = torch.cat([ys, new_y], dim=0)

    return ys

max_len = 30
X_trial, Y_trial = valid_dataloader.dataset.get_trial(0)
initial_symbol = Y_trial[-max_len-1]
X_trial, Y_trial = X_trial[-max_len:], Y_trial[-max_len:]
# pred = greedy_decode(recognition_transformer, X_trial, X_trial.shape[0], initial_symbol, len(valid_dataloader.dataset.get_target_names()))
pred = recognition_transformer(X_trial.unsqueeze(1))
print(pred.shape)
print(Y_trial)
print(torch.argmax(pred, dim=-1).view(-1))
# print(torch.argmax(pred2, dim=-1).view(-1))
# print(torch.argmax(Y_trial, dim=1))
# print(torch.mean((torch.argmax(pred, dim=-1).view(-1) == torch.argmax(Y_trial, dim=1)).to(torch.float32)))
