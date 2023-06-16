import torch
from model import Transformer


# Paper Params
p_prams = dict(
    d_enc_recognition = 38, # Obs 38 kinematic variables for JIGSAWS
    d_enc_prediction = 38 + 16, # Obs kinematic + gesture variables
    enc_positional_encoding = True, # used positional encoding
    d_dec_gesture_recognition = 16, # decoder model dimension was 16 for gesture recognition and prediction
    d_dec_prediction = 22, # decoder model dimension was 22 for trajectory prediction
    look_ahead_mask_decoder = True, # used look ahead masking for time series prediction causality
    used_padding_mask = False, # the window sizes are assumed to be fixed; no need for padding mask
    N = dict(recognition = 1, gesture_prediction = 4, traj_prediction = 1),
    h_enc = dict(recognition = 1, gesture_prediction = 1, traj_prediction = 6),
    h_dec = dict(recognition = 1, gesture_prediction = 4, traj_prediction = 11),
    train_test_split_JIGSAWS_suturing = 70/30,
    incremental_recognition_decoding = True, # they decoded recognized/predicted values incrementally, as expected
    recognition_input = "Obs kinematics",
    gesture_prediction_input = "Enc Input: Obs kinematics, Dec Input: Obs gestures, Dec Output: Future gestures",
    trajectory_prediction_input = "Obs kinematics and gestures, Dec Input: Obs position, future (assumedly predicted) gestures",
    prediction_loss = "Cumulative L2 distance (RMSE Loss)",
    optim = {
        "scheme": "same as the paper attention is all you need",
        "name" : "Adam",
        "b1" : 0.9,
        "b2" : 0.98,
        "eps" : 1e-9,
        "warmup_steps" : 2000,
        "lr" : "d_dec^-0.5 * (steps ^ -0.5, steps*warmup_steps ^ -1.5)"
    },
    gesture_metrics = "mean of accuracy per frame over test data",
    traj_metric = "RMSE and MAE",
    evaluation_scheme = "Leave-One-User-Out over subjects",
    batch_size = 64,
    epochs_recognition = 15,
    epochs_gesture_prediction = 40,
    epochs_traj_prediction = 50,
    recognition_freq = 30,
    prediction_freq = 10




)



# kinematic_features = 22 #input_dim
# dmodel = 64 #dmodel
# heads = 4
# num_layers = 2
# dropout = 0.1
# max_len = 100
# output_classes = 7

# is_train = True
# # src = torch.rand(10, 10, kinematic_features)

# model = Transformer(kinematic_features, dmodel, heads, num_layers, dropout, max_len, output_classes, is_train)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model.to(device)