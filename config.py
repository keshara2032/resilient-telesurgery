import torch


RECORD_RESULTS = True

tcn_model_params = {
    "class_num": 7,
    "decoder_params": {
        "input_size": 128,
        "kernel_size": 61,
        "layer_sizes": [
            96,
            64,
            64
        ],
        "layer_type": "TempConv",
        "norm_type": "Channel",
        "transposed_conv": True
    },
    "encoder_params": {
        "input_size": 25,
        "kernel_size": 61,
        "layer_sizes": [
            64,
            96,
            128
        ],
        "layer_type": "TempConv",
        "norm_type": "Channel"
    },
    "fc_size": 32,
    "mid_lstm_params": {
        "hidden_size": 64,
        "input_size": 128,
        "layer_num": 1
    }
}


transformer_params = {
    "d_model": 64,
    "nhead": 32,
    "num_layers": 4,
    "hidden_dim": 64,
    "layer_dim": 4,
    "encoder_params": {
        "in_channels": 14,
        "kernel_size": 29,
        "out_channels": 64,
                       },
    "decoder_params": {
        "in_channels": 64,
        "kernel_size": 29,
        "out_channels": 64
    },
    "context":2 #0-nocontext, 1-contextonly, 2-context+kin, 3-imageonly, 4-image+kin, 5-image+kin+context
}

learning_params = {
    "lr": 8.906324028628413e-5,
    # "lr": 1e-6,
    "epochs": 10,
    "weight_decay": 1e-5,
    "patience": 3
}

dataloader_params = {
    "batch_size": 10,
    "one_hot": True,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "observation_window": 32,
    "prediction_window": 10,
    "user_left_out": 2,
    "cast": True,
    "include_image_features": False,
    "normalizer": '',  # ('standardization', 'min-max', 'power', '')
    "step": 2,  # 1 - 30 Hz
}
