tcn_model_params = {
    "class_num": 7,
    "decoder_params": {
        "input_size": 128,
        "kernel_size": 29,
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
        "kernel_size": 29,
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
    "nhead": 8,
    "num_layers": 2,
    "hidden_dim": 128,
    "layer_dim":4

}

learning_params = {
    "lr":8.906324028628413e-05,
    "epochs":30,
    "weight_decay":1e5,
    "patience":3,
    "batch_size": 10,
    "seq_len": 32
}