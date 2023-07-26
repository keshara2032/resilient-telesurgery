import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .crf import CRF

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.optim as optim

from .utils import *


    

class DirectRNNRecognitionModel(nn.Module):
    def __init__(self,
                 input_dim: int, # dimension of the input to the encoder
                 hidden_dim: int,
                 num_rnn_layers: int,
                 vocab_size: int, # number of output classes
                 rnn: str, # rnn type, either lstm or gru
                 emb_dim: int = -1, # input transformation dim
                 dropout: float = 0.0): # activation function of the input dim matching linear layers
        super(DirectRNNRecognitionModel, self).__init__()

        # parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_rnn_layers = num_rnn_layers
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        # encoder input transformation to model dimension
        if emb_dim:
            self.encoder_input_transform = torch.nn.Linear(in_features=input_dim, out_features=emb_dim)
        else:
            self.emb_dim = input_dim

        # dropout
        self.dropout = nn.Dropout(p=dropout)

        # lstm encoder
        RNN = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = RNN(self.emb_dim, hidden_dim // 2, num_layers=num_rnn_layers,
                       bidirectional=True, batch_first=True, dropout=dropout)
        self.crf = CRF(hidden_dim, self.vocab_size)

    def __build_features(self, sentences):
        # masks = sentences.gt(0)
        # embeds = self.embedding(sentences.long())

        # seq_length = masks.sum(1)
        # sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        # embeds = embeds[perm_idx, :]

        # pack_sequence = pack_padded_sequence(embeds, lengths=sorted_seq_length, batch_first=True)
        # packed_output, _ = self.rnn(pack_sequence)
        # lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        # _, unperm_idx = perm_idx.sort()
        # lstm_out = lstm_out[unperm_idx, :]

        if self.emb_dim:
            sentences = self.encoder_input_transform(sentences)
        sentences = self.dropout(sentences)
        lstm_out, _ = self.rnn(sentences)
        masks = torch.ones_like(sentences[:, :, 0])

        return lstm_out, masks

    def loss(self, xs, tags):
        features, masks = self.__build_features(xs)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, xs):
        # Get the emission scores from the BiLSTM
        features, masks = self.__build_features(xs)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq
    

class Trainer:
    def __init__(self) -> None:
        pass

    def __eval_model(self, model, device, dataloader, desc):
        model.eval()
        with torch.no_grad():
            # eval
            losses, nums = zip(*[
                (model.loss(src.to(device), tgt.to(device)), len(src))
                for src, src_image, tgt, future_gesture, future_kinematics in tqdm(dataloader, desc=desc)])
            losses = [l.item() for l in losses]
            return np.sum(np.multiply(losses, nums)) / np.sum(nums)

    def __save_loss(self, losses, file_path):
        pd.DataFrame(data=losses, columns=["epoch", "batch", "train_loss", "val_loss"]).to_csv(file_path, index=False)

    def __save_model(self, model_path, model):
        torch.save(model.state_dict(), model_path)
        print("save model => {}".format(model_path))

    def train_and_evaluate(self, train_dataloader, valid_dataloader, epochs, model_dir, args, device):
        model_dir = model_dir
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        input_dim = len(train_dataloader.dataset.get_feature_names())
        vocab_dim = len(train_dataloader.dataset.get_target_names())
        hidden_dim = args['hidden_dim']
        num_rnn_layers = args['num_rnn_layers']
        rnn = args['rnn']
        emb_dim = args['emb_dim']
        dropout = args['dropout']
        model = DirectRNNRecognitionModel(
            input_dim,
            hidden_dim,
            num_rnn_layers,
            vocab_dim,
            rnn,
            emb_dim,
            dropout
        )

        # loss
        loss_path = os.path.join(model_dir, "loss.csv")
        losses = pd.read_csv(loss_path).values.tolist() if args['recovery'] and os.path.exists(loss_path) else []

        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], betas=(0.9, 0.98), eps=1e-9)

        val_loss = 0
        best_val_loss = 1e4
        for epoch in range(epochs):
            # train
            model.train()
            for bi, (src, src_image, tgt, future_gesture, future_kinematics) in enumerate(tqdm(train_dataloader)):
                model.zero_grad()

                loss = model.loss(src, tgt)
                loss.backward()
                optimizer.step()
                # print("{:2d}/{} loss: {:5.2f}, val_loss: {:5.2f}".format(
                #     epoch+1, epochs, loss, val_loss))
                losses.append([epoch, bi, loss.item(), np.nan])

            # evaluation
            val_loss = self.__eval_model(model, device, dataloader=valid_dataloader, desc="eval").item()
            # save losses
            losses[-1][-1] = val_loss
            self.__save_loss(losses, loss_path)

            # save model
            if not args['save_best_val_model'] or val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(model_dir, 'model.pth')
                self.__save_model(model_path, model)
                print("save model(epoch: {}) => {}".format(epoch, loss_path))