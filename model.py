import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import math
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.optim as optim
from torch import Tensor
from torch.nn import Transformer, TransformerEncoderLayer, LayerNorm, TransformerEncoder
import math





class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
    

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    

class ScheduledOptim():
    ''' A simple wrapper class for learning rate scheduling
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        sched = ScheduledOptim(optimizer, d_model=..., n_warmup_steps=...)
    '''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class CustomEncoder:
    def __init__(self, input_dim: int, num_encoder_layers: int, emb_size: int, nhead: int, dim_feedforward: int, dropout: float):
        self.input_encoder = torch.nn.Linear(in_features=input_dim, out_features=emb_size)
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_norm = LayerNorm(d_model=emb_size)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def forward(self, x):
        x1 = self.input_encoder(x)
        x2 = self.encoder(x1)
        return x2


class RecognitionModel(nn.Module):
    def __init__(self,
                 encoder_input_dim: int,
                 decoder_input_dim: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 decoder_embedding_dim: int = -1,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(RecognitionModel, self).__init__()

        self.decoder_embedding_dim = decoder_embedding_dim
        # custom encoder
        self.encoder = CustomEncoder(encoder_input_dim, num_encoder_layers, emb_size, nhead, dim_feedforward, dropout)
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       custom_encoder=self.encoder) # we are using a custom encoder to transform the input to emb_dize dimensions
        
        self.fc_output = nn.Linear(emb_size, tgt_vocab_size)
        if self.decoder_embedding_dim > 0:
            self.tgt_tok_emb = TokenEmbedding(decoder_input_dim, decoder_embedding_dim)
            self.decoder_embedding_transform = torch.nn.Linear(decoder_embedding_dim, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                tgt_mask: Tensor):
        
        src_emb = self.positional_encoding(src) # add positional encoding to the kinematics data
        if self.decoder_uses_embeddings:
            trg = self.tgt_tok_emb(trg) # if using label encoding as well, encode the gesture inputs to the decoder
            trg = self.decoder_embedding_transform(trg) # transform from decoder embedding dim to emb_size
        tgt_emb = self.positional_encoding(trg) # add positional encoding to the targets (gestures)
        outs = self.transformer(src_emb, tgt_emb, None, tgt_mask)
        return self.fc_output(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            src))

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
    

def get_tgt_mask(window_size, device):
    return Transformer.generate_square_subsequent_mask(window_size, device)