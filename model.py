import torch
import torch.nn as nn
import math 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.optim as optim
from torch import Tensor
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
import math





class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model%2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

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



class RecognitionModel(nn.Module):
    def __init__(self,
                 encoder_input_dim: int, # dimension of the input to the encoder
                 decoder_input_dim: int, # dimension of the input to the decoder (before embedding)
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 tgt_vocab_size: int, # number of output classes
                 max_len: int, # maximum length of the encoder input
                 decoder_embedding_dim: int = -1, # embedding dimension for the decoder input (if > 0)
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 activation=torch.nn.ReLU()): # activation function of the input dim matching linear layers
        super(RecognitionModel, self).__init__()

        # parameters
        self.decoder_embedding_dim = decoder_embedding_dim
        self.activation = activation

        # positional encoding
        self.encoder_positional_encoding = PositionalEncoding(
            encoder_input_dim, dropout=dropout, max_len=max_len)
        
        # encoder input transformation to model dimension
        self.encoder_input_transform = torch.nn.Linear(in_features=encoder_input_dim, out_features=emb_size)
         

        # output layer    
        self.fc_output = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                tgt_mask: Tensor):
        
        # encoder
        src_emb = self.encoder_positional_encoding(src) # add positional encoding to the kinematics data
        src_emb = self.encoder_input_transform(src_emb)
        src_emb = self.activation(src_emb)

        # decoder
        if self.decoder_embedding_dim > 0:
            trg = self.tgt_tok_emb(trg) # if using label encoding as well, encode the gesture inputs to the decoder
        tgt_emb = self.decoder_positional_encoding(trg) # add positional encoding to the targets (gestures)
        tgt_emb = self.decoder_embedding_transform(tgt_emb) # transform tgt to model hidden dimension (d_model = emb_size)
        tgt_emb = self.activation(tgt_emb)

        # output
        outs = self.transformer(src_emb, tgt_emb, None, tgt_mask)

        return self.fc_output(outs)

    def encode(self, src: Tensor, src_mask: Tensor = None):
        x = self.activation(self.encoder_input_transform(self.encoder_positional_encoding(src)))
        return self.transformer.encoder(x)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        if self.decoder_embedding_dim > 0:
            x = self.tgt_tok_emb(tgt)
        else:
            x = tgt
        x = self.decoder_embedding_transform(self.decoder_positional_encoding(x))
        x = self.activation(x)
        return self.transformer.decoder(x, memory, tgt_mask)
    

class DirectRecognitionModel(nn.Module):
    def __init__(self,
                 encoder_input_dim: int, # dimension of the input to the encoder
                 num_encoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 tgt_vocab_size: int, # number of output classes
                 max_len: int, # maximum length of the encoder input
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 activation=torch.nn.ReLU()): # activation function of the input dim matching linear layers
        super(DirectRecognitionModel, self).__init__()

        # parameters
        self.activation = activation

        # positional encoding
        self.encoder_positional_encoding = PositionalEncoding(
            encoder_input_dim, dropout=dropout, max_len=max_len)
        
        # encoder input transformation to model dimension
        self.encoder_input_transform = torch.nn.Linear(in_features=encoder_input_dim, out_features=emb_size)

        # encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_encoder_layers
        )

        # output layer    
        self.fc_output = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src: Tensor):
        
        # src transformation
        src_emb = self.encoder_positional_encoding(src) # add positional encoding to the kinematics data
        src_emb = self.encoder_input_transform(src_emb)
        src_emb = self.activation(src_emb)

        # encoder
        encoded = self.transformer_encoder(src_emb)

        # output
        out = self.fc_output(encoded)

        return out
    

def get_tgt_mask(window_size, device):
    return Transformer.generate_square_subsequent_mask(window_size, device)