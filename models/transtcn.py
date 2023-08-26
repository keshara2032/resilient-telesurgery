
import torch
import torch.nn as nn


class GlobalMaxPooling1D(nn.Module):

    def __init__(self, data_format='channels_last'):
        super(GlobalMaxPooling1D, self).__init__()
        self.data_format = data_format
        self.step_axis = 1 if self.data_format == 'channels_last' else 2

    def forward(self, input):
        return torch.max(input, axis=self.step_axis).values
    

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)

        return out
    
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden
    
class CNN_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CNN_Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=96, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride = 2),
            nn.Conv1d(in_channels=96, out_channels=128, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride = 2),
            nn.Conv1d(in_channels=128, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride = 2)
        )
    

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape input to [batch_size, features, seq_len]
        x = self.encoder(x)
        # print('encoder_out',x.shape)
        return x
    
    
class CNN_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CNN_Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2.0),
            nn.ConvTranspose1d(in_channels=in_channels, out_channels=96, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0),
            nn.ConvTranspose1d(in_channels=96, out_channels=64, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0),
            nn.ConvTranspose1d(in_channels=64, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.decoder(x)
        # print('decoder_out',x.shape)
        return x
    
    
class TransformerModel(nn.Module):
    
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, hidden_dim, layer_dim,encoder_params, decoder_params,dropout=0.1):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout), num_layers=num_layers
        )
        
        # self.lstm = LSTMModel(2*d_model, hidden_dim, layer_dim, int(2*d_model))
       
        self.encoder = CNN_Encoder(**encoder_params)
        self.decoder = CNN_Decoder(**decoder_params)
        
        self.max_pool = GlobalMaxPooling1D()
        # self.fc = nn.Linear(input_dim, 2*d_model)
        # self.out = nn.Linear(int(d_model/2), output_dim)
        self.out = nn.Linear(d_model, output_dim) # vanilla + gru
        self.relu = nn.ReLU()
        
    # tcn + transformer
    def forward(self, x):
        
        x = self.encoder(x)
        
        x = self.decoder(x)
        
        x = x.permute(0, 2, 1)  # Reshape input to [batch_size, seq_len,features, ]
        
        x = self.transformer(x)
        
        x = self.max_pool(x) # gets rid of seq_len

        x = self.out(x)
        
        x = self.relu(x)
        
        
        return x
        

    
