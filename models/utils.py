import math

import torch
from torch import Tensor
from torch.nn import Transformer
import torch.nn as nn
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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


    def step(self):
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

def plot_sequences_as_horizontal_bar(sequence1, sequence2):
    colors = colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    unique_numbers1 = sorted(set(sequence1))
    unique_numbers2 = sorted(set(sequence2))
    all_unique_numbers = sorted(set(unique_numbers1 + unique_numbers2))

    color_map = {num: color for num, color in zip(all_unique_numbers, colors)}

    def plot_sequence(sequence, offset):
        current_number = sequence[0]
        start_idx = 0
        spans = []

        for i, num in enumerate(sequence):
            if num != current_number:
                spans.append((start_idx, i - 1, current_number))
                start_idx = i
                current_number = num

        # Add the last span
        spans.append((start_idx, len(sequence) - 1, current_number))

        for i, (start, end, num) in enumerate(spans):
            color = color_map[num]
            plt.barh(offset, end - start + 1, left=start, height=0.5, color=color, edgecolor='black')
            # plt.text(start + (end - start + 1) / 2, offset, str(num), ha='center', va='center', color='white', fontsize=12)

    plt.figure(figsize=(10, 4))
    plt.yticks([])
    plot_sequence(sequence1, 0)
    plot_sequence(sequence2, 1)

    plt.xlabel('Sequence Index')
    plt.title('Horizontal Bar Plot of Sequences')
    plt.show()

def get_tgt_mask(window_size, device):
    return Transformer.generate_square_subsequent_mask(window_size, device)

def get_classification_report(pred, gt, target_names):

    # get the classification report
    labels=np.arange(0, len(target_names) ,1)
    report = classification_report(gt, pred, target_names=target_names, labels=labels, output_dict=True)

    # plot computation matrix
    # from sklearn import metrics
    # import matplotlib.pyplot as plt
    # label_map = {i: t for i, t in enumerate(target_names)}
    # pred = list(map(lambda x: label_map[x], pred))
    # gt = list(map(lambda x: label_map[x], gt))
    # disp = metrics.ConfusionMatrixDisplay.from_predictions(
    #     gt, pred, labels=target_names
    # )
    # disp.plot()
    # plt.show()

    return pd.DataFrame(report).transpose()

def merge_gesture_sequence(seq):
    import itertools
    merged_seq = list()
    for g, _ in itertools.groupby(seq): merged_seq.append(g)
    return merged_seq

def compute_edit_score(gt, pred):
    import editdistance
    max_len = max(len(gt), len(pred))
    return 1.0 - editdistance.eval(gt, pred)/max_len

