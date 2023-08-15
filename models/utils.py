import math

import torch
from torch import Tensor
from torch.nn import Transformer
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



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

def plot_bars(gt, pred):

    print(gt, pred)

    def plot_sequence_as_horizontal_bar(sequence, title, ax):
        if not sequence:
            print(f"Error: Empty sequence for {title}!")
            return

        # Initialize variables
        unique_elements = [sequence[0]]
        span_lengths = [1]

        # Calculate the span lengths of each element in the sequence
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i - 1]:
                span_lengths[-1] += 1
            else:
                unique_elements.append(sequence[i])
                span_lengths.append(1)

        # Create the horizontal bar plot
        current_position = 0

        for i in range(len(unique_elements)):
            element = unique_elements[i]
            span_length = span_lengths[i]
            ax.barh(0, span_length, left=current_position, height=1, color=f'C{element}')
            current_position += span_length

        ax.set_yticks([])
        ax.set_xticks([])
        # ax.set_xlabel("Sequence")
        # ax.set_title(title)

    def plot_difference_bar(true_sequence, pred_sequence, ax):
        if not true_sequence or not pred_sequence:
            print("Error: Empty sequences!")
            return

        # Create a horizontal bar plot to indicate differences between sequences
        current_position = 0
        for true_elem, pred_elem in zip(true_sequence, pred_sequence):
            color = 'red' if true_elem != pred_elem else 'white'
            ax.barh(0, 1, left=current_position, height=1, color=color)
            current_position += 1

        ax.set_yticks([])
        ax.set_xticks([])
        # ax.set_title("Difference")
    
    # Replace these with your actual sequences
    true_sequence = gt
    pred_sequence = pred

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))

    plot_sequence_as_horizontal_bar(true_sequence, "Ground Truth", axes[0])
    plot_sequence_as_horizontal_bar(pred_sequence, "Predictions", axes[1])
    plot_difference_bar(true_sequence, pred_sequence, axes[2])

    plt.tight_layout()
    plt.show()

def get_tgt_mask(window_size, device):
    return Transformer.generate_square_subsequent_mask(window_size, device)

def plot_confusion_matrix(conf_matrix, labels):
    row_normalized_conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    row_normalized_conf_matrix = np.round(row_normalized_conf_matrix, 2)

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    sns.heatmap(row_normalized_conf_matrix, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels, annot_kws={'size': 12, 'ha': 'center'})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Row-Normalized Confusion Matrix (with 2 decimal places)')
    plt.show()

def get_classification_report(pred, gt, target_names):

    # get the classification report
    labels=np.arange(0, len(target_names) ,1)
    report = classification_report(gt, pred, target_names=target_names, labels=labels, output_dict=True)

    # plot computation matrix
    conf_matrix = confusion_matrix(gt, pred)
    # plot_confusion_matrix(conf_matrix, target_names)

    pd.DataFrame(report).transpose().to_csv("metrics.csv")

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

