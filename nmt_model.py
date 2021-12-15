import torch
from torch import nn


class NMT(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.1):
        super(NMT, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.encode = nn.LSTM()
