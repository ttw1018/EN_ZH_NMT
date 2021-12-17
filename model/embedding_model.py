from torch import nn
class EmbeddingModel(nn.Module):
    def __init__(self, embed_size, vocab):
        super(EmbeddingModel, self).__init__()
        src_pad_token = vocab.src['<pad>']
        tgt_pad_token = vocab.tgt['<pad>']
        self.source = nn.Embedding(
            len(vocab.src), embed_size, padding_idx=src_pad_token)
        self.target = nn.Embedding(
            len(vocab.tgt), embed_size, padding_idx=tgt_pad_token)