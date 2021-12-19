import json

import nltk
import torch
from torch import nn
from collections import Counter
from itertools import chain
import os
import sentencepiece as smp

from utils import pad_sents


class VocabEntry(object):
    def __init__(self, word2id=None):
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0
            self.word2id['<s>'] = 1
            self.word2id['</s>'] = 2
            self.word2id['<unk>'] = 3
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, item):
        return self.word2id.get(item, self.unk_id)

    def __contains__(self, item):
        return item in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is read only')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size%d]' % len(self)

    def id2word(self, id):
        return self.id2word[id]

    def add(self, word):
        if not self.__contains__(word):
            id = len(self.word2id)
            self.word2id[word] = id
            self.id2word[id] = word
            return id
        return self.word2id[word]

    def words2indices(self, sentences):
        if type(sentences[0]) == list:
            return [[self[word] for word in sentence] for sentence in sentences]
        else:
            return [self[word] for word in sentences]

    def indices2words(self, indices):
        if type(indices[0]) == list:
            return [[self.id2word[index] for index in indice] for indice in indices]
        else:
            return [self.id2word[index] for index in indices]

    def to_input_tensor(self, sentences):
        word_ids = self.words2indices(sentences)
        pad_sents(word_ids, self['<pad>'])
        sent_var = torch.tensor(word_ids, dtype=torch.long)
        return sent_var

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        for k, v in word_freq.items():
            print(k, v)
            break
        valid_words = [k for k, v in word_freq.items() if v >= freq_cutoff]
        print("number of word types: {}, number of words w / frequency {}: {}".format(len(word_freq), freq_cutoff,
                                                                                      len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry


class Vocab(object):
    def __init__(self, src_vocab, tgt_vocab):
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents, tgt_sents, vocab_size, freq_cutoff):
        assert len(src_sents) == len(tgt_sents)
        print('init source vocab')
        src = VocabEntry.from_corpus(src_sents, vocab_size, freq_cutoff)
        print('init target vocab')
        tgt = VocabEntry.from_corpus(tgt_sents, vocab_size, freq_cutoff)
        return Vocab(src, tgt)

    def save(self, file_path):
        json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), open(file_path, 'w'), indent=2)

    @staticmethod
    def load(file_path):
        entry = json.load(open(file_path, "r"))
        src_word2id = entry["src_word2id"]
        tgt_word2id = entry["tgt_word2id"]
        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def __repr__(self):
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))


def read_corpus(file_path):
    src_sents = []
    tgt_sents = []
    fin = open(file_path, 'r', encoding='utf').readlines()
    cnt = 100
    for f in fin:
        line = json.loads(f)
        src = nltk.word_tokenize(line["english"])
        tgt = nltk.word_tokenize(line["chinese"])
        src_sents.append(src)
        tgt_sents.append(["<s>"] + tgt + ["<\s>"])
        cnt = cnt - 1
        if cnt <= 0:
            break
    return src_sents, tgt_sents


def get_vocab_list(file_path, source, vocab_size):
    smp.SentencePieceTrainer.Train(input=file_path, model_prefix=source, vocab_size=vocab_size)
    sp = smp.SentencePieceProcessor()
    sp.Load("{}.model".format(source))
    print(type(sp))
    sp_list = [sp.IdToPiece(id) for id in range(vocab_size)]
    print(sp_list[:20])
    return sp_list

if __name__ == '__main__':
    # read_corpus("E:\Datasets\translation2019zh\translation2019zh_train.json")
    # get_vocab_list("E:\\Datasets\\translation2019zh\\translation2019zh_train.json", "source", 10000)

    src_sents, tgt_sents = read_corpus("/Users/tianwentang/Datasets/translation2019zh/translation2019zh_train.json")
    print(len(src_sents))
    vocab = Vocab.build(src_sents, tgt_sents, 10000, 3)
    print("generate vocabulary, source %d words, target word %d" % (len(vocab.src), len(vocab.tgt)))
    vocab.save("vocab.json")