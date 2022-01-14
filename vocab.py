import json
import nltk
import torch
from collections import Counter
from itertools import chain

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


    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size%d]' % len(self)

    # def id2word(self, id):
    #     return self.id2word[id]

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

    def to_input_tensor(self, sentences, device):
        word_ids = self.words2indices(sentences)
        if type(word_ids[0]) == list:
            pad_sents(word_ids, self['<pad>'])
        sent_var = torch.tensor(word_ids, dtype=torch.long, device=device)
        return sent_var

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [k for k, v in word_freq.items() if v >= freq_cutoff]
        print("number of word types: {}, number of words w / frequency {}: {}".format(len(word_freq), freq_cutoff,
                                                                                      len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry

    @staticmethod
    def from_subwords(words):
        vocab = VocabEntry()
        for word in words:
            vocab.add(word)
        return vocab



class Vocab(object):
    def __init__(self, src_vocab, tgt_vocab):
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents, tgt_sents, vocab_size, freq_cutoff):
        print('init source vocab')
        src = VocabEntry.from_corpus(src_sents, vocab_size, freq_cutoff)
        print('init target vocab')
        tgt = VocabEntry.from_corpus(tgt_sents, vocab_size, freq_cutoff)
        return Vocab(src, tgt)

    def save(self, file_path):
        json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), open(file_path, 'w', encoding='utf-8'), indent=2)

    @staticmethod
    def load(file_path):
        entry = json.load(open(file_path, "r"))
        src_word2id = entry["src_word2id"]
        tgt_word2id = entry["tgt_word2id"]
        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def __repr__(self):
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))



def convert_json(filename):
    src_sents = []
    tgt_sents = []
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()[:50000]
        for line in lines:
            line = json.loads(line)
            src_sents.append(line['english'])
            tgt_sents.append(line['chinese'])
    with open("E://Datasets//translation2019zh//en.txt", "w", encoding="utf-8") as f:
        f.write(str.join("\n", src_sents))

    with open("E://Datasets//translation2019zh//zh.txt", "w", encoding="utf-8") as f:
        f.write(str.join("\n", tgt_sents))


def read_corpus(filename, source):
    data = []
    for line in open(filename, "r", encoding='utf-8'):
        sent = nltk.word_tokenize(line)
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)
    return data
        
    
    

if __name__ == '__main__':
    src_sents = read_corpus("/Users/tianwentang/Datasets/chr_en_data/train.chr", 'src')
    tgt_sents = read_corpus("/Users/tianwentang/Datasets/chr_en_data/train.en", 'tgt')
    print(len(src_sents), len(tgt_sents))
    vocab = Vocab.build(src_sents, tgt_sents, 200000, 2)
    vocab.save("vocab.json")
    
    
    
