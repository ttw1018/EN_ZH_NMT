import sys

import time

import nltk

from nltk.translate import bleu_score

from torch import cuda

from torch import optim

from nmt_model import NMT

from utils import batch_iter

from vocab import Vocab, read_corpus

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def parser():
    argv = dict()
    for i in sys.argv[1:]:
        assert i[:2] == "--", "command samples: [test | train] - -command = type"
        lst = i[2:].split("=")
        argv[lst[0]] = "None" if len(lst) <= 1 else lst[1]
    return argv


def train(argv):
    src_sents = read_corpus(argv["src-sents-path"], 'src')
    tgt_sents = read_corpus(argv["tgt-sents-path"], 'tgt')
    batch_size = int(argv["batch-size"])
    epoches = int(argv["epoches"])

    print("loading vocab")
    vocab = Vocab.load("vocab.json")
    print("load vocab success")
    device = "cuda" if cuda.is_available() else "cpu"
    print("device: ", device)
    model = NMT(
        int(argv["embed-size"]),
        int(argv["hidden-size"]),
        vocab,
        float(argv["dropout-rate"]),
        device,
    )

    model = model.to(device)
    model.train()
    # model = model.load("model.bin")

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch = 0
    # batch_size = 1
    # epoches = 20
    while epoch < epoches:
        epoch = epoch + 1
        cnt = 0
        for src_sent, tgt_sent in batch_iter((src_sents, tgt_sents), batch_size):
            loss = model(src_sent, tgt_sent)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if cnt % 20 == 0:
                print("epoch: {} cnt: {}, loss: {}".format(epoch, cnt, loss.mean()), flush=True)
            cnt = cnt + 1
            # break
        if epoch % 20 == 0:
            model.save("model" + str(epoch) + ".bin")

    # model.save("model.bin")
    # hypothesis = model.beam_search(src_sents[0])
    # print(" ".join(hypothesis[0].value), "\n", " ".join(tgt_sents[0][1:-1]))


def main():
    argv = parser()
    if "train" in argv:
        train(argv)
    else:
        print("wrong command")


if __name__ == "__main__":
    main()
