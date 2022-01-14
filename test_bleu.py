import nltk

from nmt_model import NMT
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def test_bleu(src_file, tgt_file, id):
    model = "model" + str(id) + ".bin"
    model = NMT.load(model)

    src = open(src_file, "r", encoding="utf-8")
    tgt = open(tgt_file, "r", encoding="utf-8")

    sum = 0
    cnt = 0
    labels = []
    lst = []
    for line in zip(src, tgt):
        source = nltk.word_tokenize(line[0])
        label = nltk.word_tokenize(line[1])
        hypotheses = model.beam_search(source)
        now = max(sentence_bleu([label], i.value) for i in hypotheses)
        sum += now
        cnt = cnt + 1
        # lst.extend([i.value for i in hypotheses])
        lst.append(hypotheses[0].value)
        labels.append([label])
        # for i in hypotheses:
        #     print(i.score, " ".join(i.value))
        # print(line[1], end="\n\n")
        # if cnt > 100:
        #     break
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join([" ".join(i) for i in lst]))
    print(f"bleu: {corpus_bleu(labels, lst)}")


    print(f"epoches: {id}  bleu: {sum / cnt}")

if __name__ == "__main__":
    # src_file = "/Users/tianwentang/Datasets/chr_en_data/test.chr"
    # tgt_file = "/Users/tianwentang/Datasets/chr_en_data/test.en"
    src_file = "/data2/twtang/Datasets/chr_en_data/test.chr"
    tgt_file = "/data2/twtang/Datasets/chr_en_data/test.en"
    id = ["20", "40", "60", "80", "100", "120", "140", "160", "180"]
    for i in id[:1]:
        test_bleu(src_file, tgt_file, i)