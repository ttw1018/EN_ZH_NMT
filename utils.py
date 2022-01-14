def pad_sents(sents, pad_tokens):
    max_len = max([len(i) for i in sents])
    sents_len = len(sents)
    [sents[i].extend([pad_tokens] * (max_len - len(sents[i]))) for i in range(sents_len)]
    # return sents


def batch_iter(data, batch_size):
    assert(len(data[0]) == len(data[1]))
    total = len(data[0])
    for i in range(0, total, batch_size):
        yield data[0][i : min(total, i + batch_size)], data[1][i : min(total, i + batch_size)]


