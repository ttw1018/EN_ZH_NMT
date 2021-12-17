def pad_sents(sents, pad_tokens):
    max_len = max([len(i) for i in sents])
    sents_len = len(sents)
    [sents[i].extend([pad_tokens] * (max_len - len(sents[i]))) for i in range(sents_len)]
    # return sents


def batch_iter(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i : min(len(data), i + batch_size)]