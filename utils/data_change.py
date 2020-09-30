import torch
import numpy as np

def prepare_sequence(seqs, to_ix):
    idxs = []
    tmp = []
    length = []
    max_length = 0
    # seqs_tmp= []
    # seqs_tmp.append(seqs)
    for seq in seqs:
        idx = [to_ix[w[0]] if w[0] in to_ix else 0 for w in seq]
        tmp.append(idx)
        max_length = max(len(idx), max_length)
    #     数据维度不统一，不能被转换成tensor

    for idx in tmp:
        pad_len = max_length - len(idx)
        [idx.append(0) for i in range(pad_len)]
        idxs.append(idx)
        length.append(max_length)
    return torch.from_numpy(np.array(idxs)), torch.from_numpy(np.array(length)), max_length

# def prepare_sequence(seq, to_ix):
#     idxs = [to_ix[w] for w in seq]
#     return idxs


def prepare_label(labels, to_ix, length):
    idxs = []
    tmp = []
    for label in labels:
        idx = [to_ix[w] for w in label]
        idx.append(to_ix["O"])
        idx.append(to_ix["O"])
        tmp.append(idx)
    for idx in tmp:
        pad_len = length - len(idx)
        [idx.append(to_ix["[PAD]"]) for i in range(pad_len)]
        idxs.append(idx)

    return torch.from_numpy(np.array(idxs))
