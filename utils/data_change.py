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
    return np.array(idxs), np.array(length), max_length

def prepare_test_sequence(txt, to_ix, length):
    array_list = []
    for seqs in txt:
        array_list.append(prepare_single_sequence(seqs, to_ix, length))
    return np.array(array_list)

def prepare_pre_sequence(txt, to_ix, length):
    array_list = []
    for seqs in txt:
        tmp = [prepare_single_sequence(seqs, to_ix, length)]
        array_list.append(np.array(tmp))
    return array_list

def prepare_single_sequence(seqs, to_ix, length):
    idxs = []
    tmp = []
    # length = []
    # max_length = 0
    # seqs_tmp= []
    # seqs_tmp.append(seqs)

    idx = [to_ix[w[0]] if w[0] in to_ix else 0 for w in seqs]
        # max_length = max(len(idx), max_length)
    #     数据维度不统一，不能被转换成tensor
    pad_len = length - len(idx)
    if pad_len > 0:
        [idx.append(0) for i in range(pad_len)]
    else:
        idx = idx[:length]
    idxs.append(idx)
        # length.append(max_length)
    return np.array(idx)

# def prepare_sequence(seq, to_ix):
#     idxs = [to_ix[w] for w in seq]
#     return idxs


def prepare_label(labels, to_ix, length):
    idxs = []
    tmp = []
    for label in labels:
        idx = [to_ix[w] for w in label]
        tmp.append(idx)
    for index, idx in enumerate(tmp):
        pad_len = length - len(idx)
        [idx.append(to_ix["[PAD]"]) for i in range(pad_len)]
        idxs.append(np.array(idx))
    return np.array(idxs)

