import torch
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from entity_contract import normal_param
from keras_bert import Tokenizer,load_trained_model_from_checkpoint
from keras.preprocessing import sequence

import codecs

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


def auto_pad(seqs, to_ix, length, is_label = False, model = "bilstm"):

        if model == "bilstm" or model == "lstm" or model == "rnn":
            x = [[to_ix.get(w, 1) for w in s] for s in seqs]

            if is_label is False:
                max_length = len(x)
                x = pad_sequences(x, length)  # left padding
                return x, max_length
            else:
                y_chunk = pad_sequences(x, length, value=-1)
                return y_chunk
        elif model == "bert":
            x = [[to_ix.get(w, 1) for w in s] for s in seqs]
            labels = pad_sequences(x, length, padding= 'post', value=0)
            return labels
        elif model == "wordvec":
            x = [[to_ix[w] for w in s] for s in seqs]
            if is_label is False:
                max_length = len(x)
                x = pad_sequences(x, length)  # left padding
                return x, max_length
            else:
                y_chunk = pad_sequences(x, length, value=88)
                return y_chunk

def auto_single_test_pad(seq, to_ix, length, is_label = False, mode = "bilstm"):
    # seq = seq.split("")

    if mode == "bilstm" or mode == "lstm" or mode == "rnn":
        x = [to_ix.get(w, 1) for w in seq]
        if is_label is False:
            max_length = len(x)
            x = pad_sequences([x], length)  # left padding

            return x, max_length
        else:
            y_chunk = pad_sequences(x, length, value=-1)
            return y_chunk
    elif mode == "bert_bilstm":
        x = [to_ix.get(w, 1) for w in seq]
        if is_label is False:
            # wordvec = txtpad_use_bert([seq], length)
            max_length = len(seq)
            wordvec = txtpad_use_bert_cxy([seq], length)
            return wordvec, max_length
        else:
            labels = pad_sequences(x, length, padding='post', value=0)
            return labels
    elif mode == "wordvec":
        x = [to_ix.get(w, 1) for w in seq]
        if is_label is False:
            max_length = len(x)
            x = pad_sequences([x], length)  # left padding

            return x, max_length
        else:
            y_chunk = pad_sequences(x, length, value=0)
            return y_chunk


def prepare_test_sequence(txt, to_ix, length):
    array_list = []
    for seqs in txt:
        array_list.append(prepare_single_sequence(seqs, to_ix, length))
    return np.array(array_list)



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


def txtpad_use_bert(txt_list, length):
    '''
    对txt文本做bert的embeding处理
    :param txt_list:
    :param length:
    :return:
    '''
    token_dict = get_token_dict(normal_param.dict_path)

    [indices, segments] = get_encode(txt_list, token_dict, length)
    bert_model = load_trained_model_from_checkpoint(normal_param.config_path, normal_param.checkpoint_path,seq_len=None)
    wordvec = bert_model.predict([indices, segments])
    # keras.backend.clear_session()
    return wordvec

bert_model =  None
def  getBertModle_cxy():
    global bert_model
    if bert_model is not None:
        return bert_model
    bert_model = load_trained_model_from_checkpoint(normal_param.config_path, normal_param.checkpoint_path,seq_len=None)
    return bert_model

def txtpad_use_bert_cxy(txt_list, length):
    '''
    对txt文本做bert的embeding处理
    :param txt_list:
    :param length:
    :return:
    '''
    token_dict = get_token_dict(normal_param.dict_path)

    [indices, segments] = get_encode(txt_list, token_dict, length)
    bert_model = getBertModle_cxy()
    wordvec = bert_model.predict([indices, segments])
    # keras.backend.clear_session()
    return wordvec


def get_token_dict(path):
    token_dict ={}
    with codecs.open(path,'r','utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


def get_encode(list,token_dict, length):
    indices = []
    segments = []
    tokenizer = OurTokenizer(token_dict)
    for item in list:
        x1,x2 = tokenizer.encode(first=item)
        indices.append(x1)
        segments.append(x2)
    indices = sequence.pad_sequences(indices,maxlen=length,padding='post',truncating='post')
    segments = sequence.pad_sequences(segments,maxlen=length,padding='post',truncating='post')
    return [indices,segments]

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R
