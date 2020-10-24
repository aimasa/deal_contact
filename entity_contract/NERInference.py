import re
import numpy as np
from entity_contract import NER_pre_data, normal_param
from utils import data_change, normal_util
from model import keras_BILSTM_CEF
from entity_contract import process_data_for_keras
from keras_bert import Tokenizer,load_trained_model_from_checkpoint
from keras.preprocessing import sequence
import os
import tqdm
import codecs
class NERInference:
    def __init__(self, model, word2idx, tags, n_words, path, mode = "bilstm", split_pattern="(,|!|\.| +)"):
        self.model = model
        # self.words = words
        self.word2idx = word2idx
        self.tags = tags
        self.n_words = n_words
        self.pattern = split_pattern
        self.path = path
        self.mode = mode

    def predict_all(self, is_eval):
        if is_eval:
            contents, lengths, labels, contents_txtlist = self.read_data(is_eval)
        else:
            contents, lengths, contents_txtlist = self.read_data()
        result = []
        result_ix = []
        length = len(contents)
        for index in range(length):
            tag_result, ix_result = self.predict(contents[index], lengths[index])
            # print(tag_result)
            # print(contents_txtlist[index])
            # print(tag_result)
            result.append(tag_result)
            result_ix.append(ix_result)
        return result, result_ix




    def predict(self, content, length):
        # preds = []
        # pred_ner = np.argmax(self.model.predict(padded), axis=-1)
        # for w, pred in zip(padded[0], pred_ner[0]):
        #     if w == self.n_words - 1:
        #         break
        #     # print("{:15}: {}".format(self.words[w], self.tags[pred]))
        #     preds.append(list(self.tags.keys())[list(self.tags.values()).index(pred)])
        if self.mode == "lstm" or self.mode == "bilstm" or self.mode == "wordvec":
            raw = self.model.predict(content)[0][-length:]
        else:
            raw = self.model.predict(content)[0][:length]
        result = [np.argmax(row) for row in raw]
        result_tags = [self.tags[i] for i in result]
        return result_tags, result

    def read_data(self, is_evl = False):
        # labels_to_ix = NER_pre_data.build_label(normal_param.labels)
        # vocab = normal_util.read_vocab(normal_param.lstm_vocab)
        if is_evl is False:
            contents, lengths, contents_txtlist= read_single_data(self.path, self.word2idx, length=normal_param.max_length, mode=self.mode)

            return contents, lengths, contents_txtlist
        else:
            contents, lengths, labels, contents_txtlist = read_single_data(self.path, self.word2idx, length=normal_param.max_length, mode=self.mode, is_evl= is_evl)

            return contents, lengths, labels, contents_txtlist

def read_single_data(path, vocab, length, mode = "bilstm", is_evl = False):
    txt_array = []
    lengths = []
    labels_array = []
    if is_evl:
        contents = NER_pre_data.read_content(path, mode = mode)
        labels = NER_pre_data.read_content(path)

        for tmp in contents:
            content, max_length = data_change.auto_single_test_pad(tmp, vocab, length, mode=mode)
            txt_array.append(content)
            lengths.append(max_length)
        for label in labels:
            label_array = data_change.auto_single_test_pad(label, vocab, length, is_label=True, mode=mode)
            labels_array.append(label_array)
        return txt_array, lengths, labels_array, contents

    else:
        contents = NER_pre_data.read_content(path, mode="txt")

        for tmp in tqdm.tqdm(contents):
            # print(tmp)
            content, max_length = data_change.auto_single_test_pad(tmp, vocab, length, mode=mode)
            txt_array.append(content)
            lengths.append(max_length)
        return txt_array, lengths, contents

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
