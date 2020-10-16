import re
import numpy as np
from entity_contract import NER_pre_data, normal_param
from utils import data_change, normal_util
from model import keras_BILSTM_CEF

class NERInference:
    def __init__(self, model, word2idx, tags, n_words, path, split_pattern="(,|!|\.| +)"):
        self.model = model
        # self.words = words
        self.word2idx = word2idx
        self.tags = tags
        self.n_words = n_words
        self.pattern = split_pattern
        self.path = path

    def predict_all(self):
        contents, lengths = self.read_data()
        result = []
        result_ix = []
        length = len(contents)
        for index in range(length):
            tag_result, ix_result = self.predict(contents[index], lengths[index])
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
        raw = self.model.predict(content)[0][-length:]
        result = [np.argmax(row) for row in raw]
        result_tags = [self.tags[i] for i in result]
        return result_tags, result

    def read_data(self):
        # labels_to_ix = NER_pre_data.build_label(normal_param.labels)
        vocab = normal_util.read_vocab(normal_param.lstm_vocab)
        contents, lengths = read_single_data(self.path, vocab, length=normal_param.max_length)

        return contents, lengths

def read_single_data(path, vocab, length):
    txt_array = []
    lengths = []
    contents = NER_pre_data.read_content(path, mode="txt")
    for tmp in contents:
        content, max_length = data_change.auto_single_test_pad(tmp, vocab, length)
        txt_array.append(content)
        lengths.append(max_length)
    return txt_array, lengths

