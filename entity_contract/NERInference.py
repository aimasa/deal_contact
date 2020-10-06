import re
import numpy as np
from entity_contract import NER_pre_data, normal_param
from utils import data_change, normal_util
from model import keras_BILSTM_CEF

class NERInference:
    def __init__(self, model, word2idx, tags, n_words, maxlen, path, split_pattern="(,|!|\.| +)"):
        self.model = model
        # self.words = words
        self.word2idx = word2idx
        self.tags = tags
        self.n_words = n_words
        self.pattern = split_pattern
        self.maxlen = maxlen
        self.path = path

    def predict(self):
        preds = []
        padded = self.read_data()

        pred_ner = np.argmax(self.model.predict(padded), axis=-1)
        for w, pred in zip(padded[0], pred_ner[0]):
            if w == self.n_words - 1:
                break
            # print("{:15}: {}".format(self.words[w], self.tags[pred]))
            preds.append(list(self.tags.keys())[list(self.tags.values()).index(pred)])
        return preds

    def read_data(self):
        labels_to_ix = NER_pre_data.build_label(normal_param.labels)
        vocab = normal_util.read_vocab(normal_param.lstm_vocab)
        content = read_single_data(self.path, vocab, length=normal_param.max_length)

        return content

def read_single_data(path, vocab, length):
    txts = []
    tmp = NER_pre_data.read_content(path, mode="txt")
    txts.append(tmp)
    content = data_change.prepare_test_sequence(txts,vocab, length)
    return content
