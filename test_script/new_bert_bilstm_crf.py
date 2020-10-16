import numpy as np
from keras_bert import Tokenizer, load_trained_model_from_checkpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, TimeDistributed
from keras.preprocessing import sequence

import codecs
import yaml
from keras_contrib.layers import CRF

# 下面三个path是电脑中bert预训练模型文件路径，我这个是windows下的路径
config_path = 'F:\\phython workspace\\deal_contact\\bert\\bert_config.json'
checkpoint_path = 'F:\\phython workspace\\deal_contact\\bert\\bert_model.ckpt'
dict_path = 'F:\\phython workspace\\deal_contact\\bert\\vocab.txt'

tag2label = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}


def get_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        original_data = f.readlines()
        wordlist = []
        taglist = []
        for line in original_data:
            if len(line.split('\t')) > 1:
                word = line.replace('\n', '').split('\t')[0]
                tag = line.replace('\n', '').split('\t')[1]
            else:
                word = line
                tag = line
            wordlist.extend(word)
            taglist.append(tag)
        return wordlist, taglist


def get_sentence(list):
    sentencelist = []
    sentences = ''
    for word in list:
        if word != '\n':
            sentences = sentences + word
        else:
            sentencelist.append(sentences)
            sentences = ''
    return sentencelist


def tag(list):
    totallist = []
    childrenlist = []
    for tag in list:
        if tag != '\n':
            childrenlist.append(tag2label[tag])
        else:
            totallist.append(childrenlist)
            childrenlist = []
    return totallist


def tag_padding(list):
    leng = [len(x) for x in list]
    maxlen = 120
    for i, tag in enumerate(list):
        if len(tag) < maxlen:
            tag.extend([0] * (maxlen - len(tag)))
        else:
            list[i] = tag[:120]
    return list


def inputtag(list):
    totallist = []
    tagseq = ''
    for taglist in list:
        for tag in taglist:
            tagseq = tagseq + str(tag)
        totallist.append(tagseq)
        tagseq = ''
    return totallist


def get_token_dict(path):
    token_dict = {}
    with codecs.open(path, 'r', 'utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


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


def get_encode(list, token_dict):
    X1 = []
    X2 = []
    tokenizer = OurTokenizer(token_dict)
    for item in list:
        x1, x2 = tokenizer.encode(first=item)
        X1.append(x1)
        X2.append(x2)
    X1 = sequence.pad_sequences(X1, maxlen=120, padding='post', truncating='post')
    X2 = sequence.pad_sequences(X2, maxlen=120, padding='post', truncating='post')
    return [X1, X2]


def build_bert_model(X1, X2):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    wordvec = bert_model.predict([X1, X2])
    return wordvec


def build_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True)))


    model.add(TimeDistributed(Dense(7)))
    model.add(Dropout(0.5))
    crf_layer = CRF(7, sparse_target=True)
    model.add(crf_layer)
    model.compile(loss=crf_layer.loss_function, optimizer='rmsprop', metrics=[crf_layer.accuracy])
    return model


def train(wordvec, y, wordvec1, y1):
    model = build_model()
    model.fit(wordvec, y, batch_size=64, epochs=7, validation_split=0.1)
    yaml_string = model.to_yaml()
    with open('keras_bert_ner5.yml', 'w') as f:
        f.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('keras_bert_ner5.h5')
    model.save('ner5.pkl')
    print('start test ....')
    loas, acc = model.evaluate(wordvec1, y1, batch_size=64)
    print(acc)


if __name__ == '__main__':
    token_dict = get_token_dict(dict_path)
    wordlist, taglist = get_data('F:/data/SMART/msra_test_bio.txt')
    sentence = get_sentence(wordlist)
    print('start encoding...')
    [X1, X2] = get_encode(sentence, token_dict)
    print('start getting wordvec...')
    wordvec = build_bert_model(X1, X2)
    list = tag(taglist)
    y = np.array(tag_padding(list))
    y = np.expand_dims(y, 2)
    testword, testtag = get_data(('F:/data/SMART/msra_train_bio.txt'))
    testsen = get_sentence(testword)
    [X3, X4] = get_encode(testsen, token_dict)
    wordvec1 = build_bert_model(X3, X4)
    list1 = tag(testtag)
    y1 = np.array(tag_padding(list1))
    y1 = np.expand_dims(y1, 2)
    print('start training')
    train(wordvec, y, wordvec1, y1)