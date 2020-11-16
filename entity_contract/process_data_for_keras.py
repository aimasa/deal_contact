from entity_contract import NER_pre_data, normal_param
from utils import data_change, normal_util, check_utils
import pickle
import numpy as np
from keras_bert import Tokenizer,load_trained_model_from_checkpoint
from keras.preprocessing import sequence
import codecs
import gc
import keras
from gensim.models import Word2Vec
import sys
# f = open('lstm_crf.log', 'a')
# sys.stdout = f
# sys.stderr = f		# redirect std err, if necessary

def read_vocab(vocab_path):
    '''
    读取词表内容
    :param vocab_path: 词表路径
    :return: 词表dic文件
    '''
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def gain_max_length(trn_contents, tst_contents):
    '''
    获得最大长度，以便于将所有句子调整至长度一致
    :param trn_contents: 训练文本数据
    :param tst_content: 测试文本数据
    :return: 训练集与测试集中数据最长的文本
    '''
    max_length = 0
    for trn_content in trn_contents:
        max_length = max(max_length, len(trn_content))
    for tst_content in tst_contents:
        max_length = max(max_length, len(tst_content))
    return max_length

def read_data(head_path, vocab = 1, label_to_ix = 2, max_length = 0):
    '''
    读取数据部分
    :return:txt labels
    '''
    new_label_path, new_txt_paths = NER_pre_data.concat_path(head_path)
    txts, labels = NER_pre_data.load_data(new_label_path, new_txt_paths)
    # arrys, length, num_length = data_change.auto_pad(txts, vocab)
    # max_length = max(num_length, max_length)
    # targets = data_change.prepare_label(labels, label_to_ix, num_length)
    return txts, labels


def read_data_part( new_txt_paths, new_label_path,):
    '''
    读取数据部分
    :return:txt labels
    '''
    txts, labels = NER_pre_data.load_data(new_label_path, new_txt_paths)
    # arrys, length, num_length = data_change.auto_pad(txts, vocab)
    # max_length = max(num_length, max_length)
    # targets = data_change.prepare_label(labels, label_to_ix, num_length)
    return txts, labels


# def split_follow_length(array, length):


def split_tst_trn(x, y, num_of_tst):
    '''
    将读取的数据切割
    :param x: 文本内容
    :param y: label内容
    :param num_of_tst: 测试集数量
    :return: 训练集（content， label）和测试集（content， label）
    '''
    x, y = normal_util.shuffle(x, y)
    length = len(x)
    x_train = x[0:length - num_of_tst]
    y_train = y[0:length - num_of_tst]
    x_test = x[length - num_of_tst : length]
    y_test = y[length - num_of_tst : length]
    return x_train, y_train, x_test, y_test



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

def list_to_array(x_train, y_train, x_test, y_test, vocab, labels_to_ix, length, wordembeding = None):
    '''
    将测试数据和训练数据一起转换成长度一致的array
    :param x_train: 训练数据
    :param y_train: 训练数据对应的标签
    :param x_test: 测试数据
    :param y_test: 测试数据对应的标签
    :param vocab: 词表
    :param labels_to_ix: label对应的标签
    :param length: 所有数据的最大长度
    :param wordembeding bert 使用bert对数据做pad和embeding   None 只做pad
    :return: 训练集（content， label）和测试集（content， label）array数组形式
    '''
    # if wordembeding is None:
    #     x_train, _ = data_change.auto_pad(x_train, vocab, length)
    #     x_test, _ = data_change.auto_pad(x_test, vocab, length)
    # elif wordembeding == 'bert':
    #     x_train = get_sentence(x_train)
    #     x_test = get_sentence(x_test)
    #     x_train, x_test = bert_embeding(x_train, x_test, length)
    #
    # y_train = data_change.auto_pad(y_train, labels_to_ix, length, is_label=True)
    # y_test = data_change.auto_pad(y_test, labels_to_ix, length, is_label=True)
    y_train, x_train = deal_txt_label_to_array(x_train, y_train, vocab, labels_to_ix, length, mode = wordembeding)
    y_test, x_test = deal_txt_label_to_array(x_test, y_test, vocab, labels_to_ix, length, mode = wordembeding)
    return x_train, y_train, x_test, y_test

def deal_txt_label_to_array(txt, label, vocab, labels_to_ix, length, mode = "bilstm"):
    '''
    将文本和对应的label转换成array
    :param txt: 需要被文本embeding的文本
    :param label: 标签
    :param vocab: 提供可转换的词表
    :param labels_to_ix: 提供可转换的label
    :param length: 文本中最大的长度
    :param mode: embeding模式
    :return: label_array  label的数组, array_txt 文本的数组
    '''
    if mode is "bilstm" or mode == "lstm" or mode == "rnn":
        array_txt, _ = data_change.auto_pad(txt, vocab, length)
    elif mode == 'bert':
        array_txt = get_sentence(txt)
        array_txt = txtpad_use_bert(array_txt, length)
    else:
        length = normal_param.max_length
        array_txt, _ = data_change.auto_pad(txt, vocab, length)
    label_array = data_change.auto_pad(label, labels_to_ix, length, is_label=True, model= mode)
    return label_array, array_txt

def txtpad_use_word2vec():
    '''通过word2vec获得对应的词嵌入矩阵'''

    model = Word2Vec.load("word2vec.h5")

    word2idx = {"_PAD": 0} # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。

    vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]

    # 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
    embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1
        embeddings_matrix[i + 1] = vocab_list[i][1]
    return embeddings_matrix, word2idx

def get_sentence(list):
    sentencelist = ["".join(str(i) for i in a) for a in list]
    return sentencelist


def bert_embeding(x_train, x_test, length):
    '''
    对训练数据和测试数据做bert的embeding
    :param x_train: 训练txt数据
    :param x_test: 测试txt数据
    :return: 被embeding之后的训练 + 测试 txt文本
    '''
    wordvec_train = txtpad_use_bert(x_train, length)
    wordvec_test = txtpad_use_bert(x_test, length)
    return wordvec_train, wordvec_test

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
    keras.backend.clear_session()
    return wordvec

# data_generator只是一种为了节约内存的数据方式
class data_generator:
    def __init__(self, data, label, n_part, batch_size=18, shuffle=True):
        self.data = data
        self.label = label
        self.part_num = len(data) // n_part
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start = 0
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):

        while True:
            end = min(len(self.data), self.start + self.batch_size)
            x_train, y_train, _, _ = process_data_gen(self.data[self.start : end], self.data[self.start : end], embeding="bert")
            if end >= len(self.data):
                self.start = 0
                break
            self.start = end
            yield x_train, y_train


def process_test_data():
    '''
    对测试集数据进行
    :return:
    '''
    labels_to_ix, _ = NER_pre_data.build_label(normal_param.labels)
    vocab = read_vocab(normal_param.lstm_vocab)
    x, y = read_data(normal_param.head_test_path, vocab, labels_to_ix)
    y_test, x_test = deal_txt_label_to_array(x, y, vocab, labels_to_ix, normal_param.max_length, mode = "bert")
    return x_test, y_test

def read_test_data_from_path():
    # labels_to_ix, _ = NER_pre_data.build_label(normal_param.labels)
    x, y = read_data(normal_param.head_path)
    return x, y

def process_data_gen(data, label, embeding = None):
    '''
    根据不同的embeding方法处理数据。
    :param embeding: embeding方法：bert、wordvec、不用embeding方法
    :return:
    '''
    labels_to_ix, _ = NER_pre_data.build_label(normal_param.labels)
    vocab = read_vocab(normal_param.lstm_vocab)
    # x, y = read_data_part(start_path, end_path)

    # x_test, y_test = read_data(normal_param.head_test_path, vocab, labels_to_ix)
    # x_train, y_train, x_test, y_test = split_tst_trn(x, y, 50)
    data, label = normal_util.shuffle(data, label)
    length = normal_param.max_length
    x_train, y_train = deal_txt_label_to_array(data, label, vocab, labels_to_ix, length, mode = None)
    # y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    # # y_train = np.expand_dims(y_train, 2)
    y_train = np.expand_dims(y_train, 2)
    return x_train, y_train, len(vocab), len(labels_to_ix)






def process_data(embeding = None, is_train = True, vocab2 = None):
    '''
    根据不同的embeding方法处理数据。
    :param embeding: embeding方法：bert、wordvec、不用embeding方法
    :return:
    '''
    labels_to_ix, _ = NER_pre_data.build_label(normal_param.labels)
    vocab = read_vocab(normal_param.lstm_vocab)
    # x_test, y_test = read_data(normal_param.head_test_path, vocab, labels_to_ix)
    if is_train:
        x, y = read_data(normal_param.head_path, vocab, labels_to_ix)

        x_train, y_train, x_test, y_test = split_tst_trn(x, y, 50)
        length = gain_max_length(x_train, x_test)
        if embeding == "wordvec":
            x_train, y_train, x_test, y_test = list_to_array(x_train, y_train, x_test, y_test, vocab2, labels_to_ix,
                                                             length, wordembeding=embeding)
        else:
            x_train, y_train, x_test, y_test = list_to_array(x_train, y_train, x_test, y_test, vocab, labels_to_ix, length, wordembeding = embeding)
        y_test = np.expand_dims(y_test, 2)
        y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))

        return x_train, y_train, x_test, y_test, len(vocab), len(labels_to_ix)
    else:
        x, y = read_data(normal_param.head_test_path, vocab, labels_to_ix)

        length = gain_max_length(x, [])
        y_test, x_test = deal_txt_label_to_array(x, y, vocab, labels_to_ix, length, mode = embeding)
        return x_test, y_test






