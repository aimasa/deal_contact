from entity_contract import NER_pre_data, normal_param
from utils import data_change, normal_util, check_utils
import pickle
from model import keras_BILSTM_CEF
import numpy as np
import keras
from entity_contract import NERInference

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

def read_data(head_path, vocab, label_to_ix, max_length = 0):
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

def read_single_data(path, vocab, length):
    txts = []
    tmp = NER_pre_data.read_content(path, mode="txt")
    txts.append(tmp)
    content = data_change.prepare_test_sequence(txts,vocab, length)
    return content

def read_test_data(head_path, vocab, label_to_ix, length):
    '''
    读取测试数据
    :param head_path: 测试数据路径
    :param vocab: 词表dic
    :param labels_to_ix: label的one-hot转换
    :param length: 句子长度
    :return: x_test test数据one-hot表示, y_test test数据对应的label
    '''
    new_label_path, new_txt_paths = NER_pre_data.concat_path(head_path)
    txts, labels = NER_pre_data.load_data(new_label_path, new_txt_paths)
    arrys = data_change.prepare_test_sequence(txts, vocab, length)
    targets = data_change.prepare_label(labels, label_to_ix, length)
    return arrys, targets


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

def list_to_array(x_train, y_train, x_test, y_test, vocab, labels_to_ix, length):
    '''
    将测试数据和训练数据一起转换成长度一致的array
    :param x_train: 训练数据
    :param y_train: 训练数据对应的标签
    :param x_test: 测试数据
    :param y_test: 测试数据对应的标签
    :param vocab: 词表
    :param labels_to_ix: label对应的标签
    :param length: 所有数据的最大长度
    :return: 训练集（content， label）和测试集（content， label）array数组形式
    '''
    x_train, _ = data_change.auto_pad(x_train, vocab, length)
    y_train = data_change.auto_pad(y_train, labels_to_ix, length, is_label=True)
    x_test, _ = data_change.auto_pad(x_test, vocab, length)
    y_test = data_change.auto_pad(y_test, labels_to_ix, length, is_label=True)
    return x_train, y_train, x_test, y_test

def process_data(embeding = None):
    labels_to_ix, _ = NER_pre_data.build_label(normal_param.labels)
    vocab = read_vocab(normal_param.lstm_vocab)
    x, y = read_data(normal_param.head_path, vocab, labels_to_ix)
    # x_test, y_test = read_data(normal_param.head_test_path, vocab, labels_to_ix)
    x_train, y_train, x_test, y_test = split_tst_trn(x, y, 50)
    length = gain_max_length(x_train, x_test)

    x_train, y_train, x_test, y_test = list_to_array(x_train, y_train, x_test, y_test, vocab, labels_to_ix, length)
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    # y_train = np.expand_dims(y_train, 2)
    y_test = np.expand_dims(y_test, 2)
    return x_train, y_train, x_test, y_test, len(vocab), len(labels_to_ix)





