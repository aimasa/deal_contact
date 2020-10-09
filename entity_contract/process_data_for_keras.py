from entity_contract import NER_pre_data, normal_param
from utils import data_change
import pickle
from model import keras_BILSTM_CEF
import numpy as np
import keras
from entity_contract import NERInference

import sys
f = open('lstm_crf.log', 'a')
sys.stdout = f
sys.stderr = f		# redirect std err, if necessary

def read_vocab(vocab_path):
    '''
    读取词表内容
    :param vocab_path: 词表路径
    :return: 词表dic文件
    '''
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def read_data(head_path, vocab, label_to_ix, max_length = 0):
    '''
    读取数据部分
    :return:txt labels
    '''
    new_label_path, new_txt_paths, _ = NER_pre_data.concat_path(head_path)
    txts, labels = NER_pre_data.load_data(new_label_path, new_txt_paths)
    arrys, length, num_length = data_change.prepare_sequence(txts, vocab)
    max_length = max(num_length, max_length)
    targets = data_change.prepare_label(labels, label_to_ix, num_length)
    return arrys, targets, max_length

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
    new_label_path, new_txt_paths, _ = NER_pre_data.concat_path(head_path)
    txts, labels = NER_pre_data.load_data(new_label_path, new_txt_paths)
    arrys = data_change.prepare_test_sequence(txts, vocab, length)
    targets = data_change.prepare_label(labels, label_to_ix, length)
    return arrys, targets


# def split_follow_length(array, length):





def run():
    labels_to_ix = NER_pre_data.build_label(normal_param.labels)
    vocab = read_vocab(normal_param.lstm_vocab)
    x_train, y_train, length = read_data(normal_param.head_path, vocab, labels_to_ix)
    x_test, y_test = read_test_data(normal_param.head_test_path, vocab, labels_to_ix, length)
    model = keras_BILSTM_CEF.build_embedding_bilstm2_crf_model(len(vocab), len(labels_to_ix), length)
    # x_train, y_train = read_data(normal_param.head_path, vocab, labels_to_ix)
    # x_test, y_test = read_data(normal_param.head_test_path, vocab, labels_to_ix)
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    # y_train = np.expand_dims(y_train, 2)
    y_test = np.expand_dims(y_test, 2)
    model.fit(x_train, y_train, batch_size=18, epochs=7, validation_data = (x_test, y_test), shuffle = False, validation_split=0.2, verbose=1)
    keras_BILSTM_CEF.save_embedding_bilstm2_crf_model(model, normal_param.save_path)

def prediction(path):
    labels_to_ix = NER_pre_data.build_label(normal_param.labels)
    vocab = read_vocab(normal_param.lstm_vocab)
    model = keras_BILSTM_CEF.load_embedding_bilstm2_crf_model(normal_param.save_path, len(vocab), len(labels_to_ix), normal_param.max_length)
    myNerInfer = NERInference.NERInference(model, vocab, labels_to_ix, len(vocab), normal_param.max_length, path)
    new_string4_pred = myNerInfer.predict()
    print(new_string4_pred)
    # result = model.predict(content)
    # print(result)

if __name__ == '__main__':
    run()
    # prediction("F:/data/test/pred_contant/txt/0.txt")