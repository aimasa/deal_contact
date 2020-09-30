import pickle
from model import LSTM_CRF, BERT_LSTM_CRF
import torch
import torch.optim as optim
import jieba
import os
from entity_contract import normal_param
from torch.utils.data import Dataset
from utils import normal_util, data_change

import math
import random
from torch.utils.data import DataLoader
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
START_TAG = "[CLS]"
STOP_TAG = "[SEP]"
PAD_TAG = "[PAD]"
EPOCH = 4
tag_to_ix = {}
vocab = {}
ech_size = 100
batch_size = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Make up some training data
# training_data = [(
#     "the wall street journal reported today that apple corporation made money".split(),
#     "B I I I O O O B I O O".split()
# ), (
#     "georgia tech is a university in georgia".split(),
#     "B I O O O O B".split()
# )]
#
# word_to_ix = {}
# for sentence, tags in training_data:
#     for word in sentence:
#         if word not in word_to_ix:
#             word_to_ix[word] = len(word_to_ix)
#
# tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
#
# model = LSTM_CRF.BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
# optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
#
# # Check predictions before training
# with torch.no_grad():
#     precheck_sent = LSTM_CRF.prepare_sequence(training_data[0][0], word_to_ix)
#     precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
#     print(model(precheck_sent))
#
# # Make sure prepare_sequence from earlier in the LSTM section is loaded
# for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
#     for sentence, tags in training_data:
#         # Step 1. Remember that Pytorch accumulates gradients.
#         # We need to clear them out before each instance
#         model.zero_grad()
#
#         # Step 2. Get our inputs ready for the network, that is,
#         # turn them into Tensors of word indices.
#         sentence_in = LSTM_CRF.prepare_sequence(sentence, word_to_ix)
#         targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
#
#         # Step 3. Run our forward pass.
#         loss = model.neg_log_likelihood(sentence_in, targets)
#
#         # Step 4. Compute the loss, gradients, and update the parameters by
#         # calling optimizer.step()
#         loss.backward()
#         optimizer.step()
#
# # Check predictions after training
# with torch.no_grad():
#     precheck_sent = LSTM_CRF.prepare_sequence(training_data[0][0], word_to_ix)
#     print(model(precheck_sent))
# # We got it!
# 1、词组对应转换成index
# 2、标签转换成对应的one-hot格式
# 3、用以上数据初始化模型
# 4、把句子转换成下标形式，并将数据喂进模型

import argparse


parser = argparse.ArgumentParser(description='LSTM_CRF')
parser.add_argument('--epochs', type=int, default=32,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size for training')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--use-cuda', action='store_true',
                    help='enables cuda')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--use-crf', action='store_true',
                    help='use crf')

parser.add_argument('--mode', type=str, default='train',
                    help='train mode or test mode')

parser.add_argument('--save', type=str, default='./checkpoints/lstm_crf.pth',
                    help='path to save the final model')
parser.add_argument('--save-epoch', action='store_true',
                    help='save every epoch')
parser.add_argument('--data', type=str, default='dataset',
                    help='location of the data corpus')

parser.add_argument('--word-ebd-dim', type=int, default=300,
                    help='number of word embedding dimension')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='the probability for dropout')
parser.add_argument('--lstm-hsz', type=int, default=300,
                    help='BiLSTM hidden size')
parser.add_argument('--lstm-layers', type=int, default=2,
                    help='biLSTM layer numbers')
parser.add_argument('--l2', type=float, default=0.005,
                    help='l2 regularization')
parser.add_argument('--clip', type=float, default=.5,
                    help='gradient clipping')
parser.add_argument('--result-path', type=str, default='./result',
                    help='result-path')

args = parser.parse_args()

def build_label(labels):
    '''根据已有的label构建标签
    :param labels 已有标签种类
    :return tag_to_index dic 加上边界标记符组成的字典[key : 加上边界标记符的标签, value : 标签对应的下标]'''
    tag_heads = {'B', 'I', 'E', 'S'}
    last_index = 0
    for label in labels:
        for tag_head in tag_heads:
            tag = '%s-%s' % (tag_head, label)
            tag_to_ix[tag] = last_index
            last_index += 1
    special_tag(tag_to_ix)
    return tag_to_ix

def special_tag(tag_to_ix):
    '''添加特殊tag---index的key-value组合
    :param tag_to_ix  dic 加上边界标记符组成的字典[key : 加上边界标记符的标签, value : 标签对应的下标]
    :param index 特殊字符对应下标'''
    tag_to_ix['O'] = 0
    tag_to_ix[START_TAG] = 1
    tag_to_ix[STOP_TAG] = 2
    tag_to_ix[PAD_TAG] = 3


def read_content(file, mode = "label"):
    '''对file的内容进行读取，建立单词列表'''
    with open(file, "r", encoding="utf-8") as f:
        contents = f.read()
        # content = split(content)
    if mode is "txt":
        return read_txt(contents)

    return read_label(contents)

def read_txt(contents):
    '''
    读取txt文件，并返回
    :param contents:
    :return:
    '''
    result = []
    for content in contents:
        if content is "\n" or content is " ":
            continue
        # create_vocab(content)
        result.append(content)
    return result

def read_label(contents):
    contents = contents.replace("\n", " ")
    result = []
    for content in contents.split(" "):
        if content is "" or content is " ":
            continue
        result.append(content)
    return result



def get_path(head_path):
    names = os.listdir(head_path)
    return [os.path.join(head_path, name) for name in names]

# def creat_vocab(head_path):
#     txt_paths = get_path(head_path)
#
#     for txt_path in txt_paths:
#         txt_content = read_content(txt_path)
#         words = txt_content
#         for word in words:
#             if word not in vocab:
#                 vocab[word] = len(vocab)
#     return vocab

def create_vocab(word):
    if word not in vocab:
        vocab[word] = len(vocab)

def load_data(label_paths, txt_paths):
    '''
    :param label_paths: 标签路径
    :param txt_paths: 文本路径
    :return:
    '''
    labels = []
    txts = []
    build_label(normal_param.labels)
    max_length = 0
    if len(label_paths) is not len(txt_paths):
        return None
    for index in range(len(label_paths)):
        label_array = read_content(label_paths[index], "label")
        txt_array = read_content(txt_paths[index], "txt")
        labels.append(label_array)
        max(max_length, len(txt_array))
        txts.append(txt_array)
    return txts, labels


def split(content):
    '''用结巴对句子内容进行分词处理
    :param content 需要被分词的句子内容
    :param dic_path 新词表词典路径'''
    if os.path.exists(normal_param.dic_path):
        jieba.load_userdict(normal_param.dic_path)
    content = " ".join(jieba.cut(content))
    return content

def train(model, optimizer, train_data):

    # word_to_ix = creat_vocab(head_path)

    model.train()
    total_loss = 0

    for txts, labels, length  in train_data:
        optimizer.zero_grad()
        # arrys, length = data_change.prepare_sequence(txts, vocab)
        # sentence_in = torch.tensor(arrys, dtype=torch.long)
        # targets = torch.tensor(data_change.prepare_label(labels, tag_to_ix), dtype=torch.long)
        loss, _ = model(txts.squeeze(0).long().cuda(), labels.squeeze(0).long().cuda(), torch.as_tensor(length.cpu(), dtype=torch.int64).squeeze(0))
        loss.backward()
        optimizer.step()
        total_loss += loss.detach()
    return total_loss / train_data._stop_step

def evaluate(model, test_data):
    model.eval()


    # for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    #     for sentence, tags in training_data:
    #         # Step 1. Remember that Pytorch accumulates gradients.
    #         # We need to clear them out before each instance
    #         model.zero_grad()
    #
    #         # Step 2. Get our inputs ready for the network, that is,
    #         # turn them into Tensors of word indices.
    #         sentence_in = LSTM_CRF.prepare_sequence(sentence, word_to_ix)
    #     #         targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
    #
    #         # Step 3. Run our forward pass.
    #         loss = model.neg_log_likelihood(sentence_in, targets)
    #
    #         # Step 4. Compute the loss, gradients, and update the parameters by
    #         # calling optimizer.step()
    #         loss.backward()
    #         optimizer.step()

class MyIterableDataset(torch.utils.data.IterableDataset):

    def __init__(self, head_path, ech_size, batch_size, vocab):
        super(MyIterableDataset).__init__()
        # assert end > start, "this example code only works with end >= start"
        self.head_path = head_path
        self.ech_size = ech_size
        self.batch_size = batch_size
        self.vocab = vocab

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is None:  # single-process data loading, return the full iterator
        iter_head_path = self.head_path
        iter_ech_size = self.ech_size
        batch_size = self.batch_size
        # else:  # in a worker process
        #      # split workload
        #     # per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
        #     worker_id = worker_info.id
        #     # iter_start = self.iter_ech_size + worker_id * per_worker
        #     iter_end = min(iter_start + per_worker, self.end)
        #     batch_size = self.batch_size
        print("一次性迭代：\n")
        return iter(iterate_data(iter_head_path, iter_ech_size, batch_size, self.vocab))

# def read_file(path):
#
def concat_path(head_path):
    '''
    通过拼接路径得到相应的路径名称
    :param head_path: label和txt的总路径
    :return: label的路径和txt的路径
    '''
    label_head_path = os.path.join(head_path, "label")
    txt_head_path = os.path.join(head_path, "txt")
    label_paths = [os.path.join(label_head_path, path_name) for path_name in os.listdir(label_head_path)]
    txt_paths = [os.path.join(txt_head_path, path_name) for path_name in os.listdir(txt_head_path)]
    index =  [i for i in range(len(txt_paths))]
    random.shuffle(index)
    new_label_path = [label_paths[i] for i in index]
    new_txt_paths = [txt_paths[i] for i in index]
    return new_label_path, new_txt_paths



def iterate_data(head_path, ech_size, batch_size, vocab):
    '''
    迭代器，边读取ech_size个文件，边返回batch_size大小的tensor
    :param head_path: txt和label的头路径
    :param ech_size: 一次性读取的文件个数
    :param batch_size: 训练批次大小
    :return: label的tensor和txt的tensor
    '''
    label_paths, txt_paths = concat_path(head_path)

    iter_start = 0

    read_index_start = 0
    read_index_end = len(label_paths) - 1
    while True:
        next_index = min(read_index_end, read_index_start + ech_size)
        read_txt_paths = txt_paths[read_index_start: next_index]
        read_label_path = label_paths[read_index_start: next_index]
        txts, labels = load_data(read_label_path, read_txt_paths)
        # length = len(txts)
        # indexs = [i for i in range(length)]
        # random.shuffle(indexs)
        # txts = [txts[i] for i in indexs]
        # labels = [labels[i] for i in indexs]
        iter_end = len(txts) - 1
        while True:
            next_iter_index = iter_start + batch_size
            part_txts = txts[iter_start : next_iter_index]
            part_labels = labels[iter_start : next_iter_index]
            arrys, length, max_length = data_change.prepare_sequence(part_txts, vocab)
            targets = data_change.prepare_label(part_labels, tag_to_ix, max_length)
            if iter_start >= iter_end:
                break
            # yield torch.tensor(arrys, dtype=torch.long), torch.tensor(targets, dtype=torch.long), torch.tensor(length, dtype=torch.long)
            yield arrys, targets,  length
        if next_index >= read_index_end:
            break

def run(head_path):
    with open("vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    tag_to_ix = build_label(normal_param.labels)
    model : LSTM_CRF.Model = LSTM_CRF.Model(len(vocab), 300, 300, 1, normal_param.dropout1, normal_param.batch_size, len(tag_to_ix), tag_to_ix)
    # word_size, word_ebd_dim, lstm_hsz, lstm_layers, dropout, batch_size, label_size, use_cuda = True
    # model:BERT_LSTM_CRF.BERT_LSTM_CRF = BERT_LSTM_CRF.BERT_LSTM_CRF(normal_param.bert_path, len(tag_to_ix), EMBEDDING_DIM, HIDDEN_DIM, 1, 0.5, 0.5, use_cuda=True)
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    train_data = DataLoader(dataset=MyIterableDataset(head_path, ech_size, batch_size, vocab), shuffle=False)

if __name__ == '__main__':
    # build_label(labels)
    # split("他已经有五个月没有回来了", "F:/phython workspace/deal_contact/script/txt_process/dic_word.txt")
    train("F:/data/test/pred_contant")
    # load_data(label_path, txt_path)
