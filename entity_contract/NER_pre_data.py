import pickle
from model import LSTM_CRF, BERT_LSTM_CRF
import torch
import torch.optim as optim
import jieba
import os
from entity_contract import normal_param
from torch.utils.data import Dataset
from utils import normal_util, data_change
import time
import tqdm
import math
import random
from torch.utils.data import DataLoader
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
START_TAG = "[CLS]"
STOP_TAG = "[SEP]"
PAD_TAG = "[PAD]"
ix_to_label = {}
tag_to_ix = {}
vocab = {}
ech_size = 100
# batch_size = 2
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# with open("vocab.pkl", 'rb') as f:
#     vocab = pickle.load(f)


def build_label(labels):
    if len(tag_to_ix) > 0 : return tag_to_ix, ix_to_label
    '''根据已有的label构建标签
    :param labels 已有标签种类
    :return tag_to_index dic 加上边界标记符组成的字典[key : 加上边界标记符的标签, value : 标签对应的下标]'''
    tag_heads = ['B', 'I', 'E', 'S']
    labs = ["O", START_TAG, STOP_TAG, PAD_TAG]
    for label in labels:
        for tag_head in tag_heads:
            tag = '%s-%s' % (tag_head, label)
            tag_to_ix[tag] = len(tag_to_ix)
            ix_to_label[tag_to_ix[tag]] = tag
    for lab in labs:
        tag_to_ix[lab] = len(tag_to_ix)
        ix_to_label[tag_to_ix[lab]] = lab

    return tag_to_ix, ix_to_label

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

    :param contents:
    :return:
    '''
    result = []
    for content in contents.split("\n"):
        tmp = split_txt(content)
        if(len(tmp)):
            result.append(tmp)
    return result



def split_txt(contents):
    '''
    读取txt文件，并返回
    :param contents:
    :return:
    '''
    result = []
    for content in contents:
        if content is "\n" or content is " " or content is "":
            continue
        # create_vocab(content)
        result.append(content)
    return result

def split_label(contents):
    '''
    将自然段中的label通过空格截断，整理成label的list格式
    :param contents: 自然段的全部内容
    :return: 被整理成label的list<>
    '''
    result = []
    for content in contents.split(" "):
        if content is "" or content is " ":
            continue
        result.append(content)
    return result

def read_label(contents):
    '''
    根据"\n"切割contents，根据自然段分割句子
    :param contents: 一篇文章的全部内容
    :return: 分段后的文章句子
    '''
    contents = contents.split("\n")
    result = []
    for content in contents:
        if content is "" or content is " ":
            continue
        tmp = split_label(content)
        if len(tmp):
            result.append(tmp)
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
        labels += label_array
        max(max_length, len(txt_array))
        txts += txt_array
    return txts, labels


def split(content):
    '''用结巴对句子内容进行分词处理
    :param content 需要被分词的句子内容
    :param dic_path 新词表词典路径'''
    if os.path.exists(normal_param.dic_path):
        jieba.load_userdict(normal_param.dic_path)
    content = " ".join(jieba.cut(content))
    return content

# class MyIterableDataset(torch.utils.data.IterableDataset):
#
#     def __init__(self, label_paths, txt_paths, ech_size, batch_size, vocab, is_train = True):
#         super(MyIterableDataset).__init__()
#         # assert end > start, "this example code only works with end >= start"
#         self.ech_size = ech_size
#         self.batch_size = batch_size
#         self.vocab = vocab
#         self.label_paths = label_paths
#         self.txt_paths = txt_paths
#         self.is_train = is_train
#
#     def __iter__(self):
#         # worker_info = torch.utils.data.get_worker_info()
#         # if worker_info is None:  # single-process data loading, return the full iterator
#         iter_txt_path = self.txt_paths
#         iter_label_path = self.label_paths
#         iter_ech_size = self.ech_size
#         batch_size = self.batch_size
#         # else:  # in a worker process
#         #      # split workload
#         #     # per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
#         #     worker_id = worker_info.id
#         #     # iter_start = self.iter_ech_size + worker_id * per_worker
#         #     iter_end = min(iter_start + per_worker, self.end)
#         #     batch_size = self.batch_size
#         print("一次性迭代：\n")
#         return iter(iterate_data(iter_txt_path, iter_label_path, iter_ech_size, batch_size, self.vocab, self.is_train))
#
# # def read_file(path):
# #
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
    # index =  [i for i in range(len(txt_paths))]
    # random.shuffle(index)
    # new_label_path = [label_paths[i] for i in index]
    # new_txt_paths = [txt_paths[i] for i in index]
    return label_paths, txt_paths
