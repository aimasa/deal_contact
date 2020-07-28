
from model import LSTM_CRF
import torch
import torch.optim as optim
import jieba
import os
from entity_contract import normal_param
from torch.utils.data import Dataset
import math


EMBEDDING_DIM = 5
HIDDEN_DIM = 4
START_TAG = "[CLS]"
STOP_TAG = "[SEP]"

tag_to_ix = {}
vocab = {}

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


def read_content(file, mode = "label"):
    '''对file的内容进行读取，建立单词列表'''
    with open(file, "r", encoding="utf-8") as f:
        contents = f.read()
        # content = split(content)
    result = []
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
        create_vocab(content)
        result.append(vocab[content])
    return result

def read_label(contents):
    contents = contents.replace("\n", "")
    result = []
    for content in contents.split(" "):
        if content is "":
            continue
        result.append(tag_to_ix[content])
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
    if len(label_paths) is not len(txt_paths):
        return None
    for index in range(len(label_paths)):
        label_array = read_content(label_paths[index], "label")
        txt_array = read_content(txt_path[index], "txt")
        labels.append(label_array)
        txts.append(txt_array)
    return labels, txts





def split(content):
    '''用结巴对句子内容进行分词处理
    :param content 需要被分词的句子内容
    :param dic_path 新词表词典路径'''
    if os.path.exists(normal_param.dic_path):
        jieba.load_userdict(normal_param.dic_path)
    content = " ".join(jieba.cut(content))
    return content

def train(head_path):
    # word_to_ix = creat_vocab(head_path)
    tag_to_ix = build_label(normal_param.labels)
    model : LSTM_CRF.BiLSTM_CRF = LSTM_CRF.BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    print()
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

    def __init__(self, start, end, batch_size):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
        self.batch_size = batch_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
            batch_size = self.batch_size
        else:  # in a worker process
             # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
            batch_size = self.batch_size
        print("一次性迭代：\n")
        return iter(iterate_data(iter_start, iter_end, batch_size))

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
    return label_paths, txt_paths


def iterate_data(head_path, ech_size, batch_size):
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
        read_txt_paths = txt_path[read_index_start: next_index]
        read_label_path = label_paths[read_index_start: next_index]
        txts, labels = load_data(read_label_path, read_txt_paths)
        iter_end = len(txts) - 1
        while True:
            next_iter_index = iter_start + batch_size
            part_txts = txts[iter_start : next_iter_index]
            part_labels = labels[iter_start : next_iter_index]

            if iter_start >= iter_end:
                break
            yield part_labels, part_txts
        if next_index >= read_index_end:
            break





if __name__ == '__main__':
    # build_label(labels)
    # split("他已经有五个月没有回来了", "F:/phython workspace/deal_contact/script/txt_process/dic_word.txt")
    label_path, txt_path = concat_path("F:/data/test/pred_contant")
    load_data(label_path, txt_path)
