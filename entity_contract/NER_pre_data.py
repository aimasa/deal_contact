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

#
#
# def iterate_data(iter_txt_path, iter_label_path, ech_size, batch_size, vocab, is_train):
#     '''
#     迭代器，边读取ech_size个文件，边返回batch_size大小的tensor
#     :param head_path: txt和label的头路径
#     :param ech_size: 一次性读取的文件个数
#     :param batch_size: 训练批次大小
#     :return: label的tensor和txt的tensor
#     '''
#
#
#     iter_start = 0
#     content_length = len(label_paths)
#     read_index_start = 0
#     read_index_end = len(label_paths) - 1
#     while True:
#         next_index = min(read_index_end, read_index_start + ech_size)
#         read_txt_paths = iter_txt_path[read_index_start: next_index]
#         read_label_path = iter_label_path[read_index_start: next_index]
#         read_index_start = next_index
#         txts, labels = load_data(read_label_path, read_txt_paths)
#         # length = len(txts)
#         # indexs = [i for i in range(length)]
#         # random.shuffle(indexs)
#         # txts = [txts[i] for i in indexs]
#         # labels = [labels[i] for i in indexs]
#         iter_end = len(txts)
#         while True:
#             next_iter_index = min((iter_start + batch_size), iter_end)
#             part_txts = txts[iter_start : next_iter_index]
#             part_labels = labels[iter_start : next_iter_index]
#             iter_start = next_iter_index
#             arrys, length, max_length = data_change.prepare_sequence(part_txts, vocab)
#             targets = data_change.prepare_label(part_labels, tag_to_ix, max_length)
#             if iter_start >= iter_end:
#                 break
#             # yield torch.tensor(arrys, dtype=torch.long), torch.tensor(targets, dtype=torch.long), torch.tensor(length, dtype=torch.long)
#             if is_train:
#                 yield torch.from_numpy(arrys), torch.from_numpy(targets), length
#             else:
#                 yield torch.from_numpy(arrys), torch.from_numpy(targets),  length, part_txts, part_labels, content_length
#         if read_index_start >= read_index_end:
#             break
#
# # tag_to_ix = build_label(normal_param.labels)
# # model: LSTM_CRF.Model = LSTM_CRF.Model(len(vocab), 300, 300, 1, normal_param.dropout1, normal_param.batch_size,
# #                                        len(tag_to_ix), tag_to_ix)
# # # word_size, word_ebd_dim, lstm_hsz, lstm_layers, dropout, batch_size, label_size, use_cuda = True
# # # model:BERT_LSTM_CRF.BERT_LSTM_CRF = BERT_LSTM_CRF.BERT_LSTM_CRF(normal_param.bert_path, len(tag_to_ix), EMBEDDING_DIM, HIDDEN_DIM, 1, 0.5, 0.5, use_cuda=True)
# # model.cuda()
# # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
# # label_paths, txt_paths, _ = concat_path(normal_param.head_path)
# # dev_label_paths, dev_txt_path, unsort_idx = concat_path(normal_param.head_test_path)
# # train_data = DataLoader(dataset=MyIterableDataset(label_paths, txt_paths, ech_size, normal_param.batch_size, vocab), shuffle=False)
# # test_data = DataLoader(dataset=MyIterableDataset(dev_label_paths, dev_txt_path, ech_size, normal_param.batch_size, vocab, is_train=False),
# #                        shuffle=False)
#
# def train():
#
#     model.train()
#     total_loss = 0
#     num = 0
#     for txts, labels, length  in train_data:
#         optimizer.zero_grad()
#         cor_time = time.time()
#         # arrys, length = data_change.prepare_sequence(txts, vocab)
#         # sentence_in = torch.tensor(arrys, dtype=torch.long)
#         # targets = torch.tensor(data_change.prepare_label(labels, tag_to_ix), dtype=torch.long)
#         loss, _ = model(txts.squeeze(0).long().cuda(), labels.squeeze(0).long().cuda(), torch.as_tensor(length.cpu(), dtype=torch.int64).squeeze(0))
#         print("训练次数：", num, loss, "spend time", cor_time - time.time())
#         num = num + 1
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.detach()
#     return total_loss / (len(txt_paths)//normal_param.batch_size)
#
# def evaluate():
#     model.eval()
#     label_list = []
#     eval_loss = 0
#     model_predict = []
#     for txts, labels, length, txts_part, labels_part in test_data:
#         loss, _ = model(txts.squeeze(0).long().cuda(), labels.squeeze(0).long().cuda(), torch.as_tensor(length.cpu(), dtype=torch.int64).squeeze(0))
#         pred = model.predict(txts, length)
#         pred = pred[unsort_idx]
#         seq_lengths = length[unsort_idx]
#
#         for i, seq_len in enumerate(seq_lengths.cpu().numpy()):
#             pred_ = list(pred[i][:seq_len].cpu().numpy())
#             label_list.append(pred_)
#
#         eval_loss += loss.detach().item()
#         for label_, sent, tag in zip(label_list, txts_part, labels_part):
#             tag_ = [tag_to_ix[label__] for label__ in label_]
#             sent_res = []
#             if len(label_) != len(sent):
#                 # print(sent)
#                 print(len(sent))
#                 print(len(label_))
#                 # print(tag)
#             for i in range(len(sent)):
#                 sent_res.append([sent[i], tag[i], tag_[i]])
#             model_predict.append(sent_res)
#
#         label_path = os.path.join(normal_param.result_path, 'label_' + str(normal_param.EPOCH))
#         metric_path = os.path.join(normal_param.result_path, 'result_metric_' + str(normal_param.EPOCH))
#
#         for line in conlleval(model_predict, label_path, metric_path):
#             print(line)
#
#         return eval_loss / (len(dev_txt_path)//normal_param.batch_size)
# import os
#
# def conlleval(label_predict, label_path, metric_path):
#     """
#     :param label_predict:
#     :param label_path:
#     :param metric_path:
#     :return:
#     """
#     eval_perl = "./conlleval_rev.pl"
#     with open(label_path, "w") as fw:
#         line = []
#         for sent_result in label_predict:
#             for char, tag, tag_ in sent_result:
#                 tag = '0' if tag == 'O' else tag
#                 char = char.encode("utf-8")
#                 line.append("{} {} {}\n".format(char, tag, tag_))
#             line.append("\n")
#         fw.writelines(line)
#     os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
#     with open(metric_path) as fr:
#         metrics = [line.strip() for line in fr]
#     return metrics
#     # for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
#     #     for sentence, tags in training_data:
#     #         # Step 1. Remember that Pytorch accumulates gradients.
#     #         # We need to clear them out before each instance
#     #         model.zero_grad()
#     #
#     #         # Step 2. Get our inputs ready for the network, that is,
#     #         # turn them into Tensors of word indices.
#     #         sentence_in = LSTM_CRF.prepare_sequence(sentence, word_to_ix)
#     #     #         targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
#     #
#     #         # Step 3. Run our forward pass.
#     #         loss = model.neg_log_likelihood(sentence_in, targets)
#     #
#     #         # Step 4. Compute the loss, gradients, and update the parameters by
#     #         # calling optimizer.step()
#     #         loss.backward()
#     #         optimizer.step()
#
# def run(is_train = True):
#
#     train_loss = []
#     if is_train:
#         total_start_time = time.time()
#
#         print('-' * 90)
#         for epoch in range(1, normal_param.EPOCH + 1):
#             epoch_start_time = time.time()
#             loss = train()
#             train_loss.append(loss * 1000.)
#
#             print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(
#                 epoch, time.time() - epoch_start_time, loss))
#             eval_loss = evaluate()
#             print("eval_loss: ", eval_loss)
#             torch.save(model.state_dict(), normal_param.save_path)
#
# if __name__ == '__main__':
#     # build_label(labels)
#     # split("他已经有五个月没有回来了", "F:/phython workspace/deal_contact/script/txt_process/dic_word.txt")
#     run()
#     # load_data(label_path, txt_path)
