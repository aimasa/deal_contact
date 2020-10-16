import re
import random
import numpy as np
import jieba
from tensorflow.contrib import learn
from contact_classify import normal_param, evaluation
from progressbar import *
from cache import cache
from tqdm import tqdm
from docx import Document
from utils import normal_util, check_utils
import pickle
# import gensim
class process_data(object):
    def __init__(self):
        self.all_text_path = []
        self.part = None

    @cache.cache
    def build_datas_and_labels(self, parted_text_path, is_word_split=False):
        '''输入数据，得到输出'''
        contents_and_labels = []

        # progress = ProgressBar()
        for text_path, label in tqdm(parted_text_path):
            try:
                # content_and_label = self.load_data(text_path, label=label, is_word_split=is_word_split)
                content_and_label = self.load_data_docx(text_path, label=label, is_word_split=is_word_split)
                contents_and_labels.append(content_and_label)
            except Exception as e:
                print(e)
                print("\n")
                continue
        return contents_and_labels

    def clean_data(self, text):
        '''清洗数据，删掉无用字符'''
        # print(text)
        text = re.sub("\\n", " ", text)
        text = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]|\\（.*?）|\\【.*?】", " ", text)
        text = re.sub(u"_", " ", text)
        # text = re.sub("/(.*?)/", "", text)

        # text = re.sub(":", " ", text)
        text = re.sub("\\u3000|\\xa0", " ", text)
        text = re.sub("[^\u4e00-\u9fff]", " ", text)
        text = re.sub("\s{2,}", " ", text)
        # text_str = [i for i in text.split(" ")]
        return text


    def batch_iter(self, batch_size, num_epochs, shuffle=True):
        '''迭代器'''
        # num = 1
        # data = np.array(data)
        # data_size = len(data)
        # num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
        echo_part_num = len(self.all_text_path) // normal_param.num_database
        for epoch in range(num_epochs):
            print("epoch:", epoch, "/", num_epochs)
            for part_n in range(normal_param.num_database):
                is_save = False
                # train_data, train_label, dev_data, dev_label, vocal_size_train = self.deal_data(part=echo_part_num,n_part=part_n)
                train_data, train_label, vocal_size_train = self.deal_data(part=echo_part_num, n_part=part_n)
                dev_data, dev_label = evaluation.run()
                data = list(zip(train_data, train_label))
                data = np.array(data)
                data_size = len(data)
                num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
                if shuffle:
                    shuffle_indices = np.random.permutation(np.arange(data_size))
                    shuffle_data = data[shuffle_indices]
                else:
                    shuffle_data = data
                for batch_num in range(num_batches_per_epoch):
                    start_idx = batch_num * batch_size
                    end_idx = min((batch_num + 1) * batch_size, data_size)
                    if batch_num + 1 == num_batches_per_epoch:
                        is_save = True
                    yield shuffle_data[start_idx:end_idx], dev_data, dev_label, is_save
                    # yield shuffle_data[start_idx:end_idx], is_save

    def dense_to_one_hot(self, labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        temp = index_offset + labels_dense.ravel()
        labels_one_hot.flat[temp] = 1
        return labels_one_hot

    # @cache.cache
    def load_data(self, text_path, label=None, is_word_split=False):
        '''根据输入的文件路径以及文件对应的label，载入数据，输出label以及文件内容'''
        content_and_label = []
        with open(text_path, "rb") as f:
            content = ""
            for line in f:
                content = content + line.decode("utf-8")
            # print(content)
            content = self.clear_stop_word(content)
            content = self.clean_data(content)
            # content = [i for i in content.split(" ")]
            # if label is None:
            #     return content
            content_and_label = [content, label]
        return content_and_label

    def load_data_docx(self, text_path, label=None, is_word_split=False):
        '''根据输入的文件路径以及文件对应的label，载入数据，输出label以及文件内容'''
        content_and_label = []
        # with open(text_path, "rb") as f:
        #     content = ""
        #     for line in f:
        #         content = content + line.decode("utf-8")
        #     # print(content)
        #     content = self.clear_stop_word(content)
        #     content = self.clean_data(content)
        #     # content = [i for i in content.split(" ")]
        #     # if label is None:
        #     #     return content
        #     content_and_label = [content, label]

        document = Document(text_path)
        document_content = ""
        for content_para in document.paragraphs:
            content = content_para.text
            if check_utils.check_useless_seq(content):
                continue
            document_content += " " + content
        document_content = self.clear_stop_word(document_content)
        document_content = self.clean_data(document_content)
            # if label is None:
            #     return content
        content_and_label = [document_content, label]
        return content_and_label

    def split_data_file(self, path):
        '''将路径整理存放在self.all_text_path中，计算出需要被切割成多少part'''
        labels = os.listdir(path)
        labels_and_contents = [(os.listdir(os.path.join(path, label)), label, index) for index, label in
                               enumerate(labels)]
        for texts_filename, label, index in labels_and_contents:
            text_path = [(os.path.join(path, label, text_filename), index) for text_filename in texts_filename]
            self.all_text_path += text_path
        random.shuffle(self.all_text_path)
        # print(self.all_text_path)

    def file_path_split(self, part_n, part):
        '''将需要输入的数据根据batch size大小切割，边训练模型边读取后面的数据，但有个缺点，就是没办法知道全部数据中的最大的sequence size是多少'''

        first_path_num = part_n * part
        last_path_num = min(len(self.all_text_path), first_path_num + part)

        return self.all_text_path[first_path_num:last_path_num]

    def deal_data(self, n_part, part=0, need_label=False):
        '''加载数据，并且转成对应数据下标id数组输出'''

        # part = self.num_part()
        path_split = self.file_path_split(n_part, part)
        x_texts_labels = self.build_datas_and_labels(path_split)
        # 是取文本中对应的词语在该文本中的对应下标
        x_texts = [x for x, label in x_texts_labels]
        # list_arr_text = self.deal_text(x_texts)
        labels = np.array([label for x, label in x_texts_labels])
        if os.path.exists(os.path.join("./", "vocab")):
            vocab_processor = learn.preprocessing.VocabularyProcessor.restore(os.path.join("./", "vocab"))
            x = np.array(list(vocab_processor.fit_transform(x_texts)))
            print('not exist vocab')
        else:
            vocab_processor = learn.preprocessing.VocabularyProcessor(normal_param.max_document_length)
            x = np.array(list(vocab_processor.fit_transform(x_texts)))
            vocab_processor.save(os.path.join("./", "vocab"))

        # x = np.array(list_arr_text)

        print("x::", x)
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        text_shuffled = x[shuffle_indices]
        label_shuffled = labels[shuffle_indices]

        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("Train/Dev split: {:d}/{:d}".format(len(text_shuffled), len(label_shuffled)))
        # return train_text, self.dense_to_one_hot(train_label, normal_param.num_class), dev_text, self.dense_to_one_hot(
        #     dev_label, normal_param.num_class), len(
        #     vocab_processor.vocabulary_)
        if need_label:
            return text_shuffled, self.dense_to_one_hot(label_shuffled, normal_param.num_class), label_shuffled
        return text_shuffled, self.dense_to_one_hot(label_shuffled, normal_param.num_class), len(
            vocab_processor.vocabulary_)

    def clear_stop_word(self, text_content):
        '''清除停用词'''
        cleared_stop_word = []
        seg_list = jieba.cut(text_content, cut_all=False)
        seg_str = "/".join(list(seg_list))
        with open(normal_param.stop_words_path, "rb") as f:
            stop_word = f.read().decode("utf-8")
        stop_word_list = stop_word.split("\r\n")
        for word in seg_str.split("/"):
            if not (word.strip() in stop_word_list) and len(word.strip()) > 1:
                cleared_stop_word.append(word)
        return " ".join(cleared_stop_word)


if __name__ == "__main__":
    # clean_data("[我是共产主义接班人]对我就说（哈哈哈），不是我::::说的【略略略】，哈哈哈？{jahaha}，吃葡萄，不吐葡萄皮不     哈哈哈？？")
    # split_data_file(2, "F:/实验数据暂存/tempNew", 0)
    pro = process_data()
    # pro.deal_data(normal_param.train_path, 1, is_word_split=True)
    # pro.deal_data("F:/实验数据暂存/tempNew", 1)
    # new_word = pro.clear_stop_word("我今天好开心但是没有什么可以说的，不外乎也就是那么几种原因罢了，不过我今天还是有些许烦心事，之所以那么开心，还是因为开心事大过于难过的事情罢了")
    # print(new_word)
    pro.deal_data("F:/实验数据暂存/tempNew", 3, is_word_split=True)
