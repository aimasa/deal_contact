
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import re
import os
from pro_data import process_data as process
import contact_classify.normal_param as normal_param
from matplotlib import pyplot as plt
from tensorflow.contrib import learn
from tqdm import tqdm
from utils import normal_util

# dict = {"财经":2, "彩票":0, "房产":1}
def pro_data(texts_path):
    '''读取测试路径中数据（texts_path）里面的待测试txt文件
    :return contents 所有待测试文件内容列表
    '''
    pro = process.process_data()
    # pro.split_data_file(normal_param.dev_path)
    contents_and_label = []
    contents = []
    for text_path in texts_path:
        content = pro.load_data_docx(text_path)
        contents += [content]
        # contents += [contents_and_label]
    # dev_data, dev_label, _=    pro.deal_data(part=len(pro.all_text_path), n_part=0)

    return contents


def get_array(texts_content):
    '''将待测试文件内容列表中文字转换为对应vocab中的文字下标数组
    :param texts_content 测试文件内容列表
    :return x 待测试文件内容对应下标数组
    '''
    # list_arr_text = self.deal_text(x_texts)
    x_texts = [x for x, label in texts_content]
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore("F:/phython workspace/deal_contact/contact_classify/vocab")
    x = np.array(list(vocab_processor.fit_transform(x_texts)))
    # shuffle_indices = np.random.permutation(np.arange(len(x_labels)))
    # text_shuffled = x[shuffle_indices]
    # label_shuffled = x_labels[shuffle_indices]
    return x


def test(data_array):
    '''使用模型对数据进行预测
    :param data_array 待预测数据数组
    :return pred 预测待预测数据对应的类别
    '''
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('F:/phython workspace/deal_contact/contact_classify/runs/' + normal_param.model_name + '/checkpoint/model-1000.meta')
        saver.restore(sess, tf.train.latest_checkpoint(
            os.path.join("F:/phython workspace/deal_contact/contact_classify/runs", normal_param.model_name, "checkpoint")))  # .data文件
        # pred = tf.get_collection('predictions')[0]

        graph = tf.get_default_graph()
        # var_list = [v.name for v in tf.global_variables()]
        # print(var_list)
        # for names in graph._names_in_use:
        #     print(names)
        # embeding = graph.get_tensor_by_name("embedding/W:0")
        # input_y = graph.get_tensor_by_name('input_y:0')
        input_x = graph.get_tensor_by_name('input_x:0')
        pred = graph.get_tensor_by_name("output/predictions:0")
        dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
        # score = graph.get_tensor_by_name("output/scores:0")
        # input_y = graph.get_operation_by_name('input_y').outputs[0]
        # sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())
        # accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')
        # W = graph.get_tensor_by_name('output/W:0')

        pred = sess.run(pred,
                        feed_dict={input_x: data_array, dropout_keep_prob: 1.0})

    print('Successfully load the pre-trained model!')
    return pred


def read_files(path):
    '''获取待预测测试文件的路径下的所有txt文件路径
    :param 待预测全部测试文件路径
    :return files_path 所有文件对应路径
    '''
    files_name = os.listdir(path)
    files_path = [os.path.join(path, file_name) for file_name in files_name]
    # file_label = [file_name for file_name in files_name]
    # print(files_path)
    return files_path


def run(path, classfiy_path):
    texts_path = read_files(path)
    dev_data = pro_data(texts_path)
    text_content = get_array(dev_data)

    # return dev_data, dev_label
    pro = test(text_content)
    print(pro)
    classfiy(pro,texts_path, classfiy_path)
    # print("this is acc", accuracy)

def classfiy(pros, texts_path, classfiy_path):
    '''将文档根据预测结果分类
    :param pro(array) 预测分类结果
    :param texts_path 被预测文件路径
    :param classfiy_path 被分类文件待存放路径'''
    files_path = normal_util.concat_file_path(classfiy_path, normal_util.get_all_path(classfiy_path))
    for index, pro in enumerate(pros.tolist()):
        normal_util.copy_replace(texts_path[index], files_path[pro])





if __name__ == "__main__":
    # path = "F:/实验数据暂存/tree_test"
    path = "G:/正保法律实务-买卖"
    classfiy_path = "F:/contact_classfiy"

    # path = "F:/实验数据暂存/tree/test-caipiao.txt"
    run(path, classfiy_path)
