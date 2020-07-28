import codecs
import os
from utils import normal_util
from tqdm import tqdm
import re

tag_dic = {
   "location" : "LOCATION",
    "area" : "AREA",
    "rent" : "RENT",
    "type" : "TYPE",
    "startTerm" : "STARTTERM",
    "endTerm" : "ENDTERM",
    "deadline" : "DEADLINE"
}


def read_file(r_ann_path, r_txt_path, w_path, store_path):
    '''读取ann内容转成数组'''
    q_dic = {}
    split = []
    print("开始读取文件:%s" % r_ann_path)
    with codecs.open(r_ann_path, "r", encoding="utf-8") as f:
        line = f.readline()
        line = line.strip("\n\r")
        while line != "":
            line_arr = line.split()
            print(line_arr)
            cls = tag_dic[line_arr[1]]
            start_index = int(line_arr[2])
            end_index = int(line_arr[3])
            length = end_index - start_index
            if length is 1:
                q_dic[start_index] = ("S-%s" % cls)
            else:
                for r in range(length - 1):
                    if r == 0:
                        q_dic[start_index] = ("B-%s" % cls)
                    else:
                        q_dic[start_index + r] = ("I-%s" % cls)
                q_dic[start_index + length - 1] = ("E-%s" % cls)

            line = f.readline()
            line = line.strip("\t")

    print("开始读取文件:%s" % r_txt_path)
    with codecs.open(r_txt_path, "r", encoding="utf-8") as f:
        content_str = f.read()
        # content_str = content_str.replace("\n", "").replace("\r", "").replace("//////", "\n")


    print("开始写入文本%s" % w_path)
    with codecs.open(store_path, "a", encoding="utf-8") as store:
        with codecs.open(w_path, "w", encoding="utf-8") as wp:
            phrase = ""
            for i, str in enumerate(content_str):

                if i in q_dic:
                    tag = q_dic[i]
                    phrase += str
                else:
                    if len(phrase) > 0:
                        phrase = re.sub("\s","_", phrase)
                        store.write("%s\n" % phrase)
                        print("存入内容：-----------》", phrase)
                        phrase = ""
                    tag = "O"
                wp.write('%s %s\n' % (str, tag))
            wp.write('%s\n' % "END O")

def get_file_path(base_path, target_path):
    '''将ann和txt文件分开；来获取路径列表
    :param base_path (str) 基础路径（包括ann和txt的）
    :return anns ann后缀的路径
    :return txt txt后缀的路径
    :return target 文件转换完地址'''
    path_lists = os.listdir(base_path)
    ann_files = normal_util.names_filter(path_lists, ["ann"])
    txt_files = normal_util.names_filter(path_lists, ["txt"])
    anns = [os.path.join(base_path, i) for i in ann_files]
    txts = [os.path.join(base_path, i) for i in txt_files]
    targets = [os.path.join(target_path, i) for i in txt_files]
    return anns, txts, targets

def deal_files(base_path, target_path, dic_path):
    '''把文件转换成BIOES已打好标签内容的格式
    :param base_path ann,txt文件存放总路径
    :param target_path 已经打好标签的特殊文件路径
    :param dic_path 存放自定义字典的词组'''
    anns, txts, targets = get_file_path(base_path, target_path)
    if not (len(anns) is len(txts) and len(txts) is len(targets)):
        print("ann与txt数目不一致")
        return


    for index in tqdm(range(len(anns))):
        read_file(anns[index], txts[index], targets[index], dic_path)


def store_split(split_list, store_path):
    with codecs.open(store_path, "w", encoding="utf-8") as wp:
        wp.write("%s \n" % split_list)






if __name__ == '__main__':
    '''test'''
    # read_ann("F:/data/test/contact/0.ann","F:/data/test/contact/0.txt","F:/data/test/pred_contant/0.txt")
    deal_files("F:/data/test/contact","F:/data/test/pred_contant", "./dic_word.txt")
