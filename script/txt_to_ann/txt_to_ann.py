from utils import normal_util, check_utils
import os
from entity_contract import normal_param
import re
from tqdm import tqdm
def read_txt(path):
    '''
    读取path中的txt文档，并返回文档的list表list<>
    :param path: 文档的路径
    :return: 文档内容的list表
    '''
    content = normal_util.read_txt(path)
    content = content.replace("\r\n", "\n")
    content = content.replace("\n", "\r\n")
    content_by_char = []
    for i in content:
        content_by_char.append(i)
    return content_by_char

def read_label(path):
    '''
    读取path中的label文档，并返回文档中的label的list表list<>
    :param path: label的path
    :return: label的list表
    '''
    labels = normal_util.read_txt(path)
    labels = labels.replace("\r\n", "\n")
    labels = labels.replace("\n", "\r\n")
    labels = labels.split(" ")
    label_by_char = []
    for i in labels:
        tmp = i.split("\r\n")
        if len(tmp) > 1:
            for i in range(len(tmp)):
                if tmp[i] == "" :
                    label_by_char.append("\r")
                    label_by_char.append("\n")
                    continue
                elif i - 1 >= 0 and tmp[i - 1] != "":
                    label_by_char.append("\r")
                    label_by_char.append("\n")
                label_by_char.append(tmp[i])
        else:
            label_by_char.append(tmp[0])
    return label_by_char

def read_path(head_path):
    '''
    根据头路径获取不同路径下对应的txt和label文件路径信息
    :param head_path: 头路径的文件信息
    :return: label_path, content_path
    '''
    content_head_path = os.path.join(head_path, "txt")
    label_head_path = os.path.join(head_path, "label")
    names = os.listdir(content_head_path)
    content_paths = []
    label_paths = []
    for name in names:
        content_path = os.path.join(content_head_path, name)
        label_path = os.path.join(label_head_path, name)
        if not check_utils.check_path(label_path):
            continue
        content_paths.append(content_path)
        label_paths.append(label_path)
    return label_paths, content_paths


def ann_content(label, content):
    '''
    结合label和content的内容对ann进行填写
    :param label: 标签list<>
    :param content: 内容list<>
    :return: ann信息
    '''
    length = len(content)
    index = 0
    anns = []
    i = 0
    while i < length:
        start, end, tag = label_parse(label, i)
        i = end
        if tag is None:
            continue
        head = ("%s%s"%("T", index))
        str_index = ("%s %s %s"% (tag, start, end))
        str_content = content_parse(content, start, end)
        index = index + 1
        ann = [head, str_index, str_content]
        anns.append(ann)
    return anns



def content_parse(content, start, end):
    '''
    通过标签标注的下标，找到对应的content文本
    :param content: 文本
    :param start:起始下标
    :param end:结束下标
    :return: 对应的文本
    '''
    content_ = ""
    for i in range(start, end):
        content_ = content_ + content[i]
    return content_

def label_parse(label, start):
    '''
    解析当前label队列中存在哪些标签需要被标记
    :param label_info: label序列
    :return: label中的需要被标记的标签【start, end】
    '''
    if start >= len(label):
        return start, start + 1, None
    if re.match("\A(B-|I-|E-)", label[start]):
        label_ = label[start].replace("B-", "").replace("I-", "").replace("E-", "")
        tag = normal_param.label_to_tag[label_]
        end = start + 1
        head = ["B-", "E-", "S-"]
        while end < len(label) and (check_title_head(label[end], head) and label[end] != "O" and label[end] != "\r"):
            end = end + 1
        if  end < len(label) and label[end].startswith("E-"):
            end = end + 1
    elif label[start].startswith("S-"):
        end = start + 1
        tag = label[start].replace("S-", "")
        tag = normal_param.label_to_tag[tag]
    else:
        end = start + 1
        tag = None
    return start, end, tag


def check_title_head(title, head):
    '''
    判断字符串头中是否包含head里面任何一个头
    :param title: 需要被判断的字符串
    :param head: 需要被匹配的所有头列表list<>
    :return: 只要匹配上一个头就返回true，否则false
    '''
    for i in head:
        if title.startswith(i):
            return False

    return True


def run(head_path, ann_path):
    '''

    :param head_path:
    :param ann_path:
    :return:
    '''
    label_path, content_path = read_path(head_path)
    length = len(label_path)
    for i in tqdm(range(length)):
        id = re.search("(\\d+)\\.txt", content_path[i]).group(1)
        label = read_label(label_path[i])
        content = read_txt(content_path[i])
        ann_info = ann_content(label, content)
        write_ann_file(ann_info, os.path.join(check_utils.check_and_build(ann_path), ("%s.ann"% id)))


def write_ann_file(ann_info, ann_path):
    for index, info in enumerate(ann_info):
        str = ("%s\t%s\t%s"%(info[0], info[1], info[2]))
        with open(ann_path, "a", encoding="utf-8") as f:
            f.write(str)
            if index < len(ann_info) - 1:
                f.write("\n")










if __name__ == '__main__':
    run("F:/contract", "F:/contract/ann")
    # str = "A\nB\n"
    # tmp = str.split("\n")
    # print(tmp)