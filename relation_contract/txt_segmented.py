from utils import normal_util
import os
from relation_contract import ann_to_example
from collections import defaultdict

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, locations, labels, num_relations):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: unused in the entity task
          locations: entity localtion
          labels: relation label
          num_relations: number of relations in the text
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.locations = locations
        self.labels = labels
        self.num_relations = num_relations


def gain_label(path, word_count):
    '''
    获得label列表
    :param path: ann路径
    :param word_count: label对应的txt有多少个字
    :return: label列表
    '''
    label_list = []
    label_list.extend(["O"] * word_count)
    contents = normal_util.read_txt(path)
    for content in contents.split("\n"):
        if len(content.split("\t")) <= 1 or content.split("\t")[0].find("T") < 0:
            continue
        label_no = content.split("\t")[0]
        label_content = content.split("\t")[1]
        list = label_content.split(" ")
        # label_name = list[0]
        start_index = int(list[1])
        end_index = int(list[2])
        for i in range(start_index, end_index):
            label_list[i] = label_no
    return label_list


def gain_txt(path):
    '''
    通过路径获取路径对应的txt中的文本内容，并转换成list
    :param path: txt对应路径
    :return: txt对应的list列表
    '''
    content = normal_util.read_txt(path)

    content_list = [i for i in content.replace("\n", "\r\n")]
    return content_list


def get_file_path(base_path):
    '''将ann和txt文件分开；来获取路径列表
    :param base_path (str) 基础路径（包括ann和txt的）
    :return anns ann后缀的路径
    :return txt txt后缀的路径'''
    path_lists = os.listdir(base_path)
    ann_files = normal_util.names_filter(path_lists, ["ann"])
    txt_files = normal_util.names_filter(path_lists, ["txt"])
    anns = [os.path.join(base_path, i) for i in ann_files]
    txts = [os.path.join(base_path, i) for i in txt_files]
    return anns, txts, txt_files

def gain_format_data(type, path):
    '''
    获得格式化的训练数据
    :param path:
    :return:
    '''
    ann_paths, txt_paths, txts_files_name = get_file_path(path)
    assert len(ann_paths) == len(txt_paths)
    entity_T = defaultdict()
    txt_segments, label_segments, entity_T = gain_entity_and_split_segments(ann_paths, txt_paths, entity_T)
    all_labels = updata_entity_T(entity_T, label_segments)
    example, max_label_num = format_relation_and_entity(type, txt_segments, all_labels, entity_T)
    normal_util.shuffle_(example)
    return example

def gain_entity_and_split_segments(ann_paths, txt_paths, entity_T):
    '''
    获得实体dic及将路径中获取的文本整体及标签划分成句子的list。
    :param ann_paths: ann文本的所有路径
    :param txt_paths: txt文本的所有路径
    :param entity_T: 实体dic
    :return: txt_segments和其对应的label_segments
    '''
    txt_segments = []
    label_segments = []
    entity_infos = []
    for i in range(len(ann_paths)):
        entity_info = {}
        ann_to_example.gain_relation_contact_entity(entity_info, ann_paths[i])
        txt_dic = gain_txt(txt_paths[i])
        label_dic = gain_label(ann_paths[i], len(txt_dic))
        txt_segment, label_segment = segment_text(txt_dic, label_dic)
        txt_segments.append(txt_segment)
        label_segments.append(label_segment)
        entity_infos.append(entity_info)
    return txt_segments, label_segments, entity_infos


def format_relation_and_entity(type, txt_segments, all_labels, entity_T):
    '''
    将实体和关系通过entity dic格式化成example
    :param txt_segment: 句子list
    :param all_labels: 每个句子中的所有label
    :param entity_T: 实体dic
    :param num_relations: 关系类别数量
    :return: example
    '''
    example = []
    num = 0
    max_label_name = 0
    for index in range(len(txt_segments)):
        for next_index in range(len(txt_segments[index])):
            guid = "%s-%s"%(type, num)


            labels_name, entity_local = gain_label_and_local(all_labels[index][next_index], entity_T[index])
            if len(labels_name) <= 0:
                continue
            max_label_name = max(max_label_name, len(labels_name))
            example.append(InputExample(guid=guid, text_a="".join(txt_segments[index][next_index]), text_b=None,
                             locations=entity_local, labels=labels_name, num_relations=len(labels_name)))
            num += 1
    return example, max_label_name



def gain_label_and_local(all_label, entity_T):
    '''
    获得当前所有labels对应的关系和labels组成关系的第二个实体的local
    :param all_labels: 当前句子中存在的所有label
    :param entity_T: 实体dic
    :return: 实体对 对应的关系, entity_local 实体对 中实体对应的local（（），（））
    '''
    labels_name = []
    entity_local = []
    for label_no in all_label:
        T_dic = entity_T[label_no]
        if "relation_info" not in T_dic:
            continue
        relation_dic = T_dic["relation_info"]
        for key in relation_dic:
            labels_name.append(relation_dic[key])
            T2_dic = entity_T[key]
            entity_local.append((T_dic["local"], T2_dic["local"]))
    return labels_name, entity_local





def segment_text(txt_dic, label_dic):

    '''
    将数据根据段落切割，连同label
    :param txt_dic:
    :param label_dic:
    :return:
    '''
    txt_segments = []
    label_segments = []
    txt_segment = []
    label_segment = []
    for i in range(len(txt_dic)):
        if txt_dic[i] == '\n' or txt_dic[i] == '\r':
            if len(txt_segment) <= 0:
                continue
            txt_segments.append(txt_segment)
            label_segments.append(label_segment)
            txt_segment = []
            label_segment = []
            continue
        if txt_dic[i] == '\u3000':
            continue
        txt_segment.append(txt_dic[i])
        label_segment.append(label_dic[i])
    return txt_segments, label_segments

def updata_entity_T(entity_T, label_segments):
    '''
    对entity_T的entity的位置进行更新
    :param entity_T: 实体dic
    :param label_segments: 被分句后的label
    :return: 每个句子中存在的实体label数组   [[Tn,Tn+u……],[]……]
    '''
    all_labels = []
    for index in range(len(label_segments)):
        segment_labels = []
        for label_segment in label_segments[index]:

            labels = []

            left = 0
            while left < len(label_segment):
                right = left + 1
                while right < len(label_segment) and label_segment[right] == label_segment[left]:
                    right += 1
                if label_segment[left] != 'O':
                    adjustment_entity_local(label_segment[left], left, right, entity_T[index])
                    labels.append(label_segment[left])
                left = right
            segment_labels.append(labels)
        all_labels.append(segment_labels)
    return all_labels

def adjustment_entity_local(T_no, left, right, entity_T):
    entity_dic = entity_T[T_no]
    entity_dic["local"] = (left, right - 1)



if __name__ == "__main__":
    gain_format_data(True, "F:/data/test/test/ann")