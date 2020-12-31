import os
from utils import normal_util
import random
import random
import time
import datetime

def regiun():
    '''生成身份证前六位'''
    #列表里面的都是一些地区的前六位号码
    first_list = ['362402','362421','362422','362423','362424','362425','362426','362427','362428','362429','362430','362432','110100','110101','110102','110103','110104','110105','110106','110107','110108','110109','110111']
    first = random.choice(first_list)
    return first

def year(min_year = 1948):
    '''生成年份'''
    return random.randint(min_year, int(time.strftime('%Y')))

def year_for_ID(start = 1948):
    '''生成年份'''
    now = time.strftime('%Y')
    #1948为第一代身份证执行年份,now-18直接过滤掉小于18岁出生的年份
    second = random.randint(start, int(now)-18)
    age = int(now) - second
    print('随机生成的身份证人员年龄为：'+str(age))
    return second

def month():
    '''生成月份'''
    three = random.randint(1,12)
    #月份小于10以下，前面加上0填充
    if three < 10:
        three = '0' + str(three)
        return three
    else:
        return three


def day():
    '''生成日期'''
    four = random.randint(1,28)
    #日期小于10以下，前面加上0填充
    if four < 10:
        four = '0' + str(four)
        return four
    else:
        return str(four)


def randoms():
    '''生成身份证后四位'''
    #后面序号低于相应位数，前面加上0填充
    five = random.randint(1,9999)
    if five < 10:
        five = '000' + str(five)
        return five
    elif 10 < five < 100:
        five = '00' + str(five)
        return five
    elif 100 < five < 1000:
        five = '0' + str(five)
        return five
    else:
        return five


def gain_IDcard():
    '''
    生成身份证信息
    :return: 身份证信息
    '''
    first = regiun()
    second = year_for_ID()
    three = month()
    four = day()
    last = randoms()
    IDcard = str(first)+str(second)+str(three)+str(four)+str(last)
    return IDcard


def concat_path(head_path, name):
    '''
    通过拼接获得对应类型数据的所有路径
    :param head_path: 头路径
    :param name: 数据类型名称
    :return: 对应类型数据的所有路径
    '''
    head_path = os.path.join(head_path, name)
    path = [os.path.join(head_path, path_name) for path_name in os.listdir(head_path)]
    # index =  [i for i in range(len(txt_paths))]
    # random.shuffle(index)
    # new_label_path = [label_paths[i] for i in index]
    # new_txt_paths = [txt_paths[i] for i in index]

    return path

def read_file(paths):
    '''
    获得路径下所有内容
    :param path: 路径list
    :return: 内容list
    '''
    data = normal_util.read_txt(paths)
    dic_value = []
    for values in data.split("\n"):

        values = values.split("\t")[-1]
        if len(values) == 0:
            continue
        dic_value.append(values)
    return dic_value



def gain_dic_data(path, dic_name):
    '''
    词典路径
    :param path: 路径
    :return: list 词典内容
    '''
    paths = concat_path(path, dic_name)
    values = []
    for dic_path in paths:
        values += read_file(dic_path)
    return values

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
        label_content = content.split("\t")[1]
        list = label_content.split(" ")
        label_name = list[0]
        if "replace" not in label_name:
            continue
        start_index = int(list[1])
        end_index = int(list[2])
        for i in range(start_index, end_index):
            label_list[i] = label_name
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

def gain_time(min_num):
    '''
    获取时间
    :param min_num 最小时间
    :return: 获取大于最小时间的时间
    '''

    min_strftime = datetime.datetime.strptime(min_num, "%Y-%m-%d")

    year_data  = year(min_strftime.year)
    month_data = month()
    day_data = day()
    data_time = "%s-%s-%s"%(year_data, month_data, day_data)
    strftime = datetime.datetime.strptime(data_time, "%Y-%m-%d")

    if strftime < min_strftime:
        gain_time(min_num)
    return data_time

def gain_timenumber(min_number, max_number):
    '''
    获取时间期限数字
    :param min_number: 最小期限
    :param max_number: 最大期限
    :return:
    '''
    time = ["天", "个月"]

    return "%s%s" %(random.randint(min_number, max_number), time[random.randint(0, 1)])

def gain_number(min_number, max_number):
    '''
    获取时间期限数字
    :param min_number: 最小期限
    :param max_number: 最大期限
    :return:
    '''
    time = ["天", "月"]

    return "%s" %(random.randint(min_number, max_number))

def gain_money(min_number, max_number):
    '''
    获取金额值
    :param min_number: 最小金额
    :param max_number: 最大金额
    :return:
    '''
    return random.randint(min_number, max_number)

def gain_local(path):
    '''
    获取地理位置
    :param path: 地理位置存放词典前一路径
    :return: 词典中任意一地理位置
    '''
    location_list = gain_dic_data(path, "location")
    index = random.randint(0, len(location_list))
    return location_list[index]

def gain_name(path):
    '''
    获取姓名
    :param path: 存放姓名词典的前一路径
    :return: 词典中任意一姓名
    '''
    name_list = gain_dic_data(path, "name")
    index = random.randint(0, len(name_list))
    return name_list[index]

def gain_case_money(path):
    '''
    获取大写金额总数
    :param path:
    :return:
    '''
    location_list = gain_dic_data(path, "case_money")
    index = random.randint(0, len(location_list))
    return location_list[index]


def gain_kind_of_data(mode, path = "", min_num = 0, max_num = 0):
    '''
    返回对应mode的data列表
    :param path: 对应data路径, default = ""
    :param mode: 对应data类型
    :param min_num: 最小值
    :param max_num: 最大值
    :return:
    '''
    if mode == "replace_money":
        return "%s%s"%(gain_money(1000, 20000), "元")
    elif mode == "replace_phone":
        return createPhone()
    elif mode == "replace_id":
        return gain_IDcard()
    elif mode == "replace_time":
        return gain_time("2017-9-7")
    elif mode == "replace_timenumber":
        return gain_timenumber(1, 30)
    elif mode == "replace_number":
        return gain_number(1, 30)
    elif mode == "replace_local":
        return gain_local(path)
    elif mode == "replace_name":
        return gain_name(path)
    elif mode == "replace_case_money":
        return gain_case_money(path)

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

def fill_all_data(path):
    '''
    填充txt中被标记的数据
    :param path: 总路径头【里面必须包含ann文件夹、txt文件夹、dic文件夹
    :return: 所有的被填充好的txt文本【转换成list格式
    '''
    ann_paths, txt_paths, txts_files_name = get_file_path(os.path.join(path, "txt_and_ann"))
    assert len(ann_paths) == len(txt_paths)
    all_txt_dic = []
    for i in range(len(ann_paths)):
        txt_dic = gain_txt(txt_paths[i])
        label_dic = gain_label(ann_paths[i], len(txt_dic))
        txt_dic = fill_data(txt_dic, label_dic, path)
        all_txt_dic.append(txt_dic)
    return all_txt_dic, txts_files_name

def write_all_data_to_txt(path, all_txt_dic, list_txt_name):
    '''
    把所有的被填充的文本list转换成新path路径下的txt文本
    :param path: 存储被填充的文本的路径
    :param all_txt_dic: 所有被填充好的文本
    :return:
    '''
    for i in range(len(all_txt_dic)):
        tmp_path = os.path.join(path, list_txt_name[i])
        content = "".join(all_txt_dic[i])
        normal_util.write_content(content, tmp_path)





def fill_data(txt_dic, label_dic, path):
    '''
    根据标签填充数据
    :param txt_dic: 需要被填充的数据组
    :param label_dic: 标记需要填充的标签组
    :return: 已经被填充好的数据
    '''
    last_label = ""
    filled_txt_dic = []
    for i in range(len(label_dic)):
        if label_dic[i] is 'O':
            filled_txt_dic += txt_dic[i]
            last_label = label_dic[i]
        else:
            if label_dic[i] is not last_label:
                last_label = label_dic[i]
                entity = gain_kind_of_data(last_label, path)
                filled_txt_dic += [i for i in entity]
    return filled_txt_dic







def createPhone():
    prelist = ["130", "131", "132", "133", "134", "135", "136", "137", "138", "139",
               "147", "150", "151", "152", "153", "155", "156", "157", "158", "159",
               "186", "187", "188", "189"]
    return random.choice(prelist) + "".join(random.choice("0123456789") for i in range(8))


def run(path):
    '''
    最终运行方法。
    :param path:
    :param new_path:
    :return:
    '''

    all_txt_content, txt_files_name = fill_all_data(path)
    write_all_data_to_txt(os.path.join(path, "txt"), all_txt_content, txt_files_name)


if __name__ == "__main__":
    # value = gain_dic_data("F:\毕业设计涉及论文\毕业设计数据", "姓名")
    # gain_label("F:/data/test/contact/0.ann", 10)
    run("F:/data/test/filldata_test")