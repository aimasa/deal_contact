import re
import os
from utils import normal_util, check_utils
from tqdm import tqdm

def get_path(path):
    '''文件路径处理，从总文件路径中给出txtpath和labelpath
    :param path 文件总路径'''
    txt_path = os.path.join(path, "txt")
    label_path = os.path.join(path, "label")
    return check_utils.check_and_build(txt_path), check_utils.check_and_build(label_path)

def read_content(file, txt_path, label_path):
    '''将file文件中的内容切割成txt和label部分，分别存进相应文件
    :param file 原始文件---》 我 E\n(此格式
    :param txt_path 存放txt部分的路径
    :param label_path 存放label部分的路径'''
    with open(file, "r", encoding="utf-8") as f:
        contents = f.readlines()
        for row in contents:
            if row is "\n" or row is "E O":
                write_txt(row, txt_path)
                write_txt(row, label_path)
            else:
                row = row.replace("\n", "")
                row = row.split(" ")
                if len(row[0]) == 0 or re.search("\s", row[0]):
                    continue
                write_txt(row[0], txt_path)
                write_txt(row[1] + " ", label_path)

def run(path, files):
    '''将path读取，创建txt和label目录，并且分割源文件成txt和label部分
    :param path 原路径'''
    txt_path, label_path = get_path(path)
    file_names = normal_util.get_all_path(files)
    for file_name in tqdm(file_names):
        print("现在已处理到文件：", file_name)
        file, txt_name, label_name = get_file_name(files, file_name, txt_path, label_path)
        read_content(file, txt_name, label_name)
        print("该文件已处理完成：", file_name)



def get_file_name(files, file_name, txt_path, label_path):
    '''把label和txt的文件夹路径拼接起来
    :param files 总路径头
    :param file_name 文件名字
    :param txt_path 被分离开的txt存放文件目录
    :param label_path 被分离开的label存放文件目录
    '''
    file = os.path.join(files, file_name)
    txt_name = os.path.join(txt_path, file_name)
    label_name = os.path.join(label_path, file_name)
    return file, txt_name, label_name

def write_txt(content, path):
    '''把文本写入对应路径的文本中
    :param content 写入文本内容
    :param path 写入路径'''
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    # read_content("F:/data/test/pred_contant/0.txt")
    # x = "xxx\nyyy".replace("\n", "000")
    # result  = re.search("\n", "O\n")
    # print(result)
    # run("F:/data/test/pred_contant", "F:/data/test/pred_contant/txt_and_label")
    run("F:/data/test/new_test", "F:/data/test/other_content/txt_and_label")