import shutil
import os
import datetime
from utils import check_utils
import re

def copy_move(src_path, dest_path):
    '''
    :param src_path 源文件
    :param dest_path 需要剪切到的路径
    将文件剪切到另一个文件夹
    '''
    try:
        check_utils.check_path(dest_path)
        shutil.move(src_path, dest_path)
    except:
        pass

def copy_replace(src_path, dest_path):
    '''
    :param src_path 源文件
    :param dest_path 需要剪切到的路径
    将文件剪切到另一个文件夹
    '''
    try:
        check_utils.check_path(dest_path)
        shutil.copy(src_path, dest_path)
    except:
        pass



def get_file_name(path_name):
    '''对有后缀的文件名的后缀进行删除
    :param path_name 待处理文件名
    :return 已删除后缀名的文件名 '''
    if not check_utils.check_path_style(path_name, "path"):
        return path_name.split(".")[0]

def get_suffix(path_name):
    '''对有后缀的文件名的后缀进行获取
    :param path_name 待处理文件名
    :return 已获取文件的后缀名 '''
    return path_name.split(".")[1]

def names_filter(names,list_filts):
    '''根据条件过滤返回需要的文件名
    :param names 需要被过滤的文件名
    :param list_filt(list) 过滤条件
    :return names 过滤完成的文件名'''
    filted_name = []
    for name in names:
        for filt in list_filts:
            if re.search(filt, name):
                filted_name += [name]
    return filted_name

def concat_file_path(path, files_name):
    '''对所有的files_name进行拼接，返回该path文件夹下所有的文件（夹）的路径
    :param path 总路径
    :param files_name(list) 路径下文件（夹）名称列表
    :return 该path文件夹下所有的文件（夹）的路径'''
    files_path = []
    for file_name in files_name:
        if not check_utils.check_path_style(file_name, mode="suffix",is_flag=False):
            continue
        file_path = os.path.join(path, file_name)
        files_path += [file_path]
    return files_path

def get_all_path(path):
    return os.listdir(path)

def get_doc_file(path):
    files = []
    for file in os.listdir(path):
        if file.endswith(".doc"):  # 排除文件夹内的其它干扰文件，只获取".doc"后缀的word文件
            files.append(path + file)
    return files

