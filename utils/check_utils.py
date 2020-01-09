import re
import os
from utils import normal_util


def check_path_style(name, mode, is_flag=True):
    '''检查path路径下的文件名称
    :param name(str) 文件名,如path 检测是否有后缀   suffix 检测是否是doc
    :param mode(str) 需要检查的种类 path or suffix
    :param is_flag 是否只做判断返回 True or False default:True
    :return true/false  只做判断返回 path 有后缀则false suffix 是文件docx则false
    :return true/false  不只做判断返回 path 有后缀则false，并且返回后缀位置。 suffix 是文件docx则false，并且返回后缀位置'''
    if mode == 'path' and re.search("\.", name):
        if is_flag:
            return False
        return re.search("\.", name), False
    elif mode == 'suffix' and re.search(".docx", name):
        if is_flag:
            return False
        return False, re.search(".docx", name)
    return True


def check_path_name(paths_name):
    '''检查文件名字，剃去重名文件
    :param paths_name(list) 路径中的文件名列表
    :return paths_name_without_repeat(list) 不重复的文件名列表
    :return paths_name_repeat(list) 重复的文件名'''
    paths_name_without_sub = []
    paths_name_without_repeat = []
    paths_name_repeated = []
    for path_name in paths_name:
        path_name_without_sub = normal_util.get_file_name(path_name)
        if path_name_without_sub:
            continue
        if path_name_without_sub not in paths_name_without_sub:
            paths_name_without_sub += [path_name_without_sub]
            paths_name_without_repeat += [path_name]
        else:
            paths_name_repeated += [path_name]
    return paths_name_without_repeat, paths_name_repeated




def check_path(path):
    '''
    :param path 被检查是否存在的路径
    :return 已经存在的路径（不存在则创建）
    检查路径是否存在，不存在则创建
    '''
    if os.path.exists(path):
        return path
    else:
        os.makedirs(path)
        return path