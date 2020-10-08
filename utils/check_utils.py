import re
import os
from utils import normal_util
import difflib

def check_path_style(name, mode="none", is_flag=True):
    '''检查path路径下的文件名称
    :param name(str) 文件名,如path 检测是否有后缀   suffix 检测是否是doc
    :param mode(str) 需要检查的种类 path or suffix default "none"
    :param is_flag 是否只做判断返回 True or False default:True
    :return true/false  只做判断返回 path 有后缀则false suffix 是文件docx则false
    :return true/false  不只做判断返回 path 有后缀则false，并且返回后缀位置。 suffix 是文件docx则false，并且返回后缀位置'''
    if mode == 'none':
        return True
    if mode == 'path' and re.search("\.", name):
        if is_flag:
            return False
        return re.search("\.", name), False
    elif mode == 'suffix' and re.search(".docx", name):
        if is_flag:
            return False
        return False, re.search(".docx", name)
    return True


def check_path_name(paths_name, mode="docx"):
    '''检查文件名字，剃去重名文件
    :param paths_name(list) 路径中的文件名列表
    :param mode 需要去掉重复的哪种后缀的文件 default docx
    :return paths_name_without_repeat(list) 不重复的文件名列表
    :return paths_name_repeat(list) 重复的文件名'''
    paths_name_without_sub = []
    paths_name_without_repeat = []
    paths_name_repeated = []
    get_replect_file_names = []
    for path_name in paths_name:
        path_name_without_sub = normal_util.get_file_name(path_name)
        if not path_name_without_sub:
            continue
        if path_name_without_sub not in paths_name_without_sub:
            paths_name_without_sub += [path_name_without_sub]
            paths_name_without_repeat += [path_name]
        else:

            paths_name_repeated += [path_name]
    # check_file_style(paths_name_without_repeat, paths_name_repeated, mode)


    return paths_name_without_repeat, paths_name_repeated

def check_file_style(srcs, dests, mode):
    '''对dest中和src中文件名相同的文件，检查其后缀是否是自己想要的后缀名，整理并返回
    :param srcs(list) 源文件名
    :param dests (list) 需转移文件名
    :param mode (str) 后缀名'''
    for dest in dests:
        if check_path_style(dest, "suffix"):
            a = difflib.get_close_matches(dest, srcs, 1, cutoff=0.7)
            print(a)


def check_useless_seq(content):
    if re.match(".*律师365|.*第一范文网|.*相关推荐|.*文章来源", content):
        return True
    return False






def check_path(path):
    '''
    :param path 被检查是否存在的路径
    :return 是否存在的路径
    检查路径是否存在
    '''
    return os.path.exists(path)

def check_and_build(path):
    if not check_path(path):
        os.makedirs(path)
    return path

if __name__ == "__main__":
    src = ["hahaah.doc"]