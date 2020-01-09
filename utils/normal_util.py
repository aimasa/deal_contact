import shutil
import os
import datetime
from utils import check_utils


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


def get_file_name(path_name):
    '''对有后缀的文件名的后缀进行删除
    :param path_name 待处理文件名
    :return 已删除后缀名的文件名 '''
    if not check_utils.check_path_style(path_name, "path"):
        return path_name.split(".")[0]
