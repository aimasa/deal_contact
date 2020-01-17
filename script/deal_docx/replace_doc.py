import os
from utils import normal_util
def get_same_file_name(src, dest):
    '''获取原文件夹中与目标文件夹中文件名相同的文件'''
    src_path_name_without_sub = []
    replaced_file = []
    for dest_path_name in os.listdir(dest):
        if normal_util.get_file_name(dest_path_name) in src_path_name_without_sub:
            replaced_file += dest_path_name
    return replaced_file

