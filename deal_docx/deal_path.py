import os
from utils import check_utils


class deal_path():
    def __init__(self):
        self.all_without_repeat_paths = []
        self.all_repeated_paths = []
        self.all_paths = []

    def get_paths(self, path, is_repeat = False):
        '''读取文件夹下的所有路径
        :param '''
        names = os.listdir(path)
        if is_repeat:
            for name in names:
                self.all_paths = os.path.join(path, name)
            return
        for paths_name_without_repeat,paths_name_repeat  in check_utils.check_path_name(names):
            self.all_without_repeat_paths += os.path.join(path, paths_name_without_repeat)
            self.all_repeated_paths += os.path.join(path, paths_name_repeat)

    def split_repeat_path(self, src, dest):
        self.get_paths(src)




