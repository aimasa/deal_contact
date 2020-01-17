import os
from utils import check_utils
from utils import normal_util

class deal_path(object):
    def __init__(self, path):
        self.all_without_repeat_paths = []
        self.all_repeated_paths = []
        self.all_paths = []
        self.path = path

    def get_paths(self, is_repeat = False):
        '''读取文件夹下的所有路径
        :param path 需要被读取路径'''
        names = os.listdir(self.path)
        if is_repeat:
            for name in names:
                self.all_paths = os.path.join(self.path, name)
            return
        try:
            paths_name_without_repeat,paths_name_repeat  = check_utils.check_path_name(names)
            self.all_without_repeat_paths = self.concat_path(paths_name_without_repeat)
            self.all_repeated_paths = self.concat_path( paths_name_repeat)
        except Exception as e:
            print("error：",e)

    def concat_path(self,paths_name):
        return [os.path.join(self.path, path) for path in paths_name]

    def split_repeat_path(self, src, dest):
        self.get_paths()
        for all_repeated_path in self.all_repeated_paths:
            normal_util.copy_move(all_repeated_path, dest)

if __name__=="__main__":
    path = "F:/rent_house_contract_pos/rent_house_contract_pos"
    deal_path = deal_path(path)
    deal_path.split_repeat_path(path, "F:/repeat_contract")




