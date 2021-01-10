from utils import normal_util
import os
def copy(path, path_name):
    content = normal_util.read_txt(path)
    normal_util.write_content(content, path_name)



def gain_path(head_path, target_path, copy_num):
    ann_path, txt_path, _ = normal_util.get_file_path(head_path)
    for i in range(copy_num):
        index = i % 2
        copy(ann_path[index], os.path.join(target_path, ("%s.ann"%(i))))
        copy(txt_path[index], os.path.join(target_path, ("%s.txt"%(i))))


if __name__ == "__main__":
    gain_path("F:/data/test/test/ann", "F:/data/test/test/train", 1000)

