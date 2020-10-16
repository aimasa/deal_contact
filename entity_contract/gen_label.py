from entity_contract import val
from utils import normal_util, check_utils
import os
from tqdm import tqdm
def clean_txt(path):
    '''
    将文件中的空格去除干净
    :param path: 文件路径
    '''
    txts = normal_util.read_txt(path)
    list_txts = txts.split("\n")
    with open(path, "w", encoding="utf-8") as f:
        for index, list_txt in enumerate(list_txts):
            list_txt = list_txt.replace(" ", "").replace("\u3000", "")
            if len(list_txt) == 0:
                continue
            f.write(list_txt)
            if index < len(list_txts) - 1:
                f.write("\n")



def gen_label(path, write_path):
    labels = val.prediction(path)
    with open(write_path, 'a', encoding="utf-8") as f:
        for index, label in enumerate(labels):
            f.write(" ".join(str(i) for i in label))
            if index < len(labels) - 1:
                f.write("\n")



def run(head_path, write_path):
    paths = normal_util.concat_path(head_path)
    for path in tqdm(paths):
        clean_txt(path)
        file_name = normal_util.gain_filename_from_path(path, 'txt')
        corr_write_path = os.path.join(check_utils.check_and_build(write_path), file_name)
        gen_label(path, corr_write_path)



if __name__ == "__main__":
    # run("F:/contract/txt", "F:/contract/label")
    gen_label("F:/contract/txt/17.txt", "F:/contract/label/17.txt")