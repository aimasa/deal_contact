import docx
from utils import check_utils
from utils import normal_util
import os
import tqdm
import progressbar

def docx_content(docx_path:str) -> list:
    '''
    读取docx文本内容
    :param docx_path 文本存放路径
    :return contents 从docx中读取出来的文本内容'''
    file_content = docx.Document(docx_path)
    contents = []
    for para in file_content.paragraphs:
        text = para.text
        if check_utils.check_useless_seq(text):
            break
        contents.append(para.text)
    return contents

def write_in_txt(contents:list, path:str):
    '''
    将文件内容读取存进对应的txt文件中
    :param contents（list） 需要被存入txt的文件内容
    :param path(str) txt存储路径
    '''
    with open(path, "w", encoding='utf-8') as f:
        for content in contents:
            f.write(content)
            f.write('\n')

def get_path(root_path:str) -> list:
    '''获取docx文件列表
    :param root_path docx文件存放总路径
    :return docx文件路径列表'''

    paths = normal_util.concat_file_path(root_path, os.listdir(root_path))
    return paths

def run(root_path, head_path, error_path):
    paths = get_path(root_path)
    widgets = [ 'txt load :',' [', progressbar.Timer(), '] ',progressbar.Bar('#'), ' (', progressbar.ETA(), ') ']
    N = paths.__len__()
    bar = progressbar.ProgressBar(widgets=widgets, maxval=N)
    bar = bar.start()


    for index, path in enumerate(paths):

        try:
            contents = docx_content(path)
            txt_path = os.path.join(head_path, str(index) + '.txt')
            write_in_txt(contents, txt_path)
        except:
            normal_util.copy_move(path, check_utils.check_path(error_path))
        bar.update(index + 1)
    bar.finish()


if __name__ == '__main__':
    root_path = "F:/rent_house_contract_pos/tst/rent_house_contract_pos"
    head_path = "F:/contract/dev"
    error_path = "F:/error_contract"
    run(root_path, head_path, error_path)