from entity_contract import NER_pre_data
import pickle
import os
vocab = {}

def read_txt(contents):
    '''
    读取txt文件
    :param contents: 文本内容
    '''
    # result = []
    for content in contents:
        if content is "\n" or content is " ":
            continue
        create_vocab(content)
        # result.append(content)
    # return result

def create_vocab(word):
    '''
    创建词—下标 词典
    :param word: 需要生成对应下标的word
    '''
    if word not in vocab:
        vocab[word] = len(vocab)

def get_path(head_path):
    names = os.listdir(head_path)
    return [os.path.join(head_path, name) for name in names]
def read_content_and_create_vocab(head_path):
    '''根据路径获得需要读取的文件位置，并读取文件，生成词-下标 词典
    :param head_path 文件root地址'''
    paths = get_path(head_path)
    for index, path in enumerate(paths):
        print("now is " , index , "and the sum is ", len(paths))
        read_content(path)


def read_content(file):
    '''对file的内容进行读取，建立单词列表'''
    with open(file, "r", encoding="utf-8") as f:
        contents = f.read()
        # content = split(content)
        return read_txt(contents)


def run():
    '''运行方法，创建词_下标词典'''
    read_content_and_create_vocab("F:/data/test/pred_contant/txt")
    with open("vocab.pkl", 'wb') as f:
        pickle.dump(vocab, f)

if __name__=='__main__':
    run()