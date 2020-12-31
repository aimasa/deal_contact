from utils import normal_util

from collections import defaultdict

'''
entity_T :{
                no: 
                {
                   name:
                   local:()
                   relation_info : {
                        entity_no: relation_name
                    }
                 }
           }
'''

def gain_relation_contact_entity(entity_T, path):
    '''
    通过ann文件获取关系和实体的联系
    :param path: ann路径
    :return:
    '''
    ann_contents = normal_util.read_txt(path)
    # entity_T = defaultdict()
    for ann_content in ann_contents.split("\n"):
        if len(ann_content) <= 1:
            continue
        if not filter_relation(ann_content.split("\t")[0]):
            gain_entity(entity_T, ann_content)
        else:
            gain_relation(entity_T, ann_content)



def gain_entity(entity_T, ann_content):
    '''
    通过实体标签获取实体Tn对应的信息
    :param entity_T: 实体Tn对应的信息
    :param ann_content: 实体标签条例
    :return:
    '''
    entity_no, entity_name, entity_start_local, entity_end_local = gain_entity_info(ann_content)
    filled_entity_info(entity_T, entity_no, entity_name, entity_start_local, entity_end_local)


def filled_entity_info(entity_T, entity_no, entity_name, entity_start_local, entity_end_local):
    '''
    填充实体dic中的实体信息
    :param entity_T: 实体dic
    :param entity_no: 实体对应的Tn
    :param entity_name: 实体名字
    :param entity_start_local: 实体开始位置
    :param entity_end_local: 实体结束位置
    :return:
    '''
    if entity_no not in entity_T:
        entity_T[entity_no] = {}
    entity_dic = entity_T[entity_no]
    entity_dic["name"] = entity_name
    entity_dic["local"] = (entity_start_local, entity_end_local)
    entity_T[entity_no] = entity_dic

def gain_entity_info(str):
    '''

    通过ann_content获得关系对应实体代号Tn 及实体名称等数据
    :param ann_content: 读取ann文件中的关系标签获取的文件信息
    '''
    entity_info = str.split("\t")
    entity_no = entity_info[0]
    entity_name_and_local = entity_info[1].split(" ")
    entity_name = entity_name_and_local[0]
    entity_start_local = entity_name_and_local[1]
    entity_end_local = entity_name_and_local[2]
    return entity_no, entity_name, entity_start_local, entity_end_local


def gain_relation(entity_T, ann_content):
    '''
    通过关系标签获取对应关系对应的信息， 并将实体dic中涉及到的关系信息补全
    :param entity_T: 实体dic
    :param ann_content: 关系标签对应的关系
    :return:
    '''
    relation_name, relation_Arg1, relation_Arg2 = gain_relation_info(ann_content)
    filled_relation_dic(entity_T, relation_name, relation_Arg1, relation_Arg2)

def filled_relation_dic(entity_T, relation_name, relation_Arg1, relation_Arg2):
    '''
    通过关系标签填充entity中Tn对应的信息
    :param entity_T: entity中Tn对应的信息
    :param relation_name: 关系名称
    :param relation_Arg1: 关系对应的Tn1
    :param relation_Arg2: 关系对应的Tn2

    '''
    if relation_Arg1 not in entity_T:
        entity_T[relation_Arg1] = {}
    relation_info = entity_T[relation_Arg1]
    if "relation_info" not in relation_info:
        relation_info["relation_info"] = {}
    relation_list = relation_info["relation_info"]
    relation_list[relation_Arg2] = relation_name
    relation_info["relation_info"] = relation_list
    entity_T[relation_Arg1] = relation_info





def gain_relation_info(ann_content):
    '''
    通过ann_content获得关系对应实体代号Tn 及关系名称
    :param ann_content: 读取ann文件中的关系标签获取的文件信息
    :return:
    '''
    relation_info = ann_content.split("\t")[1].split(" ")
    relation_name = relation_info[0]
    relation_Arg1 = relation_info[1].split(":")[-1]
    relation_Arg2 = relation_info[2].split(":")[-1]
    return relation_name, relation_Arg1, relation_Arg2




def filter_relation(str):
    '''
    过滤出关系标签
    :param str: 需要被过滤语句
    :return:是否是关系标签
    '''
    if "R" in str:
        return True
    return False


# if __name__ == "__main__":
#     gain_relation_contact_entity("F:/data/test/test/ann/1.ann")