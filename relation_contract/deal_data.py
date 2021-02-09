from build_data import build_data
import utils
import numpy as np
timestamp = "date" + "%d.%m.%Y_%H.%M.%S"
output_dir = './logs/'
config_file = './configs/CoNLL04/bio_config'

def deal_data(mode):
    config = build_data(config_file)
    if mode == "train":
        data = utils.HeadData(config.train_id_docs, np.arange(len(config.train_id_docs)))
    elif mode == "dev":
        data = utils.HeadData(config.dev_id_docs, np.arange(len(config.dev_id_docs)))
    else:
        data = utils.HeadData(config.test_id_docs, np.arange(len(config.test_id_docs)))
    relation_list = config.dataset_set_relations
    return data, relation_list

def process_single_data_to_example(data):
    '''
    将data中的文本转换成example的格式
    :param data:
    :return: examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None,
                             locations=locations, labels=labels, num_relations=num_relations))
    '''
    relation_token_ids = data.heads
    locations = []
    labels = []
    for index_e1, entity_pair_ids in enumerate(relation_token_ids):
        if len(entity_pair_ids) <= 1 and entity_pair_ids[0] == index_e1:
            continue
        entity_e1 = find_loc(index_e1, data.ecs)
        for index_e2, entity_pair_id in enumerate(entity_pair_ids):
            entity_e2 = find_loc(entity_pair_id, data.ecs)
            locations.append((entity_e1, entity_e2))
            labels.append(data.relations[index_e1][index_e2])
    num_relations = len(locations)
    text_a = " ".join(data.tokens)
    return text_a, locations, labels, num_relations

def process_single_data_to_example_all_label(data):
    '''
    将data中的文本转换成example的格式
    :param data:
    :return: examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None,
                             locations=locations, labels=labels, num_relations=num_relations))
    '''
    locations = []
    labels = []
    # for index_e1, entity_pair_ids in enumerate(relation_token_ids):
    #     if len(entity_pair_ids) <= 1 and entity_pair_ids[0] == index_e1:
    #         continue
    #     entity_e1 = find_loc(index_e1, data.ecs)
    #     for index_e2, entity_pair_id in enumerate(entity_pair_ids):
    #         entity_e2 = find_loc(entity_pair_id, data.ecs)
    #         locations.append((entity_e1, entity_e2))
    #         labels.append(data.relations[index_e1][index_e2])
    entity_pairs = combination_entity(find_entity(data))
    for (first_entity, last_entity) in entity_pairs:
        first_entity_last_loc = first_entity[1] - 1
        next_entity_locs = data.heads[first_entity_last_loc]
        relations = []
        for id, next_entity_loc in enumerate(next_entity_locs):
            if next_entity_loc == last_entity[1] - 1:
                relations.append(data.relations[first_entity_last_loc][id])
        if len(relations) == 0:
            relations.append("N") # 如果实体对没有对应的关系，那就把下标为0的label：N 设置为下标
        labels += relations
    num_relations = len(labels)

    text_a = " ".join(data.tokens)
    return text_a, entity_pairs, labels, num_relations

def find_entity(data):
    '''
    找到句子中所有实体
    :param ec_ids:
    :return:
    '''
    entity_locations = []
    first_loc = 0
    while first_loc < len(data.ec_ids):
        num = data.ec_ids[first_loc]
        tmp_loc = first_loc
        while tmp_loc < len(data.ec_ids) and data.ec_ids[tmp_loc] == num:
            tmp_loc += 1
        last_loc = tmp_loc - 1 # 计算当前相同的标签的first_loc 和last_loc

        if num == 1: # 表示当前实体为“O”
            first_loc = tmp_loc
            continue
        entity_locations.append(((first_loc + 1),(last_loc + 1)))
        first_loc = tmp_loc
    return entity_locations



def combination_entity(entity_locations):
    '''
    将实体组合
    :return: 实体组合的所有可能对
    '''
    entity_pairs = []
    for first_entity_loc in entity_locations:
        for last_entity_loc in entity_locations:
            if last_entity_loc == first_entity_loc:
                continue
            entity_pairs.append((first_entity_loc, last_entity_loc))
    return entity_pairs


def find_loc(last_loc, ecs):
    '''
    找实体对位置
    :param first_loc: 实体第一个位置
    :param ecs: 句子中对应的标签
    :return:
    '''
    str = ecs[last_loc]
    first_loc = last_loc
    i = first_loc
    while i >= 0 and ecs[i] == str:
        first_loc = i
        i -= 1
    return (first_loc + 1, last_loc + 1)

if __name__ == "__main__":
    find_loc(2, [1,1,2,2])