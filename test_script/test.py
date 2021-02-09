# from model import keras_BILSTM_CEF
# from entity_contract import NER_pre_data, normal_param
# from utils import normal_util, data_change
# import numpy as np
#
# list_label = ["LOC", "ORG", "PER"]
# label_to_ix, ix_to_label = NER_pre_data.build_label(list_label)
# vocab = normal_util.read_vocab("vocab.pkl")
# model = keras_BILSTM_CEF.load_embedding_bilstm2_crf_model(normal_param.save_path, len(vocab), len(label_to_ix), 0)
# predict_text = '张东升是中国人'
# maxlength = 128
# x, length = data_change.auto_single_test_pad(predict_text, vocab, maxlength)
# # model.load_weights('model/crf.h5')
# raw = model.predict(x)[0][-length:]
# result = [np.argmax(row) for row in raw]
# result_tags = [ix_to_label[i] for i in result]
#
# per, loc, org = '', '', ''
#
# for s, t in zip(predict_text, result_tags):
#     if t in ('B-PER', 'I-PER'):
#         per += ' ' + s if (t == 'B-PER') else s
#     if t in ('B-ORG', 'I-ORG'):
#         org += ' ' + s if (t == 'B-ORG') else s
#     if t in ('B-LOC', 'I-LOC'):
#         loc += ' ' + s if (t == 'B-LOC') else s
#
# print(['person:' + per, 'location:' + loc, 'organzation:' + org])
from relation_contract import tokenization
import os
# label_list = []
# filein = open(os.path.join("F:/data/test/test/relation_label", "label.txt"))
# for line in filein:
#     label = line.strip()
#     label_list.append(tokenization.convert_to_unicode(label))
# print(label_list)

from textda.data_expansion import *
print(data_expansion('(1)租赁期间，房屋和土地的产权税由甲方依法交纳。如果发生政府有关部门征收本合同中未列出项目但与该房屋有关的费用，应由甲方负担。', alpha_ri=0.3, alpha_rs=0))