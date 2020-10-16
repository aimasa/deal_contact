from model import keras_BILSTM_CEF
from entity_contract import NER_pre_data, normal_param
from utils import normal_util, data_change
import numpy as np

list_label = ["LOC", "ORG", "PER"]
label_to_ix, ix_to_label = NER_pre_data.build_label(list_label)
vocab = normal_util.read_vocab("vocab.pkl")
model = keras_BILSTM_CEF.load_embedding_bilstm2_crf_model(normal_param.save_path, len(vocab), len(label_to_ix), 0)
predict_text = '张东升是中国人'
maxlength = 128
x, length = data_change.auto_single_test_pad(predict_text, vocab, maxlength)
# model.load_weights('model/crf.h5')
raw = model.predict(x)[0][-length:]
result = [np.argmax(row) for row in raw]
result_tags = [ix_to_label[i] for i in result]

per, loc, org = '', '', ''

for s, t in zip(predict_text, result_tags):
    if t in ('B-PER', 'I-PER'):
        per += ' ' + s if (t == 'B-PER') else s
    if t in ('B-ORG', 'I-ORG'):
        org += ' ' + s if (t == 'B-ORG') else s
    if t in ('B-LOC', 'I-LOC'):
        loc += ' ' + s if (t == 'B-LOC') else s

print(['person:' + per, 'location:' + loc, 'organzation:' + org])