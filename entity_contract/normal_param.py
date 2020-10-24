START_TAG = "[CLS]"
STOP_TAG = "[SEP]"
tag_dic = {
    "person": "PERSON",
    "house": "HOUSE",
    "location": "LOCATION",
    "area": "AREA",
    "time": "TIME",
    "term": "TERM",
    "rent": "RENT",
    "money": "MONEY",
    "rule": "RULE",
    "invoice": "INVOICE",
    "used": "USED",
    "paperwork": "PAPERWORK",
    "org": "ORG",
    "contract": "CONTRACT",
    "duty": "DUTY",
    "structure": "STRUCTURE",
    "name": "NAME",
    "license_number": "LICENSE_NUMBER"
}
label_to_tag = {
    "PERSON" : "person",
    "HOUSE": "house",
     "LOCATION": "location",
     "AREA": "area",
    "TIME": "time",
    "TERM": "term",
     "RENT": "rent",
    "MONEY":"money",
    "RULE": "rule",
    "INVOICE": "invoice",
    "USED": "used",
    "PAPERWORK": "paperwork",
    "ORG": "org",
    "CONTRACT": "contract",
     "DUTY":"duty",
    "STRUCTURE": "structure",
    "NAME": "name",
    "LICENSE_NUMBER": "license_number",
    "STARTTERM" : "startterm",
    "ENDTERM" : "endterm",
    "TYPE" : "type",
     "DEADLINE" : "deadline"
}
# tag_dic = {
#     "org" : "ORG",
#     "location" : "LOCATION",
#     "time" : "TIME",
#     "money" : "MONEY",
#     "person" : "PERSON",
#     "number" : "NUMBER",
#     "object" : "OBJECT"
# }


# label_to_tag = {
#      "ORG":"org" ,
#      "LOCATION":"location" ,
#      "TIME":"time",
#     "MONEY":"money",
#     "PERSON":"person",
#     "NUMBER":"number",
#     "OBJECT":"object"
# }
dic_path = ""

labels = ["PERSON","HOUSE","LOCATION", "AREA", "TIME", "TERM", "RENT", "MONEY", "RULE", "INVOICE","USED","PAPERWORK","ORG","CONTRACT","DUTY","STRUCTURE","NAME","LICENSE_NUMBER" ,"STARTTERM", "ENDTERM","TYPE" ,"DEADLINE"]

# labels=["ORG","LOCATION","TIME","MONEY","PERSON","NUMBER","OBJECT"]
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
head_path = "F:/data/test/pred_contant"
# head_path = "F:/data/test/test"
head_test_path = "F:/data/test/test"
test_path = "F:/contract/txt/30.txt"
result_path = "result"
EPOCH = 14
save_path_bilstm = 'checkpoints/bilstm_crf.h5py'
save_path_bert_bilstm = 'checkpoints/bert_bilstm_crf.h5py'
save_path_wordVEC_bilstm = 'checkpoints/wordvec_bilstm_crf.h5py'
save_path_lstm = 'checkpoints/lstm_crf.h5py'
save_path_keras = 'checkpoints/lstm_crf_keras.pth'
config_path = 'F:\\phython workspace\\deal_contact\\bert\\bert_config.json'
checkpoint_path = 'F:\\phython workspace\\deal_contact\\bert\\bert_model.ckpt'
dict_path = 'F:\\phython workspace\\deal_contact\\bert\\vocab.txt'

# label_file = './data/tag.txt'
# train_file = './data/train.txt'
# dev_file = './data/dev.txt'
# test_file = './data/test.txt'
max_length = 248
vocab = 'bert/vocab.txt'
lstm_vocab = 'vocab.pkl'
use_cuda = True
gpu = 0
batch_size = 4
bert_path = 'bert'
rnn_hidden = 500
bert_embedding = 768
dropout1 = 0.5
dropout_ratio = 0.5
rnn_layer = 1
lr = 0.0001
lr_decay = 0.00001
weight_decay = 0.00005
checkpoint = 'result/'
optim = 'Adam'
load_model = False
load_path = None
base_epoch = 100
n_part = 4