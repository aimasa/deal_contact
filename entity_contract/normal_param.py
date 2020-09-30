START_TAG = "[CLS]"
STOP_TAG = "[SEP]"
tag_dic = {
   "location" : "LOCATION",
    "area" : "AREA",
    "rent" : "RENT",
    "type" : "TYPE",
    "startTerm" : "STARTTERM",
    "endTerm" : "ENDTERM",
    "deadline" : "DEADLINE"
}
dic_path = ""

labels = {"LOCATION","AREA","RENT","STARTTERM","ENDTERM", "TYPE", "DEADLINE"}

EMBEDDING_DIM = 5
HIDDEN_DIM = 4
label_file = './data/tag.txt'
train_file = './data/train.txt'
dev_file = './data/dev.txt'
test_file = './data/test.txt'
vocab = 'bert/vocab.txt'
max_length = 300
use_cuda = True
gpu = 0
batch_size = 50
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