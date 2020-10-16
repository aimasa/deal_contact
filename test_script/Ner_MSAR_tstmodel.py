from utils import normal_util, data_change
from entity_contract import NER_pre_data
from model import keras_BILSTM_CEF
from entity_contract import normal_param
import numpy as np
import pickle
from keras.callbacks import ReduceLROnPlateau
list_label = ["LOC", "ORG", "PER"]
def read_content(txts):
    labels = []
    contents = []
    label = []
    content = []
    length = 0

    for txt in txts:

        if txt == "":
            if len(label) == 0:
                continue
            labels.append(label)
            contents.append(content)
            label = []
            content = []
        else:
            splited_txt = txt.split("\t")
            if splited_txt[0] == "" or splited_txt[-1] == "":
                length = max(len(content), length)
                if len(label) == 0:
                    continue
                labels.append(label)
                contents.append(content)
                label = []
                content = []
                continue
            label.append(splited_txt[-1])
            content.append(splited_txt[0])
    return labels, contents, length

def read_txt(path):
    txts = normal_util.read_txt(path)
    list_txts = txts.split("\n")
    labels, contents, length = read_content(list_txts)
    return labels, contents, length


def change_to_array(txts, label, vocab, label_to_ix, length):
    idxs, _ = data_change.auto_pad(txts, vocab, length)
    labels = data_change.auto_pad(label, label_to_ix, length, is_label=True)
    return idxs, labels




def change_tst_to_array(txts, label, vocab, label_to_ix, length):
    idxs = data_change.prepare_test_sequence(txts, vocab, length)
    labels = data_change.prepare_label(label, label_to_ix, length)
    return idxs, labels

def gain_tst_and_trn(tst_path, trn_path):
    label_to_ix, ix_to_label = NER_pre_data.build_label(list_label)
    trn_labels, trn_contents, trn_length = read_txt(trn_path)
    tst_labels, tst_contents, tst_length = read_txt(tst_path)
    maxlength = max(trn_length, tst_length)


    vocab = {}
    read_vocab(trn_contents, vocab)
    read_vocab(tst_contents, vocab)
    with open("vocab.pkl", 'wb') as f:
        pickle.dump(vocab, f)
    trn_contents, trn_labels = normal_util.shuffle(trn_contents, trn_labels)
    trn_content_idx, trn_label_idx = change_to_array(trn_contents, trn_labels, vocab, label_to_ix, maxlength)
    tst_content_idx, tst_label_idx = change_to_array(tst_contents, tst_labels, vocab, label_to_ix, maxlength)
    return trn_content_idx, trn_label_idx, tst_content_idx, tst_label_idx, vocab, label_to_ix, maxlength




def run(tst_path, trn_path):
    x_train, y_train, x_test, y_test, vocab, labels_to_ix, length = gain_tst_and_trn(tst_path, trn_path)


    model = keras_BILSTM_CEF.build_embedding_bilstm2_crf_model(len(vocab), len(labels_to_ix), length)
    # x_train, y_train = read_data(normal_param.head_path, vocab, labels_to_ix)
    # x_test, y_test = read_data(normal_param.head_test_path, vocab, labels_to_ix)
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    # y_train = np.expand_dims(y_train, 2)
    y_test = np.expand_dims(y_test, 2)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    model.fit(x_train, y_train, batch_size=18, epochs=4, validation_data = (x_test, y_test), shuffle = False, validation_split=0.2, verbose=1)
    keras_BILSTM_CEF.save_embedding_bilstm2_crf_model(model, normal_param.save_path)


def read_vocab(contents, vocab):
    for content in contents:
        for single_char in content:
            if single_char in vocab.keys():
                continue
            vocab[single_char] = len(vocab)



if __name__ == "__main__":
    trn_path = "F:/data/SMART/msra_train_bio.txt"
    tst_path = "F:/data/SMART/msra_test_bio.txt"
    run(tst_path, trn_path)