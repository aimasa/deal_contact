from entity_contract import process_data_for_keras, NER_pre_data
from entity_contract import normal_param
from model import keras_LSTM_CRF, keras_Bert_bilstm_crf, keras_BILSTM_CEF
from sklearn.metrics import classification_report
import numpy as np

def evl(mode):
    x_train, y_train = process_data_for_keras.process_data(
                embeding=mode, is_train=False)
    labels_to_ix, ix_to_label = NER_pre_data.build_label(normal_param.labels)
    vocab = process_data_for_keras.read_vocab(normal_param.lstm_vocab)
    if mode == "lstm":
        save_path = normal_param.save_path_lstm
        model = keras_LSTM_CRF.load_embedding_bilstm2_crf_model(save_path, len(vocab), len(labels_to_ix),
                                                                normal_param.max_length)
    elif mode == "bilstm":
        save_path = normal_param.save_path_bilstm
        model = keras_BILSTM_CEF.load_embedding_bilstm2_crf_model(save_path, len(vocab), len(labels_to_ix),
                                                                  normal_param.max_length)
    elif mode == "bert_bilstm":
        save_path = normal_param.save_path_bert_bilstm
        model = keras_Bert_bilstm_crf.load_embedding_bilstm2_crf_model(save_path, len(labels_to_ix))
    else:
        save_path = normal_param.save_path_wordVEC_bilstm
        model = keras_BILSTM_CEF.load_embedding_bilstm2_crf_model(save_path, len(vocab), len(labels_to_ix),
                                                                  normal_param.max_length)
    pre_label = model.predict(x_train)
    return pre_label, y_train

def pre_class(pre_label, y_train):

    # Y_test = np.array(Y_test).reshape(len(Y_test), -1)
    # enc = OneHotEncoder()
    # enc.fit(Y_test)
    # targets = enc.transform(Y_test).toarray()
    print(classification_report(pre_label, y_train))

def run(mode):
    pre_label, y_train = evl(mode)
    pre_class(pre_label, y_train)

if __name__ == "__main__":
    run("bilstm")
