from model import keras_BILSTM_CEF, keras_Bert_bilstm_crf, keras_LSTM_CRF, keras_word2vec_bilstm_crf, keras_RNN_CRF
from entity_contract import process_data_for_keras
from entity_contract import normal_param
from utils import check_utils
import os
from entity_contract import NER_pre_data


def run(model_name):
    '''
    根据模型的不同选择使用的模型代码
    :param model: 模型类型 bilstm_crf    bert_bilstm_crf   lstm_crf     wordvec_bilstm_crf
    :return:
    '''
    if model_name == 'bilstm_crf':
        x_train, y_train, x_test, y_test, vocab_length, labels_to_ix_length = process_data_for_keras.process_data(embeding= "bilstm")
        model = keras_BILSTM_CEF.build_embedding_bilstm2_crf_model(vocab_length, labels_to_ix_length, 0)
        save_path = normal_param.save_path_bilstm
    elif model_name == "rnn_crf":
        x_train, y_train, x_test, y_test, vocab_length, labels_to_ix_length = process_data_for_keras.process_data(embeding= "rnn")
        model = keras_RNN_CRF.build_embedding_lstm2_crf_model(vocab_length, labels_to_ix_length, 0)
        save_path = normal_param.save_path_gru
    elif model_name == 'bert_bilstm_crf':
        x_train, y_train, x_test, y_test, vocab_length, labels_to_ix_length = process_data_for_keras.process_data(
            embeding="bert")
        model = keras_Bert_bilstm_crf.build_bilstm_crf_model(labels_to_ix_length)
        save_path = normal_param.save_path_bert_bilstm
    elif model_name == 'lstm_crf':
        x_train, y_train, x_test, y_test, vocab_length, labels_to_ix_length = process_data_for_keras.process_data(embeding="lstm")
        model = keras_LSTM_CRF.build_embedding_lstm2_crf_model(vocab_length, labels_to_ix_length, 0)
        save_path = normal_param.save_path_lstm
    else:
        embeddings_matrix, vocab = process_data_for_keras.txtpad_use_word2vec()
        x_train, y_train, x_test, y_test, vocab_length, labels_to_ix_length = process_data_for_keras.process_data(embeding="wordvec", vocab2=vocab)

        model = keras_word2vec_bilstm_crf.build_embedding_bilstm2_crf_model(labels_to_ix_length, embeddings_matrix, normal_param.max_length)
        save_path = normal_param.save_path_wordVEC_bilstm

    if check_utils.check_path(save_path):
        model.load_weights(save_path)
    model.fit(x_train, y_train, batch_size=18, epochs=20, validation_data = (x_test, y_test), shuffle = False, validation_split=0.2, verbose=1)
    keras_BILSTM_CEF.save_embedding_bilstm2_crf_model(model, save_path)



#
# def run_by_gen():
#     vocab = process_data_for_keras.read_vocab(normal_param.lstm_vocab)
#     model = keras_Bert_bilstm_crf.build_bilstm_crf_model(len(vocab))
#     # txt_path = os.path.join(normal_param.head_path, "txt")
#     # label_path = os.path.join(normal_param.head_path, "label")
#     label_path, txt_path = NER_pre_data.concat_path(normal_param.head_path)
#     data, label = process_data_for_keras.read_data_part(txt_path, label_path)
#     x_test, y_test = process_data_for_keras.process_test_data()
#     model.fit_generator(process_data_for_keras.data_generator(data, label, normal_param.n_part, normal_param.batch_size), steps_per_epoch= len(data) // normal_param.batch_size + (len(data) % normal_param.batch_size), epochs=50, validation_data = (x_test, y_test), shuffle = False, validation_steps=0.2, verbose=1)
#     keras_BILSTM_CEF.save_embedding_bilstm2_crf_model(model, normal_param.save_path_bert_bilstm)


if __name__ == '__main__':
    # run_by_gen()
    run(model_name = "bert_bilstm_crf")