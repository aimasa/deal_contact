from entity_contract import NER_pre_data, process_data_for_keras, NERInference, normal_param
from model import keras_BILSTM_CEF
from model import keras_BILSTM_CEF, keras_Bert_bilstm_crf, keras_LSTM_CRF, keras_word2vec_bilstm_crf


def prediction(path, mode = "bilstm", is_eval = False):
    labels_to_ix, ix_to_label = NER_pre_data.build_label(normal_param.labels)
    vocab = process_data_for_keras.read_vocab(normal_param.lstm_vocab)
    if mode == "lstm":
        save_path = normal_param.save_path_lstm
        model = keras_LSTM_CRF.load_embedding_bilstm2_crf_model(save_path, len(vocab), len(labels_to_ix), normal_param.max_length)
    elif mode == "bilstm":
        save_path = normal_param.save_path_bilstm
        model = keras_BILSTM_CEF.load_embedding_bilstm2_crf_model(save_path, len(vocab), len(labels_to_ix), normal_param.max_length)
    elif mode == "bert_bilstm":
        save_path = normal_param.save_path_bert_bilstm
        model = keras_Bert_bilstm_crf.load_embedding_bilstm2_crf_model(save_path, len(labels_to_ix))
    else:
        save_path = normal_param.save_path_wordVEC_bilstm
        embeddings_matrix, vocab = process_data_for_keras.txtpad_use_word2vec()
        # NUM_CLASS, embeddings_matrix, input_length
        model = keras_word2vec_bilstm_crf.load_embedding_bilstm2_crf_model(save_path, len(labels_to_ix), embeddings_matrix, normal_param.max_length)

    myNerInfer = NERInference.NERInference(model, vocab, ix_to_label, len(vocab), path, mode= mode)
    new_string4_pred, ix = myNerInfer.predict_all(is_eval)
    return new_string4_pred
    # result = model.predict(content)
    # print(result)



# def prediction_data(path):
#     labels_to_ix, ix_to_label = NER_pre_data.build_label(normal_param.labels)
#     vocab = process_data_for_keras.read_vocab(normal_param.lstm_vocab)
#     model = keras_BILSTM_CEF.load_embedding_bilstm2_crf_model(normal_param.save_path, len(vocab), len(labels_to_ix), normal_param.max_length)
#     myNerInfer = NERInference.NERInference(model, vocab, ix_to_label, len(vocab), path)
#     new_string4_pred, ix = myNerInfer.predict_all()

if __name__ == '__main__':
    # run()
    tmp = prediction(normal_param.head_test_path, mode="bert_bilstm", is_eval=True)
    print(tmp)
