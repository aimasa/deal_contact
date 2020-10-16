from entity_contract import NER_pre_data, process_data_for_keras, NERInference, normal_param
from model import keras_BILSTM_CEF

def prediction(path):
    labels_to_ix, ix_to_label = NER_pre_data.build_label(normal_param.labels)
    vocab = process_data_for_keras.read_vocab(normal_param.lstm_vocab)
    model = keras_BILSTM_CEF.load_embedding_bilstm2_crf_model(normal_param.save_path, len(vocab), len(labels_to_ix), normal_param.max_length)
    myNerInfer = NERInference.NERInference(model, vocab, ix_to_label, len(vocab), path)
    new_string4_pred, ix = myNerInfer.predict_all()
    return new_string4_pred
    # result = model.predict(content)
    # print(result)

def prediction_data(path):
    labels_to_ix, ix_to_label = NER_pre_data.build_label(normal_param.labels)
    vocab = process_data_for_keras.read_vocab(normal_param.lstm_vocab)
    model = keras_BILSTM_CEF.load_embedding_bilstm2_crf_model(normal_param.save_path, len(vocab), len(labels_to_ix), normal_param.max_length)
    myNerInfer = NERInference.NERInference(model, vocab, ix_to_label, len(vocab), path)
    new_string4_pred, ix = myNerInfer.predict_all()

if __name__ == '__main__':
    # run()
    prediction("F:/data/test/pred_contant/txt/0.txt")