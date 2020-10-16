from model import keras_BILSTM_CEF
from entity_contract import process_data_for_keras
from entity_contract import normal_param
from utils import check_utils
def run_by_bilstm_crf():
    x_train, y_train, x_test, y_test, vocab_length, labels_to_ix_length = process_data_for_keras.process_data()
    model = keras_BILSTM_CEF.build_embedding_bilstm2_crf_model(vocab_length, labels_to_ix_length, 0)
    if check_utils.check_path(normal_param.save_path):
        model.load_weights(normal_param.save_path)
    model.fit(x_train, y_train, batch_size=18, epochs=50, validation_data = (x_test, y_test), shuffle = False, validation_split=0.2, verbose=1)
    keras_BILSTM_CEF.save_embedding_bilstm2_crf_model(model, normal_param.save_path)

def run_by_bert_bilstm_crf():
    '''
    需要对数据进行bert做embeding
    '''


def run(model):
    if model == "bilstm_crf":
        run_by_bilstm_crf()


if __name__ == '__main__':
    run("bilstm_crf")