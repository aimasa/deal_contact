from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras_contrib.layers.crf import CRF
from keras_contrib.utils import save_load_utils
import keras.backend as K
from keras.optimizers import Adam
from sklearn.metrics import precision_recall_fscore_support as score
from entity_contract import normal_param
EMBEDDING_OUT_DIM = 128
HIDDEN_UNITS = 200
DROPOUT_RATE = 0.2


def build_embedding_bilstm2_crf_model(VOCAB_SIZE, NUM_CLASS, TIME_STAMPS):
    """
    带embedding的双向LSTM + crf
    """
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, output_dim=EMBEDDING_OUT_DIM, mask_zero=True))
    # model.add(Dropout(DROPOUT_RATE))
    model.add(Bidirectional(LSTM(HIDDEN_UNITS // 2, return_sequences=True)))

    # model.add(Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)))
    # model.add(Dropout(DROPOUT_RATE))
    model.add(TimeDistributed(Dense(NUM_CLASS)))
    crf_layer = CRF(NUM_CLASS, sparse_target=True)
    model.add(crf_layer)
    # model.summary()
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(adam, loss=crf_layer.loss_function,
                    metrics=[
                        crf_layer.accuracy,
                        f1_m
                    ])
    return model

def save_embedding_bilstm2_crf_model(model, filename):
    save_load_utils.save_all_weights(model,filename)

def load_embedding_bilstm2_crf_model(filename, VOCAB_SIZE, NUM_CLASS, TIME_STAMPS):
    model = build_embedding_bilstm2_crf_model(VOCAB_SIZE, NUM_CLASS, TIME_STAMPS)
    save_load_utils.load_all_weights(model, filename, include_optimizer=False)
    return model

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.cast(K.equal(y_true, y_pred), 'float32')))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    recall = K.variable(0.)
    precision = K.variable(0.)
    for i in range(0,32):
        i_true = K.cast(K.equal(y_true, i), "float32")
        i_pre = K.cast(K.equal(y_pred, i), "float32")
        recall = recall + recall_m(i_true, i_pre) / 32.
        precision = precision + precision_m(i_true, i_pre) / 32.

    return 2 * recall * precision / (recall + precision + K.epsilon())