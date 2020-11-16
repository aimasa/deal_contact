from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM, recurrent
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras_contrib.layers.crf import CRF
from keras_contrib.utils import save_load_utils
from keras.optimizers import Adam

VOCAB_SIZE = 2500
EMBEDDING_OUT_DIM = 128
TIME_STAMPS = 100
HIDDEN_UNITS = 200
DROPOUT_RATE = 0.3
NUM_CLASS = 5


def build_embedding_lstm2_crf_model(VOCAB_SIZE, NUM_CLASS, TIME_STAMPS):
    """
    带embedding的双向LSTM + crf
    """
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, output_dim=EMBEDDING_OUT_DIM, mask_zero=True))
    # model.add(Dropout(DROPOUT_RATE))
    model.add(recurrent.GRU(HIDDEN_UNITS // 2, return_sequences=True))

    # model.add(Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)))
    # model.add(Dropout(DROPOUT_RATE))
    model.add(TimeDistributed(Dense(NUM_CLASS)))
    crf_layer = CRF(NUM_CLASS, sparse_target=True)
    model.add(crf_layer)
    # model.summary()
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(adam, loss=crf_layer.loss_function,
                    metrics=[
                        crf_layer.accuracy
                    ])
    return model
def save_embedding_bilstm2_crf_model(model, filename):
    save_load_utils.save_all_weights(model,filename)

def load_embedding_bilstm2_crf_model(filename, VOCAB_SIZE, NUM_CLASS, TIME_STAMPS):
    model = build_embedding_lstm2_crf_model(VOCAB_SIZE, NUM_CLASS, TIME_STAMPS)
    save_load_utils.load_all_weights(model, filename, include_optimizer=False)
    return model

