from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import TimeDistributed, Dropout
from keras_bert import Tokenizer,load_trained_model_from_checkpoint
from keras_contrib.layers.crf import CRF
from keras.optimizers import Adam
from keras_contrib import losses, metrics
EMBEDDING_OUT_DIM = 128
HIDDEN_UNITS = 128
DROPOUT_RATE = 0.2

def build_bilstm_crf_model(NUM_CLASS):
    """
    带embedding的双向LSTM + crf
    """

    model = Sequential()
    # model.add(Embedding(VOCAB_SIZE, output_dim=EMBEDDING_OUT_DIM, mask_zero=True))
    # model.add(Dropout(DROPOUT_RATE))
    model.add(LSTM(HIDDEN_UNITS , return_sequences=True))

    # model.add(Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)))
    # model.add(Dropout(DROPOUT_RATE))
    model.add(TimeDistributed(Dense(NUM_CLASS)))
    model.add(Dropout(0.5))
    crf_layer = CRF(NUM_CLASS, sparse_target=True)
    model.add(crf_layer)
    # model.build((None, 248, ))
    # model.summary()
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=losses.crf_loss, optimizer='rmsprop',
                    metrics=[
                        metrics.crf_accuracy
                    ])
    return model

def build_bert_model(X1,X2, config_path, checkpoint_path):
    bert_model = load_trained_model_from_checkpoint(config_path,checkpoint_path,seq_len=None)
    wordvec = bert_model.predict([X1,X2])
    return wordvec