from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM, RNN
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras_contrib.layers.crf import CRF
from keras_contrib.utils import save_load_utils
from keras.optimizers import Adam

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

VOCAB_SIZE = 2500
EMBEDDING_OUT_DIM = 128
TIME_STAMPS = 100
HIDDEN_UNITS = 200
DROPOUT_RATE = 0.3
NUM_CLASS = 5
batch_size = 128
nb_classes = 10
epochs = 12
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


def build_embedding_lstm2_crf_model(VOCAB_SIZE, NUM_CLASS, TIME_STAMPS):
    """
    带embedding的双向LSTM + crf
    """
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, output_dim=EMBEDDING_OUT_DIM, mask_zero=True))
    # model.add(Dropout(DROPOUT_RATE))
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                            padding='same',
                            input_shape=input_shape))  # 卷积层1
    model.add(Activation('relu'))  # 激活层
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))  # 卷积层2
    model.add(Activation('relu'))  # 激活层
    model.add(MaxPooling2D(pool_size=pool_size))  # 池化层
    model.add(Dropout(0.25))  # 神经元随机失活
    model.add(Flatten())  # 拉成一维数据
    model.add(Dense(128))  # 全连接层1
    model.add(Activation('relu'))  # 激活层
    model.add(Dropout(0.5))  # 随机失活
    model.add(Dense(nb_classes))  # 全连接层2
    model.add(Activation('softmax'))  # Softmax评分

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

