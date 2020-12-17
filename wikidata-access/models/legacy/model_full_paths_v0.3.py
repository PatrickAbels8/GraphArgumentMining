
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import Embedding, Bidirectional, Dropout, LSTM, Dense, Lambda, TimeDistributed, AveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from models.layers.attention_keras import attention_knowledge
import os
import keras.backend as K
import random as rn


batch_size = 40
sent_len = 60
lstm_cell_size = 32
emb_dim_we = 50
emb_dim_kge = 100
max_paths = 5
max_path_len = 10
max_index = 45000

lookup = np.random.rand(max_index + 1, emb_dim_we)

def fill_X(samples, sent_len, max_paths, max_len_path, max_index):
    res = np.zeros((samples, sent_len, max_paths, max_len_path))
    for sample in range(samples):
        for token in range(sent_len):
            for path in range(max_paths):
                res[sample, token, path] = np.random.randint(1, max_index, size=max_len_path)
    return res


X_train = fill_X(18000, sent_len, max_paths, max_path_len, emb_dim_we)
y_train = np.random.rand(18000, 2)
X_dev = fill_X(1000, sent_len, max_paths, max_path_len, emb_dim_we)
y_dev = np.random.rand(1000, 2)

# TODO geschaltelte for loop extrem langsam
def func_2(x): # x = [bs, max_paths, max_path_len, emb_dim]
    liste = []
    for i in range(max_paths):
        liste.append(lstm_test(x[:, i, :, :])) # [bs, max_path_len, emb_dim] => [[bs, 32], ...sent_len times]
    stacked = K.stack(liste, axis=1) #[bs, max_path_len, 32]
    return stacked

def func(x):
    liste = []
    for i in range(sent_len):
        #lam1 = Lambda(func_2,
         #    output_shape=(sent_len, 32))(x[:, i, :, :, :]) # [bs, max_paths, max_path_len, emb_dim] * sent_len
        #liste.append(lam1)
        #l_for = TimeDistributed(lstm_for)(x[:, i, :, :, :])
        #l_back = TimeDistributed(lstm_back)(x[:, i, :, :, :])
        #concat_sequences = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([l_for, l_back])
        #liste.append(concat_sequences)
        liste.append(TimeDistributed(lstm_test)(x[:, i, :, :, :])) # [bs, max_paths, max_path_len, emb_dim] * sent_len
    stacked = K.stack(liste, axis=1)
    return stacked

# input for all concepts of a sentence
knowledge_inputs = Input(shape=(sent_len, max_paths, max_path_len,), dtype='int32', name="knowledge_inputs")
emb_knowledge_ids = Embedding(lookup.shape[0], lookup.shape[1], mask_zero=False,
                              weights=[lookup], trainable=False)(knowledge_inputs)


lstm_test = Bidirectional(LSTM(lstm_cell_size))
lstm_for = LSTM(lstm_cell_size)
lstm_back = LSTM(lstm_cell_size, go_backwards=True)

lam = Lambda(func,
             output_shape=(sent_len, max_paths, 2*lstm_cell_size))(emb_knowledge_ids) # TD out: [bs, sent_len, max_path, lstm_size]


attention = AveragePooling2D(pool_size=(1, max_paths))(lam)
attention = Lambda(lambda x: K.squeeze(x, 2))(attention)

last_lstm = Bidirectional(LSTM(lstm_cell_size))(attention)

output_layer = Dense(2, activation='softmax')(last_lstm)

model = Model(inputs=knowledge_inputs, outputs=output_layer)

adam = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, epochs=10,
          validation_data=(X_dev, y_dev), verbose=1)