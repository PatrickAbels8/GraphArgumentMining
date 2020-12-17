
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
emb_dim = 100
max_paths = 12
max_path_len = 10
max_index = 45000

lookup = np.random.rand(max_index+1, emb_dim)

def fill_X(samples, sent_len, max_paths, max_len_path, max_index):
    res = np.zeros((samples, sent_len, max_paths, max_len_path))
    for sample in range(samples):
        for token in range(sent_len):
            for path in range(max_paths):
                res[sample, token, path] = np.random.randint(1, max_index, size=max_len_path)
    return res


X_train = fill_X(1000, sent_len, max_paths, max_path_len, emb_dim)
y_train = np.random.rand(1000, 2)
X_dev = fill_X(100, sent_len, max_paths, max_path_len, emb_dim)
y_dev = np.random.rand(100, 2)

def func(x): # [bs, max_paths, max_path_len, emb_dim]
    liste = []
    for i in range(max_paths):
        #lam1 = Lambda(func_2,
         #    output_shape=(sent_len, 32))(x[:, i, :, :, :]) # [bs, max_paths, max_path_len, emb_dim] * sent_len
        #liste.append(lam1)
        liste.append(my_LSTM(x[:, i, :, :]))
    stacked = K.stack(liste, axis=1) # [bs, max_paths, 32]
    return stacked


# input for all concepts of a sentence
knowledge_inputs = Input(shape=(sent_len, max_paths, max_path_len,), dtype='int32', name="knowledge_inputs")


knowledge_inputs_td = Input(shape=(max_paths, max_path_len,), dtype='int32', name="knowledge_inputs_td")
emb_knowledge_ids = Embedding(lookup.shape[0], lookup.shape[1], mask_zero=False,
                              weights=[lookup], trainable=False)(knowledge_inputs_td) # [bs, max_paths, max_path_len, emb_dim]

my_LSTM = Bidirectional(LSTM(32))
lam = Lambda(func,
             output_shape=(max_paths, 32))(emb_knowledge_ids) # TD out:  [bs, max_paths, 32]

decoder = Model(knowledge_inputs_td, lam)# in: [max_paths, max_path_len, emb_dim], out: [max_paths, emb_dim]

evids_doc_encoder = TimeDistributed(decoder, name='evids_sent_timedistributed')(knowledge_inputs) # out [bs, sent_len, max_paths, 32]

attention = AveragePooling2D(pool_size=(1, max_paths))(evids_doc_encoder)
attention = Lambda(lambda x: K.squeeze(x, 2))(attention)

last_lstm = LSTM(32)(attention)

output_layer = Dense(2, activation='softmax')(last_lstm)

model = Model(inputs=knowledge_inputs, outputs=output_layer)

adam = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, epochs=10,
          validation_data=(X_dev, y_dev), verbose=1)