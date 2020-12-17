from keras import Input, Model
from keras.layers import Embedding, Bidirectional, Dropout, LSTM, Dense, Lambda
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from models.layers.attention_keras import AttentionHerrmann
import numpy as np
import tensorflow as tf
import os
import random as rn

"""
MLP model for BERT sentence embeddings.
"""

def train_model(data, topic, PROCESSED_DIR, SEED_FOLDER, **kwargs):
    dropout = kwargs['model_settings']["dropout"]
    lstm_size = kwargs['model_settings']["lstm_size"]
    monitor = kwargs['model_settings']["monitor"]
    batch_size = kwargs['model_settings']["batch_size"]
    epochs = kwargs['model_settings']["epochs"]
    learning_rate = kwargs['model_settings']["learning_rate"]
    train_embeddings = kwargs['model_settings']["train_embeddings"]
    # model file eg: 'results/only_sub_and_inst/model_runs/EvLSTM/seed_0/death_penalty_threelabel_crossdomain_monitor-f1_macro_do-0.3_lsize-32_bs-32_epochs-20_lr-0.001_trainemb-False_kl-only_sub_and_inst'
    model_file = SEED_FOLDER+topic+"_"+kwargs['model_settings']["model_file_suffix"]
    seed = kwargs['model_settings']['current_seed']

    # clear default graph (new model now)
    #tf.reset_default_graph()

    # set configs for memory usage and reproducibility: https://stackoverflow.com/questions/38469632/tensorflow-non-repeatable-results
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    np.random.seed(seed)
    #graph_level_seed = seed
    operation_level_seed = seed
    #tf.set_random_seed(graph_level_seed)

    # load data
    X_train, X_dev, X_test = data["X_train"], data["X_dev"], data["X_test"]
    y_train, y_dev, y_test = data["y_train"], data["y_dev"], data["y_test"]

    # some constants
    num_labels = y_train.shape[1]

    sentence_inputs = Input(shape=(X_train.shape[1], ), dtype='float32', name="sentence_inputs")
    dense = Dense(128)(sentence_inputs)
    dropout_dense = Dropout(dropout)(dense)
    output_layer = Dense(num_labels, activation='softmax')(dropout_dense)

    model = Model(inputs=sentence_inputs, outputs=output_layer)

    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    #e = EarlyStopping(monitor=monitor, mode='auto')
    e = ModelCheckpoint(model_file, monitor=monitor, verbose=0, save_best_only=True, save_weights_only=True,
                        mode='auto', period=1)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(X_dev, y_dev), callbacks=[e], verbose=1)
    model.load_weights(model_file)

    y_pred_test = model.predict(X_test, verbose=False)
    y_pred_dev = model.predict(X_dev, verbose=False)

    return [np.argmax(pred) for pred in y_pred_test], [np.argmax(pred) for pred in y_pred_dev]