from keras import Input, Model
from keras.layers import Embedding, Bidirectional, Dropout, LSTM, Dense, Lambda, TimeDistributed, AveragePooling1D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from models.layers.attention_keras import attention_knowledge
import os
import keras.backend as K
import random as rn

def train_model(data, topic, PROCESSED_DIR, SEED_FOLDER, **kwargs):
    def func(x):
        liste = []
        for i in range(sent_len):
            liste.append(
                TimeDistributed(
                    AveragePooling1D(pool_size=max_path_len)
                )(x[:, i, :, :, :]))  # [bs, max_paths, max_path_len, emb_dim] * sent_len
        stacked = K.stack(liste, axis=1)
        stacked = Lambda(lambda x: K.squeeze(x, 3))(stacked)
        return stacked

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

    # load embeddings
    emb_sents = np.load(PROCESSED_DIR + "index_to_vec_we"+kwargs['model_settings']['word_embeddings'][1]+".npy")
    emb_knowledge = np.load(PROCESSED_DIR + "index_to_vec_kge"+kwargs['model_settings']['kg_embeddings'][1]+".npy")

    # load data
    X_train, X_dev, X_test = data["X_train"], data["X_dev"], data["X_test"] # [samples, sent_len]
    kX_train, kX_dev, kX_test = data["kX_train"], data["kX_dev"], data["kX_test"] # [samples, sent_len, max_concepts]
    y_train, y_dev, y_test = data["y_train"], data["y_dev"], data["y_test"]
    val_y_non_one_hot = [np.argmax(pred) for pred in y_dev]

    # some constants
    sent_len = X_train.shape[1]
    max_paths = kX_train.shape[2]
    max_path_len = kX_train.shape[3]
    num_labels = y_train.shape[1]
    attention_size = kwargs['model_settings'].get('attention_size', emb_sents.shape[1])

    ############################
    #   KNOWLEDGE PROCESSING   #
    ############################

    # input for all concepts of a sentence
    sentence_inputs = Input(shape=(sent_len, ), dtype='int32', name="sentence_inputs")
    knowledge_inputs = Input(shape=(sent_len, max_paths, max_path_len,), dtype='int32', name="knowledge_inputs")

    emb_knowledge_ids = Embedding(emb_knowledge.shape[0], emb_knowledge.shape[1], mask_zero=True,
                               weights=[emb_knowledge], trainable=train_embeddings)(knowledge_inputs) # [samples, sent_len, max_concepts, kge_dim]

    embedded_word_ids = Embedding(emb_sents.shape[0], emb_sents.shape[1], mask_zero=True,
                               weights=[emb_sents], trainable=train_embeddings,
                               input_length=sent_len)(sentence_inputs) # [samples, sent_len, we_dim]

    # function that reduces the paths to a single vector => from there on, model is equal to the shallow model
    # in: [bs, sent_len, max_concepts, max_path_len, kge_dim], out: [bs, sent_len, max_concepts, 2*lstm_size]
    #paths_bilstm = Bidirectional(LSTM(lstm_size)) # define lstm that reduces the paths to one vector
    reduce_paths_to_vector = Lambda(func, output_shape=(sent_len, max_paths, emb_knowledge.shape[1]))(emb_knowledge_ids)

    attended_knowledge = attention_knowledge(embedded_word_ids, None, attention_size, return_alphas=False, summed_up=True)(reduce_paths_to_vector)

    concat_sequences = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([embedded_word_ids, attended_knowledge])

    # define bilstm + dropout
    sent_bilstm = Bidirectional(LSTM(lstm_size, input_shape=(None, sent_len, attention_size+emb_sents.shape[1])))(concat_sequences)
    sent_bilstm_dropout = Dropout(dropout)(sent_bilstm)

    output_layer = Dense(num_labels, activation='softmax')(sent_bilstm_dropout)

    model = Model(inputs=[sentence_inputs, knowledge_inputs], outputs=output_layer)

    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    #e = EarlyStopping(monitor=monitor, mode='auto')
    e = ModelCheckpoint(model_file, monitor=monitor, verbose=0, save_best_only=True, save_weights_only=True,
                        mode='auto', period=1)
    model.fit([X_train, kX_train], y_train, batch_size=batch_size, epochs=epochs,
              validation_data=([X_dev, kX_dev], y_dev), callbacks=[e], verbose=1)
    model.load_weights(model_file)

    y_pred_test = model.predict([X_test, kX_test], verbose=False)
    y_pred_dev = model.predict([X_dev, kX_dev], verbose=False)

    return [np.argmax(pred) for pred in y_pred_test], [np.argmax(pred) for pred in y_pred_dev]