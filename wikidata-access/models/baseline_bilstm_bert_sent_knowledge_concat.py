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
Model that puts the flattened knowledge concepts for a sentence through a LSTM and concatenates the last hidden state with the bert sentence embedding.
"""


def flatten_shallow_knowledge_indexed_data(data):
    temp_sents = []
    max_sent_len = -1
    for i in range(len(data)):
        sample = [t for t in data[i].flatten().tolist() if t != 0]
        max_sent_len = len(sample) if len(sample) > max_sent_len else max_sent_len
        temp_sents.append(sample)

    return temp_sents, max_sent_len
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

    # load embeddings
    emb_knowledge = np.load(PROCESSED_DIR + "index_to_vec_kge"+kwargs['model_settings']['kg_embeddings'][1]+".npy")

    # load data
    X_train, X_dev, X_test = data["X_train"], data["X_dev"], data["X_test"]
    kX_train, kX_dev, kX_test = data["kX_train"], data["kX_dev"], data["kX_test"] # [samples, sent_len, max_concepts]
    y_train, y_dev, y_test = data["y_train"], data["y_dev"], data["y_test"]

    # flatten
    kX_train, max_sent_len_train = flatten_shallow_knowledge_indexed_data(kX_train)
    kX_dev, max_sent_len_dev = flatten_shallow_knowledge_indexed_data(kX_dev)
    kX_test, max_sent_len_test = flatten_shallow_knowledge_indexed_data(kX_test)

    max_sent_len = max([max_sent_len_dev, max_sent_len_test, max_sent_len_train])

    # padding
    kX_train = pad_sequences(kX_train, maxlen=max_sent_len, dtype='int32', padding='pre', truncating='pre', value=0.0)
    kX_dev = pad_sequences(kX_dev, maxlen=max_sent_len, dtype='int32', padding='pre', truncating='pre', value=0.0)
    kX_test = pad_sequences(kX_test, maxlen=max_sent_len, dtype='int32', padding='pre', truncating='pre', value=0.0)


    # some constants
    num_labels = y_train.shape[1]

    ############################
    #   KNOWLEDGE PROCESSING   #
    ############################

    # input for all concepts of a sentence
    knowledge_inputs = Input(shape=(max_sent_len, ), dtype='int32', name="knowledge_inputs")
    sentence_inputs = Input(shape=(X_train.shape[1], ), dtype='float32', name="sentence_inputs")

    embedded_word_ids = Embedding(emb_knowledge.shape[0], emb_knowledge.shape[1], mask_zero=True,
                               weights=[emb_knowledge], trainable=train_embeddings,
                               input_length=max_sent_len)(knowledge_inputs) # [samples, sent_len, we_dim]

    # define bilstm + dropout
    #sent_bilstm = Bidirectional(LSTM(lstm_size))(embedded_word_ids)
    #sent_bilstm_dropout = Dropout(dropout)(sent_bilstm)

    attended_knowledge = AttentionHerrmann(representation_claim=sentence_inputs, only_attended_vector=True,
                                           topic_shape=sentence_inputs.shape)(embedded_word_ids)

    concat_sequences = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([sentence_inputs, attended_knowledge])

    output_layer = Dense(num_labels, activation='softmax')(concat_sequences)

    model = Model(inputs=[sentence_inputs, knowledge_inputs], outputs=output_layer)

    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    #e = EarlyStopping(monitor=monitor, mode='auto')
    e = ModelCheckpoint(model_file, monitor=monitor, verbose=0, save_best_only=True, save_weights_only=True,
                        mode='auto', period=1)
    model.fit([X_train, kX_train], y_train, batch_size=batch_size, epochs=epochs,
              validation_data=([X_dev, kX_dev], y_dev), callbacks=[e], verbose=1)
    model.load_weights(model_file)

    y_pred_test = model.predict([X_test, kX_test], verbose=False)
    y_pred_dev = model.predict([X_dev, kX_dev], verbose=False)

    return [np.argmax(pred) for pred in y_pred_test], [np.argmax(pred) for pred in y_pred_dev]