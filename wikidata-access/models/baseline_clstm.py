from keras import Input, Model
from keras.layers import Embedding, Bidirectional, Dropout, Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import keras.backend as K
import os
import numpy as np
from utils.helper import load_from_pickle
import tensorflow as tf
import random as rn
from models.layers.CLSTM import custom_LSTM_fo
from helpers.classification.generate_features import get_avg_embedding


def train_model(data, topic, PROCESSED_DIR, SEED_FOLDER, **kwargs):
    dropout = kwargs['model_settings']["dropout"]
    lstm_size = kwargs['model_settings']["lstm_size"]
    monitor = kwargs['model_settings']["monitor"]
    batch_size = kwargs['model_settings']["batch_size"]
    epochs = kwargs['model_settings']["epochs"]
    learning_rate = kwargs['model_settings']["learning_rate"]
    train_embeddings = kwargs['model_settings']["train_embeddings"]
    return_probs = False
    return_model = False
    model_file = SEED_FOLDER+topic+"_"+kwargs['model_settings']["model_file_suffix"]
    seed = kwargs['model_settings']['current_seed']

    # set reproducibility
    # set configs for memory usage and reproducibility: https://stackoverflow.com/questions/38469632/tensorflow-non-repeatable-results
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    np.random.seed(seed)
    graph_level_seed = 1
    operation_level_seed = 1
    tf.set_random_seed(graph_level_seed)
    sess = tf.Session(config=config)
    K.set_session(sess)

    # load vocab we and get indices for topic
    vocab_we = load_from_pickle(PROCESSED_DIR+"vocab_we.pkl")

    # load word embeddings
    embeddings_lookup = np.load(PROCESSED_DIR + "index_to_vec_we"+kwargs['model_settings']['word_embeddings'][1]+".npy")

    # load data
    X_train, X_dev, X_test = data["X_train"], data["X_dev"], data["X_test"]
    y_train, y_dev, y_test = data["y_train"], data["y_dev"], data["y_test"]
    
    # generate topic data
    data['X_topic_train'] = [get_avg_embedding(topic.split('_'), embeddings_lookup, vocab_we)] * len(data['X_train'])
    data['X_topic_dev'] = [get_avg_embedding(topic.split('_'), embeddings_lookup, vocab_we)] * len(data['X_dev'])
    data['X_topic_test'] = [get_avg_embedding(topic.split('_'), embeddings_lookup, vocab_we)] * len(data['X_test'])
    
    X_topic_train, X_topic_dev, X_topic_test = data["X_topic_train"], data["X_topic_dev"], data["X_topic_test"]
    
    # some constants
    sent_len = X_train.shape[1]
    num_labels = y_train.shape[1]

    sentence_input = Input(shape=(sent_len,), dtype='int32', name="text_input")
    gate_vector_input = Input(shape=(300,), dtype='float32', name="gate_vectors_each_sentence")
    embedded_layer = Embedding(embeddings_lookup.shape[0], embeddings_lookup.shape[1], mask_zero=True,
                               trainable=train_embeddings, input_length=sent_len,
                               weights=[embeddings_lookup])(sentence_input)


    bilstm_layer = Bidirectional(custom_LSTM_fo(lstm_size))([embedded_layer, gate_vector_input])


    dropout_layer = Dropout(dropout)(bilstm_layer)
    output_layer = Dense(num_labels, activation='softmax')(dropout_layer)
    model = Model(inputs=[sentence_input,gate_vector_input], output=output_layer)

    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    #e = EarlyStopping(monitor=monitor, mode='auto')
    e = ModelCheckpoint(model_file, monitor=monitor, verbose=0, save_best_only=True, save_weights_only=True,
                        mode='auto', period=1)
    model.fit([X_train, X_topic_train], y_train, batch_size=batch_size, epochs=epochs,
              validation_data=([X_dev, X_topic_dev], y_dev), callbacks=[e], verbose=1)
    model.load_weights(model_file)

    if return_model == True:
        return model
    else:
        test_predictions = model.predict([X_test, X_topic_test], verbose=False)
        val_predictions = model.predict([X_dev, X_topic_dev], verbose=False)
        if return_probs == False:
            test_predictions = [np.argmax(pred) for pred in test_predictions]
            val_predictions = [np.argmax(pred) for pred in val_predictions]
        return test_predictions, val_predictions
