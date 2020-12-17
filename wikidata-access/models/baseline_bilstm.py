from keras import Input, Model
from keras.layers import Embedding, Bidirectional, Dropout, LSTM, Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import keras.backend as K
import os
import numpy as np
from utils.helper import load_from_pickle
import tensorflow as tf
from models.layers.models_tf import DynamicBiRNN
from sklearn.metrics import f1_score
import random as rn


def train_model(data, topic, PROCESSED_DIR, SEED_FOLDER, **kwargs):
    dropout = kwargs['model_settings']["dropout"]
    lstm_size = kwargs['model_settings']["lstm_size"]
    monitor = kwargs['model_settings']["monitor"]
    batch_size = kwargs['model_settings']["batch_size"]
    epochs = kwargs['model_settings']["epochs"]
    learning_rate = kwargs['model_settings']["learning_rate"]
    train_embeddings = kwargs['model_settings']["train_embeddings"]
    recurrent_dropout = kwargs['model_settings'].get("recurrent_dropout", 0.)
    return_probs = False
    return_model = False
    model_file = SEED_FOLDER + topic + "_" + kwargs['model_settings']["model_file_suffix"] + ".h5"
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

    sentence_input = Input(shape=(X_train.shape[1],), dtype='int32', name="text_input")

    embedded_layer = Embedding(embeddings_lookup.shape[0], embeddings_lookup.shape[1], mask_zero=True,
                               weights=[embeddings_lookup], trainable=train_embeddings,
                               input_length=X_train.shape[1])(sentence_input)

    bilstm_layer = Bidirectional(LSTM(lstm_size, recurrent_dropout=recurrent_dropout))(embedded_layer)

    if dropout > 0.:
        dropout_layer = Dropout(dropout)(bilstm_layer)
    else:
        dropout_layer = bilstm_layer

    output_layer = Dense(y_train.shape[1], activation='softmax')(dropout_layer)

    model = Model(inputs=sentence_input, outputs=output_layer)

    if "gradient_clipnorm" in kwargs['model_settings']:
        adam = Adam(lr=learning_rate, clipnorm=kwargs['model_settings']['gradient_clipnorm'])
    else:
        adam = Adam(lr=learning_rate)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    #e = EarlyStopping(monitor=monitor, mode='auto')
    e = ModelCheckpoint(model_file, monitor=monitor, verbose=1, save_best_only=True, save_weights_only=True,
                        mode='auto', period=1)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(X_dev, y_dev), callbacks=[e], verbose=1)
    model.load_weights(model_file)

    if return_model == True:
        return model
    else:
        test_predictions = model.predict(X_test, verbose=False)
        val_predictions = model.predict(X_dev, verbose=False)
        if return_probs == False:
            test_predictions = [np.argmax(pred) for pred in test_predictions]
            val_predictions = [np.argmax(pred) for pred in val_predictions]
        return test_predictions, val_predictions

def train_model_tf(data, topic, PROCESSED_DIR, SEED_FOLDER, **kwargs):
    # example from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
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
    tf.reset_default_graph()

    # set configs for memory usage and reproducibility: https://stackoverflow.com/questions/38469632/tensorflow-non-repeatable-results
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    np.random.seed(seed)
    graph_level_seed = seed
    operation_level_seed = seed
    tf.set_random_seed(graph_level_seed)

    # load embeddings
    emb_sents = np.load(PROCESSED_DIR + "index_to_vec_we"+kwargs['model_settings']['word_embeddings'][1]+".npy")

    # load data
    X_train, X_dev, X_test = data["X_train"], data["X_dev"], data["X_test"] # [samples, sent_len]
    y_train, y_dev, y_test = data["y_train"], data["y_dev"], data["y_test"]
    val_y_non_one_hot = [np.argmax(pred) for pred in y_dev]

    # some constants
    sent_len = X_train.shape[1]
    num_labels = y_train.shape[1]

    # calculate how often the model has to run for 1 epoch, given the batch_size
    periods_ukp = int(len(X_train) / batch_size)
    periods_ukp_val = int(len(X_dev) / batch_size)

    # tf Graph input
    X = tf.placeholder(tf.int32, [None, sent_len])
    Y = tf.placeholder(tf.float32, [None, num_labels])
    EMB_SENTS = tf.placeholder(tf.float32, [emb_sents.shape[0], emb_sents.shape[1]]) #https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
    dropout_const = tf.placeholder(tf.float32)

    # Define weights
    weights = {
        # Hidden layer weights => 2*n_hidden because of forward + backward cells and *2 again because of concat of shared&private
        #'dense': tf.Variable(tf.random_normal([(lstm_size*2)+emb_knowledge.shape[1], num_labels])),
        'dense': tf.Variable(tf.random_normal([lstm_size*2, num_labels])),
        'emb_sents': tf.Variable(tf.constant(0.0, shape=[emb_sents.shape[0], emb_sents.shape[1]]),
                trainable=train_embeddings, name="emb_sents"),
    }
    biases = {
        'dense': tf.Variable(tf.random_normal([num_labels])),
    }

    # Embedd sentences and knowledge
    # Source: https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
    emb_sents_init = weights['emb_sents'].assign(EMB_SENTS)
    embedded_word_ids = tf.nn.embedding_lookup(emb_sents_init, X) # [samples, sent_len, we_dim]

    bilstm, bilstm_last = DynamicBiRNN(embedded_word_ids, sent_len, lstm_size, name="bilstm")
    #bilstm_do_last = tf.nn.dropout(bilstm_last, dropout_const, seed=operation_level_seed)
    bilstm_do = tf.nn.dropout(bilstm_last, dropout_const, seed=operation_level_seed)

    dense = tf.matmul(bilstm_do, weights['dense']) + biases['dense']

    prediction_ukp = tf.nn.softmax(dense)

    # Define loss and optimizer (https://stats.stackexchange.com/questions/327348/how-is-softmax-cross-entropy-with-logits-different-from-softmax-cross-entropy-wi)
    loss_ukp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=prediction_ukp, labels=Y))
    optimizer_ukp = tf.train.AdamOptimizer(learning_rate=learning_rate) # adam with 0.01 => no learning, GDC w/ 0.01 good
    train_ukp = optimizer_ukp.minimize(loss_ukp)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred_ukp = tf.argmax(prediction_ukp, 1)
    correct_pred_ukp_eq = tf.equal(correct_pred_ukp, tf.argmax(Y, 1))
    accuracy_ukp = tf.reduce_mean(tf.cast(correct_pred_ukp_eq, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # init saver
    saver = tf.train.Saver()

    # Start training
    with tf.Session(config=config) as sess:
        best_f1_score = 0

        # Run the initializer
        sess.run(init)

        for epoch in range(epochs): # iterate over epochs
            loss, acc, loss_val, acc_val, pred_list_ukp = 0, 0, 0, 0, np.array([])
            print("====== Start epoch " + str(epoch) + " for UKP ======")
            # https://stackoverflow.com/questions/44565186/how-to-implement-next-batch-function-for-custom-data-in-python
            for p in range(periods_ukp+1): # each period trains samples of length batch_size
                # Run optimization op (backprop)
                _, loss_t, acc_t = sess.run([train_ukp, loss_ukp, accuracy_ukp],
                                          feed_dict={X: X_train[p*batch_size:(p+1)*batch_size],
                                                     Y: y_train[p*batch_size:(p+1)*batch_size],
                                                     EMB_SENTS: emb_sents,
                                                     dropout_const: 1-dropout})
                loss += loss_t
                acc += acc_t
                print("Train_period= "+str(p)+"/"+str(periods_ukp)+", train_loss= " + "{:.4f}".format(loss / p) + ", train_acc= " + "{:.3f}".format(acc/p), end='\r')

            for p in range(periods_ukp_val+1):
                loss_val_t, acc_val_t, pred_list_ukp_t = sess.run([loss_ukp, accuracy_ukp, correct_pred_ukp],
                                                                  feed_dict={X: X_dev[p*batch_size:(p+1)*batch_size],
                                                                             Y: y_dev[p*batch_size:(p+1)*batch_size],
                                                                             EMB_SENTS: emb_sents,
                                                                             dropout_const: 1.0})
                loss_val += loss_val_t
                acc_val += acc_val_t
                pred_list_ukp = np.concatenate([pred_list_ukp, pred_list_ukp_t], axis=-1)


            temp_F1_score = f1_score(val_y_non_one_hot, pred_list_ukp, average='macro')
            if temp_F1_score > best_f1_score:
                best_f1_score = temp_F1_score
                # save model if better than previous one
                # check if current model is better, if yes => save
                save_path = saver.save(sess, model_file)

            print("train_loss= " + "{:.4f}".format(loss) + ", train_acc= " + \
                  "{:.3f}".format(acc/(periods_ukp+1)) + ", val_loss= " + "{:.4f}".format(loss_val) + ", val_acc= " + \
                  "{:.3f}".format(acc_val/(periods_ukp_val+1)) + ", val_F1= " + "{:.3f}".format(temp_F1_score)
                  )

            # variables_names = [v.name for v in tf.trainable_variables() if "bilstm_shared/bidirectional_rnn/fw/" in v.name]
            # print(variables_names)
            # values = sess.run(variables_names)
            # for k, v in zip(variables_names, values):
            #    print(k, v)

        saver.restore(sess, model_file)

        return sess.run(correct_pred_ukp, feed_dict={X: X_test, Y: y_test,
                                                     EMB_SENTS: emb_sents, dropout_const: 1.0}).tolist(),\
               sess.run(correct_pred_ukp, feed_dict={X: X_dev, Y: y_dev,
                                                     EMB_SENTS: emb_sents, dropout_const: 1.0}).tolist()