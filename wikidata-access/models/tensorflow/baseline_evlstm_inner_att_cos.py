import numpy as np

import tensorflow as tf
from models.layers.models_tf import DynamicBiRNN
from sklearn.metrics import f1_score
from models.layers.attention_tf import attention_knowledge
import os
from utils.helper import load_from_pickle
from helpers.classification.generate_features import get_cosine_sim_input
import random as rn

def train_model(data, topic, PROCESSED_DIR, SEED_FOLDER, **kwargs):
    """
    Trains bilstm on DIP2016 and UKP in an alternating fashion. Rebuilds the idea of Shared-private model from
    "Adversarial Multi-task Learning for Text Classification" by Pengfei. Also includes ideas of https://jg8610.github.io/Multi-Task/.
    Consists of 2 private models (UKP, DIP) and 1 shared model.

    Basic implementation by:
    Links:
        [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
        [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
    Author: Aymeric Damien
    Project: https://github.com/aymericdamien/TensorFlow-Examples/
    """
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

    # load vocab we and get indices for topic
    vocab_we = load_from_pickle(PROCESSED_DIR+"vocab_we.pkl")

    # load embeddings
    emb_sents = np.load(PROCESSED_DIR + "index_to_vec_we"+kwargs['model_settings']['word_embeddings'][1]+".npy")
    emb_knowledge = np.load(PROCESSED_DIR + "index_to_vec_kge"+kwargs['model_settings']['kg_embeddings'][1]+".npy")

    # load data
    X_train, X_dev, X_test = data["X_train"], data["X_dev"], data["X_test"] # [samples, sent_len]
    kX_train, kX_dev, kX_test = data["kX_train"], data["kX_dev"], data["kX_test"] # [samples, sent_len, max_concepts]
    X_topic_train, X_topic_dev, X_topic_test = data["X_topic_train"], data["X_topic_dev"], data["X_topic_test"]
    X_topic_cos_train = get_cosine_sim_input(X_train, X_topic_train, emb_sents, vocab_we)
    X_topic_cos_dev = get_cosine_sim_input(X_dev, X_topic_dev, emb_sents, vocab_we)
    X_topic_cos_test = get_cosine_sim_input(X_test, X_topic_test, emb_sents, vocab_we)
    y_train, y_dev, y_test = data["y_train"], data["y_dev"], data["y_test"]
    val_y_non_one_hot = [np.argmax(pred) for pred in y_dev]

    # some constants
    sent_len = X_train.shape[1]
    max_concepts = kX_train.shape[2]
    num_labels = y_train.shape[1]
    attention_size = kwargs['model_settings'].get('attention_size', emb_sents.shape[1])

    # calculate how often the model has to run for 1 epoch, given the batch_size
    periods_ukp = int(len(X_train) / batch_size)
    periods_ukp_val = int(len(X_dev) / batch_size)

    # tf Graph input
    X = tf.placeholder(tf.int32, [None, sent_len])
    X_topic_cos = tf.placeholder(tf.float32, [None, sent_len, 1])
    KX = tf.placeholder(tf.int32, [None, sent_len, max_concepts])
    Y = tf.placeholder(tf.float32, [None, num_labels])
    EMB_SENTS = tf.placeholder(tf.float32, [emb_sents.shape[0], emb_sents.shape[1]]) #https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
    EMB_KNOWLEDGE = tf.placeholder(tf.float32, [emb_knowledge.shape[0], emb_knowledge.shape[1]]) #https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
    dropout_const = tf.placeholder(tf.float32)

    # Define weights
    weights = {
        # Hidden layer weights => 2*n_hidden because of forward + backward cells and *2 again because of concat of shared&private
        #'dense': tf.Variable(tf.random_normal([(lstm_size*2)+emb_knowledge.shape[1], num_labels])),
        'dense': tf.Variable(tf.random_normal([lstm_size*2, num_labels])),
        'emb_sents': tf.Variable(tf.constant(0.0, shape=[emb_sents.shape[0], emb_sents.shape[1]]),
                trainable=train_embeddings, name="emb_sents"),
        'emb_knowledge': tf.Variable(tf.constant(0.0, shape=[emb_knowledge.shape[0], emb_knowledge.shape[1]]),
                trainable=train_embeddings, name="emb_knowledge")
    }
    biases = {
        'dense': tf.Variable(tf.random_normal([num_labels])),
    }

    # Embedd sentences and knowledge
    # Source: https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
    emb_sents_init = weights['emb_sents'].assign(EMB_SENTS)
    embedded_word_ids = tf.nn.embedding_lookup(emb_sents_init, X) # [samples, sent_len, we_dim]
    emb_knowledge_init = weights['emb_knowledge'].assign(EMB_KNOWLEDGE)
    emb_knowledge_ids = tf.nn.embedding_lookup(emb_knowledge_init, KX) # [samples, sent_len, max_concepts, kge_dim]

    #bilstm, bilstm_last = DynamicBiRNN(embedded_word_ids, sent_len, lstm_size, name="bilstm")
    #bilstm_do_last = tf.nn.dropout(bilstm_last, dropout_const, seed=operation_level_seed)
    #bilstm_do = tf.nn.dropout(bilstm, dropout_const, seed=operation_level_seed)

    # attention
    initial_t = tf.constant(0)
    initial_outputs = tf.TensorArray(dtype=tf.float32, size=sent_len)
    #shit = tf.placeholder(tf.float32, [None, emb_knowledge.shape[1]])
    condition = lambda t, _: tf.less(t, sent_len)
    #body = lambda i: tf.add(i, 1)
    def body(t, outputs_):
        kX_sliced = emb_knowledge_ids[:, t,:,:] # [samples, token at t, concepts, concept_embs]
        X_topic = embedded_word_ids[:, t,:]
        outputs_ = outputs_.write(t, attention_knowledge(kX_sliced, X_topic, attention_size, return_alphas=False)) #[ [samples, kge], [samples, kge], ...] =>[sent_len, samples, kge]
        return t + 1, outputs_

    # do the loop:
    _, outputs = tf.while_loop(condition, body, loop_vars=[initial_t, initial_outputs])#,
    stack = outputs.stack() # [sent_len, samples, kge_dim]
    attended_knowledge = tf.transpose(stack, [1,0,2]) #[samples, sent_len, kge_dim]

    concat_sequences = tf.concat([embedded_word_ids, attended_knowledge, X_topic_cos], axis=2)  # the dimension where I want to add two tensors can be different, all others have to be equal

    bilstm_2, bilstm_last_2 = DynamicBiRNN(concat_sequences, sent_len, lstm_size, name="bilstm_2")

    bilstm_do_2 = tf.nn.dropout(bilstm_last_2, dropout_const, seed=operation_level_seed)

    dense = tf.matmul(bilstm_do_2, weights['dense']) + biases['dense']

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
                                                     X_topic_cos: X_topic_cos_train[p * batch_size:(p + 1) * batch_size],
                                                     KX: kX_train[p * batch_size:(p + 1) * batch_size],
                                                     Y: y_train[p*batch_size:(p+1)*batch_size],
                                                     EMB_SENTS: emb_sents, EMB_KNOWLEDGE: emb_knowledge,
                                                     dropout_const: 1-dropout})
                loss += loss_t
                acc += acc_t
                print("Train_period= "+str(p)+"/"+str(periods_ukp)+", train_loss= " + "{:.4f}".format(loss / p) + ", train_acc= " + "{:.3f}".format(acc/p), end='\r')

            for p in range(periods_ukp_val+1):
                loss_val_t, acc_val_t, pred_list_ukp_t = sess.run([loss_ukp, accuracy_ukp, correct_pred_ukp],
                                                                  feed_dict={X: X_dev[p*batch_size:(p+1)*batch_size],
                                                                             X_topic_cos: X_topic_cos_dev[p * batch_size:(p + 1) * batch_size],
                                                                             KX: kX_dev[p*batch_size:(p+1)*batch_size],
                                                                             Y: y_dev[p*batch_size:(p+1)*batch_size],
                                                                             EMB_SENTS: emb_sents, EMB_KNOWLEDGE: emb_knowledge,
                                                                             dropout_const: 1.0})
                loss_val += loss_val_t
                acc_val += acc_val_t
                pred_list_ukp = np.concatenate([pred_list_ukp, pred_list_ukp_t], axis=-1)


            temp_F1_score = f1_score(val_y_non_one_hot, pred_list_ukp, average='macro')
            if temp_F1_score > best_f1_score:
                best_f1_score = temp_F1_score
                # save model if better than previous one
                # check if current model is better, if yes => save
                save_path = saver.save(sess, model_file, latest_filename="checkpoint_" + topic+"_"+kwargs['model_settings']["model_file_suffix"])

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

        return sess.run(correct_pred_ukp, feed_dict={X: X_test, X_topic_cos: X_topic_cos_test, KX: kX_test, Y: y_test,
                                                     EMB_SENTS: emb_sents, EMB_KNOWLEDGE: emb_knowledge, dropout_const: 1.0}).tolist(),\
               sess.run(correct_pred_ukp, feed_dict={X: X_dev, X_topic_cos: X_topic_cos_dev, KX: kX_dev, Y: y_dev,
                                                     EMB_SENTS: emb_sents, EMB_KNOWLEDGE: emb_knowledge, dropout_const: 1.0}).tolist()

def train_model_tf_knowldge_baseline(data, topic, PROCESSED_DIR, SEED_FOLDER, **kwargs):
    """
    Trains bilstm on DIP2016 and UKP in an alternating fashion. Rebuilds the idea of Shared-private model from
    "Adversarial Multi-task Learning for Text Classification" by Pengfei. Also includes ideas of https://jg8610.github.io/Multi-Task/.
    Consists of 2 private models (UKP, DIP) and 1 shared model.

    Basic implementation by:
    Links:
        [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
        [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
    Author: Aymeric Damien
    Project: https://github.com/aymericdamien/TensorFlow-Examples/
    """
    # example from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
    dropout = kwargs['model_settings'].get("dropout", 0.3)
    lstm_size = kwargs['model_settings'].get("lstm_size", 32)
    monitor = kwargs['model_settings'].get("monitor", "val_loss")
    batch_size = kwargs['model_settings'].get("batch_size", 32)
    epochs = kwargs['model_settings'].get("epochs", 10)
    learning_rate = kwargs['model_settings'].get("learning_rate", 0.005)
    train_embeddings = kwargs['model_settings'].get("train_embeddings", False)
    return_probs = False
    return_model = False
    model_file = SEED_FOLDER+topic
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
    graph_level_seed = 1
    operation_level_seed = 1
    tf.set_random_seed(graph_level_seed)

    # load embeddings
    emb_sents = np.load(PROCESSED_DIR+"index_to_vec_we.npy")
    emb_knowledge = np.load(PROCESSED_DIR+"index_to_vec_kge.npy")

    # load data
    X_train, X_dev, X_test = data["X_train"], data["X_dev"], data["X_test"] # [samples, sent_len]
    kX_train, kX_dev, kX_test = data["kX_train"], data["kX_dev"], data["kX_test"] # [samples, sent_len, max_concepts]
    y_train, y_dev, y_test = data["y_train"], data["y_dev"], data["y_test"]
    val_y_non_one_hot = [np.argmax(pred) for pred in y_dev]

    # some constants
    sent_len = X_train.shape[1]
    max_concepts = kX_train.shape[2]
    num_labels = y_train.shape[1]

    # calculate how often the model has to run for 1 epoch, given the batch_size
    periods_ukp = int(len(X_train) / batch_size)
    periods_ukp_val = int(len(X_dev) / batch_size)

    # tf Graph input
    X = tf.placeholder(tf.int32, [None, sent_len])
    KX = tf.placeholder(tf.int32, [None, sent_len, max_concepts])
    Y = tf.placeholder(tf.float32, [None, num_labels])
    EMB_SENTS = tf.placeholder(tf.float32, [emb_sents.shape[0], emb_sents.shape[1]]) #https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
    EMB_KNOWLEDGE = tf.placeholder(tf.float32, [emb_knowledge.shape[0], emb_knowledge.shape[1]]) #https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
    dropout_const = tf.placeholder(tf.float32)

    # Define weights
    weights = {
        # Hidden layer weights => 2*n_hidden because of forward + backward cells and *2 again because of concat of shared&private
        'dense': tf.Variable(tf.random_normal([(lstm_size*2)+emb_knowledge.shape[1], num_labels])),
        #'dense': tf.Variable(tf.random_normal([lstm_size*2, num_labels])),
        'emb_sents': tf.Variable(tf.constant(0.0, shape=[emb_sents.shape[0], emb_sents.shape[1]]),
                trainable=train_embeddings, name="emb_sents"),
        'emb_knowledge': tf.Variable(tf.constant(0.0, shape=[emb_knowledge.shape[0], emb_knowledge.shape[1]]),
                trainable=train_embeddings, name="emb_knowledge")
    }
    biases = {
        'dense': tf.Variable(tf.random_normal([num_labels])),
    }

    # Embedd sentences and knowledge
    # Source: https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
    emb_sents_init = weights['emb_sents'].assign(EMB_SENTS)
    embedded_word_ids = tf.nn.embedding_lookup(emb_sents_init, X) # [samples, sent_len, we_dim]
    emb_knowledge_init = weights['emb_knowledge'].assign(EMB_KNOWLEDGE)
    emb_knowledge_ids = tf.nn.embedding_lookup(emb_knowledge_init, KX) # [samples, sent_len, max_concepts, kge_dim]

    bilstm, bilstm_last = DynamicBiRNN(embedded_word_ids, sent_len, lstm_size, name="bilstm")
    #bilstm_do_last = tf.nn.dropout(bilstm_last, dropout_const, seed=operation_level_seed)
    bilstm_do = tf.nn.dropout(bilstm, dropout_const)

    # attention
    initial_t = tf.constant(0)
    initial_outputs = tf.TensorArray(dtype=tf.float32, size=sent_len)
    #shit = tf.placeholder(tf.float32, [None, emb_knowledge.shape[1]])
    condition = lambda t, _: tf.less(t, sent_len)
    #body = lambda i: tf.add(i, 1)
    def body(t, outputs_):
        kX_sliced = emb_knowledge_ids[:, t,:,:]
        X_topic = bilstm_do[:, t,:]
        outputs_ = outputs_.write(t, attention_knowledge(kX_sliced, X_topic, 2*lstm_size, return_alphas=False)) #[ [samples, kge], [samples, kge], ...] =>[sent_len, samples, kge]
        return t + 1, outputs_

    # do the loop:
    _, outputs = tf.while_loop(condition, body, loop_vars=[initial_t, initial_outputs])#,
    stack = outputs.stack() # [sent_len, samples, kge_dim]
    attended_knowledge = tf.transpose(stack, [1,0,2]) #[samples, sent_len, kge_dim]
    sum = tf.reduce_sum(attended_knowledge, axis=1)  # [samples, kge_dim+we_dim]

    concat = tf.concat([bilstm_do[:,-1,:], sum], axis=1)  # the dimension where I want to add two tensors can be different, all others have to be equal

    dense = tf.matmul(concat, weights['dense']) + biases['dense']

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
                                                     KX: kX_train[p * batch_size:(p + 1) * batch_size],
                                                     Y: y_train[p*batch_size:(p+1)*batch_size],
                                                     EMB_SENTS: emb_sents, EMB_KNOWLEDGE: emb_knowledge,
                                                     dropout_const: 1-dropout})
                loss += loss_t
                acc += acc_t
                print("Train_period= "+str(p)+"/"+str(periods_ukp)+", train_loss= " + "{:.4f}".format(loss / p) + ", train_acc= " + "{:.3f}".format(acc/p), end='\r')

            for p in range(periods_ukp_val+1):
                loss_val_t, acc_val_t, pred_list_ukp_t = sess.run([loss_ukp, accuracy_ukp, correct_pred_ukp],
                                                                  feed_dict={X: X_dev[p*batch_size:(p+1)*batch_size],
                                                                             KX: kX_dev[p*batch_size:(p+1)*batch_size],
                                                                             Y: y_dev[p*batch_size:(p+1)*batch_size],
                                                                             EMB_SENTS: emb_sents, EMB_KNOWLEDGE: emb_knowledge,
                                                                             dropout_const: 1-dropout})
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

        return sess.run(correct_pred_ukp, feed_dict={X: X_test, KX: kX_test, Y: y_test,
                                                     EMB_SENTS: emb_sents, EMB_KNOWLEDGE: emb_knowledge, dropout_const: 1.0}),\
               sess.run(correct_pred_ukp, feed_dict={X: X_dev, KX: kX_dev, Y: y_dev,
                                                     EMB_SENTS: emb_sents, EMB_KNOWLEDGE: emb_knowledge, dropout_const: 1.0})
'''
def train_model_keras(data, topic, PROCESSED_DIR, SEED_FOLDER, **kwargs):
    dropout = kwargs['model_settings'].get("dropout", 0.7)
    lstm_size = kwargs['model_settings'].get("lstm_size", 32)
    monitor = kwargs['model_settings'].get("monitor", "val_loss")
    batch_size = kwargs['model_settings'].get("batch_size", 32)
    epochs = kwargs['model_settings'].get("epochs", 10)
    learning_rate = kwargs['model_settings'].get("learning_rate", 0.005)
    train_embeddings = kwargs['model_settings'].get("train_embeddings", False)
    return_probs = False
    return_model = False
    model_file = SEED_FOLDER+topic

    # load embeddings
    embeddings_lookup = np.load(PROCESSED_DIR+"index_to_vec_we.npy")
    knowledge_embeddings_lookup = np.load(PROCESSED_DIR+"index_to_vec_kge.npy")

    # load data
    X_train, X_dev, X_test = data["X_train"], data["X_dev"], data["X_test"] # [samples, sent_len]
    kX_train, kX_dev, kX_test = data["kX_train"], data["kX_dev"], data["kX_test"] # [samples, sent_len, max_concepts]
    y_train, y_dev, y_test = data["y_train"], data["y_dev"], data["y_test"]

    # some constants
    sent_len = X_train.shape[1]
    max_concepts = kX_train.shape[2]

    ############################
    #   KNOWLEDGE PROCESSING   #
    ############################

    # input for all concepts of a sentence
    sentence_inputs = Input(shape=(sent_len, ), dtype='int32', name="sentence_inputs")
    knowledge_inputs = Input(shape=(sent_len, max_concepts,), dtype='int32', name="knowledge_inputs")

    concept_embedded_layer = Embedding(knowledge_embeddings_lookup.shape[0], knowledge_embeddings_lookup.shape[1], mask_zero=True,
                               weights=[knowledge_embeddings_lookup], trainable=train_embeddings,
                               input_length=max_concepts)(knowledge_inputs) # [samples, sent_len, max_concepts, kge_dim]

    sent_embedded = Embedding(embeddings_lookup.shape[0], embeddings_lookup.shape[1], mask_zero=True,
                               weights=[embeddings_lookup], trainable=train_embeddings,
                               input_length=sent_len)(sentence_inputs) # [samples, sent_len, we_dim]

    #concepts_input = Input(shape=(max_concepts, knowledge_embeddings_lookup.shape[1]), dtype='float32', name="concepts_input")
    token_inputs = Input(shape=(embeddings_lookup.shape[1],), dtype='float32', name="sentence_inputs")

    attended_concept = TimeDistributedAttentionHerrmann(representation_claim=token_inputs,
                                         only_attended_vector=False,
                                         summed_up=True,
                                        input_shape=(sent_len, max_concepts,knowledge_embeddings_lookup.shape[1],))(concept_embedded_layer)

    concept_encoder = Model(token_inputs, attended_concept[0])
    knowledge_encoder = TimeDistributed(concept_encoder)(sent_embedded)


    # define bilstm + dropout
    sent_bilstm = Bidirectional(LSTM(lstm_size))(sent_embedded)
    sent_bilstm_dropout = Dropout(dropout)(sent_bilstm)


    output_layer = Dense(y_train.shape[1], activation='softmax')(sent_bilstm_dropout)

    model = Model(inputs=[sentence_inputs, knowledge_inputs], outputs=output_layer)

    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    #e = EarlyStopping(monitor=monitor, mode='auto')
    e = ModelCheckpoint(model_file, monitor=monitor, verbose=0, save_best_only=True, save_weights_only=True,
                        mode='auto', period=1)
    model.fit([X_train, kX_train], y_train, batch_size=batch_size, epochs=epochs,
              validation_data=([X_dev, kX_dev], y_dev), callbacks=[e], verbose=1)
    model.load_weights(model_file)

    if return_model == True:
        return model
    else:
        test_predictions = model.predict([X_test, kX_test], verbose=False)
        val_predictions = model.predict([X_dev, kX_dev], verbose=False)
        if return_probs == False:
            test_predictions = [np.argmax(pred) for pred in test_predictions]
            val_predictions = [np.argmax(pred) for pred in val_predictions]
        return test_predictions, val_predictions'''
