from keras import Input, Model
from keras.layers import Embedding, Bidirectional, Dropout, LSTM, Dense, Lambda, TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from models.layers.attention_keras import attention_knowledge
import os
import random as rn

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
    emb_sents = np.load(PROCESSED_DIR + "index_to_vec_we"+kwargs['model_settings']['word_embeddings'][1]+".npy")
    emb_knowledge = np.load(PROCESSED_DIR + "index_to_vec_kge"+kwargs['model_settings']['kg_embeddings'][1]+".npy")

    # load data
    X_train, X_dev, X_test = data["X_train"], data["X_dev"], data["X_test"] # [samples, sent_len]
    kX_train, kX_dev, kX_test = data["kX_train"], data["kX_dev"], data["kX_test"] # [samples, sent_len, max_concepts]
    y_train, y_dev, y_test = data["y_train"], data["y_dev"], data["y_test"]
    val_y_non_one_hot = [np.argmax(pred) for pred in y_dev]

    # some constants
    sent_len = X_train.shape[1]
    max_concepts = kX_train.shape[2]
    num_labels = y_train.shape[1]
    attention_size = kwargs['model_settings'].get('attention_size', emb_sents.shape[1])

    ############################
    #   KNOWLEDGE PROCESSING   #
    ############################

    # input for all concepts of a sentence
    sentence_inputs = Input(shape=(sent_len, ), dtype='int32', name="sentence_inputs")
    knowledge_inputs = Input(shape=(sent_len, max_concepts,), dtype='int32', name="knowledge_inputs")

    emb_knowledge_ids = Embedding(emb_knowledge.shape[0], emb_knowledge.shape[1], mask_zero=True,
                               weights=[emb_knowledge], trainable=train_embeddings)(knowledge_inputs) # [samples, sent_len, max_concepts, kge_dim]

    embedded_word_ids = Embedding(emb_sents.shape[0], emb_sents.shape[1], mask_zero=True,
                               weights=[emb_sents], trainable=train_embeddings,
                               input_length=sent_len)(sentence_inputs) # [samples, sent_len, we_dim]

    attended_knowledge = attention_knowledge(embedded_word_ids, None, attention_size, return_alphas=False, summed_up=True)(emb_knowledge_ids)


    concat_sequences = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([embedded_word_ids, attended_knowledge])

    # define bilstm + dropout
    sent_bilstm = Bidirectional(LSTM(lstm_size, input_shape=(None, sent_len, 150)))(concat_sequences)
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

    # attention
    # initial_outputs = tf.TensorArray(dtype=tf.float32, size=sent_len, element_shape=(None, 100), infer_shape=False)
    # initial_outputs = tf.Variable([])

    # def slice_t(x):
    #   return x[:, 0,:,:]

    # def slice_zero(x):
    #    return x[:,0,:]

    # slize_t_layer = Lambda(slice_t)
    # slize_zero_layer = Lambda(slice_zero)
    """
    @autograph.convert()
    def loop(tensor):
        initial_outputs = []
        for t in range(sent_len):
            kX_sliced = tensor[:, 0,:,:] # [samples, token at t, concepts, concept_embs]
            initial_outputs.append(kX_sliced[:,0,:])
        return K.stack(initial_outputs, 0)

    layer = tf.keras.layers.Lambda(loop, output_shape=(sent_len, None, 100))
    initial_outputs = layer(concept_embedded_layer)


    initial_t = tf.constant(0)
    initial_outputs = tf.TensorArray(dtype=tf.float32, size=sent_len)
    #shit = tf.placeholder(tf.float32, [None, emb_knowledge.shape[1]])
    condition = lambda t, _: tf.less(t, sent_len)
    #body = lambda i: tf.add(i, 1)
    def body(t, outputs_):
        kX_sliced = Lambda(lambda x: x[:, t, :, :], output_shape=(None, max_concepts, 100))(concept_embedded_layer)
        kX_sliced_temp = Lambda(lambda x: x[:, 0, :], output_shape=(None, 100))(
            kX_sliced)
        outputs_ = outputs_.write(t, kX_sliced_temp) #[ [samples, kge], [samples, kge], ...] =>[sent_len, samples, kge]
        return t + 1, outputs_

    # do the loop:
    _, outputs = tf.while_loop(condition, body, loop_vars=[initial_t, initial_outputs])#,

    initial_outputs = []
    for t in range(sent_len):
        if t == 0:
            with tf.variable_scope("tf", reuse=False):
                #kX_sliced = concept_embedded_layer[:, t, :, :]  # [samples, token at t, concepts, concept_embs]
                kX_sliced = Lambda(lambda x: x[:, t, :, :], output_shape=(None, max_concepts, 100))(concept_embedded_layer)
                X_token = sent_embedded[:, t, :]
                # temp = AttentionHerrmann_2(representation_claim=X_token, only_attended_vector=True, topic_size=50,
                #                  summed_up=True, self_attention=False, name='attention',
                #                  bias_1=True)(kX_sliced)
                # temp = attention_knowledge(kX_sliced, X_token, attention_size, return_alphas=False)
                kX_sliced_temp = Lambda(lambda x: x[:, 0, :], output_shape=(None, 100))(
                    kX_sliced)
                initial_outputs.append(kX_sliced_temp)
        else:
            with tf.variable_scope("tf", reuse=True):
                #kX_sliced = concept_embedded_layer[:, t, :, :]  # [samples, token at t, concepts, concept_embs]
                kX_sliced = Lambda(lambda x: x[:, t, :, :], output_shape=(None, max_concepts, 100))(
                    concept_embedded_layer)
                X_token = sent_embedded[:, t, :]
                # temp = AttentionHerrmann_2(representation_claim=X_token, only_attended_vector=True, topic_size=50,
                #                 summed_up=True, self_attention=False, name='attention',
                #                  bias_1=True)(kX_sliced)
                # temp = attention_knowledge(kX_sliced, X_token, attention_size, return_alphas=False)
                kX_sliced_temp = Lambda(lambda x: x[:, 0, :], output_shape=(None, 100))(
                    kX_sliced)
                initial_outputs.append(kX_sliced_temp)
    initial_outputs = K.stack(initial_outputs, 0)
    attended_knowledge = K.permute_dimensions(initial_outputs, (1, 0, 2))


    #shit = tf.placeholder(tf.float32, [None, emb_knowledge.shape[1]])
    condition = lambda t, _: tf.less(t, sent_len)
    #body = lambda i: tf.add(i, 1)
    def body(t, outputs_):
        kX_sliced = concept_embedded_layer[:, t,:,:] # [samples, token at t, concepts, concept_embs]
        X_token = sent_embedded[:, t,:]
        #outputs_=outputs_[t].assign(attention_knowledge(kX_sliced, X_token, attention_size, return_alphas=False))
        outputs_ = outputs_.write(t, attention_knowledge(kX_sliced, X_token, attention_size, return_alphas=False)) #[ [samples, kge], [samples, kge], ...] =>[sent_len, samples, kge]
        return t + 1, outputs_

    # do the loop:
    _, outputs = tf.scan(body, (), loop_vars=[initial_t, initial_outputs],
                               #shape_invariants=[initial_t.get_shape(), tf.TensorShape([None, sent_len])]
                               )#,"""

    # stack = initial_outputs.stack() # [sent_len, samples, kge_dim]
    # attended_knowledge = tf.transpose(initial_outputs, [1,0,2]) #[samples, sent_len, kge_dim]
    # attended_knowledge = Lambda(lambda x: tf.transpose(x, [1,0,2]))(initial_outputs) #[samples, sent_len, kge_dim]
    # initial_outputs = Lambda(lambda x: x.stack(), output_shape=(60, None, 100))(outputs)
    # initial_outputs = initial_outputs.stack()
    # initial_outputs = K.stack(initial_outputs, 0)