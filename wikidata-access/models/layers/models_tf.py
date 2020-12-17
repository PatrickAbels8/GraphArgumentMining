from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from scipy import spatial

def get_cosine_sim_input(data, topics, embedding_weights):
    result = []
    for j in range(len(data)):
        sent = data[j]
        topic = topics[j]
        sent_cosine = []
        for i in range(len(sent)):
            if sent[i] > 0:
                index = sent[i]
                word_vec = embedding_weights[index]
                sim = [1 - spatial.distance.cosine(topic, word_vec)]
            else:
                sim = [0.0]
            sent_cosine.append(sim)
        result.append(np.array(sent_cosine))
    return np.array(result)

def DynamicBiCRNN(x, X_topic, max_len, lstm_size, name="biclstm"):
    lstm_fw_cell = CLSTMCell(lstm_size, forget_bias=1.0)
    lstm_bw_cell = CLSTMCell(lstm_size, forget_bias=1.0)

    with tf.variable_scope(name):
        (output_fw, output_bw), (last_output_fw, last_output_bw) = bidirectional_dynamic_crnn(lstm_fw_cell, lstm_bw_cell, x, X_topic,
                                                         dtype=tf.float32)

    return tf.concat([last_output_fw[1], last_output_bw[1]], axis=-1)

def BiRNN(x, max_len, lstm_size, name="bilstm"):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
    x = tf.unstack(x, max_len, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0)

    # Get lstm cell output
    with tf.variable_scope(name):
        try:
            outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                         dtype=tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                   dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return outputs

def DynamicBiRNN(x, max_len, lstm_size, name="bilstm"):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
    #x = tf.unstack(x, max_len, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(lstm_size)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(lstm_size)

    # Get lstm cell output
    with tf.variable_scope(name):
        (output_fw, output_bw), (last_output_fw, last_output_bw) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                         dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.concat([output_fw, output_bw], axis=2), tf.concat([last_output_fw[1], last_output_bw[1]], axis=-1)