import tensorflow as tf
import numpy as np
from keras import initializers, regularizers, constraints, activations
from keras.engine.topology import Layer
from keras import backend as K

def attention_knowledge(sentence_inputs, topic_input, attention_size, return_alphas=False):
    """
    Modified from: https://github.com/ilivans/tf-rnn-attention/blob/master/attention.py
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article

    Args:
        inputs_private, inputs_shared: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    `[batch_size, max_time, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    outputs_fw is a `Tensor` shaped:
                    `[batch_size, max_time, cell_fw.output_size]`
                    and outputs_bw is a `Tensor` shaped:
                    `[batch_size, max_time, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    # inputs_private.shape = (None, 60, 192) if lstm size is 96
    hidden_size = sentence_inputs.shape[2].value  # H value - hidden size of the RNN layer
    # inputs_topic.shape = (?, 300) => why "?"
    embedding_size = topic_input.shape[1].value


    # => see https://github.com/tensorflow/tensorflow/issues/8604

    with tf.name_scope('v'):
        # Trainable parameters
        # W_private = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        # W_private = tf.get_variable("W_private", initializer=tf.random_normal([hidden_size, attention_size], stddev=0.1))
        # W_topic = tf.Variable(tf.random_normal([embedding_size, attention_size], stddev=0.1))
        # bias = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        # w_reduce = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        W_private = tf.get_variable("W_private", shape=(hidden_size, attention_size),
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        W_topic = tf.get_variable("W_topic", shape=(embedding_size, attention_size),
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias", shape=(attention_size), initializer=tf.truncated_normal_initializer(stddev=0.1))
        w_reduce = tf.get_variable("w_reduce", shape=(attention_size),
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        # => see https://stackoverflow.com/questions/45789822/tensorflow-creating-variables-in-fn-of-tf-map-fn-returns-value-error

        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,H)*(H,A)=(B,T,A), where A=attention_size (A should be H?)
        #W_private_print = tf.Print(W_private, [W_private, tf.shape(W_private)])
        a = tf.tensordot(sentence_inputs, W_private, axes=1)
        b = tf.expand_dims(tf.tensordot(topic_input, W_topic, axes=1), 1)
        v = tf.tanh(a + b + bias)

        # For each of the timesteps its vector of size A from `v` is reduced with `u` vector
        vw = tf.tensordot(v, w_reduce, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vw, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,H) shape
        # (B,T,H)*(B,T,1) => (B,T,H) w/ T's of input multiplied by alphas => reduce sum of T => (B,H)
        exp_alpha = tf.expand_dims(alphas, -1)
        output = sentence_inputs * exp_alpha
        output = tf.reduce_sum(output, 1)

        if not return_alphas:
            return output
        else:
            return output, alphas

def attention_knowledge_4dim(knowledge_inputs, token_input, attention_dim, return_alphas=False):
    """
    Modified from: https://github.com/ilivans/tf-rnn-attention/blob/master/attention.py
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article

    Args:
        inputs_private, inputs_shared: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    `[batch_size, max_time, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    outputs_fw is a `Tensor` shaped:
                    `[batch_size, max_time, cell_fw.output_size]`
                    and outputs_bw is a `Tensor` shaped:
                    `[batch_size, max_time, cell_bw.output_size]`.
        attention_dim: Linear size of the Attention weights.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    # inputs_private.shape = (None, 60, 192) if lstm size is 96
    kge_dim = knowledge_inputs.shape[3].value  # H value - hidden size of the RNN layer
    # inputs_topic.shape = (?, 300) => why "?"
    embedding_size = token_input.shape[2].value


    # => see https://github.com/tensorflow/tensorflow/issues/8604

    with tf.name_scope('v'):
        # Trainable parameters
        W_private = tf.get_variable("W_private", shape=(kge_dim, attention_dim),
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        W_token = tf.get_variable("W_token", shape=(embedding_size, attention_dim),
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias", shape=(attention_dim), initializer=tf.truncated_normal_initializer(stddev=0.1))
        w_reduce = tf.get_variable("w_reduce", shape=(attention_dim),
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        # => see https://stackoverflow.com/questions/45789822/tensorflow-creating-variables-in-fn-of-tf-map-fn-returns-value-error

        # [bs, sent_len, max_concepts, kge_dim] x [kge_dim, attention_dim] = [bs, sent_len, max_concepts, attention_dim] (bs, 60, 8, 50)
        a = tf.tensordot(knowledge_inputs, W_private, axes=((3), (0)))

        # [bs, sent_len, we_dim] x [we_dim, attention_dim] = [bs, sent_len, attention_dim] (bs, 60, 50)
        # expand dim: [bs, sent_len, 1, attention_dim] (bs, 60, 1, 50)
        b = tf.expand_dims(tf.tensordot(token_input, W_token, axes=((2), (0))), 2)

        # bias: [attention_dim] (50) => [bs, sent_len, max_concepts, attention_dim] (bs, 60, 8, 50)
        v = tf.tanh(a + b + bias)

        # In 3d with while loop: [bs, max_concepts, attention_dim] x [attention_dim] = [bs, max_concepts] (bs, 8)
        # In 4d without while loop: [bs, sent_len, max_concepts, attention_dim] x [attention_dim] = [bs, sent_len, max_concepts] (bs, 60, 8)
        vw = tf.tensordot(v, w_reduce, axes=((3), (0)), name='vu')
        alphas = tf.nn.softmax(vw, name='alphas')

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,H) shape
        # (B,T,H)*(B,T,1) => (B,T,H) w/ T's of input multiplied by alphas => reduce sum of T => (B,H)

        # [bs, sent_len, max_concepts] = [bs, sent_len, max_concepts, 1] (bs, 60, 8, 1)
        exp_alpha = tf.expand_dims(alphas, -1)

        # [bs, sent_len, max_concepts, attention_dim] * [bs, sent_len, max_concepts, 1] =  [bs, sent_len, max_concepts, attention_dim]
        output = knowledge_inputs * exp_alpha

        # [bs, sent_len, max_concepts, attention_dim] => [bs, sent_len, attention_dim]
        output = tf.reduce_sum(output, 2)

        if not return_alphas:
            return output
        else:
            return output, alphas

def attention_topic_and_knowledge(sentence_inputs, topic_input, token_input, attention_size, return_alphas=False):
    """
    Modified from: https://github.com/ilivans/tf-rnn-attention/blob/master/attention.py
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article

    Args:
        inputs_private, inputs_shared: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    `[batch_size, max_time, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    outputs_fw is a `Tensor` shaped:
                    `[batch_size, max_time, cell_fw.output_size]`
                    and outputs_bw is a `Tensor` shaped:
                    `[batch_size, max_time, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    # inputs_private.shape = (None, 60, 192) if lstm size is 96
    hidden_size = sentence_inputs.shape[2].value  # H value - hidden size of the RNN layer
    # inputs_topic.shape = (?, 300) => why "?"
    embedding_size = topic_input.shape[1].value


    # => see https://github.com/tensorflow/tensorflow/issues/8604

    with tf.name_scope('v'):
        # Trainable parameters
        # W_private = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        # W_private = tf.get_variable("W_private", initializer=tf.random_normal([hidden_size, attention_size], stddev=0.1))
        # W_topic = tf.Variable(tf.random_normal([embedding_size, attention_size], stddev=0.1))
        # bias = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        # w_reduce = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        W_private = tf.get_variable("W_private", shape=(hidden_size, attention_size),
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        W_topic = tf.get_variable("W_topic", shape=(embedding_size, attention_size),
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        W_token = tf.get_variable("W_token", shape=(embedding_size, attention_size),
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias", shape=(attention_size), initializer=tf.truncated_normal_initializer(stddev=0.1))
        w_reduce = tf.get_variable("w_reduce", shape=(attention_size),
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        # => see https://stackoverflow.com/questions/45789822/tensorflow-creating-variables-in-fn-of-tf-map-fn-returns-value-error

        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,H)*(H,A)=(B,T,A), where A=attention_size (A should be H?)
        #W_private_print = tf.Print(W_private, [W_private, tf.shape(W_private)])
        a = tf.tensordot(sentence_inputs, W_private, axes=1)
        b = tf.expand_dims(tf.tensordot(topic_input, W_topic, axes=1), 1)
        c = tf.expand_dims(tf.tensordot(token_input, W_token, axes=1), 1)
        v = tf.tanh(a + b + c + bias)

        # For each of the timesteps its vector of size A from `v` is reduced with `u` vector
        vw = tf.tensordot(v, w_reduce, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vw, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,H) shape
        # (B,T,H)*(B,T,1) => (B,T,H) w/ T's of input multiplied by alphas => reduce sum of T => (B,H)
        exp_alpha = tf.expand_dims(alphas, -1)
        output = sentence_inputs * exp_alpha
        output = tf.reduce_sum(output, 1)

        if not return_alphas:
            return output
        else:
            return output, alphas

def attention(inputs_private, inputs_shared, inputs_topic, attention_size, return_alphas=False):
    """
    Modified from: https://github.com/ilivans/tf-rnn-attention/blob/master/attention.py
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article

    Args:
        inputs_private, inputs_shared: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    `[batch_size, max_time, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    outputs_fw is a `Tensor` shaped:
                    `[batch_size, max_time, cell_fw.output_size]`
                    and outputs_bw is a `Tensor` shaped:
                    `[batch_size, max_time, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    # inputs_private.shape = (None, 60, 192) if lstm size is 96
    hidden_size = inputs_private.shape[2].value  # H value - hidden size of the RNN layer
    # inputs_topic.shape = (?, 300) => why "?"
    embedding_size = inputs_topic.shape[1].value

    # Trainable parameters
    W_private = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    W_shared = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    W_topic = tf.Variable(tf.random_normal([embedding_size, attention_size], stddev=0.1))
    bias = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    w_reduce = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,H)*(H,A)=(B,T,A), where A=attention_size (A should be H?)
        a = tf.tensordot(inputs_private, W_private, axes=1)
        b = tf.tensordot(inputs_shared, W_shared, axes=1)
        c = tf.expand_dims(tf.tensordot(inputs_topic, W_topic, axes=1), 1)
        v = tf.tanh(a + b + c + bias)

    # For each of the timesteps its vector of size A from `v` is reduced with `u` vector
    vw = tf.tensordot(v, w_reduce, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vw, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,H) shape
    # (B,T,H)*(B,T,1) => (B,T,H) w/ T's of input multiplied by alphas => reduce sum of T => (B,H)
    exp_alpha = tf.expand_dims(alphas, -1)
    output = inputs_private * exp_alpha
    output = tf.reduce_sum(output, 1)

    if not return_alphas:
        return output
    else:
        return output, alphas