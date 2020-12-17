import tensorflow as tf
from keras import initializers
from keras.engine.topology import Layer
from keras import backend as K

class attention_knowledge(Layer):
    """
    Keras Layer that implements an Attention mechanism, with a context/query vector,
    for temporal data. Supports Masking. Follows the work of Yang et al.
    [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf] "Hierarchical Attention Networks for Document Classification"

    Taken from the following source with permission of the author:
        https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2

    only_attended_vector: If true, returns only attended vector, otherwise returns also the weights  as [attended_vector, weights}
    summed_up: If true, sums up the weighted word representations, otherwise returns the sequence
    representation_claim: topic vector, e.g. the last hidden state of the claim BiLSTM. Has to have the size as the hidden states of the
                            evid/doc BiLSTMs

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    :param kwargs:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.

    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self, token_input, topic_input, attention_dim, return_alphas=False, summed_up=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.return_alphas = return_alphas
        self.token_input = token_input  # representation vector of the claim
        self.topic_input = topic_input

        self.summed_up = summed_up
        self.attention_dim = attention_dim
        super(attention_knowledge, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 4

        # weights for the sentences of original docs / evidences
        # TODO weights that are defined as trainable, have to be used or None value error due to gradients occur
        self.W_private = self.add_weight((input_shape[3], self.attention_dim,),
                                         initializer=self.init,
                                         name='{}_W_private'.format(self.name))

        self.W_token = self.add_weight((self.token_input.shape[2].value, self.attention_dim,),
                                         initializer=self.init,
                                         name='{}_W_token'.format(self.name))

        if self.topic_input != None:
            self.W_topic = self.add_weight((self.token_input.shape[2].value, self.attention_dim,),
                                           initializer=self.init,
                                           name='{}_W_token'.format(self.name))

        self.bias = self.add_weight((self.attention_dim,), initializer=self.init,
                                    name='{}_bias'.format(self.name))


        self.w_reduce = self.add_weight((self.attention_dim,),
                                        initializer=self.init,
                                        name='{}_w_reduce'.format(self.name))

        super(attention_knowledge, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, knowledge_inputs, mask=None):
        a = tf.tensordot(knowledge_inputs, self.W_private, axes=((3), (0)))
        b = K.expand_dims(tf.tensordot(self.token_input, self.W_token, axes=((2), (0))), 2)

        if self.topic_input != None:
            c = K.expand_dims(K.expand_dims(tf.tensordot(self.topic_input, self.W_topic, axes=1), 1), 1) # = axes=((2), (0))
            v = K.tanh(a + b + c + self.bias)
        else:
            v = K.tanh(a + b + self.bias)

        vw = tf.tensordot(v, self.w_reduce, axes=((3), (0)))

        alphas = K.exp(vw)  # Softmax part 1

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            alphas *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number epsilon to the sum.
        alphas /= K.cast(K.sum(alphas, axis=2, keepdims=True) + K.epsilon(), K.floatx())  # Softmax part 2
        # or K.max(sum(...), K.epsilon()) from comments https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2

        exp_alpha = K.expand_dims(alphas)  # (bs, 50, 1)
        output = knowledge_inputs * exp_alpha

        if self.summed_up == True:
            final = tf.reduce_sum(output, 2)# (bs, 200)
        else:
            final = output

        if self.return_alphas == False:
            return final
        else:
            return [final, alphas]

    def compute_output_shape(self, input_shape):
        final_out = tuple([input_shape[0], input_shape[1], input_shape[-1]])

        if self.summed_up == False:
            final_out = tuple([input_shape[0], input_shape[1], input_shape[2], input_shape[-1]])

        if self.return_alphas == False:
            return final_out
        else:
            return [final_out, tuple([input_shape[0], input_shape[1], input_shape[2]])]


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionHerrmann(Layer):
    """
    Keras Layer that implements an Attention mechanism, with a context/query vector,
    for temporal data. Supports Masking. Follows the work of Yang et al.
    [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf] "Hierarchical Attention Networks for Document Classification"

    Taken from the following source with permission of the author:
        https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2

    only_attended_vector: If true, returns only attended vector, otherwise returns also the weights  as [attended_vector, weights}
    summed_up: If true, sums up the weighted word representations, otherwise returns the sequence
    representation_claim: topic vector, e.g. the last hidden state of the claim BiLSTM. Has to have the size as the hidden states of the
                            evid/doc BiLSTMs

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    :param kwargs:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.

    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self, representation_claim=None, only_attended_vector=False, topic_shape=None, #see hiac_biLSTM
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None, self_attention=False,
                 bias_1=True, bias_2=False, summed_up=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.only_attended_vector = only_attended_vector
        self.r_c = representation_claim  # representation vector of the claim

        self.bias_1 = bias_1
        self.bias_2 = bias_2
        self.summed_up = summed_up
        self.self_attention = self_attention
        self.topic_shape = topic_shape
        super(AttentionHerrmann, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        if self.topic_shape == None:
            self.topic_shape = input_shape

        # weights for the sentences of original docs / evidences
        # TODO weights that are defined as trainable, have to be used or None value error due to gradients occur
        self.W_s = self.add_weight((input_shape[-1], input_shape[-1],),
                                   initializer=self.init,
                                   name='{}_W_s'.format(self.name))

        if self.self_attention == False:
            self.W_rc = self.add_weight((input_shape[-1], int(self.topic_shape[-1]),),
                                        initializer=self.init,
                                        name='{}_W_rc'.format(self.name))

        if self.bias_1:
            self.b_1 = self.add_weight((input_shape[-1],),
                                       initializer='zero',
                                       name='{}_b_1'.format(self.name))
        if self.bias_2:
            self.b_2 = self.add_weight((input_shape[1],),
                                       initializer='zero',
                                       name='{}_b_2'.format(self.name))

        self.w_m = self.add_weight((input_shape[-1],),
                                   initializer=self.init,
                                   name='{}_w_m'.format(self.name))

        super(AttentionHerrmann, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):  # W_s = W_rc = (200, 200), x = (bs, 50, 200)
        # TODO ggf hier mit transpose usw. arbeiten wie christian stab, weil falsch herum multipliziert wird
        m_a = dot_product(x, self.W_s)  # (bs, 50, 200) * (200, 200) = (bs, 50, 200)

        if self.self_attention == False:
            m_b = dot_product(self.r_c, self.W_rc)  # (bs, 200) * (200, 200) = (bs, 200)
            # m_b = K.reshape(m_b, [-1, 1, K.shape(x)[2]]) # (bs, 1, 200), same as expand_dims below (same results)
            m_b = K.expand_dims(m_b, axis=1)  # (bs, 1, 200)
            m_ab = m_a + m_b  # (bs, 50, 200) + (bs, 1, 200) => element wise addition? => (50, 200)
        else:
            m_ab = m_a

        if self.bias_1:  # (200,)
            m_ab += self.b_1  # (bs, 50, 200) + (200,) => element wise addition? => (bs, 50, 200)

        m = K.tanh(m_ab)  # (bs, 50, 200)
        s = dot_product(m, self.w_m)  # (bs, 50, 200) * (200,) = (bs, 50)

        if self.bias_2:  # (50,)
            s += self.b_2  # (bs, 50) + (50,) = (bs, 50)

        s = K.exp(s)  # Softmax part 1

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            s *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        s /= K.cast(K.sum(s, axis=1, keepdims=True) + K.epsilon(), K.floatx())  # Softmax part 2 TODO check this
        # or K.max(sum(...), K.epsilon()) from comments https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2

        s_exp = K.expand_dims(s)  # (bs, 50, 1)
        weighted_input = x * s_exp  # (bs, 50, 200) * (bs, 50, 1) = (bs, 50, 200) => element wise multiplication?

        if self.summed_up == True:
            final = K.sum(weighted_input, axis=1)  # (bs, 200)
        else:
            final = weighted_input

        if self.only_attended_vector == True:
            return final
        else:
            return [final, s]

    def compute_output_shape(self, input_shape):
        final_out = tuple([input_shape[0], input_shape[-1]])

        if self.summed_up == False:
            final_out = tuple([input_shape[0], input_shape[1], input_shape[2]])

        if self.only_attended_vector == True:
            return final_out
        else:
            return [final_out, tuple([input_shape[0], input_shape[1]])]
