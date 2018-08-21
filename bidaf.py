#borrowed from https://github.com/allenai/document-qa
import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def compute_attention_mask(x_mask, mem_mask, x_word_dim, key_word_dim):
    """ computes a (batch, x_word_dim, key_word_dim) bool mask for clients that want masking """
    if x_mask is None and mem_mask is None:
        return None
    elif x_mask is None or mem_mask is None:
        raise NotImplementedError()

    #x_mask = tf.sequence_mask(x_mask, x_word_dim)
    #mem_mask = tf.sequence_mask(mem_mask, key_word_dim)
    join_mask = tf.logical_and(tf.expand_dims(x_mask, 2), tf.expand_dims(mem_mask, 1))
    return join_mask


class _WithBias_no_var():
    def __init__(self, bias: bool):
        # Note since we typically do softmax on the result, having a bias is usually redundant
        self.bias = bias

    def get_scores(self, tensor_1, tensor_2, weights=None, prefix=""):
        out = self._distance_logits(tensor_1, tensor_2, weights, prefix)
        if self.bias:
            #bias = tf.get_variable("bias", shape=(), dtype=tf.float32)
            bias = weights[prefix+"bias"]
            out += bias
        return out

    def _distance_logits(self, tensor_1, tensor_2):
        raise NotImplemented()


class TriLinear_no_var(_WithBias_no_var):
    """ Function used by BiDaF, bi-linear with an extra component for the dots of the vectors """
    def __init__(self, init="glorot_uniform", bias=False):
        super().__init__(bias)
        self.init = init

    def _distance_logits(self, x, keys, weights=None, prefix=""):
        #init = get_keras_initialization(self.init)

        #key_w = tf.get_variable("key_w", shape=keys.shape.as_list()[-1], initializer=init, dtype=tf.float32)
        key_w = weights[prefix+"key_w"]
        #print(keys.shape)
        #print(key_w.shape)
        key_logits = tf.tensordot(keys, key_w, axes=[[2], [0]])  # (batch, key_len)

        #x_w = tf.get_variable("input_w", shape=x.shape.as_list()[-1], initializer=init, dtype=tf.float32)
        x_w = weights[prefix+"input_w"]
        x_logits = tf.tensordot(x, x_w, axes=[[2], [0]])  # (batch, x_len)

        #dot_w = tf.get_variable("dot_w", shape=x.shape.as_list()[-1], initializer=init, dtype=tf.float32)
        dot_w = weights[prefix+"dot_w"]

        # Compute x * dot_weights first, the batch mult with x
        x_dots = x * tf.expand_dims(tf.expand_dims(dot_w, 0), 0)
        dot_logits = tf.matmul(x_dots, keys, transpose_b=True)

        return dot_logits + tf.expand_dims(key_logits, 1) + tf.expand_dims(x_logits, 2)

    @property
    def version(self):
        return 1

class BiAttention_no_var():
    """ Bi-Directonal Attention from https://arxiv.org/abs/1611.01603 """

    def __init__(self, sim=TriLinear_no_var()):
        self.sim = sim
        self.q2c = FLAGS.q2c
        self.query_dots = FLAGS.query_dots

    def apply(self, is_train, x, keys, memories, x_mask=None, mem_mask=None, weights=None, prefix=""):
        VERY_NEGATIVE_NUMBER = -1e29
        x_word_dim = tf.shape(x)[1]
        key_word_dim = tf.shape(keys)[1]

        dist_matrix = self.sim.get_scores(x, keys, weights, prefix)
        joint_mask = compute_attention_mask(x_mask, mem_mask, x_word_dim, key_word_dim)
        #print(dist_matrix.shape)
        #print(joint_mask.shape)
        if joint_mask is not None:
            dist_matrix += VERY_NEGATIVE_NUMBER * (1 - tf.cast(joint_mask, dist_matrix.dtype))
        query_probs = tf.nn.softmax(dist_matrix)  # probability of each mem_word per x_word

        # Batch matrix multiplication to get the attended vectors
        select_query = tf.matmul(query_probs, memories)  # (batch, x_words, q_dim)

        if not self.q2c:
            if self.query_dots:
                return tf.concat([x, select_query, x * select_query], axis=2)
            else:
                return tf.concat([x, select_query], axis=2)

        # select query-to-context
        context_dist = tf.reduce_max(dist_matrix, axis=2)  # (batch, x_word``s)
        context_probs = tf.nn.softmax(context_dist)  # (batch, x_words)
        select_context = tf.einsum("ai,aik->ak", context_probs, x)  # (batch, x_dim)
        select_context = tf.expand_dims(select_context, 1)

        if self.query_dots:
            return tf.concat([x, select_query, x * select_query, x * select_context], axis=2)
        else:
            return tf.concat([x, select_query, x * select_context], axis=2)


