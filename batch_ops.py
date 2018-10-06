import tensorflow as tf
import numpy as np

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

def batch_embedding_lookup(w, ids):
    w_t = tf.transpose(w, perm=[1,0,2])
    tmp = tf.transpose(tf.nn.embedding_lookup(w_t, ids), perm=[0,2,1,3])
    result = tf.gather_nd(tmp, [[i,i] for i in range(FLAGS.meta_batch_size//FLAGS.gpu_num)])
    return result
    

def batch_embedding_lookup_2(w,ids):
    w_t = tf.transpose(w, perm=[1,0,2])
    tmp = tf.transpose(tf.nn.embedding_lookup(w_t, ids), perm=[1,0,2])
    result = tf.gather_nd(tmp, [[i,i] for i in range(FLAGS.meta_batch_size//FLAGS.gpu_num)])
    return result

def batch_matmul(x, w):
    x_tmp = tf.expand_dims(x, axis=1)
    result = tf.matmul(x_tmp,w)
    return tf.squeeze(result)
