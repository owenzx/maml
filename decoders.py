""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf

from tensorflow.python.platform import flags
from tensorflow.python.ops.rnn import _reverse_seq #at least this can be imported for tensorflow 1.10
from utils import mse, xent, conv_block, normalize, read_pretrained_embeddings, rnncell, build_cells, xent_onehot, bi_seq_xent, seq_xent, get_approx_2nd_grad, convert_list_to_tensor, convert_tensor_to_list
from customized_cells import Customized_BasicLSTMCell
from bidaf import BiAttention_no_var
from multihead_attention import multihead_attention_no_var
from constants import *
from my_static_brnn import my_static_bidirectional_rnn

from batch_ops import batch_embedding_lookup, batch_matmul, batch_embedding_lookup_2

FLAGS = flags.FLAGS


class lstm_att_decoder():
    def __init__(self, dim_hidden, dim_input, cells, max_len):
        self.dim_hidden = dim_hidden
        self.dim_input = dim_input
        self.cells = cells       
        self.max_len = max_len
        self.num_layers = len(self.cells)
        self.use_attention = FLAGS.decoder_attention
        

    def __call__(self, init_state, att_vecs, att_mask, weights, reuse=None):
        out_seq =[[]] * self.max_len
        with tf.variable_scope('lstm_att_decoder', reuse=None):
            state = init_state
            output_label = [BOS_TOKEN]*FLAGS.meta_batch_size
            for i in range(self.max_len):
                #print(output_label.shape)
                cur_emb = batch_embedding_lookup_2(weights['emb'], output_label)
                (output, state) = self.cells[0](cur_emb, state)
                if self.use_attention:
                    att = self.calc_attention_vec(output, att_vecs, att_mask, weights)
                    output = batch_matmul(tf.concat([output, att],axis=-1), weights['decoder_att_w'])
                w_sm = tf.transpose(weights['emb'], perm=[0,2,1])
                logits = batch_matmul(output, w_sm)
                out_seq[i] = logits
                out_label = tf.argmax(logits, 1)
            out_seq_tensor = tf.transpose(tf.stack(out_seq), perm=[1,0,2])
        return out_seq_tensor

    
    def calc_attention_vec(self, h, att_vecs, att_mask, weights):
        h = tf.expand_dims(h, axis=1)
        att_vecs = tf.transpose(att_vecs, perm=[1,0,2])
        return tf.squeeze(multihead_attention_no_var(h, att_vecs, keys_mask=att_mask, num_units=self.dim_hidden, num_heads=FLAGS.num_attn_head, is_training=True, scope="decoder_att", weights=weights, prefix="decoder_att_"))

