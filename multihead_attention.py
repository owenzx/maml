# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Borrowed from https://www.github.com/kyubyong/transformer
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
'''

from __future__ import print_function
import tensorflow as tf
from batch_ops import batch_matmul

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

def multihead_attention_no_var(queries, 
                        keys, 
                        queries_mask=None,
                        keys_mask=None,
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None,
                        weights=None,
                        prefix=""):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        #Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        #K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        #V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        batch_size = tf.shape(queries)[0]
        queries_len = tf.shape(queries)[1]
        keys_len = tf.shape(keys)[1]



        if FLAGS.batch_mode:
            Q = tf.nn.relu(tf.matmul(queries, weights[prefix+"q"]))
            K = tf.nn.relu(tf.matmul(keys, weights[prefix+"k"]))
            V = tf.nn.relu(tf.matmul(keys, weights[prefix+"v"]))
        else:
            queries_flat = tf.reshape(queries,[batch_size*queries_len,-1])
            keys_flat = tf.reshape(keys,[batch_size*keys_len,-1])
            Q = tf.nn.relu(tf.matmul(queries_flat, weights[prefix+"q"]))
            K = tf.nn.relu(tf.matmul(keys_flat, weights[prefix+"k"]))
            V = tf.nn.relu(tf.matmul(keys_flat, weights[prefix+"v"]))

            Q = tf.reshape(Q, [batch_size, queries_len, num_units])
            K = tf.reshape(K, [batch_size, keys_len, num_units])
            V = tf.reshape(V, [batch_size, keys_len, num_units])

        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        if FLAGS.batch_mode:
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        else:
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        K_dim_float = tf.to_float(tf.shape(K_)[-1])
        outputs = outputs / (tf.sqrt(K_dim_float))
        
        # Key Masking
        if keys_mask is not None:
            key_masks = keys_mask
        else:
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(tf.cast(key_masks,dtype=tf.int32), 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        if queries_mask is not None:
            query_masks = queries_mask
        else:
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        
        outputs *= tf.cast(query_masks, dtype=tf.float32) # broadcasting. (N, T_q, C)
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        if FLAGS.batch_mode:
            outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        else:
            outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        #outputs = normalize(outputs) # (N, T_q, C)
 
    return outputs
