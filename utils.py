""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags
from customized_cells import Customized_BasicLSTMCell

FLAGS = flags.FLAGS

## Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    if FLAGS.max_pool:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
    if FLAGS.max_pool:
        normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
    return normed

def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def seq_xent(pred, label, mask):
    """pred:  [batch_size, seq_len, vocab],
       label: [batch_size, seq_len],
       mask:  [batch_size, seq_len]"""
    mask = tf.cast(mask, pred.dtype)
    return tf.contrib.seq2seq.sequence_loss(logits=pred, targets=label, weights=mask, average_across_timesteps=True, average_across_batch=True)

def get_bi_label(label):
    print(type(label))
    batch_size = tf.shape(label)[0]
    one_step_zero = tf.zeros((batch_size, 1), label.dtype)
    fw_label = tf.concat([label[:,1:], one_step_zero],axis=-1)
    bw_label = tf.concat([one_step_zero, label[:,:-1]], axis=-1)
    return fw_label, bw_label

def bi_seq_xent(pred, label, mask):
    pred_fw, pred_bw = pred
    mask_fw, mask_bw = mask
    label_fw, label_bw = get_bi_label(label)
    return seq_xent(pred_fw, label_fw, mask_fw) + seq_xent(pred_bw, label_bw, mask_bw)

def xent(pred, label):
    print("PRED.SHAPEL:"+str(pred.shape))
    print("LABEL.SHAPE:"+str(label.shape))
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size

def xent_onehot(pred, label):
    label = tf.one_hot(label, 3)
    #label = tf.Print(label, [label])
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size

def read_pretrained_embeddings():
    PAD_TOKEN = 0
    word2idx = {'PAD': PAD_TOKEN}
    weights = []
    with open(FLAGS.pretrain_embedding_path, 'r') as file:
        for index, line in enumerate(file):
            values = line.split()
            word = values[0]
            word_weights = np.asarray(values[1:], dtype = np.float32)
            word2idx[word] = index + 1
            weights.append(word_weights)
            if index + 1 == FLAGS.vocab_size:
                break
    dim_emb = len(weights[0])
    weights.insert(0,np.random.randn(dim_emb))

    UNK_TOKEN = len(weights)
    word2idx['UNK'] = UNK_TOKEN
    weights.append(np.random.randn(dim_emb))

    weights = np.asarray(weights, dtype = np.float32)
    
    return weights, word2idx

def rnncell(dim_hidden):
    #keep_prob = 1.0-FLAGS.dropout_rate
    #return tf.nn.rnn_cell.DropoutWrapper(Customized_BasicLSTMCell(num_units = dim_hidden, dtype=tf.float32), input_keep_prob=keep_prob)
    return Customized_BasicLSTMCell(num_units = dim_hidden, dtype=tf.float32)

def build_cells(cells, dim_input):
    for i,c in enumerate(cells):
        fake_inp = tf.ones((2,dim_input[i]))
        c.build(fake_inp.shape)


def get_pad_batch(a, batch_size):
    padded = []
    for i in range(len(a)//batch_size):
        batch_len = max([len(x) for x in a[i*batch_size:(i+1)*batch_size]])
        batch = np.zeros((batch_size,batch_len), dtype=np.int32)
        for j in range(batch_size):

            batch[j,:len(a[i*batch_size+j])] = a[i*batch_size + j]
        padded.append(batch)
    return padded

def get_pad_metabatch(a, batch_size):
    padded = []
    #print('A')
    #print([x.shape for x in a])
    for i in range(len(a)//batch_size):
        batch_len = max([x.shape[1] for x in a[i*batch_size:(i+1)*batch_size]])
        batch = np.zeros((batch_size, len(a[0]), batch_len), dtype=np.int32)
        for j in range(batch_size):
            #print(batch[j].shape)
            #print(batch[j,:,:a[i*batch_size+j].shape[1]].shape)
            #print(a[i*batch_size+j])
            batch[j,:,:a[i*batch_size+j].shape[1]] = a[i*batch_size+j]
        padded.append(batch)
    return padded

def get_batch_labels(a, batch_size):
    result = []
    for i in range(len(a)//batch_size):
        batch = np.stack(a[i*batch_size:(i+1)*batch_size])
        result.append(batch)
    return result

def get_metabatch_labels(a, batch_size):
    result = []
    for i in range(len(a)//batch_size):
        batch = np.stack(a[i*batch_size:(i+1)*batch_size])
        result.append(batch)
    return result
