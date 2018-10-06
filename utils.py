""" Utility functions. """
import os
import random
import tensorflow as tf
import numpy as np
import time

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags
from customized_cells import Customized_BasicLSTMCell, Customized_LayerNormBasicLSTMCell


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
    #BE CAUCIOUS: here reuse is set to tf.AUTO_REUSE instead of reuse
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=tf.AUTO_REUSE, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=tf.AUTO_REUSE, scope=scope)
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
    return tf.contrib.seq2seq.sequence_loss(logits=pred, targets=label, weights=mask, average_across_timesteps=True, average_across_batch=True, softmax_loss_function=xent_onehot)

def get_bi_label(label):
    #print(type(label))
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

#def xent_onehot(logits, labels):
#    labels = tf.one_hot(labels, )
#    return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) / FLAGS.update_batch_size


def xent_onehot(logits, labels):
    class_num = tf.shape(logits)[-1]
    labels = tf.one_hot(labels, class_num)
    #label = tf.Print(label, [label])
    return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) / FLAGS.update_batch_size

def read_pretrained_embeddings():
    PAD_TOKEN = 0
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    word2idx = {'PAD': PAD_TOKEN, 'BOS': BOS_TOKEN, 'EOS': EOS_TOKEN}
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
    for _ in range(3):
        weights.insert(0,np.random.randn(dim_emb))

    UNK_TOKEN = len(weights)
    word2idx['UNK'] = UNK_TOKEN
    weights.append(np.random.randn(dim_emb))

    weights = np.asarray(weights, dtype = np.float32)
    
    return weights, word2idx

def rnncell(dim_hidden):
    #keep_prob = 1.0-FLAGS.dropout_rate
    #return tf.nn.rnn_cell.DropoutWrapper(Customized_BasicLSTMCell(num_units = dim_hidden, dtype=tf.float32), input_keep_prob=keep_prob)
    if FLAGS.norm == 'None':
        return Customized_BasicLSTMCell(num_units = dim_hidden, dtype=tf.float32)
    elif FLAGS.norm == 'layer_norm':
        return Customized_BasicLSTMCell(num_units= dim_hidden, dtype=tf.float32, layer_norm=True)

def build_cells(cells, dim_input):
    for i,c in enumerate(cells):
        fake_inp = tf.ones((2,dim_input[i]))
        c.build(fake_inp.shape)

def get_static_pad_batch(a, batch_size, max_len):
    padded = []
    pad_len = max_len
    print("PAD_LEN: "+str(pad_len))
    for i in range(len(a)//batch_size):
        batch = np.zeros
        batch = np.zeros((batch_size,pad_len), dtype=np.int32)
        for j in range(batch_size):
            batch[j,:len(a[i*batch_size+j])] = a[i*batch_size + j]
        padded.append(batch)
    return padded

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

def add_grads(grada, gradb, coeff=1.0):
    vars_a = grada.keys()
    vars_b = gradb.keys()
    vars_all = list(set(vars_a)|set(vars_b))

    def get_sum_grad(v, coeff):
        if grada[v] is None and gradb[v] is None:
            return None
        if grada[v] is None:
            return coeff * gradb[v]
        if gradb[v] is None:
            return coeff * grada[v]
        return coeff * (grada[v] + gradb[v])

    return dict(zip(vars_all, [get_sum_grad(v, coeff) for v in vars_all]))

def minus_grads(grada, gradb, coeff=1.0):
    vars_a = grada.keys()
    vars_b = gradb.keys()
    vars_all = list(set(vars_a)|set(vars_b))

    def get_diff_grad(v, coeff):
        if grada[v] is None and gradb[v] is None:
            return None
        if grada[v] is None:
            return coeff * (-gradb[v])
        if gradb[v] is None:
            return coeff * grada[v]
        return coeff * (grada[v] - gradb[v])

    return dict(zip(vars_all, [get_diff_grad(v, coeff) for v in vars_all]))


def convert_to_stop_grad_dict(grads, theta):
    stop_grads = [tf.stop_gradient(g) if g is not None else None for g in grads]
    grads_dict = dict(zip(theta.keys(), stop_grads))
    return grads_dict
    


def get_approx_2nd_grad(optimizer, loss1, loss2, theta, eta, loss1_func, forw_model, model_inp, model_reuse, model_is_train, labela):
    #grad_loss1_theta = optimizer.compute_gradients(loss1)
    grad_loss1_theta = tf.gradients(loss1, list(theta.values()))
    #grad_loss1_theta = [tf.stop_gradient(grad) if grad is not None else None for grad in grad_loss1_theta]
    #grad_loss1_theta = dict(zip(theta.keys(), grad_loss1_theta))
    grad_loss1_theta = convert_to_stop_grad_dict(grad_loss1_theta, theta)
    #grad_loss2_theta2 = optimizer.compute_gradients(loss2)
    grad_loss2_theta2 = tf.gradients(loss2, list(theta.values()))
    #grad_loss2_theta2 = [tf.stop_gradient(grad) if grad is not None else None for grad in grad_loss2_theta2]
    #grad_loss2_theta2 = dict(zip(theta.keys(), grad_loss2_theta2))
    grad_loss2_theta2 = convert_to_stop_grad_dict(grad_loss2_theta2, theta)
    

    nu = 0.0000001 # nu should be a very small value

    def get_weight(key, theta, nu, grad_loss1_theta):
        if grad_loss1_theta[key] is None:
            return theta[key]
        if key!='emb':
            return  theta[key] + nu*grad_loss1_theta[key] 
        else:
            return theta[key] + nu * tf.convert_to_tensor(grad_loss1_theta[key])

    theta_hat = dict(zip(theta.keys(),[get_weight(key, theta, nu, grad_loss2_theta2) for key in theta.keys()]))
    #theta_hat = theta + nu * grad_loss1_theta

    def get_loss_hat(inp, reuse=True, is_train=True):
        model_inp, labela = inp
        output_hat, mask = forw_model(model_inp, theta_hat, reuse = reuse, is_train=is_train, task="aux")
        #JUST FOR TEST USE
        #output_hat, mask = forw_model(model_inp, theta, reuse = reuse, is_train=is_train, task="aux")
        loss_hat = loss1_func(output_hat, labela, mask)
        return loss_hat



    #output_hat, mask = forw_model(model_inp, theta_hat, reuse = model_reuse, is_train=model_is_train, task="aux")
    #loss_hat = loss1_func(output_hat, labela, mask)
    loss_hat = tf.map_fn(get_loss_hat, elems=(model_inp, labela), dtype=tf.float32, parallel_iterations=FLAGS.meta_batch_size)
    loss_hat = tf.reduce_sum(loss_hat) / tf.to_float(FLAGS.meta_batch_size)
    grad_loss_theta_hat = optimizer.compute_gradients(loss_hat)
    grad_loss_theta_hat = tf.gradients(loss_hat, list(theta.values()))
    grad_loss_theta_hat = convert_to_stop_grad_dict(grad_loss_theta_hat, theta)

    final_approx =  minus_grads(grad_loss2_theta2 , minus_grads(grad_loss_theta_hat, grad_loss1_theta, coeff=(eta / nu)))
    return [(final_approx[v], theta[v]) for v in final_approx.keys() if final_approx[v] is not None]
    #return final_approx


def convert_list_to_tensor(l):
    """Used to convert the format for static rnn to the format for dynamic rnn"""    
    t = tf.convert_to_tensor(l)
    t = tf.transpose(t, [1,0,2])
    return t

def convert_tensor_to_list(t, max_len):
    """Used to convert the format for dynamic rnn to the format for static rnn"""
    t = tf.transpose(t, [1,0,2])
    #t = tf.reshape(t, [-1, dim])
    #l = tf.split(t, max_len)
    l = tf.unstack(t, num=max_len)
    return l

def concat_parallel_outputs(out_split):
    results=[]
#    time.sleep(100)
    for i,v in enumerate(zip(*out_split)):
        if type(v[0]) is list:
            #results.append([tf.concat([x[i] for x in v] ,axis=0) for i in range(len(v[0]))])
            results.append([tf.stack([x[i] for x in v] ) for i in range(len(v[0]))])
        elif type(v[0]) is tuple:
            #results.append([tf.concat([x[i] for x in v] ,axis=0) for i in range(len(v[0]))])
            results.append([tf.stack([x[i] for x in v] ) for i in range(len(v[0]))])
        else:
            #results.append(tf.concat(v, axis=0))
            results.append(tf.stack(v))
    return results

def make_parallel(fn, num_gpus, **kwargs):
    in_splits = {}
    for k, v in kwargs.items():
        if type(v) is tuple:
            a, b = v
            in_splits[k] = list(zip(tf.split(a, num_gpus),tf.split(b,num_gpus)))
        else:
            in_splits[k] = tf.split(v, num_gpus)

    out_split = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                out_split.append(fn(**{k : v[i] for k, v in in_splits.items()}))

    #return tf.concat(out_split, axis=0)
    return concat_parallel_outputs(out_split)
