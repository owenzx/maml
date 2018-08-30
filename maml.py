""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize, read_pretrained_embeddings, rnncell, build_cells, xent_onehot, bi_seq_xent, seq_xent
from customized_cells import Customized_BasicLSTMCell
from bidaf import BiAttention_no_var
from multihead_attention import multihead_attention_no_var
from constants import *

FLAGS = flags.FLAGS

class MAML:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.classification = False
        self.test_num_updates = test_num_updates
        if FLAGS.task == 'usl_adapt':
            self.construct_model = self.construct_model_usl_adapt
        else:
            self.construct_model = self.construct_model_standard
        if FLAGS.datasource == 'sinusoid':
            self.dim_hidden = [40, 40]
            self.loss_func = mse
            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights
        elif FLAGS.datasource == 'omniglot' or FLAGS.datasource == 'miniimagenet':
            self.loss_func = xent
            self.classification = True
            if FLAGS.conv:
                self.dim_hidden = FLAGS.num_filters
                self.forward = self.forward_conv
                self.construct_weights = self.construct_conv_weights
            else:
                self.dim_hidden = [256, 128, 64, 64]
                self.forward=self.forward_fc
                self.construct_weights = self.construct_fc_weights
            if FLAGS.datasource == 'miniimagenet':
                self.channels = 3
            else:
                self.channels = 1
            self.img_size = int(np.sqrt(self.dim_input/self.channels))
        elif FLAGS.datasource in NLP_DATASETS:
            if FLAGS.task == 'usl_adapt':
                self.forward = self.forward_1input_rnn_usl
                self.construct_weights = self.construct_1input_rnn_weights_usl
                self.aux_loss_func = bi_seq_xent
                self.real_loss_func = xent_onehot
            else:
                if FLAGS.model == "rnn":
                    self.forward = self.forward_rnn
                    self.construct_weights = self.construct_rnn_weights
                elif FLAGS.model == "bidaf":
                    self.forward = self.forward_better_rnn
                    self.construct_weights = self.construct_better_rnn_weights
                self.loss_func = xent_onehot
            self.vocab_size = 40000 + 2 #default choice
            self.dim_hidden =200
            self.dim_output = 3
            self.classification = True
            self.dim_emb = 300
            self.num_layers = FLAGS.num_rnn_layers
            self.batch_size = FLAGS.update_batch_size
            #Other hyper-parameters
        else:
            raise ValueError('Unrecognized data source.')
    
    def set_pretrain_embedding(self, w, word2idx):
        self.pretrain_embedding = w
        self.vocab_size = len(word2idx)

    
    def construct_model_usl_adapt(self, input_tensors=None, prefix='metatrain_'):
        self.inputa = input_tensors['inputa']
        self.inputb = input_tensors['inputb']
        self.labela, _ = input_tensors['labela']
        self.labelb = input_tensors['labelb']

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights, weightsa, weightsb = self.weights, self.weightsa, self.weightsb
            else:
                # Define the weights
                self.weights, self.weightsa, self.weightsb = weights, weightsa, weightsb = self.construct_weights()

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates

            def task_metalearn(inp, reuse=True, is_train=True):
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []
                if self.classification:
                    task_accuraciesb = []
                task_outputa, mask= self.forward(inputa, weights, reuse=reuse, is_train=is_train, task="aux")
                task_lossa = self.aux_loss_func(task_outputa, labela, mask)
                grads = tf.gradients(task_lossa, list(weightsa.values()))

                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weightsa.keys(), grads))

                def get_weight(key, weighta, weightsb):
                    if key not in weightsa.keys():
                        return weightsb[key]
                    if key!='emb':
                        return  weightsb[key] - self.update_lr*gradients[key] 
                    else:
                        return weightsb[key] - self.update_lr*tf.convert_to_tensor(gradients[key])

                #fast_weights = dict(zip(weightsb.keys(), 
                #                    [weightsb[key] if key not in weightsa.keys() 
                #                    else (
                #                        weights[key] - self.update_lr*gradients[key] if key!='emb' 
                #                        else weights[key] - self.update_lr*tf.convert_to_tensor(gradients[key])
                #                    )
                #                     for key in weightsb.keys()]))
                fast_weights = dict(zip(weightsb.keys(), [get_weight(key, weightsa, weightsb) for key in weightsb.keys()]))
                output = self.forward(inputb, fast_weights, reuse=True, is_train = is_train, task="main")
                task_outputbs.append(output)
                task_lossesb.append(self.real_loss_func(output, labelb))

                #Now first suppose only one aux task
                task_output = [task_outputa, task_outputbs, task_lossa,task_lossesb]

                if self.classification:
                    for j in range(num_updates):
                        if FLAGS.datasource in NLP_DATASETS:
                            task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), labelb))
                    task_output.extend([task_accuraciesb])

                return task_output



            if FLAGS.norm != 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [(tf.float32, tf.float32), [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
            if self.classification:
                out_dtype.extend([[tf.float32]*num_updates])
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            if self.classification:
                outputas, outputbs, lossesa, lossesb, accuraciesb = result
            else:
                outputas, outputbs, lossesa, lossesb  = result

        ## Performance & Optimization
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs
            if self.classification:
                self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
#            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)
            #TODO: create new pretrain_op
            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates-1])
                if FLAGS.datasource == 'miniimagenet':
                    gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
                self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            if self.classification:
                self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

        ## Summaries
        tf.summary.scalar(prefix+'Unsupervied auxilary task loss', total_loss1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Real task loss, step ' + str(j+1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix+'Real task accuracy, step ' + str(j+1), total_accuracies2[j])


           



    def construct_model_standard(self, input_tensors=None, prefix='metatrain_'):
        # a: training data for inner gradient, b: test data for meta gradient
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates

            def task_metalearn(inp, reuse=True, is_train=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                #inputa = tf.Print(inputa, [inputa], summarize=10000)
                #labela = tf.Print(labela, [labela])
                task_outputbs, task_lossesb = [], []

                if self.classification:
                    task_accuraciesb = []

                task_outputa = self.forward(inputa, weights, reuse=reuse, is_train=is_train)  # only reuse on the first iter
                task_lossa = self.loss_func(task_outputa, labela)

                grads = tf.gradients(task_lossa, list(weights.values()))
                if FLAGS.stop_grad:
                    #TODO: allow some weights to not have grads
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                #print([(c,type(weights[c])) for c in weights.keys()])
                #print([(c,type(gradients[c])) for c in weights.keys()])
                
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] if key!='emb' else weights[key] - self.update_lr*tf.convert_to_tensor(gradients[key]) for key in weights.keys()]))
                output = self.forward(inputb, fast_weights, reuse=True, is_train=is_train)
                #output = tf.Print(output, [output])
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True, is_train=is_train), labela)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    #print([(c,type(fast_weights[c])) for c in fast_weights.keys()])
                    #print([(c,type(gradients[c])) for c in fast_weights.keys()])
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] if key!='emb' else fast_weights[key] - self.update_lr*tf.convert_to_tensor(gradients[key]) for key in fast_weights.keys()]))
                    output = self.forward(inputb, fast_weights, reuse=True, is_train=is_train)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                if self.classification:
                    if FLAGS.datasource in NLP_DATASETS:
                        task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), labela)
                    else:
                        task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1))
                    for j in range(num_updates):
                        if FLAGS.datasource in NLP_DATASETS:
                            task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), labelb))
                        else:
                            task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(labelb, 1)))
                    task_output.extend([task_accuracya, task_accuraciesb])

                return task_output

            #using is not in the original repo is actually a bug?
            if FLAGS.norm != 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
            if self.classification:
                out_dtype.extend([tf.float32, [tf.float32]*num_updates])
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            if self.classification:
                outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result
            else:
                outputas, outputbs, lossesa, lossesb  = result

        ## Performance & Optimization
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs
            if self.classification:
                self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            #self.debug_grads = tf.gradients(total_loss1, [self.weights['w2'], self.weights['text_cell_forw_0_w']])
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)
            #self.pretrain_op = tf.train.GradientDescentOptimizer(0.01*self.meta_lr).minimize(total_loss1)

            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates-1])
                if FLAGS.datasource == 'miniimagenet':
                    gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
                self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            if self.classification:
                self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

        ## Summaries
        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])

    ### Network construction functions (fc networks and conv networks)
    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1,len(self.dim_hidden)):
            weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
            weights['b'+str(i+1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    def forward_fc(self, inp, weights, reuse=False, is_train=True):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
        for i in range(1,len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
        return tf.matmul(hidden, weights['w'+str(len(self.dim_hidden)+1)]) + weights['b'+str(len(self.dim_hidden)+1)]

    def construct_conv_weights(self):
        weights = {}

        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3

        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
        if FLAGS.datasource == 'miniimagenet':
            # assumes max pooling
            weights['w5'] = tf.get_variable('w5', [self.dim_hidden*5*5, self.dim_output], initializer=fc_initializer)
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        else:
            weights['w5'] = tf.Variable(tf.random_normal([self.dim_hidden, self.dim_output]), name='w5')
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        return weights

    def forward_conv(self, inp, weights, reuse=False, scope='', is_train=True):
        # reuse is for the normalization parameters.
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope+'0')
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope+'1')
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope+'2')
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope+'3')
        if FLAGS.datasource == 'miniimagenet':
            # last hidden layer is 6x6x64-ish, reshape to a vector
            hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])
        else:
            hidden4 = tf.reduce_mean(hidden4, [1, 2])

        return tf.matmul(hidden4, weights['w5']) + weights['b5']

    
    def construct_better_rnn_weights(self):
        weights = {}
        self.cells = {}

        dtype = tf.float32
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
        if FLAGS.pretrain_embedding == 'none':
            emb_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
            weights['emb'] = tf.get_variable('emb',[self.vocab_size, self.dim_emb], initializer=emb_initializer)
        elif FLAGS.pretrain_embedding == 'glove':
            emb_initializer = tf.constant_initializer(self.pretrain_embedding)
            weights['emb'] = tf.get_variable('emb', [self.vocab_size, self.dim_emb], initializer = emb_initializer, trainable=True)
            
        



        self.cells['text_cell_forw'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]
        self.cells['text_cell_back'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]
        self.cells['ctgr_cell_forw'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]
        self.cells['ctgr_cell_back'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]
        self.cells['aftbidaf_cell_forw'] = [rnncell(4*self.dim_hidden) for _ in range(2)]
        self.cells['aftbidaf_cell_back'] = [rnncell(4*self.dim_hidden) for _ in range(2)]

        num_rnn_hiddens = [self.dim_emb] + [self.dim_hidden for _ in range(self.num_layers-1)]
        num_aftbidaf_rnn_hiddens =  [4*self.dim_hidden for _ in range(2)]
        #All the rnn cells must be built so that their weights can be accessed
        build_cells(self.cells['text_cell_forw'], num_rnn_hiddens)
        build_cells(self.cells['text_cell_back'], num_rnn_hiddens)
        build_cells(self.cells['ctgr_cell_forw'], num_rnn_hiddens)
        build_cells(self.cells['ctgr_cell_back'], num_rnn_hiddens)
        build_cells(self.cells['aftbidaf_cell_forw'], num_aftbidaf_rnn_hiddens)
        build_cells(self.cells['aftbidaf_cell_back'], num_aftbidaf_rnn_hiddens)


        #Note that currently, the following code only works for LSTM cells 
        for i, c in enumerate(self.cells['text_cell_forw']):
            weights['text_cell_forw_%d_w'%i], weights['text_cell_forw_%d_b'%i] = c.weights
        for i, c in enumerate(self.cells['text_cell_back']):
            weights['text_cell_back_%d_w'%i], weights['text_cell_back_%d_b'%i] = c.weights
        for i, c in enumerate(self.cells['ctgr_cell_forw']):
            weights['ctgr_cell_forw_%d_w'%i], weights['ctgr_cell_forw_%d_b'%i] = c.weights
        for i, c in enumerate(self.cells['ctgr_cell_back']):
            weights['ctgr_cell_back_%d_w'%i], weights['ctgr_cell_back_%d_b'%i] = c.weights
        #print([type(weights[c]) for c in weights.keys()])
        for i, c in enumerate(self.cells['aftbidaf_cell_forw']):
            weights['aftbidaf_cell_forw_%d_w'%i], weights['aftbidaf_cell_forw_%d_b'%i] = c.weights
        for i, c in enumerate(self.cells['aftbidaf_cell_back']):
            weights['aftbidaf_cell_back_%d_w'%i], weights['aftbidaf_cell_back_%d_b'%i] = c.weights

        self.biattention = BiAttention_no_var()
        #TODO: need to recheck initialization methods
        #weights["bidaf_bias"]= tf.Variable(tf.zeros([self.dim_hidden]), name = 'bidaf_bias')
        weights["bidaf_key_w"]= tf.get_variable('bidaf_key_w', [self.dim_hidden], initializer = fc_initializer)
        weights["bidaf_input_w"]= tf.get_variable('bidaf_input_w', [self.dim_hidden], initializer = fc_initializer)
        weights["bidaf_dot_w"]= tf.get_variable('bidaf_dot_w', [self.dim_hidden], initializer = fc_initializer)

        if FLAGS.num_attn_head > 0:
            weights["text_self_att_q"] = tf.get_variable("text_self_att_q", [self.dim_hidden, self.dim_hidden])
            weights["text_self_att_k"] = tf.get_variable("text_self_att_k", [self.dim_hidden, self.dim_hidden])
            weights["text_self_att_v"] = tf.get_variable("text_self_att_v", [self.dim_hidden, self.dim_hidden])
            weights["ctgr_self_att_q"] = tf.get_variable("ctgr_self_att_q", [self.dim_hidden, self.dim_hidden])
            weights["ctgr_self_att_k"] = tf.get_variable("ctgr_self_att_k", [self.dim_hidden, self.dim_hidden])
            weights["ctgr_self_att_v"] = tf.get_variable("ctgr_self_att_v", [self.dim_hidden, self.dim_hidden])
            weights["final_self_att_q"] = tf.get_variable("final_self_att_q", [4*self.dim_hidden, 4*self.dim_hidden])
            weights["final_self_att_k"] = tf.get_variable("final_self_att_k", [4*self.dim_hidden, 4*self.dim_hidden])
            weights["final_self_att_v"] = tf.get_variable("final_self_att_v", [4*self.dim_hidden, 4*self.dim_hidden])

        

        weights['w1'] = tf.get_variable('w1', [8*self.dim_hidden, self.dim_hidden], initializer = fc_initializer)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name = 'b1')
        weights['w2'] = tf.get_variable('w2', [self.dim_hidden, self.dim_output], initializer = fc_initializer)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_output]), name='b2')
        return weights

    def forward_better_rnn(self, inp, weights, reuse=False, scope= '', is_train=True):
        keep_prob = 1-FLAGS.dropout_rate

        cells=dict()
        cells['text_cell_forw'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]
        cells['text_cell_back'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]
        cells['ctgr_cell_forw'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]
        cells['ctgr_cell_back'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]
        cells['aftbidaf_cell_forw'] = [rnncell(4*self.dim_hidden) for _ in range(2)]
        cells['aftbidaf_cell_back'] = [rnncell(4*self.dim_hidden) for _ in range(2)]

        #first update the weights of all the rnn cells
        for k in self.cells.keys():
            if 'aftbidaf' in k:
                for n in range(2):
                    cells[k][n].update_weights(weights[k+"_%d_"%n+"w"], weights[k+"_%d_"%n+"b"])
                    if is_train:
                        cells[k][n] = tf.nn.rnn_cell.DropoutWrapper(cells[k][n], input_keep_prob = keep_prob)
            else:
                for n in range(self.num_layers):
                    cells[k][n].update_weights(weights[k+"_%d_"%n+"w"], weights[k+"_%d_"%n+"b"])
                    if is_train:
                        cells[k][n] = tf.nn.rnn_cell.DropoutWrapper(cells[k][n], input_keep_prob = keep_prob)

        text_tok, ctgr_tok, text_len, ctgr_len = inp

        text_vec = tf.nn.embedding_lookup(weights['emb'], text_tok)
        ctgr_vec = tf.nn.embedding_lookup(weights['emb'], ctgr_tok)
        #print(text_vec.shape)

        output = text_vec
        for n in range(self.num_layers):
            cell_fw = cells['text_cell_forw'][n]
            cell_bw = cells['text_cell_back'][n]

            state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
            state_bw = cell_bw.zero_state(self.batch_size, tf.float32)

            (output_fw, output_bw), last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output, sequence_length=text_len, initial_state_fw = state_fw, initial_state_bw = state_bw, scope = 'BLSTM_text_'+str(n), dtype = tf.float32)

            output = output_fw + output_bw
            if (n==self.num_layers-2) and (FLAGS.num_attn_head > 0):
                output= multihead_attention_no_var(output, output, num_units = self.dim_hidden * FLAGS.num_attn_head, num_heads = FLAGS.num_attn_head, is_training=is_train, scope = "text_self_att", weights= weights, prefix = "text_self_att_")
        #last_fw, last_bw = last_state
        #text_hidden = tf.concat([last_fw.h, last_bw.h], axis = -1)
        text_hidden = output

        output = ctgr_vec
        for n in range(self.num_layers):
            cell_fw = cells['ctgr_cell_forw'][n]
            cell_bw = cells['ctgr_cell_back'][n]

            state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
            state_bw = cell_bw.zero_state(self.batch_size, tf.float32)

            (output_fw, output_bw), last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output, sequence_length=ctgr_len, initial_state_fw = state_fw, initial_state_bw = state_bw, scope = 'BLSTM_ctgr_'+str(n), dtype = tf.float32)

            output = output_fw + output_bw
            if (n==self.num_layers-2) and (FLAGS.num_attn_head > 0):
                output= multihead_attention_no_var(output, output, num_units = self.dim_hidden * FLAGS.num_attn_head, num_heads = FLAGS.num_attn_head, is_training=is_train, scope = "ctgr_self_att", weights= weights, prefix = "ctgr_self_att_")
        #last_fw, last_bw = last_state
        #ctgr_hidden = tf.concat([last_fw.h, last_bw.h], axis=-1)
        ctgr_hidden = output



        max_text_len = tf.shape(text_tok)[1]
        max_ctgr_len = tf.shape(ctgr_tok)[1]
        text_mask = tf.sequence_mask(text_len, max_text_len)
        ctgr_mask = tf.sequence_mask(ctgr_len, max_ctgr_len)

        p = self.biattention.apply(is_train, text_hidden, ctgr_hidden, ctgr_hidden, x_mask = text_mask, mem_mask= ctgr_mask, weights=weights, prefix="bidaf_")

        #print(p.shape)
        output = p
        for n in range(2):
            cell_fw = cells['aftbidaf_cell_forw'][n]
            cell_bw = cells['aftbidaf_cell_back'][n]

            state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
            state_bw = cell_bw.zero_state(self.batch_size, tf.float32)

            (output_fw, output_bw), last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output, sequence_length=ctgr_len, initial_state_fw = state_fw, initial_state_bw = state_bw, scope = 'BLSTM_aftbidaf_'+str(n), dtype = tf.float32)

            #output = tf.concat([output_fw, output_bw], axis = 2)
            output = output_fw + output_bw
            if (n==0) and (FLAGS.num_attn_head > 0):
                output = multihead_attention_no_var(output, output, num_units = self.dim_hidden * FLAGS.num_attn_head, num_heads = FLAGS.num_attn_head, is_training=is_train, scope = "final_self_att", weights= weights, prefix = "final_self_att_")
        last_fw, last_bw = last_state
        final_hidden = tf.concat([last_fw.h, last_bw.h], axis=-1)
        
        #print("TEXT_HIDDEN.SHAPE:"+str(text_hidden.shape))
        #print("CTGR_HIDDEN.SHAPE:"+str(ctgr_hidden.shape))
        cat_hidden = tf.nn.relu(final_hidden)
        cat_hidden_2 = tf.nn.relu(tf.matmul(cat_hidden, weights['w1']) + weights['b1'])
        if is_train:
            cat_hidden_2 = tf.nn.dropout(cat_hidden_2, keep_prob)
        final_output = tf.matmul(cat_hidden_2, weights['w2']) + weights['b2']
        #final_output = tf.Print(final_output, [final_output])
        return final_output

    def construct_rnn_weights(self):
        weights = {}
        self.cells = {}

        dtype = tf.float32
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
        if FLAGS.pretrain_embedding == 'none':
            emb_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
            weights['emb'] = tf.get_variable('emb',[self.vocab_size, self.dim_emb], initializer=emb_initializer)
        elif FLAGS.pretrain_embedding == 'glove':
            emb_initializer = tf.constant_initializer(self.pretrain_embedding)
            weights['emb'] = tf.get_variable('emb', [self.vocab_size, self.dim_emb], initializer = emb_initializer, trainable=True)
            
        weights['w1'] = tf.get_variable('w1', [4*self.dim_hidden, self.dim_hidden], initializer = fc_initializer)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name = 'b1')
        weights['w2'] = tf.get_variable('w2', [self.dim_hidden, self.dim_output], initializer = fc_initializer)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_output]), name='b2')



        self.cells['text_cell_forw'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]
        self.cells['text_cell_back'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]
        self.cells['ctgr_cell_forw'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]
        self.cells['ctgr_cell_back'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]

        num_rnn_hiddens = [self.dim_emb] + [self.dim_hidden for _ in range(self.num_layers-1)]
        #All the rnn cells must be built so that their weights can be accessed
        build_cells(self.cells['text_cell_forw'], num_rnn_hiddens)
        build_cells(self.cells['text_cell_back'], num_rnn_hiddens)
        build_cells(self.cells['ctgr_cell_forw'], num_rnn_hiddens)
        build_cells(self.cells['ctgr_cell_back'], num_rnn_hiddens)


        #Note that currently, the following code only works for LSTM cells 
        for i, c in enumerate(self.cells['text_cell_forw']):
            weights['text_cell_forw_%d_w'%i], weights['text_cell_forw_%d_b'%i] = c.weights
        for i, c in enumerate(self.cells['text_cell_back']):
            weights['text_cell_back_%d_w'%i], weights['text_cell_back_%d_b'%i] = c.weights
        for i, c in enumerate(self.cells['ctgr_cell_forw']):
            weights['ctgr_cell_forw_%d_w'%i], weights['ctgr_cell_forw_%d_b'%i] = c.weights
        for i, c in enumerate(self.cells['ctgr_cell_back']):
            weights['ctgr_cell_back_%d_w'%i], weights['ctgr_cell_back_%d_b'%i] = c.weights
        #print([type(weights[c]) for c in weights.keys()])
        
        if FLAGS.num_attn_head > 0:
            weights["text_self_att_q"] = tf.get_variable("text_self_att_q", [self.dim_hidden, self.dim_hidden])
            weights["text_self_att_k"] = tf.get_variable("text_self_att_k", [self.dim_hidden, self.dim_hidden])
            weights["text_self_att_v"] = tf.get_variable("text_self_att_v", [self.dim_hidden, self.dim_hidden])
            weights["ctgr_self_att_q"] = tf.get_variable("ctgr_self_att_q", [self.dim_hidden, self.dim_hidden])
            weights["ctgr_self_att_k"] = tf.get_variable("ctgr_self_att_k", [self.dim_hidden, self.dim_hidden])
            weights["ctgr_self_att_v"] = tf.get_variable("ctgr_self_att_v", [self.dim_hidden, self.dim_hidden])
        return weights


    def forward_rnn(self, inp, weights, reuse=False, scope='', is_train=True):
        cells=dict()

        keep_prob = 1-FLAGS.dropout_rate

        cells['text_cell_forw'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]
        cells['text_cell_back'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]
        cells['ctgr_cell_forw'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]
        cells['ctgr_cell_back'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]

        #first update the weights of all the rnn cells
        for k in self.cells.keys():
            for n in range(self.num_layers):
                    cells[k][n].update_weights(weights[k+"_%d_"%n+"w"], weights[k+"_%d_"%n+"b"])
                    if is_train:
                        cells[k][n] = tf.nn.rnn_cell.DropoutWrapper(cells[k][n], input_keep_prob=keep_prob)

        text_tok, ctgr_tok, text_len, ctgr_len = inp
        #text_tok = tf.Print(text_tok, [text_tok], summarize=10000)
        #text_tok = tf.expand_dims(text_tok, axis=-1)
        #ctgr_tok = tf.expand_dims(ctgr_tok, axis=-1)
        text_vec = tf.nn.embedding_lookup(weights['emb'], text_tok)
        ctgr_vec = tf.nn.embedding_lookup(weights['emb'], ctgr_tok)
        #print(text_vec.shape)

        output = text_vec
        for n in range(self.num_layers):
            cell_fw = cells['text_cell_forw'][n]
            cell_bw = cells['text_cell_back'][n]

            state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
            state_bw = cell_bw.zero_state(self.batch_size, tf.float32)
            #print("OUTPUT.SHAPE:")
            #print(output.shape)
            (output_fw, output_bw), last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output, sequence_length=text_len, initial_state_fw = state_fw, initial_state_bw = state_bw, scope = 'BLSTM_text_'+str(n), dtype = tf.float32)

            #output = tf.concat([output_fw, output_bw], axis = 2)
            output = output_fw + output_bw
            if (n==self.num_layers-2) and (FLAGS.num_attn_head > 0):
                output = multihead_attention_no_var(output, output, num_units = self.dim_hidden , num_heads = FLAGS.num_attn_head, is_training=is_train, scope = "text_self_att", weights= weights, prefix = "text_self_att_")
        last_fw, last_bw = last_state
        text_hidden = tf.concat([last_fw.h, last_bw.h], axis = -1)
        #text_hidden = tf.concat([output[:,0,:],output[:,-1,:]],axis=-1)

        output = ctgr_vec
        for n in range(self.num_layers):
            cell_fw = cells['ctgr_cell_forw'][n]
            cell_bw = cells['ctgr_cell_back'][n]

            state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
            state_bw = cell_bw.zero_state(self.batch_size, tf.float32)

            (output_fw, output_bw), last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output, sequence_length=ctgr_len, initial_state_fw = state_fw, initial_state_bw = state_bw, scope = 'BLSTM_ctgr_'+str(n), dtype = tf.float32)

            #output = tf.concat([output_fw, output_bw], axis = 2)
            output = output_fw + output_bw
            if (n==self.num_layers-2) and (FLAGS.num_attn_head > 0):
                output = multihead_attention_no_var(output, output, num_units = self.dim_hidden , num_heads = FLAGS.num_attn_head, is_training=is_train, scope = "ctgr_self_att", weights= weights, prefix = "ctgr_self_att_")
        last_fw, last_bw = last_state
        ctgr_hidden = tf.concat([last_fw.h, last_bw.h], axis=-1)
        #ctgr_hidden = tf.concat([output[:,0,:],output[:,-1,:]],axis=-1)

        #print("TEXT_HIDDEN.SHAPE:"+str(text_hidden.shape))
        #print("CTGR_HIDDEN.SHAPE:"+str(ctgr_hidden.shape))

        cat_hidden = tf.nn.relu(tf.concat([text_hidden, ctgr_hidden], axis = -1))
        if is_train:
            cat_hidden = tf.nn.dropout(cat_hidden, keep_prob)
        cat_hidden_2 = tf.nn.relu(tf.matmul(cat_hidden, weights['w1']) + weights['b1'])
        if is_train:
            cat_hidden_2 = tf.nn.dropout(cat_hidden_2, keep_prob)
        final_output = tf.matmul(cat_hidden_2, weights['w2']) + weights['b2']
        #final_output = tf.Print(final_output, [final_output])
        return final_output
        
    def construct_1input_rnn_weights_usl(self):
        weights = {}
        self.cells = {}

        dtype = tf.float32
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
        if FLAGS.pretrain_embedding == 'none':
            emb_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
            weights['emb'] = tf.get_variable('emb',[self.vocab_size, self.dim_emb], initializer=emb_initializer)
        elif FLAGS.pretrain_embedding == 'glove':
            emb_initializer = tf.constant_initializer(self.pretrain_embedding)
            weights['emb'] = tf.get_variable('emb', [self.vocab_size, self.dim_emb], initializer = emb_initializer, trainable=True)
            
        weights['w1'] = tf.get_variable('w1', [2*self.dim_hidden, self.dim_hidden], initializer = fc_initializer)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name = 'b1')
        weights['w2'] = tf.get_variable('w2', [self.dim_hidden, self.dim_output], initializer = fc_initializer)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_output]), name='b2')



        self.cells['text_cell_forw'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]
        self.cells['text_cell_back'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]

        num_rnn_hiddens = [self.dim_emb] + [self.dim_hidden for _ in range(self.num_layers-1)]
        #All the rnn cells must be built so that their weights can be accessed
        build_cells(self.cells['text_cell_forw'], num_rnn_hiddens)
        build_cells(self.cells['text_cell_back'], num_rnn_hiddens)


        #Note that currently, the following code only works for LSTM cells 
        for i, c in enumerate(self.cells['text_cell_forw']):
            weights['text_cell_forw_%d_w'%i], weights['text_cell_forw_%d_b'%i] = c.weights
        for i, c in enumerate(self.cells['text_cell_back']):
            weights['text_cell_back_%d_w'%i], weights['text_cell_back_%d_b'%i] = c.weights
        #print([type(weights[c]) for c in weights.keys()])
        
        if FLAGS.num_attn_head > 0:
            weights["text_self_att_q"] = tf.get_variable("text_self_att_q", [self.dim_hidden, self.dim_hidden])
            weights["text_self_att_k"] = tf.get_variable("text_self_att_k", [self.dim_hidden, self.dim_hidden])
            weights["text_self_att_v"] = tf.get_variable("text_self_att_v", [self.dim_hidden, self.dim_hidden])

        if not FLAGS.bind_embedding_softmax:
            weights['w_sm_forw'] = tf.get_variable('w_sm_forw', [self.dim_hidden, self.vocab_size], initializer = fc_initializer)
            weights['b_sm_forw'] = tf.Variable(tf.zeros([self.vocab_size]), name='b_sm_forw')
            weights['w_sm_back'] = tf.get_variable('w_sm_back', [self.dim_hidden, self.vocab_size], initializer = fc_initializer)
            weights['b_sm_back'] = tf.Variable(tf.zeros([self.vocab_size]), name='b_sm_back')

        not_in_weightsa_keys = ['w1', 'b1', 'w2', 'b2']
        not_in_weightsb_keys = []
        if not FLAGS.bind_embedding_softmax:
            not_in_weightsb_keys.extend(['w_sm_forw', 'b_sm_forw', 'w_sm_back', 'b_sm_back'])
        
        weightsa_keys = [k for k in weights.keys() if k not in not_in_weightsa_keys]
        weightsb_keys = [k for k in weights.keys() if k not in not_in_weightsb_keys]

        weightsa = dict(zip(weightsa_keys,[weights[k] for k in weightsa_keys]))
        weightsb = dict(zip(weightsb_keys,[weights[k] for k in weightsb_keys]))

        return weights, weightsa, weightsb

    def forward_1input_rnn_usl(self, inp, weights, reuse=False, scope='', is_train=True, task="main"):
        cells = dict()
        keep_prob = 1-FLAGS.dropout_rate

        cells['text_cell_forw'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]
        cells['text_cell_back'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]

        #print(weights)
        #print(type(weights))
        #first update the weights of all the rnn cells
        for k in self.cells.keys():
            for n in range(self.num_layers):
                    cells[k][n].update_weights(weights[k+"_%d_"%n+"w"], weights[k+"_%d_"%n+"b"])
                    if is_train:
                        cells[k][n] = tf.nn.rnn_cell.DropoutWrapper(cells[k][n], input_keep_prob=keep_prob)

        text_tok, text_len = inp

        text_vec = tf.nn.embedding_lookup(weights['emb'], text_tok)
        #print(text_vec.shape)

        batch_size = tf.shape(text_vec)[0]
        max_text_len = tf.shape(text_vec)[1]

        output = text_vec
        for n in range(self.num_layers):
            cell_fw = cells['text_cell_forw'][n]
            cell_bw = cells['text_cell_back'][n]

            state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
            state_bw = cell_bw.zero_state(self.batch_size, tf.float32)
            #print("OUTPUT.SHAPE:")
            #print(output.shape)
            (output_fw, output_bw), last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output, sequence_length=text_len, initial_state_fw = state_fw, initial_state_bw = state_bw, scope = 'BLSTM_text_'+str(n), dtype = tf.float32)

            #output = tf.concat([output_fw, output_bw], axis = 2)
            output = output_fw + output_bw
            if (n==self.num_layers-2) and (FLAGS.num_attn_head > 0):
                output = multihead_attention_no_var(output, output, num_units = self.dim_hidden , num_heads = FLAGS.num_attn_head, is_training=is_train, scope = "text_self_att", weights= weights, prefix = "text_self_att_")
        last_fw, last_bw = last_state
        text_hidden = tf.concat([last_fw.h, last_bw.h], axis = -1)
        #text_hidden = tf.concat([output[:,0,:],output[:,-1,:]],axis=-1)

        if task=="aux":
            output_fw = tf.reshape(output_fw, [-1, self.dim_hidden])
            output_bw = tf.reshape(output_bw, [-1, self.dim_hidden])
            
            if not FLAGS.bind_embedding_softmax:
                logits_fw = tf.matmul(output_fw, weights['w_sm_forw']) + weights['b_sm_forw']
                logits_bw = tf.matmul(output_bw, weights['w_sm_back']) + weights['b_sm_back']
            else:
                w_sm = tf.transpose(weights['emb'])
                logits_fw = tf.matmul(output_fw, w_sm)
                logits_bw = tf.matmul(output_bw, w_sm)
            logits_fw = tf.reshape(logits_fw, [batch_size, max_text_len, self.vocab_size])
            logits_bw = tf.reshape(logits_bw, [batch_size, max_text_len, self.vocab_size])

            max_text_len = tf.shape(text_tok)[1]
            text_mask = tf.sequence_mask(text_len, max_text_len)
            mask_logits_fw = tf.sequence_mask(text_len-1, max_text_len)
            mask_logits_bw = tf.sequence_mask(text_len, max_text_len)
            return (logits_fw, logits_bw), (mask_logits_fw, mask_logits_bw)
        elif task == "main":


            cat_hidden_2 = tf.nn.relu(tf.matmul(text_hidden, weights['w1']) + weights['b1'])
            if is_train:
                cat_hidden_2 = tf.nn.dropout(cat_hidden_2, keep_prob)
            final_output = tf.matmul(cat_hidden_2, weights['w2']) + weights['b2']
            #final_output = tf.Print(final_output, [final_output])
            return final_output



