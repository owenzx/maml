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
from tensorflow.python.ops.rnn import _reverse_seq #at least this can be imported for tensorflow 1.10
from utils import mse, xent, conv_block, normalize, read_pretrained_embeddings, rnncell, build_cells, xent_onehot, bi_seq_xent, seq_xent, get_approx_2nd_grad, convert_list_to_tensor, convert_tensor_to_list
from customized_cells import Customized_BasicLSTMCell
from bidaf import BiAttention_no_var
from multihead_attention import multihead_attention_no_var
from constants import *
from my_static_brnn import my_static_bidirectional_rnn

from decoders import lstm_att_decoder

from batch_ops import batch_embedding_lookup, batch_matmul

FLAGS = flags.FLAGS

class MAML:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5, max_len=None):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.classification = False
        self.test_num_updates = test_num_updates
        self.construct_model = self.construct_model_batch
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
            if FLAGS.use_static_rnn:
                self.max_len = max_len
            if FLAGS.task == 'usl_adapt':
                self.forward = self.forward_1input_rnn_usl
                self.construct_weights = self.construct_1input_rnn_weights_usl
                if FLAGS.aux_task == "lm":
                    self.aux_loss_func = bi_seq_xent
                else:
                    self.aux_loss_func = seq_xent
                self.real_loss_func = xent_onehot
            else:
                if FLAGS.model == "rnn":
                    self.forward = self.forward_rnn
                    self.construct_weights = self.construct_rnn_weights
                elif FLAGS.model == "bidaf":
                    self.forward = self.forward_better_rnn
                    self.construct_weights = self.construct_better_rnn_weights
                self.loss_func = xent_onehot
            self.vocab_size = 30000 + 2 #default choice
            self.dim_hidden = FLAGS.hidden_dim
            self.dim_output = FLAGS.num_classes
            self.classification = True
            self.dim_emb = FLAGS.dim_emb
            self.num_layers = FLAGS.num_rnn_layers
            if FLAGS.batch_mode:
                self.batch_size = FLAGS.meta_batch_size
            else:
                self.batch_size = FLAGS.update_batch_size
            #Other hyper-parameters
        else:
            raise ValueError('Unrecognized data source.')
    
    def set_pretrain_embedding(self, w, word2idx):
        self.pretrain_embedding = w
        self.vocab_size = len(word2idx)

    
    def construct_model_batch(self, input_tensors=None, prefix='metatrain_', max_len=0):

        
        self.inputa = input_tensors['inputa']
        self.inputb = input_tensors['inputb']
        self.labela, _ = input_tensors['labela']
        self.labelb = input_tensors['labelb']


        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights, weightsa_keys, weightsb_keys = self.weights, self.weightsa_keys, self.weightsb_keys
            else:
                # Define the weights
                self.weights, self.weightsa_keys, self.weightsb_keys = weights, weightsa_keys, weightsb_keys = self.construct_weights()

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            print("NUM_UPDATES: %d"%num_updates)
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates

            self.update_lrs = [[]]*num_updates
            ini = tf.constant_initializer(self.update_lr)
            for i in range(num_updates):
                self.update_lrs[i] = tf.get_variable("update_lr_%d"%i, [], initializer=ini, trainable=True)

            def main_pretrain(inputb, labelb, w, reuse=True, is_train=True):
                output = self.forward(inputb, w, reuse=True, is_train=is_train, task="main", max_len=max_len)
                accuracy = tf.contrib.metrics.accuracy(tf.argmax(output, 1), labelb)
                loss = self.real_loss_func(output, labelb)
                return loss, accuracy

            def aux_pretrain_freeze(inputa, labela, w, reuse=True, is_train=True):
                task_outputa, mask= self.forward(inputa, w, reuse=reuse, is_train=is_train, task="aux", max_len = max_len, pretrain_freeze=True)
                task_lossa = self.aux_loss_func(task_outputa, labela, mask)
                return task_lossa

            def task_metalearn(inputa, inputb, labela, labelb, w, reuse=True, is_train=True):
                task_outputbs, task_lossesb = [], []
                if self.classification:
                    task_accuraciesb = []
                task_outputa, mask= self.forward(inputa, w, reuse=reuse, is_train=is_train, task="aux", max_len = max_len)
                task_lossa = self.aux_loss_func(task_outputa, labela, mask)
                grads = tf.gradients(task_lossa, list(w.values()), aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(w.keys(), grads))

                def get_weight(we,key, weightsa_keys, weightsb_keys, lr):
                    if key not in weightsa_keys:
                        return we[key]
                    if key!='emb':
                        return  we[key] - lr*gradients[key] 
                    else:
                        return we[key] - lr*tf.convert_to_tensor(gradients[key])
                fast_weights = dict(zip(w.keys(), [get_weight(w, key, weightsa_keys, weightsb_keys, self.update_lrs[0]) for key in w.keys()]))
                output = self.forward(inputb, fast_weights, reuse=True, is_train = is_train, task="main", max_len=max_len)
                task_outputbs.append(output)
                task_lossesb.append(self.real_loss_func(output, labelb))

                for j in range(num_updates-1):
                    task_outputa, mask= self.forward(inputa, fast_weights, reuse=reuse, is_train=is_train, task="aux", max_len = max_len)
                    loss = self.aux_loss_func(task_outputa, labela, mask)
                    grads = tf.gradients(loss, list(fast_weights.values()), aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [get_weight(fast_weights, key, weightsa_keys, weightsb_keys, self.update_lrs[j+1]) for key in fast_weights.keys()]))
                    output = self.forward(inputb, fast_weights, reuse=True, is_train = is_train, task="main", max_len=max_len)
                    task_outputbs.append(output)
                    task_lossesb.append(self.real_loss_func(output, labelb))

                    

                #Now first suppose only one aux task
                task_output = [task_outputa, task_outputbs, task_lossa,task_lossesb]

                if self.classification:
                    for j in range(num_updates):
                        if FLAGS.datasource in NLP_DATASETS:
                            task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(task_outputbs[j], 1), labelb))
                    task_output.extend([task_accuraciesb])

                return task_output
            
            def get_stacked_weights(weights):
                stacked_w = dict()
                for k,w in weights.items():
                    stacked_w[k] = tf.stack([w for _ in range(FLAGS.meta_batch_size)])
                return stacked_w



            #if FLAGS.norm != 'None':
            #    # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
            #    unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [(tf.float32, tf.float32), [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
            if self.classification:
                out_dtype.extend([[tf.float32]*num_updates])
            w = get_stacked_weights(weights)
            result = task_metalearn(self.inputa, self.inputb, self.labela, self.labelb, w)
            if self.classification:
                outputas, outputbs, lossesa, lossesb, accuraciesb = result
            else:
                outputas, outputbs, lossesa, lossesb  = result
            
            main_pretrain_loss, main_pretrain_acc = main_pretrain(self.inputb, self.labelb, w)
            aux_pretrain_loss = aux_pretrain_freeze(self.inputa, self.labela, w)

        ## Performance & Optimization
        self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
        self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
        # after the map_fn
        self.outputas, self.outputbs = outputas, outputbs
        if self.classification:
            self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) for j in range(num_updates)]
#            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)
        
        if FLAGS.pretrain_epochs > 0:

            main_optimizer = tf.train.AdamOptimizer()
            aux_optimizer = tf.train.AdamOptimizer()
            self.main_pretrain_loss = tf.reduce_sum(main_pretrain_loss) / tf.to_float(FLAGS.meta_batch_size)
            self.aux_pretrain_loss = tf.reduce_sum(aux_pretrain_loss) / tf.to_float(FLAGS.meta_batch_size)
            self.main_pretrain_op = main_optimizer.minimize(self.main_pretrain_loss)
            self.aux_pretrain_op = aux_optimizer.minimize(self.aux_pretrain_loss)
            self.main_pretrain_acc = main_pretrain_acc


        if FLAGS.metatrain_epochs > 0:
            optimizer = tf.train.AdamOptimizer(self.meta_lr)
            if FLAGS.approx_2nd_grad:
                self.gvs = gvs = get_approx_2nd_grad(optimizer, self.total_loss1, self.total_losses2[FLAGS.num_updates-1], self.weights, self.update_lr, self.aux_loss_func, self.forward, self.inputa, True, True, self.labela)
                if FLAGS.clip_grad == True:
                    gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs if grad is not None]
            else:
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates-1], aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
                if FLAGS.clip_grad == True:
                    gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs if grad is not None]
            self.metatrain_op = optimizer.apply_gradients(gvs)
        self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
        self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
        if self.classification:
            self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) for j in range(num_updates)]

        ## Summaries
        tf.summary.scalar(prefix+'Unsupervied auxilary task loss', total_loss1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Real task loss, step ' + str(j+1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix+'Real task accuracy, step ' + str(j+1), total_accuracies2[j])


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
        return final_output
        
    def construct_1input_rnn_weights_usl(self):
        weights = {}
        self.cells = {}

        not_in_weightsa_keys = ['w1', 'b1', 'w2', 'b2']
        not_in_weightsb_keys = []
        
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

        num_rnn_hiddens = [self.dim_emb] + [2 * self.dim_hidden for _ in range(self.num_layers-1)]
        #All the rnn cells must be built so that their weights can be accessed
        build_cells(self.cells['text_cell_forw'], num_rnn_hiddens)
        build_cells(self.cells['text_cell_back'], num_rnn_hiddens)


        #Note that currently, the following code only works for LSTM cells 
        for i, c in enumerate(self.cells['text_cell_forw']):
            weights['text_cell_forw_%d_w'%i], weights['text_cell_forw_%d_b'%i] = c.weights
        for i, c in enumerate(self.cells['text_cell_back']):
            weights['text_cell_back_%d_w'%i], weights['text_cell_back_%d_b'%i] = c.weights
        #print([type(weights[c]) for c in weights.keys()])
        
        if FLAGS.num_attn_head > 0 and self.num_layers>1:
            weights["text_self_att_q"] = tf.get_variable("text_self_att_q", [2*self.dim_hidden, 2*self.dim_hidden])
            weights["text_self_att_k"] = tf.get_variable("text_self_att_k", [2*self.dim_hidden, 2*self.dim_hidden])
            weights["text_self_att_v"] = tf.get_variable("text_self_att_v", [2*self.dim_hidden, 2*self.dim_hidden])
        
        if FLAGS.aux_task == 'lm':
            if not FLAGS.bind_embedding_softmax:
                weights['w_sm_forw'] = tf.get_variable('w_sm_forw', [self.dim_hidden, self.vocab_size], initializer = fc_initializer)
                weights['b_sm_forw'] = tf.Variable(tf.zeros([self.vocab_size]), name='b_sm_forw')
                weights['w_sm_back'] = tf.get_variable('w_sm_back', [self.dim_hidden, self.vocab_size], initializer = fc_initializer)
                weights['b_sm_back'] = tf.Variable(tf.zeros([self.vocab_size]), name='b_sm_back')
                not_in_weightsb_keys.extend(['w_sm_forw', 'b_sm_forw', 'w_sm_back', 'b_sm_back'])
        elif FLAGS.aux_task == 'auto_encoder':
            self.cells['decoder_cell'] =[rnncell(self.dim_hidden) for _ in range(1)]
            num_rnn_hiddens_decoder = [self.dim_emb]
            build_cells(self.cells['decoder_cell'], num_rnn_hiddens_decoder)
            for i,c in enumerate(self.cells['decoder_cell']):
                weights['decoder_cell_%d_w'%i], weights['decoder_cell_%d_b'%i] = c.weights
                not_in_weightsb_keys.extend(['decoder_cell_%d_w'%i, 'decoder_cell_%d_b'%i])
            weights['decoder_init_state_h_w']=tf.get_variable('decoder_init_state_h_w', [self.dim_hidden*2, self.dim_hidden], initializer=fc_initializer)
            weights['decoder_init_state_c_w']=tf.get_variable('decoder_init_state_c_w', [self.dim_hidden*2, self.dim_hidden], initializer=fc_initializer)
            if FLAGS.decoder_attention and FLAGS.num_attn_head > 0:
                weights["decoder_att_q"] = tf.get_variable("decoder_att_q", [self.dim_hidden, self.dim_hidden])
                weights["decoder_att_k"] = tf.get_variable("decoder_att_k", [2*self.dim_hidden, self.dim_hidden])
                weights["decoder_att_v"] = tf.get_variable("decoder_att_v", [2*self.dim_hidden, self.dim_hidden])
                weights["decoder_att_w"] = tf.get_variable("decoder_att_w", [2*self.dim_hidden, self.dim_hidden])

        weightsa_keys = [k for k in weights.keys() if k not in not_in_weightsa_keys]
        weightsb_keys = [k for k in weights.keys() if k not in not_in_weightsb_keys]

        return weights, weightsa_keys, weightsb_keys

    def forward_1input_rnn_usl(self, inp, weights, reuse=False, scope='', is_train=True, task="main", max_len=None, pretrain_freeze=False):
        #For all the batch sequence, assume the default dimension permuation is [seq_len, batch, real_dim]
        cells = dict()
        keep_prob = 1-FLAGS.dropout_rate

        cells['text_cell_forw'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]
        cells['text_cell_back'] = [rnncell(self.dim_hidden) for _ in range(self.num_layers)]

        if task=="aux":
            if FLAGS.aux_task=="auto_encoder":
                cells['decoder_cell'] = [rnncell(self.dim_hidden) for _ in range(1)]
        for k in cells.keys():
            for n in range(len(self.cells[k])):
                cells[k][n].update_weights(weights[k+"_%d_"%n+"w"], weights[k+"_%d_"%n+"b"])
                if is_train:
                    cells[k][n] = tf.nn.rnn_cell.DropoutWrapper(cells[k][n], input_keep_prob=keep_prob)

        if task=="aux":
            if FLAGS.aux_task=="auto_encoder":
                decoder = lstm_att_decoder(dim_hidden=self.dim_hidden, dim_input=self.dim_emb, cells=cells['decoder_cell'], max_len=max_len)


        text_tok, text_len = inp

        text_vec = batch_embedding_lookup(weights['emb'], text_tok)

        max_text_len = tf.shape(text_vec)[1]

        text_mask = tf.sequence_mask(text_len, max_text_len)

        output = text_vec
        if FLAGS.use_static_rnn:
            text_vec = tf.transpose(text_vec, perm=[1,0,2])
            output = tf.unstack(text_vec, num=max_len)
        for n in range(self.num_layers):
            cell_fw = cells['text_cell_forw'][n]
            cell_bw = cells['text_cell_back'][n]

            state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
            state_bw = cell_bw.zero_state(self.batch_size, tf.float32)
            #The sequence_length parameter here is not used in order to avoid using tf.while 
            (output, output_fw_l, output_bw_l), (last_output_fw, last_output_bw) = my_static_bidirectional_rnn(cell_fw, cell_bw, output, sequence_length=text_len, initial_state_fw = state_fw, initial_state_bw = state_bw, scope = 'BLSTM_text_'+str(n), dtype = tf.float32)
            if (n==self.num_layers-2) and (FLAGS.num_attn_head > 0):
                if FLAGS.use_static_rnn:
                    output = convert_list_to_tensor(output)
                output = multihead_attention_no_var(output, output, queries_mask=text_mask, keys_mask=text_mask, num_units = self.dim_hidden , num_heads = FLAGS.num_attn_head, is_training=is_train, scope = "text_self_att", weights= weights, prefix = "text_self_att_")
                if FLAGS.use_static_rnn:
                    output = convert_tensor_to_list(output, max_len)
        if not FLAGS.use_static_rnn:
            last_fw, last_bw = last_state
            text_hidden = tf.concat([last_fw.h, last_bw.h], axis = -1)
            text_hidden_c = tf.concat([last_fw.c, last_bw.c], axis=-1)
        else:
            output_fw = convert_list_to_tensor(output_fw_l)
            #batch_idx = tf.reshape(tf.range(self.batch_size, dtype=tf.int64), [-1,1])
            #text_len_idx = tf.reshape(text_len, [-1,1])
            #fw_idx = tf.concat([batch_idx, text_len_idx-1], axis=-1)
            #last_fw = tf.gather_nd(output_fw, fw_idx)
            
            output_bw = convert_list_to_tensor(output_bw_l)
            #zero_idx = tf.zeros((self.batch_size,1),dtype=tf.int64)
            #bw_idx = tf.concat([batch_idx,zero_idx], axis=-1)
            #last_bw = tf.gather_nd(output_bw, bw_idx)
            #text_hidden = tf.concat([last_fw, last_bw], axis=-1)
            text_hidden = tf.concat([last_output_fw.h, last_output_bw.h], axis=-1)
            text_hidden_c = tf.concat([last_output_fw.c, last_output_bw.c], axis=-1)

        if (task=="aux") or (task=="both"):
            if pretrain_freeze:
                output_fw = tf.stop_gradient(output_fw)
                output_bw = tf.stop_gradient(output_bw)
                text_hidden = tf.stop_gradient(text_hidden)
                text_hidden_c = tf.stop_gradient(text_hidden)

            if FLAGS.aux_task=="lm":
            
                if not FLAGS.bind_embedding_softmax:
                    logits_fw = tf.matmul(output_fw, weights['w_sm_forw']) + tf.expand_dims(weights['b_sm_forw'], axis=1)
                    logits_bw = tf.matmul(output_bw, weights['w_sm_back']) + tf.expand_dims(weights['b_sm_back'], axis=1)
                else:
                    w_sm = tf.transpose(weights['emb'],perm=[0,2,1])
                    #w_sm = tf.transpose(weights['emb'])
                    logits_fw = tf.matmul(output_fw, w_sm)
                    logits_bw = tf.matmul(output_bw, w_sm)
                #logits_fw = tf.reshape(logits_fw, [batch_size, max_text_len, self.vocab_size])
                #logits_bw = tf.reshape(logits_bw, [batch_size, max_text_len, self.vocab_size])

                max_text_len = tf.shape(text_tok)[1]
                text_mask = tf.sequence_mask(text_len, max_text_len)
                mask_logits_fw = tf.sequence_mask(text_len-1, max_text_len)
                mask_logits_bw = tf.sequence_mask(text_len, max_text_len)
                #return (logits_fw, logits_bw), (mask_logits_fw, mask_logits_bw)
            elif FLAGS.aux_task=="auto_encoder":
                init_state = batch_matmul(text_hidden_c,weights['decoder_init_state_c_w']), batch_matmul(text_hidden, weights['decoder_init_state_h_w'])
                mask_logits = tf.sequence_mask(text_len, max_text_len)
                output_logits = decoder(init_state=init_state, att_vecs=output, att_mask =mask_logits ,weights=weights, reuse=reuse)
                #return output_logits, mask_logits
        elif (task == "main") or (task=="both"):
            cat_hidden_2 = normalize(batch_matmul(text_hidden, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope="main_f_1")
            if is_train:
                cat_hidden_2 = tf.nn.dropout(cat_hidden_2, keep_prob)
            final_output = batch_matmul(cat_hidden_2, weights['w2']) + weights['b2']
            #return final_output
        
        if task=="aux":
            if FLAGS.aux_task == "lm":
                return (logits_fw, logits_bw), (mask_logits_fw, mask_logits_bw)
            elif FLAGS.aux_task == "auto_encoder":
                return output_logits, mask_logits
        elif task == "main":
            return final_output
        elif task == "both":
            if FLAGS.aux_task == "lm":
                return (logits_fw, logits_bw), (mask_logits_fw, mask_logits_bw), final_output
            elif FLAGS.aux_task == "auto_encoder":
                return output_logits, mask_logits, final_output



