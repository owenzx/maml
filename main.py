"""
Usage Instructions:
    10-shot sinusoid:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10

    10-shot sinusoid baselines:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10 --baseline=oracle
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10

    5-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/

    20-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/

    5-way 1-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True

    5-way 5-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet5shot/ --num_filters=32 --max_pool=True

    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.

    For omniglot and miniimagenet training, acquire the dataset online, put it in the correspoding data directory, and see the python script instructions in that directory to preprocess the data.
"""
import csv
import numpy as np
import pickle
import random
import tensorflow as tf
import os
import sys

from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags
from utils import read_pretrained_embeddings
from constants import *

from tensorflow.python.client import timeline

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_string('train_datasets', '', 'the datasets for training, use comma to saparate')
flags.DEFINE_string('test_datasets', '', 'the datasets for testing')
flags.DEFINE_string('multi_datasets', '', 'the last dataset is used for testing, others for training')
flags.DEFINE_bool('switch_datasets', True, 'whether switch the train and val dataset when using multiple datasets for transfer')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')

flags.DEFINE_integer('pretrain_epochs', 0, 'number of pre-training epochs')
flags.DEFINE_integer('metatrain_epochs', 0, 'number of metatraining epochs')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_integer('num_rnn_layers', 3, 'number of rnn layers')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

flags.DEFINE_bool('zeroshot', False, 'if true, set number of updates when testing to zero.')

flags.DEFINE_string('pretrain_embedding', 'none', 'what kind of pretrained word embeddings to load')
flags.DEFINE_integer('vocab_size', 40000, 'the size of vocabulary, default set to 40000, which is a common setting for GloVe')
flags.DEFINE_string('pretrain_embedding_path', '', 'the path of the pretrained embedding file')

flags.DEFINE_integer('gpu_id', -1, 'the id of the gpu to use')

flags.DEFINE_string('absa_domain', 'restaurant', 'specific domain of the absa dataset')
flags.DEFINE_string('model', 'bidaf', 'choose the model used in nlp experiments')

flags.DEFINE_bool('q2c', True, '')
flags.DEFINE_bool('query_dots', True, '')
flags.DEFINE_float('dropout_rate', 0.1, 'dropout rate')

flags.DEFINE_integer('num_attn_head', 0, 'num of head in multi-head attention, set 0 to disable')

flags.DEFINE_string('task', 'single_dataset', 'determine what task is running')

flags.DEFINE_bool('bind_embedding_softmax', False, "whether use the same parameter for embedding and the softmax layer")

flags.DEFINE_bool('approx_2nd_grad', False, "Set to true to use approximated second-order gradient when stop_grad is set to True")
flags.DEFINE_bool('clip_grad', False, 'Whether to clip the grad of not, default range is -10 to 10')
flags.DEFINE_integer('hidden_dim', 300, "default dimension for most of the hidden layers")

flags.DEFINE_bool('TF_USE_CUDNN', True, "set to True to use CUDNN")
flags.DEFINE_bool('use_static_rnn', False, 'set to True to use static rnn instead of dynamic rnn')
flags.DEFINE_bool('debug', False, 'whether run in debug mode')
flags.DEFINE_integer('dim_emb', 200, 'the dimension of the embedding')

def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    if FLAGS.datasource == 'sinusoid':
        PRINT_INTERVAL = 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
        if 'generate' in dir(data_generator):
            batch_x, batch_y, amp, phase = data_generator.generate()

            if FLAGS.baseline == 'oracle':
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                for i in range(FLAGS.meta_batch_size):
                    batch_x[i, :, 1] = amp[i]
                    batch_x[i, :, 2] = phase[i]

            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :] # b used for testing
            labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}

        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0 and FLAGS.datasource !='sinusoid':
            if 'generate' not in dir(data_generator):
                feed_dict = {}
                if model.classification:
                    input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1], model.summ_op]
                else:
                    input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1], model.summ_op]
            else:
                batch_x, batch_y, amp, phase = data_generator.generate(train=False)
                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :]
                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
                if model.classification:
                    input_tensors = [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]]
                else:
                    input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates-1]]

            result = sess.run(input_tensors, feed_dict)
            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))


def train_transfer(model, saver, sess, exp_string, data_generator, resume_epoch=0):
    SUMMARY_INTERVAL = 1
    SAVE_INTERVAL = 1

    PRINT_INTERVAL = 1
    TEST_PRINT_INTERVAL = PRINT_INTERVAL

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []
    test_prelosses, test_postlosses = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    test_init_op_inputa, test_init_op_inputb, test_init_op_labela, test_init_op_labelb = data_generator.test_init_ops

    dataset_handles = [[sess.run(itr.string_handle()) for itr in itrs] for itrs in data_generator.dataset_itrs]
    train_handles = dataset_handles[:-1]
    test_handles = dataset_handles[-1:]

    itr = 0
    for epoch in range(resume_epoch, FLAGS.pretrain_epochs + FLAGS.metatrain_epochs):
        feed_dict = {}
        #sess.run(init_op_inputa)
        #sess.run(init_op_labela)
        #sess.run(init_op_inputb)
        #sess.run(init_op_labelb)

        for i in range(100):
            handles = train_handles[i%len(train_handles)]
            #print(handles)
            #print(type(handles))
            feed_dict = {data_generator.handle_inputa:handles[0], data_generator.handle_labela:handles[1], data_generator.handle_inputb:handles[2], data_generator.handle_labelb:handles[3]}
            #print(i)
            #sess.run([model.pretrain_op], feed_dict=feed_dict)
            #sess.run([model.metatrain_op], feed_dict=feed_dict)
            #print(-i)
            if epoch < FLAGS.pretrain_epochs:
                input_tensors = [model.pretrain_op]
            else:
                input_tensors = [model.metatrain_op]
            if (itr % SUMMARY_INTERVAL == 0):
                #input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
                input_tensors.extend([model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
                if model.classification:
                    input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])                
            #print(i)
            result = sess.run(input_tensors, feed_dict=feed_dict)
            if itr % SUMMARY_INTERVAL == 0:
                    prelosses.append(result[-2])
                    #if FLAGS.log:
                    #    train_writer.add_summary(result[1], itr)
                    postlosses.append(result[-1])

        if epoch % PRINT_INTERVAL == 0:
            if epoch < FLAGS.pretrain_epochs:
                print_str = 'Pretrain Epoch ' + str(epoch)
            else:
                print_str = 'Epoch ' + str(epoch - FLAGS.pretrain_epochs)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if epoch % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(epoch))

        if epoch % TEST_PRINT_INTERVAL == 0 and FLAGS.datasource !='sinusoid':
            sess.run(test_init_op_inputa)
            sess.run(test_init_op_labela)
            sess.run(test_init_op_inputb)
            sess.run(test_init_op_labelb)
            handles = test_handles[0]
            feed_dict = {data_generator.handle_inputa:handles[0], data_generator.handle_labela:handles[1], data_generator.handle_inputb:handles[2], data_generator.handle_labelb:handles[3]}
            while True:
                try:
                    #feed_dict = {}
                    if model.classification:
                        input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1]]
                    else:
                        input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1]]
                    result = sess.run(input_tensors, feed_dict)
                    test_prelosses.append(result[-2])
                    test_postlosses.append(result[-1])
                except tf.errors.OutOfRangeError:
                    break
                
            #print('Test_prelosses: ' + str(test_prelosses) + ', Test_postlosses: ' + str(test_postlosses))
            print('Validation results: ' + str(np.mean(test_prelosses)) + ', ' + str(np.mean(test_postlosses)))
            test_prelosses, test_postlosses = [], []

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

def train_dataset(model, saver, sess, exp_string, data_generator, resume_epoch=0, train_set_init_ops=None, test_set_init_ops=None):
    SUMMARY_INTERVAL = 1
    SAVE_INTERVAL = 1

    PRINT_INTERVAL = 1
    TEST_PRINT_INTERVAL = PRINT_INTERVAL

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []
    test_prelosses, test_postlosses = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    init_op_inputa, init_op_inputb, init_op_labela, init_op_labelb = train_set_init_ops

    test_init_op_inputa, test_init_op_inputb, test_init_op_labela, test_init_op_labelb = test_set_init_ops

    itr = 0

    for epoch in range(resume_epoch, FLAGS.pretrain_epochs + FLAGS.metatrain_epochs):
        feed_dict = {}
        sess.run(init_op_inputa)
        sess.run(init_op_labela)
        sess.run(init_op_inputb)
        sess.run(init_op_labelb)


        while True:
            try:
                if epoch < FLAGS.pretrain_epochs:
                    input_tensors = [model.pretrain_op]
                else:
                    input_tensors = [model.metatrain_op]
                #input_tensors.extend([model.debug_grads])

                itr += 1
                #print(itr)
                if (itr % SUMMARY_INTERVAL == 0):
                    #input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
                    if epoch < FLAGS.pretrain_epochs:
                        input_tensors.extend([model.total_loss1])
                        if model.classification:
                            input_tensors.extend([model.total_accuracy1])                
                    else:
                        input_tensors.extend([model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
                        if model.classification:
                            input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])                
                result = sess.run(input_tensors)
                if itr % SUMMARY_INTERVAL == 0:
                    if epoch < FLAGS.pretrain_epochs:
                        prelosses.append(result[-1])
                    else:
                        prelosses.append(result[-2])
                        postlosses.append(result[-1])                

            except tf.errors.OutOfRangeError:
                break

        if epoch % PRINT_INTERVAL == 0:
            if epoch < FLAGS.pretrain_epochs:
                print_str = 'Pretrain Epoch ' + str(epoch)
                print_str += ': ' + str(np.mean(prelosses))
            else:
                print_str = 'Epoch ' + str(epoch - FLAGS.pretrain_epochs)
                print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            sys.stdout.flush()

            prelosses, postlosses = [], []

        if epoch % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(epoch))

        if FLAGS.metatrain_epochs == 0:
            no_meta = True
        else:
            no_meta = False

        if epoch % TEST_PRINT_INTERVAL == 0 and FLAGS.datasource !='sinusoid':
            sess.run(test_init_op_inputa)
            sess.run(test_init_op_labela)
            sess.run(test_init_op_inputb)
            sess.run(test_init_op_labelb)
            while True:
                try:
                    feed_dict = {}
                    if model.classification:
                        if no_meta:
                            input_tensors = [model.metaval_total_accuracy1]
                        else:
                            input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1]]
                    else:
                        if no_meta:
                            input_tensors = [model.metaval_total_loss1]
                        else:
                            input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1]]
                    result = sess.run(input_tensors, feed_dict)
                    if no_meta:
                        test_prelosses.append(result[-1])
                    else:
                        test_prelosses.append(result[-2])
                        test_postlosses.append(result[-1])
                    #print("running")
                except tf.errors.OutOfRangeError:
                    break
                
            #print('Test_prelosses: ' + str(test_prelosses) + ', Test_postlosses: ' + str(test_postlosses))
            if no_meta:
                print('Validation results: ' + str(np.mean(test_prelosses)))
            else:
                print('Validation results: ' + str(np.mean(test_prelosses)) + ', ' + str(np.mean(test_postlosses)))
            test_prelosses, test_postlosses = [], []

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

def train_usl(model, saver, sess, exp_string, data_generator, resume_epoch=0):
    SUMMARY_INTERVAL = 1
    SAVE_INTERVAL = 1

    PRINT_INTERVAL = 1
    TEST_PRINT_INTERVAL = PRINT_INTERVAL

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')

    auxlosses, reallosses, realacces = [], [], []
    test_auxlosses, test_reallosses, test_realacces = [], [], []

    train_init_op_input, train_init_op_label = data_generator.train_init_ops
    test_init_op_input, test_init_op_label = data_generator.test_init_ops

    dataset_handles = [[sess.run(itr.string_handle()) for itr in itrs] for itrs in data_generator.dataset_itrs]

    train_handles = dataset_handles[0]
    test_handles = dataset_handles[1]

    itr = 0

    for epoch in range(resume_epoch, FLAGS.pretrain_epochs + FLAGS.metatrain_epochs):
        feed_dict = {}

        sess.run(train_init_op_input)
        sess.run(train_init_op_label)
        handles = train_handles
        feed_dict = {data_generator.handle_input: handles[0],
                     data_generator.handle_label: handles[1]}
        while True:
            try:
                if epoch < FLAGS.pretrain_epochs:
                    pass
                else:
                    input_tensors = [model.metatrain_op]
                itr += 1
                if FLAGS.debug:
                    print(itr)

                if epoch < FLAGS.pretrain_epochs:
                    pass
                else:
                    input_tensors.extend([model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
                    if model.classification:
                        input_tensors.extend([model.total_accuracies2[FLAGS.num_updates-1]])                
                result = sess.run(input_tensors, feed_dict=feed_dict)
                auxlosses.append(result[1])
                reallosses.append(result[2])
                if model.classification:
                    realacces.append(result[-1])                

            except tf.errors.OutOfRangeError:
                break

        if epoch % PRINT_INTERVAL == 0:
            print_str = 'Epoch ' + str(epoch - FLAGS.pretrain_epochs)
            if model.classification:
                print_str += 'aux loss: ' + str(np.mean(auxlosses)) + ', real loss: ' + str(np.mean(reallosses)) + ', real acc: ' + str(np.mean(realacces))
            else:
                print_str += 'aux loss: ' + str(np.mean(auxlosses)) + ', real loss: ' + str(np.mean(reallosses))
            print(print_str)
            sys.stdout.flush()

            auxlosses, reallosses, realacces = [], [], []

        if epoch % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(epoch))

        feed_dict = {}

        if epoch % TEST_PRINT_INTERVAL == 0 and FLAGS.datasource !='sinusoid':
            sess.run(test_init_op_input)
            sess.run(test_init_op_label)
            handles = test_handles
            feed_dict = {data_generator.handle_input: handles[0],
                        data_generator.handle_label: handles[1]}
            while True:
                try:
                    if model.classification:
                        input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1], model.metaval_total_accuracies2[FLAGS.num_updates-1]]
                    else:
                        input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1]]
                    result = sess.run(input_tensors, feed_dict=feed_dict)
                    test_auxlosses.append(result[0])
                    test_reallosses.append(result[1])
                    if model.classification:
                        test_realacces.append(result[-1])                
                    if FLAGS.debug:
                        print("testing")
                except tf.errors.OutOfRangeError:
                    break
                
            if model.classification:
                print('Validation results: aux loss: ' + str(np.mean(test_auxlosses)) + ', real loss: ' + str(np.mean(test_reallosses)) + ', real acc: ' + str(np.mean(test_realacces)))
            else:
                print('Validation results: aux loss: ' + str(np.mean(test_auxlosses)) + ', real loss: ' + str(np.mean(test_reallosses)))
            sys.stdout.flush()
            test_auxlosses, test_reallosses, test_realacces = [], [], []

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

def test_usl(model, saver, sess, exp_string, data_generator):
    pass



# calculated for omniglot
NUM_TEST_POINTS = 600

def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []

    for _ in range(NUM_TEST_POINTS):
        if 'generate' not in dir(data_generator):
            feed_dict = {}
            feed_dict = {model.meta_lr : 0.0}
        else:
            batch_x, batch_y, amp, phase = data_generator.generate(train=False)

            if FLAGS.baseline == 'oracle': # NOTE - this flag is specific to sinusoid
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                batch_x[0, :, 1] = amp[0]
                batch_x[0, :, 2] = phase[0]

            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = batch_x[:,num_classes*FLAGS.update_batch_size:, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            labelb = batch_y[:,num_classes*FLAGS.update_batch_size:, :]

            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

        if model.classification:
            result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
        else:  # this is for sinusoid
            result = sess.run([model.total_loss1] +  model.total_losses2, feed_dict)
        metaval_accuracies.append(result)
    print(metaval_accuracies)
    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)


def test_transfer(model, saver, sess, exp_string, data_generator, resume_epoch=0):
    """currently only support testing on one dataset"""
    SUMMARY_INTERVAL = 1
    SAVE_INTERVAL = 1

    PRINT_INTERVAL = 1
    TEST_PRINT_INTERVAL = PRINT_INTERVAL

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []
    test_prelosses, test_postlosses = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    test_init_op_inputa, test_init_op_inputb, test_init_op_labela, test_init_op_labelb = data_generator.test_init_ops

    dataset_handles = [(sess.run(itr.string_handle()) for itr in itrs) for itrs in data_generator.dataset_itrs]
    train_handles = dataset_handles[:-1]
    test_handles = dataset_handles[-1:]

    sess.run(test_init_op_inputa)
    sess.run(test_init_op_labela)
    sess.run(test_init_op_inputb)
    sess.run(test_init_op_labelb)
    handles = test_handles[0]
    feed_dict = {model.meta_lr : 0.0, data_generator.handle_inputa:handles[0], data_generator.handle_labela:handles[1], data_generator.handle_inputb:handles[2], data_generator.handle_labelb:handles[3]}
    while True:
        try:
            #feed_dict = {}
            if model.classification:
                input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1]]
            else:
                input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1]]
            result = sess.run(input_tensors, feed_dict)
            test_prelosses.append(result[-2])
            test_postlosses.append(result[-1])
        except tf.errors.OutOfRangeError:
            break
        
    print('Test results: ' + str(np.mean(test_prelosses)) + ', ' + str(np.mean(test_postlosses)))


def test_dataset(model, saver, sess, exp_string, data_generator, test_num_updates=None, test_set_init_ops=None):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []

    test_init_op_inputa, test_init_op_inputb, test_init_op_labela, test_init_op_labelb = test_set_init_ops

    sess.run(test_init_op_inputa)
    sess.run(test_init_op_labela)
    sess.run(test_init_op_inputb)
    sess.run(test_init_op_labelb)

    feed_dict = {model.meta_lr : 0.0}
    test_prelosses, test_postlosses = [], []

    while True:
        try:
            feed_dict = {}
            if model.classification:
                input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1]]
            else:
                input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1]]
            result = sess.run(input_tensors, feed_dict)
            test_prelosses.append(result[-2])
            test_postlosses.append(result[-1])
        except tf.errors.OutOfRangeError:
            break
        
    print('Test results: ' + str(np.mean(test_prelosses)) + ', ' + str(np.mean(test_postlosses)))
    

def main():
    print(FLAGS.gpu_id)
    if FLAGS.gpu_id == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif FLAGS.gpu_id == -2:
        pass
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_id)

    if FLAGS.TF_USE_CUDNN == True:
        pass
    else:
        os.environ["TF_USE_CUDNN"] = False
    
    if FLAGS.datasource == 'sinusoid':
        if FLAGS.train:
            test_num_updates = 5
        else:
            test_num_updates = 10
    else:
        if FLAGS.datasource == 'miniimagenet':
            if FLAGS.train == True:
                test_num_updates = 1  # eval on at least one update during training
            else:
                test_num_updates = 10
        else:
            test_num_updates = FLAGS.num_updates
    
    
    if FLAGS.zeroshot == True:
        test_num_updates = 0

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1


    if FLAGS.pretrain_embedding != 'none':
        weights_emb, word2idx = read_pretrained_embeddings()


    if FLAGS.datasource == 'sinusoid':
        data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
    else:
        if FLAGS.metatrain_iterations == 0 and FLAGS.datasource == 'miniimagenet':
            assert FLAGS.meta_batch_size == 1
            assert FLAGS.update_batch_size == 1
            data_generator = DataGenerator(1, FLAGS.meta_batch_size)  # only use one datapoint,
        else:
            if FLAGS.datasource == 'miniimagenet': # TODO - use 15 val examples for imagenet?
                if FLAGS.train:
                    data_generator = DataGenerator(FLAGS.update_batch_size+15, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
                else:
                    data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
            elif FLAGS.datasource in NLP_DATASETS:
                data_generator = DataGenerator(FLAGS.update_batch_size, FLAGS.meta_batch_size)
            else:
                data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory


    dim_output = data_generator.dim_output
    if FLAGS.baseline == 'oracle':
        assert FLAGS.datasource == 'sinusoid'
        dim_input = 3
        FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
        FLAGS.metatrain_iterations = 0
    else:
        dim_input = data_generator.dim_input

    if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'omniglot':
        tf_data_load = True
        num_classes = data_generator.num_classes

        if FLAGS.train: # only construct training model if needed
            random.seed(5)
            image_tensor, label_tensor = data_generator.make_data_tensor()
            print(image_tensor.shape)
            print(label_tensor.shape)
            inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

        random.seed(6)
        image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
        inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
    elif FLAGS.datasource == 'absa' or FLAGS.datasource == 'SNLI':
        tf_data_load = True
        if FLAGS.train:
            random.seed(5)
            (inputa, inputb, labela, labelb), train_set_init_ops = data_generator.make_data_tensor(word2idx)

            
            input_tensors = {'inputa':inputa, 'inputb': inputb, 'labela':labela, 'labelb':labelb}
            #input_tensors = {'inputa': {'text':tf.placeholder(tf.float32), 'ctgr':tf.placeholder(tf.float32)}, 'inputb': {'text':tf.placeholder(tf.float32), 'ctgr':tf.placeholder(tf.float32)}, 'labela': tf.placeholder(tf.float32), 'labelb': tf.placeholder(tf.float32)}
        random.seed(6)
        (t_inputa, t_inputb, t_labela, t_labelb), test_set_init_ops = data_generator.make_data_tensor(word2idx, train=False)
        metaval_input_tensors = {'inputa':t_inputa, 'inputb': t_inputb, 'labela':t_labela, 'labelb':t_labelb}
        #metaval_input_tensors = {'inputa': {'text':tf.placeholder(tf.float32), 'ctgr':tf.placeholder(tf.float32)}, 'inputb': {'text':tf.placeholder(tf.float32), 'ctgr':tf.placeholder(tf.float32)}, 'labela': tf.placeholder(tf.float32), 'labelb': tf.placeholder(tf.float32)}
    elif FLAGS.datasource == 'transfer_multi':
        tf_data_load = True
        if FLAGS.train:
            random.seed(5)
            inputa, inputb, labela, labelb = data_generator.make_data_tensor(word2idx)

            
            input_tensors = {'inputa':inputa, 'inputb': inputb, 'labela':labela, 'labelb':labelb}

        metaval_input_tensors = {'inputa':inputa, 'inputb': inputb, 'labela':labela, 'labelb':labelb}
    elif FLAGS.task == 'usl_adapt':
        tf_data_load = True
        if FLAGS.train:
            random.seed(5)
            (next_text, next_label) = data_generator.make_data_tensor(word2idx)
            input_tensors = {'inputa':next_text, 'inputb':next_text, 'labela':next_text, 'labelb':next_label}
    else:
        tf_data_load = False
        input_tensors = None

        
    if FLAGS.use_static_rnn:
        max_len = data_generator.max_len
        train_max_len = data_generator.train_max_len
        if FLAGS.test_set == True:
            test_max_len = data_generator.test_max_len
        else:
            test_max_len = data_generator.dev_max_len
    else:
        max_len = train_max_len = test_max_len =  None
    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates, max_len = max_len)
    if FLAGS.pretrain_embedding!='none':
        model.set_pretrain_embedding(weights_emb, word2idx)
    if FLAGS.task == "usl_adapt": 
        model.construct_model(input_tensors=input_tensors, prefix = "train+val", max_len = max_len)
    else:
        if FLAGS.train or not tf_data_load:
            model.construct_model(input_tensors=input_tensors, prefix='metatrain_', max_len=train_max_len)
        if tf_data_load:
            model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_', max_len=test_max_len)
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    sess = tf.Session(config=sess_config)
    #sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)

    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')

    resume_itr = 0
    model_file = None

    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    if FLAGS.train:
        if FLAGS.task == 'usl_adapt':
            train_usl(model, saver, sess, exp_string, data_generator, resume_itr)
        elif FLAGS.datasource in ['absa', 'SNLI']:
            train_dataset(model, saver, sess, exp_string, data_generator, resume_itr, train_set_init_ops, test_set_init_ops)
        elif FLAGS.datasource =='transfer_multi':
            train_transfer(model, saver, sess, exp_string, data_generator, resume_itr) 
        else:
            train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        if FLAGS.task == 'usl_adapt':
            test_usl(model, saver, sess, exp_string, data_generator, resume_itr)
        if FLAGS.datasource in ['absa', 'SNLI']:
            test_dataset(model, saver, sess, exp_string, data_generator, test_num_updates, test_set_init_ops)
        elif FLAGS.datasource == 'transfer_multi':
            test_transfer(model, saver, sess, exp_string, data_generator, resume_itr)
        else:
            test(model, saver, sess, exp_string, data_generator, test_num_updates)

if __name__ == "__main__":
    main()
