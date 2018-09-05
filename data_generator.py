""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import get_images, get_pad_batch, get_pad_metabatch, get_batch_labels, get_metabatch_labels, get_static_pad_batch
from nlp_data_reader import read_absa_restaurants, read_absa_laptops, read_target_dependent, readTopic3Way, read_snli, read_sst
import nltk
from constants import *
FLAGS = flags.FLAGS

class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        #Here batch_size means META batch size
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)

        if FLAGS.datasource == 'sinusoid':
            self.generate = self.generate_sinusoid_batch
            self.amp_range = config.get('amp_range', [0.1, 5.0])
            self.phase_range = config.get('phase_range', [0, np.pi])
            self.input_range = config.get('input_range', [-5.0, 5.0])
            self.dim_input = 1
            self.dim_output = 1
        elif 'omniglot' in FLAGS.datasource:
            self.make_data_tensor = self.make_data_tensor_image
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (28, 28))
            self.dim_input = np.prod(self.img_size)
            self.dim_output = self.num_classes
            # data that is pre-resized using PIL with lanczos filter
            data_folder = config.get('data_folder', './data/omniglot_resized')

            character_folders = [os.path.join(data_folder, family, character) \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
            random.seed(1)
            random.shuffle(character_folders)
            num_val = 100
            num_train = config.get('num_train', 1200) - num_val
            self.metatrain_character_folders = character_folders[:num_train]
            if FLAGS.test_set:
                self.metaval_character_folders = character_folders[num_train+num_val:]
            else:
                self.metaval_character_folders = character_folders[num_train:num_train+num_val]
            self.rotations = config.get('rotations', [0, 90, 180, 270])
        elif FLAGS.datasource == 'miniimagenet':
            self.make_data_tensor = self.make_data_tensor_image
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size)*3
            self.dim_output = self.num_classes
            metatrain_folder = config.get('metatrain_folder', './data/miniImagenet/train')
            if FLAGS.test_set:
                metaval_folder = config.get('metaval_folder', './data/miniImagenet/test')
            else:
                metaval_folder = config.get('metaval_folder', './data/miniImagenet/val')

            metatrain_folders = [os.path.join(metatrain_folder, label) \
                for label in os.listdir(metatrain_folder) \
                if os.path.isdir(os.path.join(metatrain_folder, label)) \
                ]
            metaval_folders = [os.path.join(metaval_folder, label) \
                for label in os.listdir(metaval_folder) \
                if os.path.isdir(os.path.join(metaval_folder, label)) \
                ]
            self.metatrain_character_folders = metatrain_folders
            self.metaval_character_folders = metaval_folders
            self.rotations = config.get('rotations', [0])
        elif FLAGS.datasource in NLP_SINGLE_DATASETS:
            self.make_data_tensor = self.make_data_tensor_nlp_2inp
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.dim_output = self.num_classes
            if FLAGS.datasource == 'absa':
                if FLAGS.absa_domain == 'restaurant':
                    data_train, data_dev, data_test = read_absa_restaurants(datafolder='./data/semeval_task5')
                elif FLAGS.absa_domain == 'laptop':
                    data_train, data_dev, data_test = read_absa_laptops(datafolder='./data/semeval_task5')
            elif FLAGS.datasource == 'SNLI':
                data_train, data_dev, data_test = read_snli(datafolder = './data/SNLI/original')


            train_dataset = data_train
            if FLAGS.test_set:
                val_dataset = data_test
                val_dataset = data_train
            else:
                val_dataset = data_dev
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            
            self.dim_output = self.num_classes
            self.dim_input = -1 # do not use this value
        elif FLAGS.datasource in NLP_1SEN_SENTIMENT_DATASETS:
            self.make_data_tensor = self.make_data_tensor_1sen_senti
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.dim_output = self.num_classes
            if FLAGS.datasource == 'sst':
                data_train, data_dev, data_test = read_sst(datafolder='./data/SST-2')
            elif FLAGS.datasource == 'imdb':
                pass
            train_dataset = data_train
            if FLAGS.test_set:
                val_dataset = data_test
            else:
                val_dataset = data_dev
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.dim_input = -1 # do not use this value

        elif FLAGS.datasource == 'transfer_multi':
            self.make_data_tensor = self.make_data_tensor_transfer_multi
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.dim_output = self.num_classes

            self.dim_input = -1


        else:
            raise ValueError('Unrecognized data source')

    def preprocessed_text(self, text, word2idx):
        words = nltk.word_tokenize(text.lower())
        result = []
        for i, w in enumerate(words):
            if w in word2idx.keys():
                result.append(w)
            else:
                result.append('UNK')
        return ' '.join(result)

    def make_data_tensor_1sen_senti(self, word2idx):
        train_next_items, self.train_init_ops = self._make_data_tensor_1sen_senti(word2idx, train=True)
        test_next_items, self.test_init_ops = self._make_data_tensor_1sen_senti(word2idx, train=False)

        return train_next_items, test_next_items
        


    def _make_data_tensor_1sen_senti(self, word2idx, train=True):
        if train:
            dataset = self.train_dataset
        else:
            dataset = self.val_dataset
        
        dataset_size = len(dataset['seq1'])
        all_text = []
        all_label = []
        all_text_len = []

        shuffled_index = np.random.permutation(dataset_size)

        #dataset['labels'] = [x.lower() for x in dataset['labels']]

        print(dataset_size)
        
        for i in range(dataset_size):
            j = shuffled_index[i]
            text = np.array([word2idx.get(x,word2idx['UNK']) for x in nltk.word_tokenize(dataset['seq1'][j].lower())])
            label = np.array(dataset['labels'].index(dataset['stance'][j]))
            text_len = np.array(len(text))

            all_text.append(text)
            all_label.append(label)
            all_text_len.append(text_len)

        assert(self.num_samples_per_class==1)
        if not FLAGS.use_static_rnn:
            padded_all_text = get_pad_batch(all_text, self.num_samples_per_class)
        else:
            padded_all_text = get_static_pad_batch(all_text, self.num_samples_per_class)

        all_label = get_batch_labels(all_label, self.num_samples_per_class)
        all_text_len = get_batch_labels(all_text_len, self.num_samples_per_class)
        meta_all_text = get_pad_metabatch(padded_all_text, self.batch_size)
        meta_all_label = get_metabatch_labels(all_label, self.batch_size)    
        meta_all_text_len = get_metabatch_labels(all_text_len, self.batch_size)    

        print(len(meta_all_text))
        dataset_text = tf.data.Dataset.from_generator(lambda: meta_all_text, tf.int64, tf.TensorShape([self.batch_size, self.num_samples_per_class, None]))
        dataset_label = tf.data.Dataset.from_generator(lambda: meta_all_label, tf.int64, tf.TensorShape([self.batch_size, self.num_samples_per_class]))
        dataset_text_len = tf.data.Dataset.from_generator(lambda: meta_all_text_len, tf.int64,tf.TensorShape([self.batch_size,self.num_samples_per_class]))

        dataset_all = tf.data.Dataset.zip((dataset_text, dataset_text_len, dataset_label))

        dataset_input = dataset_all.map(lambda a,b,c:(a,b))
        dataset_label = dataset_all.map(lambda a,b,c:c)

        iterator_input = tf.data.Iterator.from_structure(dataset_input.output_types, dataset_input.output_shapes)
        iterator_label = tf.data.Iterator.from_structure(dataset_label.output_types, dataset_label.output_shapes)

        next_input = iterator_input.get_next()
        next_label = iterator_label.get_next()

        init_op_input = iterator_input.make_initializer(dataset_input)
        init_op_label = iterator_label.make_initializer(dataset_label)

        return (next_input, next_label), (init_op_input, init_op_label)

    def make_data_tensor_transfer_multi(self, word2idx, train=True):
        # Notice the shape of different input datasets must be the same
        #if train:
        #    dataset_names = FLAGS.train_datasets
        #else:
        #    dataset_names = FLAGS.test_datasets
        dataset_names = FLAGS.multi_datasets
        dataset_names_list = dataset_names.split(',')
        tfdatasets = []
        for i, name in enumerate(dataset_names_list):
            if name == 'absa-l':
                data_train, data_dev, data_test = read_absa_restaurants(datafolder='./data/semeval_task5')
            elif name == 'absa-r':
                data_train, data_dev, data_test = read_absa_laptops(datafolder='./data/semeval_task5')
            elif name == 'target':
                data_train, data_dev, data_test = read_target_dependent(datafolder='./data/target-dependent')
            elif name == 'topic-5':
                # used as topic-3
                data_train, data_dev, data_test = readTopic3Way(datafolder='./data/semeval2016-task4c-topic-based-sentiment')
            #The following setting can be changed.
            if train==True:
                dataset = data_train
            elif FLAGS.test_set == False:
                dataset = data_dev
            else:
                dataset = data_test
            if i!=len(dataset_names_list)-1:
                tfdataset = self.create_nlp_classification_dataset(dataset, word2idx)
            else:
                tfdataset = self.create_nlp_classification_dataset(dataset, word2idx, repeat_dataset=False)
            tfdatasets.append(tfdataset)
        if FLAGS.switch_datasets:
            self.switch_datasets(tfdatasets)
        self.handle_inputa = tf.placeholder(tf.string, shape=[])
        self.handle_labela = tf.placeholder(tf.string, shape=[])
        self.handle_inputb = tf.placeholder(tf.string, shape=[])
        self.handle_labelb = tf.placeholder(tf.string, shape=[])
        next_items, iterators = self.get_itr_and_next_from_dataset(tfdatasets[0])
        self.dataset_itrs = [self.get_oneshot_itr_from_datasets(tfdatasets[i]) if i!=len(tfdatasets)-1 else self.get_initializable_iter_from_datasets(tfdatasets[i]) for i in range(len(tfdatasets))]
        self.test_init_ops = self.get_init_ops_from_iters(self.dataset_itrs[-1])
        return next_items

    def switch_datasets(self, tfdatasets):
        tfdatasets[0][2], tfdatasets[0][3], tfdatasets[1][2], tfdatasets[1][3] = tfdatasets[1][2], tfdatasets[1][3], tfdatasets[0][2], tfdatasets[0][3]
        

    
    def create_nlp_classification_dataset(self, dataset, word2idx, repeat_dataset=True):
        #The input dataset should be a dict with 4 keys: seq1, seq2, labesl, stance
        dataset_size = len(dataset['seq1'])

	    #DELETE THIS AFTER DEBUGGING!!!
        #dataset_size = min(dataset_size, 50)

        all_text = []
        all_ctgr = []
        all_label = []
        all_text_len = []
        all_ctgr_len = []
        #Shuffle once here
        shuffled_index = np.random.permutation(dataset_size)
        #print(dataset_size)
        
        #Have to do batching first, so have to view everything in a meta-batch a single batch
        #total_batch_size = self.batch_size * self.num_samples_per_class

        #dataset_size = int(dataset_size / total_batch_size) * total_batch_size
        
        dataset['labels'] = [str(x).lower() for x in dataset['labels']]

        print(dataset_size)
        for i in range(dataset_size):
            j = shuffled_index[i]
            #print("BEFORE: "+str(dataset['seq2'][j]))
            #print("AFTER: "+str(self.preprocessed_text(dataset['seq2'][j], word2idx)))
            #text = np.array([word2idx.get(x,word2idx['UNK']) for x in dataset['seq2'][j].split()])
            #ctgr = np.array([word2idx.get(x,word2idx['UNK']) for x in dataset['seq1'][j].lower().split()])
            text = np.array([word2idx.get(x,word2idx['UNK']) for x in nltk.word_tokenize(dataset['seq2'][j].lower())])
            ctgr = np.array([word2idx.get(x,word2idx['UNK']) for x in nltk.word_tokenize(dataset['seq1'][j].lower())])
            label = np.array(dataset['labels'].index(str(dataset['stance'][j])))
            text_len = np.array(len(text))
            ctgr_len = np.array(len(ctgr))

            #text = np.expand_dims(text, axis=-1)
            #ctgr = np.expand_dims(ctgr, axis=-1)
            #label = np.expand_dims(label, axis=-1)
            all_text.append(text)
            all_ctgr.append(ctgr)
            all_label.append(label)
            all_text_len.append(text_len)
            all_ctgr_len.append(ctgr_len)
        padded_all_text = get_pad_batch(all_text, self.num_samples_per_class)
        padded_all_ctgr = get_pad_batch(all_ctgr, self.num_samples_per_class)
        all_label = get_batch_labels(all_label, self.num_samples_per_class)
        all_text_len = get_batch_labels(all_text_len, self.num_samples_per_class)
        all_ctgr_len = get_batch_labels(all_ctgr_len, self.num_samples_per_class)

        meta_all_text = get_pad_metabatch(padded_all_text, self.batch_size)
        meta_all_ctgr = get_pad_metabatch(padded_all_ctgr, self.batch_size)
        meta_all_label = get_metabatch_labels(all_label, self.batch_size)    
        meta_all_text_len = get_metabatch_labels(all_text_len, self.batch_size)    
        meta_all_ctgr_len = get_metabatch_labels(all_ctgr_len, self.batch_size)    
        #np.set_printoptions(threshold=np.nan)
        #for i in range(33):
        #    print(meta_all_text[12][i])
        #    print(meta_all_label[12][i])


        print(len(meta_all_text))
        #text_queue = tf.train.input_producer()
        #ctgr_queue = tf.train.input_producer()
        #label_queue = tf.train.input_producer()
        dataset_text = tf.data.Dataset.from_generator(lambda: meta_all_text, tf.int64, tf.TensorShape([self.batch_size,self.num_samples_per_class,None]))
        #dataset_text = dataset_text.padded_batch(self.batch_size, padded_shapes=[None])
        dataset_ctgr = tf.data.Dataset.from_generator(lambda: meta_all_ctgr, tf.int64,tf.TensorShape([self.batch_size,self.num_samples_per_class,None]))
        #dataset_ctgr = dataset_ctgr.padded_batch(self.batch_size, padded_shapes=[None])
        dataset_label = tf.data.Dataset.from_generator(lambda: meta_all_label, tf.int64,tf.TensorShape([self.batch_size,self.num_samples_per_class]))
        #dataset_label = dataset_label.batch(total_batch_size)
        dataset_text_len = tf.data.Dataset.from_generator(lambda: meta_all_text_len, tf.int64,tf.TensorShape([self.batch_size,self.num_samples_per_class]))
        dataset_ctgr_len = tf.data.Dataset.from_generator(lambda: meta_all_ctgr_len, tf.int64,tf.TensorShape([self.batch_size,self.num_samples_per_class]))

        dataset_alla = tf.data.Dataset.zip((dataset_text, dataset_ctgr, dataset_label, dataset_text_len, dataset_ctgr_len))
    
        dataset_allb = dataset_alla.map(lambda a,b,c,d,e:(a,b,c,d,e))

        dataset_alla = dataset_alla.shuffle(dataset_size)
        dataset_allb = dataset_allb.shuffle(dataset_size)
        if repeat_dataset:
            dataset_alla = dataset_alla.repeat()
            dataset_allb = dataset_allb.repeat()

        dataset_inputa = dataset_alla.map(lambda a,b,c,d,e :(a,b,d,e))
        dataset_labela = dataset_alla.map(lambda a,b,c,d,e:c)
        dataset_inputb = dataset_allb.map(lambda a,b,c,d,e:(a,b,d,e))
        dataset_labelb = dataset_allb.map(lambda a,b,c,d,e:c)

        return [dataset_inputa, dataset_labela, dataset_inputb, dataset_labelb]

    
    def get_itr_and_next_from_dataset(self, datasets):
        dataset_inputa, dataset_labela, dataset_inputb, dataset_labelb = datasets

        iterator_inputa = tf.data.Iterator.from_string_handle(self.handle_inputa, dataset_inputa.output_types, dataset_inputa.output_shapes)
        iterator_labela = tf.data.Iterator.from_string_handle(self.handle_labela, dataset_labela.output_types, dataset_labela.output_shapes)
        iterator_inputb = tf.data.Iterator.from_string_handle(self.handle_inputb, dataset_inputb.output_types, dataset_inputb.output_shapes)
        iterator_labelb = tf.data.Iterator.from_string_handle(self.handle_labelb, dataset_labelb.output_types, dataset_labelb.output_shapes)

        #print(dataset_inputa.output_shapes)
        #print(dataset_labela.output_shapes)
        
        next_inputa = iterator_inputa.get_next()
        next_inputb = iterator_inputb.get_next()
        next_labela = iterator_labela.get_next()
        next_labelb = iterator_labelb.get_next()

        return (next_inputa, next_inputb, next_labela, next_labelb), (iterator_inputa, iterator_inputb, iterator_labela, iterator_labelb)


    def get_initializable_iter_from_datasets(self, datasets):
        return [d.make_initializable_iterator() for d in datasets]

    def get_oneshot_itr_from_datasets(self, datasets):
        return [d.make_one_shot_iterator() for d in datasets]
    
    def get_init_ops_from_iters(self, iters):
        return (itr.initializer for itr in iters)

    
    def get_init_ops_from_dataset(self, itrs, datasets):
        dataset_inputa, dataset_labela, dataset_inputb, dataset_labelb = datasets
        iterator_inputa, iterator_labela, iterator_inputb, iterator_labelb = itrs

        init_op_inputa = iterator_inputa.make_initializer(dataset_inputa)
        init_op_labela = iterator_labela.make_initializer(dataset_labela)
        init_op_inputb = iterator_inputb.make_initializer(dataset_inputb)
        init_op_labelb = iterator_labelb.make_initializer(dataset_labelb)
        #self.init_ops = (init_op_inputa, init_op_inputb, init_op_labela, init_op_labelb)
        return (init_op_inputa, init_op_inputb, init_op_labela, init_op_labelb)



    def make_data_tensor_nlp_2inp(self, word2idx, train=True):
        if train:
            dataset = self.train_dataset
        else:
            dataset = self.val_dataset
        
        dataset_size = len(dataset['seq1'])

	#DELETE THIS AFTER DEBUGGING!!!
        #dataset_size = min(dataset_size, 50)

        all_text = []
        all_ctgr = []
        all_label = []
        all_text_len = []
        all_ctgr_len = []
        #Shuffle once here
        shuffled_index = np.random.permutation(dataset_size)
        #print(dataset_size)
        
        #Have to do batching first, so have to view everything in a meta-batch a single batch
        #total_batch_size = self.batch_size * self.num_samples_per_class

        #dataset_size = int(dataset_size / total_batch_size) * total_batch_size
        
        dataset['labels'] = [x.lower() for x in dataset['labels']]

        print(dataset_size)
        for i in range(dataset_size):
            j = shuffled_index[i]
            #print("BEFORE: "+str(dataset['seq2'][j]))
            #print("AFTER: "+str(self.preprocessRed_text(dataset['seq2'][j], word2idx)))
            #text = np.array([word2idx.get(x,word2idx['UNK']) for x in dataset['seq2'][j].split()])
            #ctgr = np.array([word2idx.get(x,word2idx['UNK']) for x in dataset['seq1'][j].lower().split()])
            text = np.array([word2idx.get(x,word2idx['UNK']) for x in nltk.word_tokenize(dataset['seq2'][j].lower())])
            ctgr = np.array([word2idx.get(x,word2idx['UNK']) for x in nltk.word_tokenize(dataset['seq1'][j].lower())])
            label = np.array(dataset['labels'].index(dataset['stance'][j]))
            text_len = np.array(len(text))
            ctgr_len = np.array(len(ctgr))

            #text = np.expand_dims(text, axis=-1)
            #ctgr = np.expand_dims(ctgr, axis=-1)
            #label = np.expand_dims(label, axis=-1)
            all_text.append(text)
            all_ctgr.append(ctgr)
            all_label.append(label)
            all_text_len.append(text_len)
            all_ctgr_len.append(ctgr_len)
        padded_all_text = get_pad_batch(all_text, self.num_samples_per_class)
        padded_all_ctgr = get_pad_batch(all_ctgr, self.num_samples_per_class)
        all_label = get_batch_labels(all_label, self.num_samples_per_class)
        all_text_len = get_batch_labels(all_text_len, self.num_samples_per_class)
        all_ctgr_len = get_batch_labels(all_ctgr_len, self.num_samples_per_class)

        meta_all_text = get_pad_metabatch(padded_all_text, self.batch_size)
        meta_all_ctgr = get_pad_metabatch(padded_all_ctgr, self.batch_size)
        meta_all_label = get_metabatch_labels(all_label, self.batch_size)    
        meta_all_text_len = get_metabatch_labels(all_text_len, self.batch_size)    
        meta_all_ctgr_len = get_metabatch_labels(all_ctgr_len, self.batch_size)    
        #np.set_printoptions(threshold=np.nan)
        #for i in range(33):
        #    print(meta_all_text[12][i])
        #    print(meta_all_label[12][i])


        print(len(meta_all_text))
        #text_queue = tf.train.input_producer()
        #ctgr_queue = tf.train.input_producer()
        #label_queue = tf.train.input_producer()
        dataset_text = tf.data.Dataset.from_generator(lambda: meta_all_text, tf.int64, tf.TensorShape([self.batch_size,self.num_samples_per_class,None]))
        #dataset_text = dataset_text.padded_batch(self.batch_size, padded_shapes=[None])
        dataset_ctgr = tf.data.Dataset.from_generator(lambda: meta_all_ctgr, tf.int64,tf.TensorShape([self.batch_size,self.num_samples_per_class,None]))
        #dataset_ctgr = dataset_ctgr.padded_batch(self.batch_size, padded_shapes=[None])
        dataset_label = tf.data.Dataset.from_generator(lambda: meta_all_label, tf.int64,tf.TensorShape([self.batch_size,self.num_samples_per_class]))
        #dataset_label = dataset_label.batch(total_batch_size)
        dataset_text_len = tf.data.Dataset.from_generator(lambda: meta_all_text_len, tf.int64,tf.TensorShape([self.batch_size,self.num_samples_per_class]))
        dataset_ctgr_len = tf.data.Dataset.from_generator(lambda: meta_all_ctgr_len, tf.int64,tf.TensorShape([self.batch_size,self.num_samples_per_class]))

        dataset_alla = tf.data.Dataset.zip((dataset_text, dataset_ctgr, dataset_label, dataset_text_len, dataset_ctgr_len))
    
        dataset_allb = dataset_alla.map(lambda a,b,c,d,e:(a,b,c,d,e))
        
        #TODO:enable shuffling for large datasets
        #dataset_alla = dataset_alla.shuffle(dataset_size)
        #dataset_allb = dataset_allb.shuffle(dataset_size)

        #dataset_alla = dataset_alla.repeat()
        #dataset_allb = dataset_allb.repeat()

        dataset_inputa = dataset_alla.map(lambda a,b,c,d,e :(a,b,d,e))
        dataset_labela = dataset_alla.map(lambda a,b,c,d,e:c)
        dataset_inputb = dataset_allb.map(lambda a,b,c,d,e:(a,b,d,e))
        dataset_labelb = dataset_allb.map(lambda a,b,c,d,e:c)

        iterator_inputa = tf.data.Iterator.from_structure(dataset_inputa.output_types, dataset_inputa.output_shapes)
        iterator_labela = tf.data.Iterator.from_structure(dataset_labela.output_types, dataset_labela.output_shapes)
        iterator_inputb = tf.data.Iterator.from_structure(dataset_inputb.output_types, dataset_inputb.output_shapes)
        iterator_labelb = tf.data.Iterator.from_structure(dataset_labelb.output_types, dataset_labelb.output_shapes)

        #print(dataset_inputa.output_shapes)
        #print(dataset_labela.output_shapes)
        
        next_inputa = iterator_inputa.get_next()
        next_inputb = iterator_inputb.get_next()
        next_labela = iterator_labela.get_next()
        next_labelb = iterator_labelb.get_next()

        init_op_inputa = iterator_inputa.make_initializer(dataset_inputa)
        init_op_labela = iterator_labela.make_initializer(dataset_labela)
        init_op_inputb = iterator_inputb.make_initializer(dataset_inputb)
        init_op_labelb = iterator_labelb.make_initializer(dataset_labelb)
        #self.init_ops = (init_op_inputa, init_op_inputb, init_op_labela, init_op_labelb)
        return (next_inputa, next_inputb, next_labela, next_labelb), (init_op_inputa, init_op_inputb, init_op_labela, init_op_labelb)


    def make_data_tensor_image(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            num_total_batches = 200000
        else:
            folders = self.metaval_character_folders
            num_total_batches = 600

        # make list of files
        print('Generating filenames')
        all_filenames = []
        for _ in range(num_total_batches):
            sampled_character_folders = random.sample(folders, self.num_classes)
            random.shuffle(sampled_character_folders)
            labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
            # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)

        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        if FLAGS.datasource == 'miniimagenet':
            image = tf.image.decode_jpeg(image_file, channels=3)
            image.set_shape((self.img_size[0],self.img_size[1],3))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
        else:
            image = tf.image.decode_png(image_file)
            image.set_shape((self.img_size[0],self.img_size[1],1))
            image = tf.reshape(image, [self.dim_input])
            image = tf.cast(image, tf.float32) / 255.0
            image = 1.0 - image  # invert
        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 256
        examples_per_batch = self.num_classes * self.num_samples_per_class
        batch_image_size = self.batch_size  * examples_per_batch
        print('Batching images')
        images = tf.train.batch(
                [image],
                batch_size = batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_image_size,
                )
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size):
            image_batch = images[i*examples_per_batch:(i+1)*examples_per_batch]

            if FLAGS.datasource == 'omniglot':
                # omniglot augments the dataset by rotating digits to create new classes
                # get rotation per class (e.g. 0,1,2,0,0 if there are 5 classes)
                rotations = tf.multinomial(tf.log([[1., 1.,1.,1.]]), self.num_classes)
            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random_shuffle(class_idxs)

                true_idxs = class_idxs*self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch,true_idxs))
                if FLAGS.datasource == 'omniglot': # and FLAGS.train:
                    new_list[-1] = tf.stack([tf.reshape(tf.image.rot90(
                        tf.reshape(new_list[-1][ind], [self.img_size[0],self.img_size[1],1]),
                        k=tf.cast(rotations[0,class_idxs[ind]], tf.int32)), (self.dim_input,))
                        for ind in range(self.num_classes)])
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

    def generate_sinusoid_batch(self, train=True, input_idx=None):
        # Note train arg is not used (but it is used for omniglot method.
        # input_idx is used during qualitative testing --the number of examples used for the grad update
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
        return init_inputs, outputs, amp, phase
