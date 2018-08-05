""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import get_images, get_pad_batch, get_pad_metabatch, get_batch_labels, get_metabatch_labels
from absa_reader import read_absa_restaurants
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
        elif FLAGS.datasource == 'absa':
            self.make_data_tensor = self.make_data_tensor_absa
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
#            from data.semeval.data_loader import SemEvalDataLoader
#            d_loader = SemEvalDataLoader()
#            self.train_data = d_loader.get_data(task="BD", years=2016, datasets = "train", only_semeval=True) 
            self.dim_output = self.num_classes
            data_train, data_dev, data_test = read_absa_restaurants(datafolder='./data/semeval_task5')
            train_dataset = data_train
            if FLAGS.test_set:
                val_dataset = data_test
            else:
                val_dataset = data_dev
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            
            self.dim_output = 3
            self.dim_input = -1 # do not use this value

        else:
            raise ValueError('Unrecognized data source')

    
    def make_data_tensor_absa(self, word2idx, train=True):
        if train:
            dataset = self.train_dataset
        else:
            dataset = self.val_dataset
        
        dataset_size = len(dataset['seq1'])

        all_text = []
        all_ctgr = []
        all_label = []
        #Shuffle once here
        shuffled_index = np.random.permutation(dataset_size)
        #print(dataset_size)
        
        #Have to do batching first, so have to view everything in a meta-batch a single batch
        #total_batch_size = self.batch_size * self.num_samples_per_class

        #dataset_size = int(dataset_size / total_batch_size) * total_batch_size
        
        dataset['labels'] = [x.lower() for x in dataset['labels']]

        for i in range(dataset_size):
            j = shuffled_index[i]
            text = np.array([word2idx.get(x,word2idx['UNK']) for x in dataset['seq2'][j].split()])
            ctgr = np.array([word2idx.get(x,word2idx['UNK']) for x in dataset['seq1'][j].lower().split()])
            label = np.array(dataset['labels'].index(dataset['stance'][j]))
            #text = np.expand_dims(text, axis=-1)
            #ctgr = np.expand_dims(ctgr, axis=-1)
            #label = np.expand_dims(label, axis=-1)
            all_text.append(text)
            all_ctgr.append(ctgr)
            all_label.append(label)
        padded_all_text = get_pad_batch(all_text, self.num_samples_per_class)
        padded_all_ctgr = get_pad_batch(all_ctgr, self.num_samples_per_class)
        all_label = get_batch_labels(all_label, self.num_samples_per_class)

        meta_all_text = get_pad_metabatch(padded_all_text, self.batch_size)
        meta_all_ctgr = get_pad_metabatch(padded_all_ctgr, self.batch_size)
        meta_all_label = get_metabatch_labels(all_label, self.batch_size)    
        #print(all_text[100])


        #text_queue = tf.train.input_producer()
        #ctgr_queue = tf.train.input_producer()
        #label_queue = tf.train.input_producer()
        dataset_text = tf.data.Dataset.from_generator(lambda: meta_all_text, tf.int32, tf.TensorShape([self.batch_size,self.num_samples_per_class,None]))
        #dataset_text = dataset_text.padded_batch(self.batch_size, padded_shapes=[None])
        dataset_ctgr = tf.data.Dataset.from_generator(lambda: meta_all_ctgr, tf.int32,tf.TensorShape([self.batch_size,self.num_samples_per_class,None]))
        #dataset_ctgr = dataset_ctgr.padded_batch(self.batch_size, padded_shapes=[None])
        dataset_label = tf.data.Dataset.from_generator(lambda: meta_all_label, tf.int32,tf.TensorShape([self.batch_size,self.num_samples_per_class]))
        #dataset_label = dataset_label.batch(total_batch_size)

        dataset_alla = tf.data.Dataset.zip((dataset_text, dataset_ctgr, dataset_label))
    
        dataset_allb = dataset_alla.map(lambda a,b,c:(a,b,c))

        dataset_alla = dataset_alla.shuffle(dataset_size)
        dataset_allb = dataset_allb.shuffle(dataset_size)
        dataset_alla = dataset_alla.repeat()
        dataset_allb = dataset_allb.repeat()

        dataset_inputa = dataset_alla.map(lambda a,b,c :(a,b))
        dataset_labela = dataset_alla.map(lambda a,b,c:c)
        dataset_inputb = dataset_allb.map(lambda a,b,c:(a,b))
        dataset_labelb = dataset_allb.map(lambda a,b,c:c)

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
