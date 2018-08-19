import tensorflow as tf
import numpy as np
def test1():
    dataset0 = tf.data.Dataset.range(10)
    dataset = dataset0.map(lambda x: tf.fill([2,tf.cast(x, tf.int32)], x))
    #dataset2 = dataset0.map(lambda x: tf.fill([2,tf.cast(x+1, tf.int32)], x))
    #dataset2 = dataset2.padded_batch(4, padded_shapes=[None])
    dataset = dataset.padded_batch(4, padded_shapes=[None])
    #dataset = tf.data.Dataset.zip((dataset, dataset2))
    dataset = dataset.shuffle(100)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    #a, b = next_element

    sess = tf.InteractiveSession()
    #print(dataset.output_shapes)
    print(next_element)
    #print(a.shape)
    print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
    #print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],

def test2():
    c = tf.nn.rnn_cell.LSTMCell(3)
    c.build(tf.TensorShape([2,2]))
    print(c.get_weights())
    e, f = c.weights
    print(type(e))
    g = e * 10
    print(type(g))
    h = f * 10
    c.set_weights(g)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    sess.run(h)
    print(c.get_weights())

def test3():
    from absa_reader import read_absa_restaurants
    data_train, data_dev, data_test = read_absa_restaurants(datafolder='./data/semeval_task5')
    print(data_train.keys() )
    print(len(data_train['seq1']))

def test4():
    a = [np.arange(3), np.arange(5)]
    dataset = tf.data.Dataset.from_generator(lambda: a, tf.int64)
    #dataset = dataset.repeat()
    #ds1=dataset.repeat(10)
    #value = dataset.make_one_shot_iterator().get_next()
    itr = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    value = itr.get_next()
    sess = tf.InteractiveSession()
    #sess.run(value) # first element
    #sess.run(value) # second element
    #sess.run(value) # Out of range message
    for _ in range(10):
        sess.run(itr.make_initializer(dataset))

        while True:
            try:
                sess.run(value)
                print("running")
            except tf.errors.OutOfRangeError:
                print("out of range!")
                break

test4()
