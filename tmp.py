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
    value2 = value
    value3 = value
    sess = tf.InteractiveSession()
    #sess.run(value) # first element
    #sess.run(value) # second element
    #sess.run(value) # Out of range message
    for _ in range(10):
        sess.run(itr.make_initializer(dataset))

        while True:
            try:
                print(sess.run([value, value2, value3]))
            except tf.errors.OutOfRangeError:
                print("out of range!")
                break
def test5():
    text_len = tf.convert_to_tensor([3,4,5,6,7])
    e = tf.sequence_mask(text_len,10)

    f = tf.sequence_mask(text_len-1, 10)
    z1 = tf.ones(text_len.shape)
    z = tf.logical_not(tf.sequence_mask(z1,10))
    g = tf.logical_and(e,z)
    #g = e * 10
    #h = g * 10
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    print(sess.run(e))
    print(sess.run(f))
    print(sess.run(g))

def test6():
    text_label = tf.convert_to_tensor([3,4,5,6,7])
    text_label = tf.reshape(text_label, [1,-1])
    text_labels = tf.concat([text_label, text_label, text_label], axis=0)
    one_step_zero = tf.zeros((3,1),tf.int32)
    fw_labels = text_labels[:,1:]
    fw_labels = tf.concat([fw_labels, one_step_zero], axis=-1)
    bw_labels = text_labels[:,:-1]
    bw_labels = tf.concat([one_step_zero, bw_labels], axis=-1)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    print(sess.run(text_labels))
    print(sess.run(fw_labels))
    print(sess.run(bw_labels))

def test7():
    c = tf.ones((10))
    e, f = c, c+10
    print(type(e))
    g = e * 10
    print(type(g))
    h = g + e
    loss = tf.reduce_sum(h)
    grads = tf.gradients(loss, [c,e,f,g,h])
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    print(sess.run(grads))

    

test7()
