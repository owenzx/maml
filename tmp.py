import tensorflow as tf
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

test2()
