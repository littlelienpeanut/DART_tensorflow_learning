# use tf.name_scope to name the group name and add the writer = tf.summary.FileWriter('logs/', sess.graph) in the session area to get logs.
# In the command line, we can use tensorboard --logdir=ADDRESS to build the tensorboard graph

import tensorflow as tf
from time import time
from tensorflow.examples.tutorials.mnist import input_data

def layer(output_dim, input_dim, input, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(input, W) + b
    if activation is None:
        outputs = XWb

    else:
        outputs = activation(XWb)

    return outputs

def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    #tensorboard name
    with tf.name_scope('input'):
        #define input layer
        x = tf.placeholder('float', [None, 784], name='x_input')
        y = tf.placeholder('float', [None, 10], name='y_input')

    with tf.name_scope('layer'):
        #define hidden layer
        h1 = layer(256, 784, x, activation = tf.nn.relu)
        y_prediction = layer(10, 256, h1, activation = None)

    #define loss function, optimizer
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_prediction, labels=y))
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('logs/', sess.graph)
        trainEpochs = 1
        batch_size = 100
        n_batch = mnist.train.num_examples // batch_size

        for epoch in range(trainEpochs):
            for batch_num in range(n_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})

            acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
            print('Iter: ' + str(epoch) + ' acc: ' + str(acc))

if __name__ == '__main__':
    main()
