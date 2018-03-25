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

#get details of variable
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    #tensorboard name
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y = tf.placeholder(tf.float32, [None, 10], name='y-input')

    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            W = tf.Variable(tf.random_normal([784, 10]), name='W')
            variable_summaries(W)
        with tf.name_scope('biases'):
            b = tf.Variable(tf.random_normal([1, 10]), name='b')
            variable_summaries(b)
        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.matmul(x, W) + b
        with tf.name_scope('softmax'):
            y_prediction = tf.nn.softmax(wx_plus_b)

    #define loss function, optimizer
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_prediction, labels=y))
        tf.summary.scalar('loss', loss) #loss just a value, we don't need to analyze loss.

    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1).minimize(loss)

    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(y, 1))

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        tf.summary.scalar('accuracy', accuracy)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        merged = tf.summary.merge_all() #merge the data to show on the tensorboard.
        writer = tf.summary.FileWriter('logs/', sess.graph) #create the directory and add summaries to it.
        trainEpochs = 21
        batch_size = 200
        n_batch = n_batch = mnist.train.num_examples // batch_size

        for epoch in range(trainEpochs):
            for batch_num in range(n_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                summary, _ = sess.run([merged, optimizer], feed_dict={x:batch_x, y:batch_y})

            writer.add_summary(summary, epoch)

            acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
            print('Iter: ' + str(epoch) + ' acc: ' + str(acc))

if __name__ == '__main__':
    main()
