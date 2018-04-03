import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def layer(output_dim, input_dim, input, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(input, W) + b
    if activation:
        output = activation(XWb)

    else:
        output = XWb

    return output

def main():
    #input
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name = 'x_input')
        y = tf.placeholder(tf.float32, [None, 10], name = 'y_input')

    with tf.name_scope('layer'):
        y_prediction = layer(10, 784, x, None)

    with tf.name_scope('loss'):
        #loss = tf.reduce_mean(tf.square(y_prediction - y))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_prediction, labels = y))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_prediction, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        #variable
        batch_size = 200
        epoch_num = 11
        batch_num = mnist.train.num_examples // batch_size
        sess.run(tf.global_variables_initializer())

        #non training
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
        print('')

        #loading trained NN
        saver.restore(sess, 'net/my_net.ckpt')
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

if __name__ == '__main__':
    main()
