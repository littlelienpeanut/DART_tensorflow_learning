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

    merged = tf.summary.merge_all()
    saver = tf.train.Saver() #saver can't show up the graph on tensorboard

    with tf.Session() as sess:
        #Variable
        batch_size = 200
        epoch_num = 11
        batch_num = mnist.train.num_examples // batch_size
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/test', sess.graph)


        sess.run(tf.global_variables_initializer())

        for epoch in range(epoch_num):
            for batch_i in range(batch_num):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict = {x: batch_xs, y: batch_ys})

            train_res = sess.run(merged, feed_dict = {x: mnist.train.images, y: mnist.train.labels})
            train_writer.add_summary(train_res, epoch)

            test_res = sess.run(merged, feed_dict = {x: mnist.test.images, y: mnist.test.labels})
            test_writer.add_summary(test_res, epoch)

            if epoch % 5 == 0:
                test_acc = sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels})
                print('testing acc: ' + str(test_acc))

        #model saved as sess
        saver.save(sess, 'net/my_net.ckpt')


if __name__ == '__main__':
    main()
