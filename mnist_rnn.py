import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def RNN(X, weight, biases):
    n_inputs = 28
    max_time = 28
    lstm_size = 50
    inputs = tf.reshape(X, [-1, max_time, n_inputs]) #[batch_size, max_time, n_inputs]
    lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(lstm_size)
    #final_state[0] is cell state ~ value in memory cell
    #final_state[1] is hidden_state ~ value of h'(memory cell)
    output, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype = tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weight) + biases)
    return results

def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    lstm_size = 50
    n_classes = 10


    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name = 'x_input')
        y = tf.placeholder(tf.float32, [None, 10], name = 'y_input')

    with tf.name_scope('Weight'):
        weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev = 0.1))

    with tf.name_scope('bias'):
        biases = tf.Variable(tf.constant(0.1, shape = [n_classes]))

    with tf.name_scope('Layer'):
        y_prediction = RNN(x, weights, biases)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_prediction, labels = y))
        tf.summary.scalar('cross_entropy', cross_entropy)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_prediction, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        batch_size = 100
        n_batch = mnist.train.num_examples // batch_size
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/test', sess.graph)

        for epoch in range(100):
            for batch_i in range(n_batch):
                print('batch num: ' + str(batch_i))
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})

            #training error in each batch
            training_acc, train_res = sess.run([accuracy, merged], feed_dict={x:mnist.train.images, y:mnist.train.labels})
            train_writer.add_summary(train_res, epoch)
            print('training acc: ' + str(training_acc))

            testing_acc, test_res = sess.run([accuracy, merged], feed_dict={x:mnist.test.images, y:mnist.test.labels})
            test_writer.add_summary(test_res, epoch)
            print('testing acc: ' + str(testing_acc))


if __name__ == '__main__':
    main()
