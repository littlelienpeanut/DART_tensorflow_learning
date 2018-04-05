import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def layer(output_dim, input_dim, input, activation=None):
    with tf.name_scope('weight'):
        W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    with tf.name_scope('bias'):
        b = tf.Variable(tf.random_normal([1, output_dim]))

    XWb = tf.matmul(input, W) + b

    if activation:
        output = activation(XWb)

    else:
        output = XWb

    return output

def main():
    #variable
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    learning_rate = 0.01


    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784])
        y = x

    with tf.name_scope('autoencoder'):
        with tf.name_scope('encoder_L1'):
            encoder_L1 = layer(256, 784, x, tf.nn.sigmoid)
        with tf.name_scope('encoder_L2'):
            encoder_L2 = layer(128, 256, encoder_L1, tf.nn.sigmoid)

        with tf.name_scope('decoder_L1'):
            decoder_L1 = layer(256, 128, encoder_L2, tf.nn.sigmoid)
        with tf.name_scope('decoder_L2'):
            y_prediction = layer(784, 256, decoder_L1, tf.nn.sigmoid)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(y_prediction - y))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('training'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    merged = tf.summary.merge_all()


    with tf.Session() as sess:
        #Variable
        epoch_num = 5
        batch_size = 200
        batch_num = mnist.train.num_examples // batch_size
        writer = tf.summary.FileWriter('logs/', sess.graph)
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(epoch_num):
            print('epoch: ' + str(epoch_i+1))
            for batch_i in range(batch_num):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch_xs})

            mse_value, mse = sess.run([loss, merged], feed_dict={x: mnist.test.images[:100]})
            print('mse:' + str(mse_value))
            writer.add_summary(mse, epoch_i)


if __name__ == '__main__':
    main()
