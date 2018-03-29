#input -> convolution -> max_pooling -> convolution -> max_pooling -> flatten -> full connective NN L1, L2


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    #create a normal distribution number as initial values
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    #x: 4-D input image [batch_size, in_height, in_weight, in_channels(black and white pic:1, color:3)]
    #W: filter weight [filter_height, filter_width, input_channels, output_channels]
    #strides: filter moving step [moving_right, right_step_value, down_step_valuemoving_down]
    #padding: SAME or VALID
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    #ksize [1, height, width, 1]
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')

def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    ### ------------------------------------------------------------------- ###
    ### convolution and max pooling ###
    #change input image to 4-D vector [batch, in_height, in_weight, in_channels]
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    #initialize the weight and bias in L1
    W_L1 = weight_variable([5, 5, 1, 32]) #5x5 window size, 1: input_channel, 32: output 32 features map
    b_L1 = bias_variable([32])

    #convl L1
    h_conv1 = tf.nn.relu(conv2d(x_image, W_L1) + b_L1)
    h_pool1 = max_pool_2x2(h_conv1)

    #initalize the weight and bias in L2
    W_L2 = weight_variable([5, 5, 32, 64])
    b_L2 = weight_variable([64])

    #convl L2
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_L2) + b_L2)
    h_pool2 = max_pool_2x2(h_conv2)

    ### ------------------------------------------------------------------ ###
    ### flatten
    h_pool2_flat = tf.reshape(h_pool2, [-1, 3136])

    ### ------------------------------------------------------------------ ###
    ### define full connective NN ###
    w_fc1 = weight_variable([3136, 1024]) #after flatten we can get 7*7*64 tensors
    b_fc1 = weight_variable([1024])
    fc_L1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    fc_L1_drop = tf.nn.dropout(fc_L1, keep_prob)

    w_fc2 = weight_variable([1024, 10])
    b_fc2 = weight_variable([10])

    y_prediction = tf.matmul(fc_L1_drop, w_fc2) + b_fc2

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_prediction))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        #Variable
        batch_size = 100
        n_batch = mnist.train.num_examples // batch_size
        epoch_num = 20

        sess.run(tf.global_variables_initializer())
        for epoch in range(epoch_num):
            for batch in range(n_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.7})

            acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
            print('Iter: ' + str(epoch) + ' testing acc: ' + str(acc))


if __name__ == '__main__':
    main()
