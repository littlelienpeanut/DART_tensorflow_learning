#input -> convolution -> max_pooling -> convolution -> max_pooling -> flatten -> full connective NN L1, L2


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape, name):
    #create a normal distribution number as initial values
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name=name)

def var_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))

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

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x_input')
        y = tf.placeholder(tf.float32, [None, 10], name='y_input')
        with tf.name_scope('x_image'):
            #change input image to 4-D vector [batch, in_height, in_weight, in_channels]
            x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

    ### ------------------------------------------------------------------- ###
    ### convolution and max pooling ###
    #initialize the weight and bias in L1
    with tf.name_scope('convolution'):
        with tf.name_scope('convolution_L1'):
            with tf.name_scope('convl1_w'):
                W_L1 = weight_variable([5, 5, 1, 32], name='convl1_w') #5x5 window size, 1: input_channel, 32: output 32 features map
            with tf.name_scope('convl1_b'):
                    b_L1 = bias_variable([32], name='convl1_b')

            #convl L1
            with tf.name_scope('conv2d_L1'):
                h_conv1 = tf.nn.relu(conv2d(x_image, W_L1) + b_L1)
            with tf.name_scope('max_pool_L1'):
                h_pool1 = max_pool_2x2(h_conv1)

        #initalize the weight and bias in L2
        with tf.name_scope('convolution_L2'):
            with tf.name_scope('convl2_w'):
                W_L2 = weight_variable([5, 5, 32, 64], name='convl2_w')
            with tf.name_scope('convl2_b'):
                b_L2 = weight_variable([64], name='convl2_b')

            #convl L2:
            with tf.name_scope('conv2d_L2'):
                h_conv2 = tf.nn.relu(conv2d(h_pool1, W_L2) + b_L2)
            with tf.name_scope('max_pool_L2'):
                h_pool2 = max_pool_2x2(h_conv2)

    ### ------------------------------------------------------------------ ###
    with tf.name_scope('flatten'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 3136])

    ### ------------------------------------------------------------------ ###
    ### define full connective NN ###
    with tf.name_scope('full_connective'):
        with tf.name_scope('fc_L1'):
            with tf.name_scope('fc1_w'):
                w_fc1 = weight_variable([3136, 1024], name='fc1_w') #after flatten we can get 7*7*64 tensors
            with tf.name_scope('fc1_b'):
                b_fc1 = weight_variable([1024], name='fc1_b')

            with tf.name_scope('fc1_wx_plus_b'):
                fc_L1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

            with tf.name_scope('fc1_keep_prob'):
                keep_prob = tf.placeholder(tf.float32)
            with tf.name_scope('fc1_drop'):
                fc_L1_drop = tf.nn.dropout(fc_L1, keep_prob)

        with tf.name_scope('fc_L2'):
            with tf.name_scope('fc2_w'):
                w_fc2 = weight_variable([1024, 10], name='fc2_w')
            with tf.name_scope('fc2_b'):
                b_fc2 = weight_variable([10], name='fc2_b')
            with tf.name_scope('fc2_wx_plus_b'):
                y_prediction = tf.matmul(fc_L1_drop, w_fc2) + b_fc2

    with tf.name_scope('cross_entropy'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_prediction))
        tf.summary.scalar('cross_entropy', loss)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_prediction, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        #Variable
        sess.run(tf.global_variables_initializer())
        batch_size = 500
        n_batch = mnist.train.num_examples // batch_size
        epoch_num = 20
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/test', sess.graph)


        for epoch_n in range(1):
            for batch_i in range(n_batch):
                print('batch num: ' + str(batch_i))
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.7})

                #training error in each batch
                train_res = sess.run(merged, feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0})
                train_writer.add_summary(train_res, batch_i)

                batch_xs, batch_ys = mnist.test.next_batch(batch_size)
                test_res = sess.run(merged, feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0})
                test_writer.add_summary(test_res, batch_i)

            training_acc = sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels, keep_prob:1.0})
            print('Iter: ' + str(epoch_n) + ' training acc: ' + str(training_acc))
            testing_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
            print('Iter: ' + str(epoch_n) + ' testing acc: ' + str(testing_acc))
            print('')



if __name__ == '__main__':
    main()
