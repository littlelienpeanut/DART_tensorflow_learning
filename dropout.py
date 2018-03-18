import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#loading datasets
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#define batch size
batch_size = 200

#get batch number
n_batch = mnist.train.num_examples // batch_size

#define placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

#create NN
W_1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1))
b_1 = tf.Variable(tf.zeros([2000]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W_1) + b_1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W_2 = tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.1))
b_2 = tf.Variable(tf.zeros([2000]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W_2) + b_2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W_3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1))
b_3 = tf.Variable(tf.zeros([1000]) + 0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W_3) + b_3)
L3_drop = tf.nn.dropout(L3, keep_prob)

W_4 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
b_4 = tf.Variable(tf.zeros([10]) + 0.1)

prediction = tf.matmul(L3_drop, W_4) + b_4

#define loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

#if the max value position in y is equal to the max value position in prediction return true(1) to correct_prediction else return false(0)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

#tf.cast() can change correct_prediction type to float32
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #dropout: 0.7 -> 70% tensor are working
            sess.run(train_step, feed_dict = {x:batch_xs, y:batch_ys, keep_prob:0.7})

        test_acc = sess.run(accuracy, feed_dict = {x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        train_acc = sess.run(accuracy, feed_dict = {x:mnist.train.images, y:mnist.train.labels, keep_prob:1.0})
        print('Iter: ' + str(epoch) + ', testing acc: ' + str(test_acc) + ' ,traning acc: ' + str(train_acc))
