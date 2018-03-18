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

#create NN
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

#define loss function
#cross entropy with softmax
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

#optimizer choosing
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

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
            sess.run(train_step, feed_dict = {x:batch_xs, y:batch_ys})

        acc = sess.run(accuracy, feed_dict = {x:mnist.test.images, y:mnist.test.labels})
        print('Iter: ' + str(epoch) + ', testing acc: ' + str(acc))
