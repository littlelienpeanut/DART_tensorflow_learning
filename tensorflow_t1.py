#first tensorflow coding
import tensorflow as tf
import numpy as np
import os
#close warning message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

### test1 ###
'''
#create matrix operations
m1 = tf.constant([[3, 3]])
m2 = tf.constant(([[2], [3]]))

#matrix multiplication
#tensor definition
product = tf.matmul(m1, m2)


sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# use with to simplify the sess.close()

with tf.Session() as sess:
    result = sess.run(product)
    print(result)
'''

### test2 ###
'''
x = tf.Variable([1, 2])
a = tf.Variable([3, 3])
#create a subtract operation
sub = tf.subtract(x, a)
#create a addition operation
add = tf.add(x, sub)

#must to initialize the global variables first
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))
'''

### test3 ###
'''
state = tf.Variable(0)
new_value = tf.add(state, 1)
#state = new_value
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for i in range(5):
        sess.run(update)
        print(sess.run(state))
'''

### fetch ###
'''
input1 = tf.Variable(3.0)
input2 = tf.Variable(2.0)
input3 = tf.Variable(5.0)

add = tf.add(input2, input3)
mul = tf.multiply(input1, add)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #fetch: run multi-operstion at the same time
    result = sess.run([mul, add])
    print(result)
'''

# feed #
'''
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)


with tf.Session() as sess:
    #feed the data in a dictionary structure
    print(sess.run(output, feed_dict = {input1:[7.0], input2:[2.0]}))
'''

### example ###

x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2

b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b

#loss: mse
loss = tf.reduce_mean(tf.square(y_data - y))
#use GradientDescent as learning rate=0.2 to optimize
optimizer = tf.train.GradientDescentOptimizer(0.2)
#minimize loss function
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for iter in range(201):
        sess.run(train)
        if iter%20 == 0:
            print(iter, sess.run([k, b]))
