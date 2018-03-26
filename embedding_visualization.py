import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

def layer(output_dim, input_dim, input, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    WXb = tf.matmul(input, W) + b
    if activation:
        output = activation(WXb)

    else:
        output = WXb

    return output

def var_summaries(var):
    with th.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.sqart(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.scalar('histogram', histogram)

def main():
    # loading data sets
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    #number of epochs
    epoch_num = 1000

    image_num = 3000

    #file path
    DIR = 'C:/Users/105522013/Desktop/DART/Tensorflow_project/'

    #loading picture
    embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')

    sess = tf.Session()

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y = tf.placeholder(tf.float32, [None, 10], name='y-input')

    #show image
    with tf.name_scope('input_reshape'):
        #-1: unknow number, 28*28 = 784, 1: black and white pic
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        #10 pictures
        tf.summary.image('input', image_shaped_input, 10)

    with tf.name_scope('layer'):
        with tf.name_scope('hidden_L1'):
            h1 = layer(output_dim=256, input_dim=784, input=x, activation=tf.nn.relu)

        # with tf.name_scope('hidden_L2'):
        #     h2 = layer(output_dim=256, input_dim=512, input=h1, activation=tf.nn.relu)

        with tf.name_scope('output_layer'):
            y_prediction = layer(output_dim=10, input_dim=256, input=h1, activation=None)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_prediction))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    with tf.name_scope('acuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(y, 1))

        with tf.name_scope('accuracy'):
            #change correct_prediction to float type
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    #create metadata
    if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
        tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')

    with open(DIR + 'projector/projector/metadata.tsv', 'w') as fout:
        labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
        for i in range(image_num):
            fout.write(str(labels[i]) + '\n')

    projector_writer = tf.summary.FileWriter(DIR + 'projector/projector', sess.graph)
    saver = tf.train.Saver()
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = embedding.name
    embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
    embed.sprite.image_path = DIR + 'projector/data/mnist_10k_sprite.png'
    embed.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(projector_writer, config)

    #merge all summaries
    merged = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    for epoch in range(epoch_num):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, optimizer], feed_dict={x:batch_xs, y:batch_ys}, options=run_options, run_metadata=run_metadata)
        projector_writer.add_run_metadata(run_metadata, 'step%03d' % epoch)
        projector_writer.add_summary(summary, epoch)

        if epoch % 100 == 0:
            acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
            print('Iter: ' + str(epoch) + ' acc: ' + str(acc))

    saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step = epoch_num)
    projector_writer.close()
    sess.close()





if __name__ == '__main__':
    main()
