import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def RNN(X, weight, biases):
    n_inputs = 28
    max_time = 28
    lstm_size = 50
    inputs = tf.reshape(X, [-1, max_time, n_inputs]) #[batch_size, max_time, n_inputs]
    #attention_states
    attention_states = tf.transpose(X, [1, 0, 2])
    attention_mechanism = tf.contrib.seq2seq2seq.LuongAttention(
        128, attention_states,
        memory_sequence_length=source_sequence_length)


    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    #final_state[0] is cell state ~ value in memory cell
    #final_state[1] is hidden_state ~ value of h'(memory cell)
    output, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype = tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weight) + biases)
    return results


if __name__ == '__main__':
    tf.reset_default_graph()
    main()
