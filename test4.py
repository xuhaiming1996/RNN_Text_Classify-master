import tensorflow as tf

input_x=tf.Variable(tf.truncated_normal([16,2000],dtype=tf.float32))
lstm_cell=tf.nn.rnn_cell.LSTMCell(1000, num_proj=1000)
word_encode_sent_basic_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(1000, forget_bias=0.0,  state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
word_encode_sent_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell,word_encode_sent_basic_lstm_cell,word_encode_sent_basic_lstm_cell], state_is_tuple=True)
state_word_encode_sent = word_encode_sent_cell.zero_state(16, dtype=tf.float32)

print(state_word_encode_sent[0][0])
output,state_word_encode_sent=word_encode_sent_cell(input_x,state_word_encode_sent)
print(output.get_shape().as_list())
print(state_word_encode_sent[1][0])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(output))
