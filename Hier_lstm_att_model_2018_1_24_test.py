
import tensorflow as tf
import numpy as np
import math

class RNN_Model(object):
    def __init__(self,config):
        self.batch_size=config.batch_size
        # self.batch_size = tf.Variable(0,dtype=tf.int32,trainable=False)
        self.hidden_neural_size = config.hidden_neural_size  # 隐藏层神经元的的个数
        vocabulary_size = config.vocabulary_size  # 单词的个数
        embed_dim = config.embed_dim  # word2Vec的的维度
        self.hidden_layer_num = config.hidden_layer_num  # 神经元的层数
        max_grad_norm =config.max_grad_norm              #搞明白这个原理 适当更改
        # self.new_batch_size = tf.placeholder(tf.int32, shape=[], name="new_batch_size")
        # self._batch_size_update = tf.assign(self.batch_size, self.new_batch_size)
        #这是编码阶段用到的
        self.max_source_sen_num = config.max_source_sen_num
        self.max_source_word_num = config.max_source_word_num
        self.train_source_set = tf.placeholder(tf.int32,[self.batch_size, self.max_source_sen_num, self.max_source_word_num])
        self.mask_train_source_set = tf.placeholder(tf.int32, [self.max_source_sen_num, self.max_source_word_num,self.batch_size])


        #解码阶段用到的
        self.max_target_sen_num = config.max_target_sen_num
        self.max_target_word_num = config.max_target_word_num

        self.train_target_set = tf.placeholder(tf.int32, [self.batch_size, self.max_target_sen_num, self.max_target_word_num])
        self.mask_train_target_set_float = tf.placeholder(tf.float32, [self.max_target_sen_num, self.max_target_word_num,
                                                               self.batch_size])
        self.mask_train_target_set = tf.placeholder(tf.int32, [self.max_target_sen_num, self.max_target_word_num,
                                                               self.batch_size])


        self.sen_mask = tf.placeholder(tf.float32, [self.batch_size, self.max_source_sen_num])  #这个变量准备一下。。。。。。。
        self.lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        self. embedding = tf.get_variable("embedding", [vocabulary_size, embed_dim], dtype=tf.float32)

        word_encode_sent_basic_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_neural_size, forget_bias=0.0,
                                                                        state_is_tuple=True,
                                                                        reuse=tf.AUTO_REUSE)
        sent_encode_doc_basic_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_neural_size, forget_bias=0.0,
                                                                       state_is_tuple=True,
                                                                       reuse=tf.AUTO_REUSE)

        word_encode_sent_cell = tf.nn.rnn_cell.MultiRNNCell([word_encode_sent_basic_lstm_cell] * self.hidden_layer_num,
                                                            state_is_tuple=True)
        sent_encode_doc_cell = tf.nn.rnn_cell.MultiRNNCell([sent_encode_doc_basic_lstm_cell] * self.hidden_layer_num,
                                                           state_is_tuple=True)

        self._initial_state_word_encode_sent = word_encode_sent_cell.zero_state(self.batch_size, dtype=tf.float32)
        self._initial_state_sent_encode_doc = sent_encode_doc_cell.zero_state(self.batch_size, dtype=tf.float32)

        # build decoder_LSTM cell
        doc_decode_sent_basic_lstm = tf.nn.rnn_cell.LSTMCell(self.hidden_neural_size, num_proj=self.hidden_neural_size,
                                                             forget_bias=0.0,
                                                             state_is_tuple=True, reuse=tf.AUTO_REUSE)
        doc_decode_sent_basic_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_neural_size, forget_bias=0.0,
                                                                       state_is_tuple=True,
                                                                       reuse=tf.AUTO_REUSE)
        sent_decode_word_basic_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_neural_size, forget_bias=0.0,
                                                                        state_is_tuple=True,
                                                                        reuse=tf.AUTO_REUSE)
        doc_decode_sent_cell = tf.nn.rnn_cell.MultiRNNCell(
            [doc_decode_sent_basic_lstm, doc_decode_sent_basic_lstm_cell, doc_decode_sent_basic_lstm_cell,
             doc_decode_sent_basic_lstm_cell],
            state_is_tuple=True)

        sent_decode_word_cell = tf.nn.rnn_cell.MultiRNNCell([sent_decode_word_basic_lstm_cell] * self.hidden_layer_num,
                                                            state_is_tuple=True)

        self._initial_state_doc_decode_sent = doc_decode_sent_cell.zero_state(self.batch_size, dtype=tf.float32)

        self._initial_state_sent_decode_word = sent_decode_word_cell.zero_state(self.batch_size, dtype=tf.float32)
        # self._initial_state_doc_decode_sent = doc_decode_sent_cell.zero_state(self.batch_size, dtype=tf.float32)
        # self._initial_state_sent_decode_word = sent_decode_word_cell.zero_state(self.batch_size, dtype=tf.float32)
        # 编码阶段
        input_for_word_encode_sent = []
        length_array_input_for_word_encode_sent = []
        with tf.device("/cpu:0"):
            for no_sen in range(self.max_source_sen_num):
                input_for_word_encode_sent.append(
                    tf.nn.embedding_lookup(self.embedding, self.train_source_set[:, no_sen, :]))
                length_array_input_for_word_encode_sent.append(tf.reduce_sum(self.mask_train_source_set[no_sen], axis=0))  # 计算每一句话的长度 就是seq_length
        #加上--------------------------------测试的东西
        output_from_word_encode_sent = []  # 单词到句子编码的输出
        state_word_encode_sent = self._initial_state_word_encode_sent  # state.shape = [layer_num, 2, batch_size, hidden_size],
        self.length_array_input_for_word_encode_sent = length_array_input_for_word_encode_sent[9]


        for no_sen in range(self.max_source_sen_num):
            state_word_encode_sent = self._initial_state_word_encode_sent  # state.shape = [layer_num, 2, batch_size, hidden_size],
            if no_sen > 0:
                tf.get_variable_scope().reuse_variables()
            outputs, state_word_encode_sent = tf.nn.dynamic_rnn(word_encode_sent_cell,
                                                                inputs=input_for_word_encode_sent[no_sen],
                                                                sequence_length=length_array_input_for_word_encode_sent[no_sen],
                                                                initial_state=state_word_encode_sent,
                                                                time_major=False)
            output_from_word_encode_sent.append(state_word_encode_sent[-1][1])  # state_word_encode_sent[-1][1]就是ht shape为[batch_size,hidden_dim]

        # train_source_each_sent=[]   #相当于matlab的source_each_sent   里买你保存的c_t和h_t 整体的shape是 [batch_size, max_time, cell_state_size]
        # output_from_sent_encode_doc = []                                             # 4层的要全部保存下来 包括c_t和h_t
        state_sent_encode_doc = self._initial_state_sent_encode_doc
        length_array_input_for_sent_encode_doc = []  # 就是一个list
        for no_batch in range(self.batch_size):
            sen_num = 0
            for no_sen in range(self.max_source_sen_num):
                sen_num += self.mask_train_source_set[no_sen][0][no_batch]
            length_array_input_for_sent_encode_doc.append(sen_num)

        self.length_array_input_for_sent_encode_doc = length_array_input_for_sent_encode_doc


        # 注意 output_from_word_encode_sent的shape:[max_sen_time, batch_size, cell_state_size]  但是传进去的需要是[batch_size, max_sen_time, cell_state_size]
        output_from_word_encode_sent = tf.transpose(output_from_word_encode_sent, [1, 0, 2])
        outputs, state_sent_encode_doc = tf.nn.dynamic_rnn(sent_encode_doc_cell,
                                                           inputs=output_from_word_encode_sent,
                                                           sequence_length=length_array_input_for_sent_encode_doc,
                                                           initial_state=state_sent_encode_doc,
                                                           time_major=False)
        train_source_each_sent = outputs
        output_from_sent_encode_doc = state_sent_encode_doc



        with tf.variable_scope("decode_RNN_attention", reuse=tf.AUTO_REUSE):
            u = tf.get_variable("u", [self.hidden_neural_size, 1])  # 1000X1
            w1 = tf.get_variable("w1", [self.hidden_neural_size, self.hidden_neural_size])  # 1000X1000
            w2 = tf.get_variable("w2", [self.hidden_neural_size, self.hidden_neural_size])  # 1000X1000
            b = tf.get_variable("b1", [self.hidden_neural_size])  # 1000

        # 解码阶段
        input_for_sent_decode_word = []
        length_array_input_for_sent_decode_word = []
        # with tf.device("/cpu:0"), tf.name_scope("target_embedding_layer"):
        with tf.device("/cpu:0"):
            for no_sen in range(self.max_target_sen_num):
                input_for_sent_decode_word.append(
                    tf.nn.embedding_lookup(self.embedding, self.train_target_set[:, no_sen, :]))
                length_array_input_for_sent_decode_word.append(
                    tf.reduce_sum(self.mask_train_target_set[no_sen], axis=0))  # 计算每一句话的长度 就是seq_length

        length_array_input_for_doc_decode_sent = []  # 就是一个list
        for no_batch in range(self.batch_size):
            sen_num = 0
            for no_sen in range(self.max_target_sen_num):
                sen_num += self.mask_train_target_set[no_sen][0][no_batch]
            length_array_input_for_doc_decode_sent.append(sen_num)
        # state_doc_decode_sent  = self._initial_state_doc_decode_sent
        # state_sent_decode_word = self._initial_state_sent_decode_word

        # Taget_sen=[] #以句子为单位 保存每一句解码的过程的输出 和matlab里面的名字一样
        state_doc_decode_sent = output_from_sent_encode_doc  # 包含四层的ht 和 Ct
        h_t_target_sen = []  # 保存在doc_decode_sent 中所有时间步的最后一层的ht 列表的长度就是文章的最大的句子数  整体的shape是[max_sen_time,batch_size,cell_state_size]
        h_t_Target_sen = []  # 将sent_decode_word中的每个时间步的最后一层的输出ht保存为一个元素 放在这个列表里面  所以这个列表的长度就是文章的最大句子数 整体的shape是[max_sen_time,batch_size, max_word_time, cell_state_size]
        train_source_each_sent = tf.transpose(train_source_each_sent, [1, 0,
                                                                       2])  # 现在train_source_each_sent的shapes是[max_sen_time,batch_size,cell_state_size]
        sen_mask = tf.transpose(self.sen_mask, [1, 0])  # shape是[max_sen_time,batch_size]
        for no_sen in range(self.max_target_sen_num):
            state_sent_decode_word = state_doc_decode_sent
            # with tf.variable_scope("sent_decode_word"):
            if no_sen > 0:
                tf.get_variable_scope().reuse_variables()
            outputs, state_sent_decode_word = tf.nn.dynamic_rnn(sent_decode_word_cell,
                                                                inputs=input_for_sent_decode_word[no_sen],
                                                                sequence_length=length_array_input_for_sent_decode_word[ no_sen],
                                                                initial_state=state_sent_decode_word,
                                                                time_major=False)
            h_t_Target_sen.append(
                outputs)  # 每一个outputs的shape都是[batch_size, max_word_time, cell_state_size] padding位置上的输出全是0
            input_for_doc_decode_sent_at_tt = state_sent_decode_word[-1][-1]  # 这个时间步的doc_decode_sent的输入

            vs = []
            for no_sen_2 in range(self.max_target_sen_num):
                # with tf.variable_scope("decode_RNN_attention", reuse=tf.AUTO_REUSE):
                #     if no_sen > 0 or no_sen_2 > 0:
                #         tf.get_variable_scope().reuse_variables()
                #     u = tf.get_variable("u", [self.hidden_neural_size, 1])  # 1000X1
                #     w1 = tf.get_variable("w1", [self.hidden_neural_size, self.hidden_neural_size])  # 1000X1000
                #     w2 = tf.get_variable("w2", [self.hidden_neural_size, self.hidden_neural_size])  # 1000X1000
                #     b = tf.get_variable("b1", [self.hidden_neural_size])  # 1000
                vi = tf.matmul(tf.tanh(tf.add(tf.add(
                    tf.matmul(state_doc_decode_sent[-1][-1], w1),
                    tf.matmul(train_source_each_sent[no_sen_2], w2)), b)), u)
                vi = tf.exp(vi)  # vi的shape是[batchsize,1]
                # vi = vi * tf.transpose(sen_mask[no_sen_2], [1, 0])
                vi = vi * tf.reshape(sen_mask[no_sen_2], [-1, 1])
                # print("vi",vi)
                # print("tf.reshape(sen_mask[no_sen_2], [-1, 1])",tf.reshape(sen_mask[no_sen_2], [-1, 1]))
                vs.append(vi)

            vs_sum = tf.add_n(vs)
            mt = tf.add_n([vs[i] * train_source_each_sent[i] for i in
                           range(self.max_target_sen_num)]) / vs_sum  # train_source_each_sent[i]的shape是
            # with tf.variable_scope("doc_decode_sent"):
            if no_sen > 0:
                tf.get_variable_scope().reuse_variables()
            # new_state_doc_decode_sent = []
            # print(state_doc_decode_sent)

            for no_lay in range(self.hidden_layer_num):
                new_state_doc_decode_sent = np.array(state_doc_decode_sent)
                new_state_doc_decode_sent[no_lay][0] = tf.reshape(sen_mask[no_sen], [-1, 1]) * \
                                                       new_state_doc_decode_sent[no_lay][0]
                new_state_doc_decode_sent[no_lay][1] = tf.reshape(sen_mask[no_sen], [-1, 1]) * \
                                                       new_state_doc_decode_sent[no_lay][1]

            state_doc_decode_sent = tuple(new_state_doc_decode_sent)




            sent_decode = tf.concat([input_for_doc_decode_sent_at_tt, mt], 1) * tf.reshape(sen_mask[no_sen],
                                                                                           [-1, 1])  # 16X2000
            # print("tf.reshape(sen_mask[no_sen], [-1, 1])",tf.reshape(sen_mask[no_sen], [-1, 1]))
            # print("input_for_doc_decode_sent_at_tt",input_for_doc_decode_sent_at_tt)
            # states的shape = [batch_size, n(LSTMStateTuple)]
            output, state_doc_decode_sent = doc_decode_sent_cell(sent_decode, state_doc_decode_sent)

            h_t_target_sen.append(output)  # output的shape是[batchsize,cell_state_size] padding的位置都是0

        word_decodes = []
        self.h_t_Target_sen = h_t_Target_sen[0][0][4]
        # 计算误差
        for no_sen in range(self.max_target_sen_num):
            if no_sen == 0:
                h_t_1 = output_from_sent_encode_doc[-1][-1]  # 16X1000
            else:
                h_t_1 = h_t_target_sen[no_sen - 1]

            word_decodes.append(h_t_1)
            # 获取第no_sen句子的在word水平的解码的输出
            target_sen = h_t_Target_sen[no_sen]  # shape是[batch_size, max_word_time, cell_state_size]
            target_sen = tf.transpose(target_sen, perm=[1, 0, 2])  # shape是[max_word_time,batch_size,cell_state_size]
            for no_word in range(self.max_target_word_num - 1):
                word_decodes.append(target_sen[no_word])

        targets = tf.reshape(self.train_target_set,
                             [self.batch_size, self.max_target_sen_num * self.max_target_word_num])
        targets = [targets[:, i] for i in range(self.max_target_sen_num * self.max_target_word_num)]
        # targets = tf.convert_to_tensor(targets, dtype=tf.int64)
        targets = tf.reshape(targets, [-1])

        weights = tf.reshape(self.mask_train_target_set_float, [-1, 1])

        word_decodes = tf.reshape(word_decodes, [-1, self.hidden_neural_size])

        word_decodes = word_decodes * weights
        with tf.variable_scope("Softmax", reuse=tf.AUTO_REUSE):
            softmax_w = tf.get_variable("softmax_w", [self.hidden_neural_size, vocabulary_size], dtype=tf.float32)
        self.logits = tf.matmul(word_decodes, softmax_w)

        with tf.name_scope("loss"):

            # print("weights:",weights)
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits + 1e-10,
                                                                       labels=targets)  # 这里加一是我们的语料的的特点
            # print("self.loss",self.loss)
            self.cost = tf.reduce_sum(tf.reshape(weights, [-1]) * self.loss) / self.batch_size

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), max_grad_norm)

        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer.apply_gradients(zip(grads, tvars))
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))




        #测试阶段的解码过程
        train_source_each_sent = tf.transpose(train_source_each_sent, [1, 0, 2])

        max_length = math.floor(1.5 * self.max_source_sen_num * self.max_target_word_num)                               #根据编码最后得到的输出 得到第一个单词
        output_words = []  #存档解码中 每一步得到的单词  最后的shape是 [Max_length,batch_size]
        self.words = tf.argmax(tf.matmul(output_from_sent_encode_doc[-1][-1], softmax_w[:, 0:vocabulary_size-2]),1)

        #测试这一步的输出 看看后面是否处理正确

        output_words.append(self.words)

        state_doc_decode_sent = output_from_sent_encode_doc
        state_sent_decode_word = state_doc_decode_sent
        # train_source_each_sent = []   整体的shape是 [batch_size, max_time, cell_state_size]
        #sen_mask = tf.transpose(self.sen_mask, [1, 0])  # shape是[max_sen_time,batch_size]
        for no_word in range(max_length):
            words = output_words[no_word]                                                              #是一个行向量大小是batch_size
            _, state_sent_decode_word = sent_decode_word_cell(tf.nn.embedding_lookup(self.embedding, words),state_sent_decode_word)
            #寻找单词是句子结束符号的
            end_sen_index=[]

            for i in range(self.batch_size):
                if words[i]==vocabulary_size-2:
                    end_sen_index.append(i)

            if len(end_sen_index)>0:
                #下面的所有操作 全部都是对这次得到的单词中是句子结束符号的文章进行的操作 并不是整个batch_size
                x_t_sen_stop=[]
                state_sen_decode_word_sen_stop=[]               #代表的是剩余的文章的信息  是解码单词中最后一层的h_t
                train_source_each_sent_sen_stop = []
                sen_mask_sen_stop=[]
                for i in range(len(end_sen_index)):
                    x_t_sen_stop.append(state_sent_decode_word[0][1][end_sen_index[i]])
                    state_sen_decode_word_sen_stop.append(state_sent_decode_word[-1][-1][end_sen_index[i]])
                    train_source_each_sent_sen_stop.append(train_source_each_sent[end_sen_index[i]])
                    sen_mask_sen_stop.append(self.sen_mask[end_sen_index[i]])



                train_source_each_sent_sen_stop = tf.transpose(train_source_each_sent_sen_stop, [1, 0, 2])  # 现在train_source_each_sent__sen_stop的shapes是[max_sen_time,batch_size_sen_stop,cell_state_size]
                sen_mask_sen_stop=tf.transpose(sen_mask_sen_stop, [1, 0])  # shape是[max_sen_time,batch_size_sen_stop]
                vs = []
                for no_sen in range(self.max_source_sen_num):
                    if no_word > 0 or no_sen > 0:
                        tf.get_variable_scope().reuse_variables()
                    vi = tf.matmul(tf.tanh(tf.add(tf.add(
                        tf.matmul(state_sen_decode_word_sen_stop, w1),
                        tf.matmul(train_source_each_sent_sen_stop[no_sen], w2)), b)), u)
                    vi = tf.exp(vi)  # vi的shape是[batchsize,1]
                    # vi = vi * tf.transpose(sen_mask[no_sen_2], [1, 0])
                    vi = vi * tf.reshape(sen_mask_sen_stop[no_sen], [-1, 1])
                    # print("vi",vi)
                    # print("tf.reshape(sen_mask[no_sen_2], [-1, 1])",tf.reshape(sen_mask[no_sen_2], [-1, 1]))
                    vs.append(vi)

                vs_sum = tf.add_n(vs)
                mt = tf.add_n([vs[i] * train_source_each_sent_sen_stop[i] for i in
                               range(self.max_source_sen_num)]) / vs_sum  # train_source_each_sent[i]的shape是

                state_doc_decode_sent_sen_stop = []
                for num_layer in range(self.hidden_layer_num):
                    for i in range(len(end_sen_index)):  #end_sen_index[i]words
                        state=[]
                        c_t_sen_stop = []
                        h_t_sen_stop = []
                        c_t_sen_stop.append(state_doc_decode_sent[num_layer][0][end_sen_index[i]])
                        state.append(c_t_sen_stop)
                        h_t_sen_stop.append(state_doc_decode_sent[num_layer][1][end_sen_index[i]])
                        state.append(h_t_sen_stop)
                        state_doc_decode_sent_sen_stop.append(state)


                state_doc_decode_sent_sen_stop=tuple(state_doc_decode_sent_sen_stop)
                sent_decode_sen_stop = tf.concat([x_t_sen_stop, mt], 1)
                _, state_doc_decode_sent_sen_stop = doc_decode_sent_cell(sent_decode_sen_stop, state_doc_decode_sent_sen_stop)


                #部分batch的注意力机制添加完毕 更新到原来的中
                new_state_doc_decode_sent = np.array(state_doc_decode_sent)
                new_state_sent_decode_word =np.array(state_sent_decode_word)
                for num_layer in range(self.hidden_layer_num):
                    for i in range(len(end_sen_index)):  #end_sen_index[i]
                        new_state_doc_decode_sent[num_layer][0][end_sen_index[i]] = state_doc_decode_sent_sen_stop[num_layer][0][i]
                        new_state_doc_decode_sent[num_layer][1][end_sen_index[i]] = state_doc_decode_sent_sen_stop[num_layer][1][i]

                        new_state_sent_decode_word[num_layer][0][end_sen_index[i]] = state_doc_decode_sent_sen_stop[num_layer][0][i]
                        new_state_sent_decode_word[num_layer][1][end_sen_index[i]] = state_doc_decode_sent_sen_stop[num_layer][1][i]


                state_doc_decode_sent=tuple(new_state_doc_decode_sent)
                state_sent_decode_word = tuple(new_state_sent_decode_word)


            all_next_words = tf.argmax(tf.matmul(state_sent_decode_word[-1][-1], softmax_w), 1)
            output_words.append(all_next_words)

        self.result = tf.transpose(output_words,[1,0])