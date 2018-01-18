
import tensorflow as tf
import numpy as np

class RNN_Model(object):



    def __init__(self,config,is_training=True):
        self.batch_size=tf.Variable(0,dtype=tf.int32,trainable=False)
        #这是编码阶段用到的
        self.max_source_sen_num=tf.Variable(0,dtype=tf.int32,trainable=False)
        self.max_source_word_num = tf.Variable(0, dtype=tf.int32, trainable=False)

        self.train_source_set=tf.placeholder(tf.int32,[self.batch_size,self.max_source_sen_num,self.max_source_word_num])
        # self.target = tf.placeholder(tf.int64,[None])
        self.mask_train_source_set = tf.placeholder(tf.int32,[self.max_source_sen_num,self.max_source_word_num,self.batch_size])

        hidden_neural_size=config.hidden_neural_size   #隐藏层神经元的的个数
        vocabulary_size=config.vocabulary_size         #单词的个数
        embed_dim=config.embed_dim                     #word2Vec的的维度
        hidden_layer_num=config.hidden_layer_num       #神经元的层数

        self.new_batch_size = tf.placeholder(tf.int32,shape=[],name="new_batch_size")
        self._batch_size_update = tf.assign(self.batch_size,self.new_batch_size)

        self.new_max_source_sen_num=tf.placeholder(tf.int32,shape=[],name="new_max_source_sen_num")
        self._max_source_sen_num_update=tf.assign(self.max_source_sen_num,self.new_max_source_sen_num)

        self.new_max_source_word_num = tf.placeholder(tf.int32, shape=[], name="new_max_source_word_num")
        self._max_source_word_num_update = tf.assign(self.max_source_word_num, self.new_max_source_word_num)


        #解码阶段用到的
        #build LSTM network
        '''
        构造 4层的单词到句子的LSTM  保存每一个句子的向量
        构造 4层的句子到文章的LSTM  
        '''
        word_encode_sent_basic_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_neural_size, forget_bias=0.0,  state_is_tuple=True)
        sent_encode_doc_basic_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_neural_size, forget_bias=0.0,   state_is_tuple=True)

        word_encode_sent_cell = tf.nn.rnn_cell.MultiRNNCell([word_encode_sent_basic_lstm_cell] * hidden_layer_num, state_is_tuple=True)
        sent_encode_doc_cell = tf.nn.rnn_cell.MultiRNNCell([sent_encode_doc_basic_lstm_cell] * hidden_layer_num, state_is_tuple=True)

        self._initial_state_word_encode_sent = word_encode_sent_cell.zero_state(self.batch_size, dtype=tf.float32)
        self._initial_state_sent_encode_doc = sent_encode_doc_cell.zero_state(self.batch_size, dtype=tf.float32)

        input_for_word_encode_sent = []
        length_array_input_for_word_encode_sent = []

        # embedding layer 同时对mask_train_source_set进行处理获取[max_train_sent_num,]
        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            embedding = tf.get_variable("embedding", [vocabulary_size, embed_dim], dtype=tf.float32)
            for no_sen in range(self.max_source_sen_num):
                input_for_word_encode_sent.append(tf.nn.embedding_lookup(embedding, self.train_source_set[:, no_sen,:]))
                length_array_input_for_word_encode_sent.append(tf.reduce_sum(self.mask_train_source_set[no_sen],axis=0))   #计算每一句话的长度 就是seq_length




        """下面是编码阶段"""
        output_from_word_encode_sent = []                                                 #单词到句子编码的输出
        state_word_encode_sent = self._initial_state_word_encode_sent                     #state.shape = [layer_num, 2, batch_size, hidden_size],
        with tf.variable_scope("word_encode_sent"):
            for no_sen in range(self.max_source_sen_num):
                #for no_word in range(self.max_source_word_num):
                    if no_sen > 0 or no_word > 0:
                        tf.get_variable_scope().reuse_variables()
                    _, state_word_encode_sent = tf.nn.dynamic_rnn(word_encode_sent_cell,
                                                                      inputs=input_for_word_encode_sent[no_sen][:,no_word, :],
                                                                      sequence_length=length_array_input_for_word_encode_sent[no_sen],
                                                                      initial_state=state_word_encode_sent,
                                                                      time_major=False)
                output_from_word_encode_sent.append(state_word_encode_sent[-1][1])       #state_word_encode_sent[-1][1]就是ht shape为[batch_size,hidden_dim]

        train_source_each_sent=[]           #相当于matlab的source_each_sent   里买你保存的c_t和h_t
        output_from_sent_encode_doc = []  # 4层的要全部保存下来 包括c_t和h_t
        state_sent_encode_doc = self._initial_state_sent_encode_doc
        length_array_input_for_sent_encode_doc = []                      #就是一个list
        for no_batch in range(self.batch_size):
            sen_num = 0
            for no_sen in range(self.max_source_sen_num):
                sen_num+=self.mask_train_source_set[no_sen][0][no_batch]
            length_array_input_for_sent_encode_doc.append(sen_num)

        with tf.variable_scope("sent_encode_doc"):
            for no_sen in range(self.max_source_sen_num):
                if no_sen > 0:
                    tf.get_variable_scope().reuse_variables()
                _, state_sent_encode_doc= tf.nn.dynamic_rnn(sent_encode_doc_cell,
                                                              inputs=output_from_word_encode_sent[no_sen],
                                                              sequence_length=length_array_input_for_sent_encode_doc,
                                                              initial_state=state_sent_encode_doc,
                                                              time_major=False)
                train_source_each_sent.append(state_sent_encode_doc)
            output_from_sent_encode_doc.append(state_sent_encode_doc)



        # 下面是解码阶段
        self.max_target_sen_num = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.max_target_word_num = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.train_target_set = tf.placeholder(tf.int32,[self.batch_size, self.max_target_sen_num, self.max_target_word_num])
        self.mask_train_target_set = tf.placeholder(tf.int32, [self.max_target_sen_num, self.max_target_word_num,self.batch_size])

        self.new_max_target_sen_num = tf.placeholder(tf.int32, shape=[], name="new_max_target_sen_num")
        self._max_target_sen_num_update = tf.assign(self.max_target_sen_num, self.new_max_target_sen_num)

        self.new_max_target_word_num = tf.placeholder(tf.int32, shape=[], name="new_max_target_word_num")
        self._max_target_word_num_update = tf.assign(self.max_target_word_num, self.new_max_target_word_num)

        doc_decode_sent_basic_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_neural_size, forget_bias=0.0,
                                                                       state_is_tuple=True)
        sent_decode_word_basic_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_neural_size, forget_bias=0.0,
                                                                        state_is_tuple=True)
        doc_decode_sent_cell = tf.nn.rnn_cell.MultiRNNCell([doc_decode_sent_basic_lstm_cell] * hidden_layer_num,
                                                           state_is_tuple=True)

        sent_decode_word_cell = tf.nn.rnn_cell.MultiRNNCell([sent_decode_word_basic_lstm_cell] * hidden_layer_num,
                                                            state_is_tuple=True)
        self._initial_state_doc_decode_sent = doc_decode_sent_cell.zero_state(self.batch_size, dtype=tf.float32)
        self._initial_state_sent_decode_word = sent_decode_word_cell.zero_state(self.batch_size, dtype=tf.float32)



        # embedding layer 同时对mask_train_target_set进行处理获取[max_train_sent_num,]
        input_for_sent_decode_word = []
        length_array_input_for_sent_decode_word = []
        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):        #这个先写在这里  最后要整合到一个里面去！******************！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            embedding = tf.get_variable("embedding", [vocabulary_size, embed_dim], dtype=tf.float32)
            for no_sen in range(self.max_target_sen_num):
                input_for_sent_decode_word.append(tf.nn.embedding_lookup(embedding, self.train_target_set[:, no_sen, :]))
                length_array_input_for_sent_decode_word.append(tf.reduce_sum(self.mask_train_target_set[no_sen], axis=0))  # 计算每一句话的长度 就是seq_length

        
        
        length_array_input_for_doc_decode_sent = []  # 就是一个list
        for no_batch in range(self.batch_size):
            sen_num = 0
            for no_sen in range(self.max_target_sen_num):
                sen_num += self.mask_train_target_set[no_sen][0][no_batch]
            length_array_input_for_doc_decode_sent.append(sen_num)
        # state_doc_decode_sent  = self._initial_state_doc_decode_sent
        # state_sent_decode_word = self._initial_state_sent_decode_word

        Taget_sen=[] #以句子为单位 保存每一句解码的过程 和matlab里面的名字一样
        #开始解码
        state_doc_decode_sent = output_from_sent_encode_doc                  #包含四层的ht 和 Ct

        #进行解码
        h_t_target_sen = []                                                    #保存在doc_decode_sent 中所有时间步的最后一层的ht 列表的长度就是文章的最大的句子数
        h_t_Target_sen = []                                                    #将sent_decode_word中的每个时间步的最后一层的输出ht保存为一个元素 放在这个列表里面  所以这个列表的长度就是文章的最大句子数
        for no_sen in range(self.max_target_sen_num):
            state_sent_decode_word = state_doc_decode_sent
            h_t_target_word = []
            with tf.variable_scope("sent_decode_word"):
                for no_word in range(self.max_target_word_num):
                    if no_word > 0 or no_sen > 0:
                        tf.get_variable_scope().reuse_variables()
                    _, state_sent_decode_word = tf.nn.dynamic_rnn(sent_decode_word_cell,
                                                                  inputs=input_for_sent_decode_word[no_sen][:, no_word,:],
                                                                  sequence_length= length_array_input_for_sent_decode_word[no_sen],
                                                                  initial_state=state_sent_decode_word,
                                                                  time_major=False)
                    h_t_target_word.append(state_sent_decode_word[-1][-1])                   #只保存最后一层的h

                h_t_Target_sen.append(h_t_target_word)
               #sjdhasdi hasi dhihsid hoisd
                input_for_doc_decode_sent_at_tt = state_sent_decode_word[-1][-1]  # 这个时间步的doc_decode_sent的输入
                vs = []
                for no_sen_2 in range(self.max_target_sen_num):
                    with tf.variable_scope("decode_RNN_attention"):
                        if no_sen > 0 or no_sen_2 > 0:
                            tf.get_variable_scope().reuse_variables()
                        u = tf.get_variable("u", [hidden_neural_size, 1])  # 1000X1
                        w1 = tf.get_variable("w1", [hidden_neural_size, hidden_neural_size])  # 1000X1000
                        w2 = tf.get_variable("w2", [hidden_neural_size, hidden_neural_size])  # 1000X1000
                        b = tf.get_variable("b1", [hidden_neural_size])  # 1000
                        vi = tf.matmul(tf.tanh(tf.add(tf.add(
                            tf.matmul(state_doc_decode_sent[-1][-1], w1),
                            tf.matmul(train_source_each_sent[no_sen_2], w2)), b)), u)
                        vi = tf.exp(vi)
                        vs.append(vi)

                vs_sum = tf.add_n(vs)
                mt = tf.add_n([vs[i] * train_source_each_sent[i] for i in range(self.max_target_sen_num)]) / vs_sum

                with tf.variable_scope("doc_decode_sent"):
                    if no_sen > 0:
                        tf.get_variable_scope().reuse_variables()
                    sent_decode = tf.concat(1, [input_for_doc_decode_sent_at_tt, mt])  # 16X2000
                    _, state_doc_decode_sent = tf.nn.dynamic_rnn(doc_decode_sent_cell,
                                                                 inputs=sent_decode,
                                                                 sequence_length=length_array_input_for_doc_decode_sent,
                                                                 initial_state=state_doc_decode_sent,
                                                                 time_major=False)

                state_doc_decode_sent = state_doc_decode_sent
                h_t_target_sen.append(state_doc_decode_sent[-1][-1])


        #
        soft_w = tf.get_variable("w", [self.size, self.vocab_size])
        soft_b = tf.get_variable("b", [self.vocab_size])


        with tf.name_scope("Softmax_layer_and_output"):
            softmax_w = tf.get_variable("softmax_w",[hidden_neural_size,class_num],dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b",[class_num],dtype=tf.float32)
            self.logits = tf.matmul(out_put,softmax_w)+softmax_b

        with tf.name_scope("loss"):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits+1e-10,self.target)
            self.cost = tf.reduce_mean(self.loss)

        with tf.name_scope("accuracy"):
            self.prediction = tf.argmax(self.logits,1)
            correct_prediction = tf.equal(self.prediction,self.target)
            self.correct_num=tf.reduce_sum(tf.cast(correct_prediction,tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name="accuracy")

        #add summary
        loss_summary = tf.scalar_summary("loss",self.cost)
        #add summary
        accuracy_summary=tf.scalar_summary("accuracy_summary",self.accuracy)

        if not is_training:
            return

        self.globle_step = tf.Variable(0,name="globle_step",trainable=False)
        self.lr = tf.Variable(0.0,trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                      config.max_grad_norm)


        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in zip(grads, tvars):
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.grad_summaries_merged = tf.merge_summary(grad_summaries)

        self.summary =tf.merge_summary([loss_summary,accuracy_summary,self.grad_summaries_merged])



        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer.apply_gradients(zip(grads, tvars))
        self.train_op=optimizer.apply_gradients(zip(grads, tvars))

        self.new_lr = tf.placeholder(tf.float32,shape=[],name="new_learning_rate")
        self._lr_update = tf.assign(self.lr,self.new_lr)



    def assign_new_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self.new_lr:lr_value})

    def assign_new_batch_size(self,session,batch_size_value):
        session.run(self._batch_size_update,feed_dict={self.new_batch_size:batch_size_value})

    def assign_new_max_source_sen_num(self,session,max_source_sen_num_value):
        session.run(self._max_source_sen_num_update,feed_dict={self.new_max_source_sen_num:max_source_sen_num_value})

    def assign_new_max_source_word_num(self,session,max_source_word_num_value):
        session.run(self._max_source_word_num_update,feed_dict={self.new_max_source_word_num:max_source_word_num_value})

    def assign_new_max_target_sen_num(self, session, max_target_sen_num_value):
        session.run(self._max_target_sen_num_update, feed_dict={self.new_max_target_sen_num: max_target_sen_num_value})

    def assign_new_max_target_word_num(self, session, max_target_word_num_value):
        session.run(self._max_target_word_num_update, feed_dict={self.new_max_target_word_num: max_target_word_num_value})
