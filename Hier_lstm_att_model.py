
import tensorflow as tf
import numpy as np

class RNN_Model(object):



    def __init__(self,config,is_training=True):

        self.batch_size=tf.Variable(0,dtype=tf.int32,trainable=False)
        self.max_source_sen_num=tf.Variable(0,dtype=tf.int32,trainable=False)
        self.max_source_word_num = tf.Variable(0, dtype=tf.int32, trainable=False)

        self.trian_source_set=tf.placeholder(tf.int32,[self.batch_size,self.max_source_sen_num,self.max_source_word_num])
        # self.target = tf.placeholder(tf.int64,[None])
        self.mask_trian_source_set = tf.placeholder(tf.int32,[self.max_source_sen_num,self.max_source_word_num,self.batch_size])

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
                length_array_input_for_word_encode_sent.append(tf.reduce_sum(self.mask_trian_source_set[no_sen],axis=0))   #计算每一句话的长度 就是seq_length




        """下面是编码阶段"""
        output_from_word_encode_sent = []                                                 #单词到句子编码的输出
        state_word_encode_sent = self._initial_state_word_encode_sent
        with tf.Variable_scope("word_encode_sent"):
            for no_sen in range(self.max_train_source_sen_num):
                for no_word in range(self.max_train_source_word_num):
                    if no_sen > 0 or no_word > 0:
                        tf.get_variable_scope().reuse_variables()
                    _, state_word_encode_sent = tf.nn.dynamic_rnn(word_encode_sent_cell,
                                                                      inputs=input_for_word_encode_sent[no_sen][:,no_word, :],
                                                                      sequence_length=length_array_input_for_word_encode_sent[no_sen],
                                                                      initial_state=state_word_encode_sent,
                                                                      time_major=False)
                output_from_word_encode_sent.append(state_word_encode_sent[-1][1])       #state_word_encode_sent[-1][1]就是ht shape为[batch_size,hidden_dim]



        output_from_sent_encode_doc=[]                               #4层的要全部保存下来 包括c_t和h_t
        with tf.variable_scope("sent_encode_doc"):
            for no_sen in range(self.max_train_source_sen_num):
                if no_sen > 0:
                    tf.get_variable_scope().reuse_variables()
                _, sent_state) = self.encode_sent_cell(output_for_sent_encode_doc[s], sent_state)

























        out_put=[]
        state=self._initial_state
        with tf.variable_scope("LSTM_layer"):
            for time_step in range(num_step):
                if time_step>0: tf.get_variable_scope().reuse_variables()
                (cell_output,state)=cell(inputs[:,time_step,:],state)
                out_put.append(cell_output)

        out_put=out_put*self.mask_x[:,:,None]

        with tf.name_scope("mean_pooling_layer"):

            out_put=tf.reduce_sum(out_put,0)/(tf.reduce_sum(self.mask_x,0)[:,None])

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



















