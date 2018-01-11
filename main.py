import data_helper_source

# file_source="##源文件##"
# file_target="###目标文件###"

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from rnn_model import RNN_Model


flags =tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer('batch_size',16,'the batch_size of the training procedure')
flags.DEFINE_integer('vocabulary_size',25003,'vocabulary_size')              #25001是句子的结束符号 25002是文章的结束符号 所有的单词是1-25000 0不用
flags.DEFINE_integer('emdedding_dim',1000,'embedding dim')
flags.DEFINE_integer('hidden_neural_size',1000,'LSTM hidden neural size')
flags.DEFINE_integer('hidden_layer_num',4,'LSTM hidden layer num')
flags.DEFINE_float('initial',0.08,'init initial')    #这个初始化参数的范围1---
flags.DEFINE_integer('num_epoch',7,'num epoch')
flags.DEFINE_integer('max_decay_epoch',30,'num epoch')
flags.DEFINE_integer('max_grad_norm',5,'max_grad_norm')
flags.DEFINE_string('out_dir',os.path.abspath(os.path.join(os.path.curdir,"runs")),'output directory')
flags.DEFINE_integer('check_point_every',10,'checkpoint every num epoch ')
flags.DEFINE_integer('max_source_sen_num',-1,'最大的句子数')
flags.DEFINE_integer('max_source_word_num',-1,'最长的一句话包含的单词数')
flags.DEFINE_string('source_dir',#,'源文件的路径')
flags.DEFINE_string('target_dir',#,'目标文件的路径')


class Config(object):                           #配置模型需要的参数  这个里面只存放和模型相关的参数

    hidden_neural_size=FLAGS.hidden_neural_size
    vocabulary_size=FLAGS.vocabulary_size
    embed_dim=FLAGS.emdedding_dim
    hidden_layer_num=FLAGS.hidden_layer_num
    batch_size=FLAGS.batch_size
    max_grad_norm=FLAGS.max_grad_norm
    num_epoch = FLAGS.num_epoch
    out_dir=FLAGS.out_dir
    checkpoint_every = FLAGS.check_point_every
    max_source_sen_num=FLAGS.max_source_sen_num
    max_source_word_num=FLAGS.max_source_word_num



def evaluate(model,session,data,global_steps=None,summary_writer=None):


    correct_num=0
    total_num=len(data[0])
    for step, (x,y,mask_x) in enumerate(data_helper.batch_iter(data,batch_size=FLAGS.batch_size)):

         fetches = model.correct_num
         feed_dict={}
         feed_dict[model.input_data]=x
         feed_dict[model.target]=y
         feed_dict[model.mask_x]=mask_x
         model.assign_new_batch_size(session,len(x))
         state = session.run(model._initial_state)
         for i , (c,h) in enumerate(model._initial_state):
            feed_dict[c]=state[i].c
            feed_dict[h]=state[i].h
         count=session.run(fetches,feed_dict)
         correct_num+=count

    accuracy=float(correct_num)/total_num
    dev_summary = tf.scalar_summary('dev_accuracy',accuracy)
    dev_summary = session.run(dev_summary)
    if summary_writer:
        summary_writer.add_summary(dev_summary,global_steps)
        summary_writer.flush()
    return accuracy

def run_epoch(model,session,data,global_steps,valid_model,valid_data,train_summary_writer,valid_summary_writer=None):
    for step, (x,y,mask_x) in enumerate(data_helper.batch_iter(data,batch_size=FLAGS.batch_size)):

        feed_dict={}
        feed_dict[model.input_data]=x
        feed_dict[model.target]=y
        feed_dict[model.mask_x]=mask_x
        model.assign_new_batch_size(session,len(x))
        fetches = [model.cost,model.accuracy,model.train_op,model.summary]
        state = session.run(model._initial_state)
        for i , (c,h) in enumerate(model._initial_state):
            feed_dict[c]=state[i].c
            feed_dict[h]=state[i].h
        cost,accuracy,_,summary = session.run(fetches,feed_dict)
        train_summary_writer.add_summary(summary,global_steps)
        train_summary_writer.flush()
        valid_accuracy=evaluate(valid_model,session,valid_data,global_steps,valid_summary_writer)
        if(global_steps%100==0):
            print("the %i step, train cost is: %f and the train accuracy is %f and the valid accuracy is %f"%(global_steps,cost,accuracy,valid_accuracy))
        global_steps+=1

    return global_steps





def train_step():

    print("loading the dataset...")
    config = Config()
    global_steps=1
    stop=0
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-1 * FLAGS.initial, 1 * FLAGS.initial)
    for i in config.num_epoch:
        f_source=open(FLAGS.source_dir) #获取源文件的文件指针
        f_target=open(FLAGS.target_dir)

        while stop==0:#当整个文件还没结束....
            train_source_set, mask_train_source_set, length_array_eachdoc_source, max_source_sen_num, max_source_word_num, batch_size,f_source,stop=data_helper_source.load_data(f_source,config.batch_size)
            if stop==1: #为了防止特殊情况，最后一个batch不进行计算 同时又可以保证我们所有的batch_size都是一样的
                f_source.close()
                f_target.close()
                break

            config.max_source_sen_num=max_source_sen_num
            config.max_source_word_num=max_source_word_num





    with tf.Graph().as_default(), tf.Session() as session:

        with tf.variable_scope("model",reuse=None,initializer=initializer):
            model = RNN_Model(config=config,is_training=True)

        with tf.variable_scope("model",reuse=True,initializer=initializer):
            valid_model = RNN_Model(config=eval_config,is_training=False)
            test_model = RNN_Model(config=eval_config,is_training=False)

        #add summary
        # train_summary_op = tf.merge_summary([model.loss_summary,model.accuracy])
        train_summary_dir = os.path.join(config.out_dir,"summaries","train")
        train_summary_writer =  tf.train.SummaryWriter(train_summary_dir,session.graph)

        # dev_summary_op = tf.merge_summary([valid_model.loss_summary,valid_model.accuracy])
        dev_summary_dir = os.path.join(eval_config.out_dir,"summaries","dev")
        dev_summary_writer =  tf.train.SummaryWriter(dev_summary_dir,session.graph)

        #add checkpoint
        checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())


        tf.initialize_all_variables().run()
        global_steps=1
        begin_time=int(time.time())

        for i in range(config.num_epoch):
            print("the %d epoch training..."%(i+1))
            lr_decay = config.lr_decay ** max(i-config.max_decay_epoch,0.0)
            model.assign_new_lr(session,config.lr*lr_decay)
            global_steps=run_epoch(model,session,train_data,global_steps,valid_model,valid_data,train_summary_writer,dev_summary_writer)

            if i% config.checkpoint_every==0:
                path = saver.save(session,checkpoint_prefix,global_steps)
                print("Saved model chechpoint to{}\n".format(path))

        print("the train is finished")
        end_time=int(time.time())
        print("training takes %d seconds already\n"%(end_time-begin_time))
        test_accuracy=evaluate(test_model,session,test_data)
        print("the test data accuracy is %f"%test_accuracy)
        print("program end!")



def main(_):
    train_step()


if __name__ == "__main__":
    tf.app.run()






