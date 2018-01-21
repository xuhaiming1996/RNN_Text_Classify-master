import data_helper_source
import  data_helper_target
# file_source="##源文件##"
# file_target="###目标文件###"
from  Hier_lstm_att_model_update import RNN_Model
import tensorflow as tf
import numpy as np
import os
import time
import datetime


flags =tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('lr',0.1,'the learning rate')
flags.DEFINE_integer('batch_size',16,'the batch_size of the training procedure')
flags.DEFINE_integer('vocabulary_size',25003,'vocabulary_size')              #25001是句子的结束符号 25002是文章的结束符号 所有的单词是1-25000 0不用
flags.DEFINE_integer('emdedding_dim',1000,'embedding dim')
flags.DEFINE_integer('hidden_neural_size',1000,'LSTM hidden neural size')
flags.DEFINE_integer('hidden_layer_num',4,'LSTM hidden layer num')
flags.DEFINE_float('initial',0.08,'init initial')    #这个初始化参数的范围1---
flags.DEFINE_integer('num_epoch',8,'num epoch')
flags.DEFINE_integer('max_decay_epoch',30,'num epoch')
flags.DEFINE_integer('max_grad_norm',5,'max_grad_norm')

flags.DEFINE_string('out_dir',os.path.abspath(os.path.join(os.path.curdir,"runs")),'output directory')

flags.DEFINE_integer('check_point_every',1,'checkpoint every num epoch ')  #每几次保存数据

flags.DEFINE_integer('max_source_sen_num',-1,'源文件最大的句子数')
flags.DEFINE_integer('max_source_word_num',-1,'源文件最长的一句话包含的单词数')

flags.DEFINE_integer('max_target_sen_num' ,-1 , '源文件最大的句子数')
flags.DEFINE_integer('max_target_word_num' ,-1 , '源文件最长的一句话包含的单词数')

flags.DEFINE_string('source_dir','data/train_source','path of train_source')
flags.DEFINE_string('target_dir','data/train_target','path of train_target')
flags.DEFINE_string('isInitializer',1,'用于判断是否进行初始化，1为进行 0 为不进行')

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
    lr = FLAGS.lr
    max_source_sen_num=FLAGS.max_source_sen_num
    max_source_word_num=FLAGS.max_source_word_num
    max_target_sen_num = FLAGS.max_target_sen_num
    max_target_word_num = FLAGS.max_target_word_num
    batch_size = FLAGS.batch_size
    isInitializer = FLAGS.isInitializer



def run_batch(session,feed_dict,fetches,global_steps):
    cost,_ = session.run(fetches, feed_dict)
    return cost


def train():
    print("loading the dataset...")
    config = Config()
    stop_source = 0
    stop_target = 0
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-1 * FLAGS.initial, 1 * FLAGS.initial)
        with tf.variable_scope("model",reuse=None,initializer=initializer):
            model = RNN_Model(config=config,is_training=True,session=session)
        # add checkpoint
        print("######################################")
        checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        global_steps = 1
        session.run(tf.global_variables_initializer())
        for i in range(config.num_epoch):
            f_source=open(FLAGS.source_dir,'r',encoding="UTF-8") #获取源文件的文件指针
            f_target=open(FLAGS.target_dir,'r',encoding="UTF-8") #获得目标文件的指针
            num_batch=0 #用于记录这次迭代中的batch的个数
            while stop_source==0 and stop_target==0:#当整个文件还没结束....
                train_source_set, mask_train_source_set, length_array_eachdoc_source, max_source_sen_num, max_source_word_num, batch_size_source, f_source, stop_source ,sen_mask_source = data_helper_source.load_data(f_source, config.batch_size)
                train_target_set, mask_train_target_set, length_array_eachdoc_target, max_target_sen_num, max_target_word_num, batch_size_target, f_target, stop_target ,sen_mask_target ,mask_train_target_set_float= data_helper_target.load_data(f_target, config.batch_size,config.vocabulary_size)
                if stop_source == 1 or stop_target == 1: #为了防止特殊情况，最后一个batch不进行计算 同时又可以保证我们所有的batch_size都是一样的
                    f_source.close()
                    f_target.close()
                    break
                config.max_source_sen_num=max_source_sen_num
                config.max_source_word_num=max_source_word_num
                config.max_target_sen_num = max_source_sen_num
                config.max_target_word_num = max_source_word_num
                if batch_size_source == batch_size_target:
                    #数据读取正常
                    print("the %d epoch training..." % (i + 1))
                    num_batch += 1
                    model.assign_new_lr(session, config.lr)

                    model.assign_new_max_source_word_num(session,max_source_word_num)
                    model.assign_new_max_source_sen_num(session,max_source_sen_num)

                    model.assign_new_max_target_word_num(session,max_target_word_num)
                    model.assign_new_max_target_sen_num(session,max_target_sen_num)
                    feed_dict = {}
                    feed_dict[model.sen_mask] = sen_mask_source

                    feed_dict[model.train_source_set] = train_source_set
                    feed_dict[model.mask_train_source_set] = mask_train_source_set

                    feed_dict[model.train_target_set] = train_target_set
                    feed_dict[model.mask_train_target_set] = mask_train_target_set
                    feed_dict[model.mask_train_target_set_float] = mask_train_target_set_float

                    fetches = [model.cost, model.train_op]
                    cost = run_batch(session,feed_dict,fetches,global_steps)
                    print("cost_debug:",cost)
                    if num_batch % 1000 == 0:#在每次迭代中没1000个batch保存一次参数
                        path = saver.save(session, checkpoint_prefix, global_steps)
                        print("num_bath_%i of %i ecpo, Saved model chechpoint to %s\n" % (num_batch,global_steps,path))
                        print("num_bath_%i of %i ecpo, train cost is: %f" % (num_batch,global_steps,cost))

                else:
                    print("这次读取的数据异常，出现batch_size_source==batch_size_target")

            if i % config.checkpoint_every == 0:#数据遍历结束后 保存数组
                path = saver.save(session, checkpoint_prefix, global_steps)
                print("Saved model chechpoint to{}\n".format(path))
            global_steps+=1

        print("program end!")

def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()