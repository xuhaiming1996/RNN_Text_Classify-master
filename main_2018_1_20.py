import data_helper_source_2018_1_20
import  data_helper_target_2018_1_20
# file_source="##源文件##"
# file_target="###目标文件###"
from  Hier_lstm_att_model_2018_1_20 import RNN_Model
import tensorflow as tf
import numpy as np
import os
import time
import datetime
try:
    from itertools import izip
except ImportError:
    izip=zip
    


flags =tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('lr',0.1,'the learning rate')
flags.DEFINE_integer('batch_size',16,'the batch_size of the training procedure')
flags.DEFINE_integer('vocabulary_size',25003,'vocabulary_size')              #25001是句子的结束符号 25002是文章的结束符号 所有的单词是1-25000 0不用
flags.DEFINE_integer('emdedding_dim',1000,'embedding dim')
flags.DEFINE_integer('hidden_neural_size',1000,'LSTM hidden neural size')
flags.DEFINE_integer('hidden_layer_num',4,'LSTM hidden layer num')
flags.DEFINE_float('initial',0.08,'init initial')    #这个初始化参数的范围1---
flags.DEFINE_integer('num_epoch',3,'num epoch')
flags.DEFINE_integer('max_decay_epoch',30,'num epoch')
flags.DEFINE_integer('max_grad_norm',5,'max_grad_norm')

flags.DEFINE_string('out_dir',os.path.abspath(os.path.join(os.path.curdir,"runs")),'output directory')

flags.DEFINE_integer('check_point_every',1,'checkpoint every num epoch ')  #每几次保存数据

flags.DEFINE_integer('max_source_sen_num',-1,'源文件最大的句子数')
flags.DEFINE_integer('max_source_word_num',-1,'源文件最长的一句话包含的单词数')

flags.DEFINE_integer('max_target_sen_num' ,-1 , '源文件最大的句子数')
flags.DEFINE_integer('max_target_word_num' ,-1 , '源文件最长的一句话包含的单词数')

flags.DEFINE_string('source_dir','data/debug_source','path of train_source')
flags.DEFINE_string('target_dir','data/debug_target','path of train_target')
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




def run_epoch(model,session,data_source,data_target):
    num_batch=0
    for source,target in zip(  data_helper_source_2018_1_20.batch_iter(data_source, FLAGS.batch_size), data_helper_target_2018_1_20.batch_iter(data_target,FLAGS.batch_size)):
        begin_time = int(time.time())
        sen_mask, train_source_set, mask_train_source_set,_=source
        train_target_set, mask_train_target_set, mask_train_target_set_float=target
        num_batch+=1
        feed_dict = {}
        feed_dict[model.sen_mask] = sen_mask
        feed_dict[model.train_source_set] = train_source_set
        feed_dict[model.mask_train_source_set] = mask_train_source_set

        feed_dict[model.train_target_set] = train_target_set
        feed_dict[model.mask_train_target_set] = mask_train_target_set
        feed_dict[model.mask_train_target_set_float] = mask_train_target_set_float
        feed_dict[model.lr] = 0.1
        fetches = [model.cost, model.train_op]
        cost, _ = session.run(fetches, feed_dict)
        print("cost_debug:", cost)
        end_time=int(time.time())
        print("training a batch %d seconds already"%(end_time-begin_time))
        # print("outputs",outputs)
        if num_batch % 1000 == 0:  # 在每次迭代中没1000个batch保存一次参数
            # path = saver.save(session, checkpoint_prefix, global_steps)
            # print("num_bath_%i of %i ecpo, Saved model chechpoint to %s\n" % (num_batch, global_steps, path))
            print("num_bath_%i, train cost is: %f" % (num_batch, cost))





def train():
    print("loading the dataset...")
    config = Config()
    # data_source的元素  train_source_set, mask_train_source_set, length_array_eachdoc_source, max_source_sen_num, max_source_word_num, num_doc, sen_mask )
    data_source = data_helper_source_2018_1_20.load_data(FLAGS.source_dir)  # 获取源文件的相关数据
    # data_target的元素  train_target_set, mask_train_target_set, length_array_eachdoc_target, max_target_sen_num, max_target_word_num, num_doc, sen_mask, mask_train_target_set_float
    data_target = data_helper_target_2018_1_20.load_data(FLAGS.target_dir,FLAGS.vocabulary_size)
    config.max_source_sen_num = data_source[3]
    config.max_source_word_num = data_source[4]
    config.max_target_sen_num =  data_target[3]
    config.max_target_word_num = data_target[4]
    print( config.max_source_sen_num,config.max_source_word_num,config.max_target_sen_num,config.max_target_word_num)
    num_doc=data_source[5]

    print("文章的个数",num_doc)

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-1 * FLAGS.initial, 1 * FLAGS.initial)
        with tf.variable_scope("model",reuse=None ,initializer=initializer):
            model = RNN_Model(config=config)
        # add checkpoint
        print("######################################")
        checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        saver = tf.train.Saver(tf.all_variables())
        session.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(session, ckpt.model_checkpoint_path)

        start = global_step.eval()

        print("start from:", start)

        for i in range(start,config.num_epoch):
            print("the %d epoch training..."%(i+1))
            run_epoch(model,session,data_source,data_target)
            global_step.assign(i).eval()
            path = saver.save(session,checkpoint_prefix,global_step)
            print("Saved model chechpoint to{}\n".format(path))

        print("the train is finished")
        print("program end!")

def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()