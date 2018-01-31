import tensorflow as tf
import numpy as np
c=[]
a=tf.Variable(tf.random_uniform(shape=[2,3]))
c.append(a)
b=tf.Variable(tf.random_uniform(shape=[2,3]))
c.append(b)
d=tf.reshape(c,[-1,3])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(d))



