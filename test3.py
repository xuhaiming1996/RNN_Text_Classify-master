import numpy as np
import tensorflow as tf

with tf.Session() as sess:
    c = tf.ones([16])
    print(sess.run(c))



