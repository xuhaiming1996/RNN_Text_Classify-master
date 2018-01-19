import numpy as np
import tensorflow as tf
a = [
               [[1, 2, 3, 4],
                [4, 5, 6, 7],
                [7, 8, 9, 10]],

               [[1, 2, 3, 4],
                [4, 5, 6, 7],
                [7, 8, 9, 10]],

               [[1, 2, 3, 4],
                [4, 5, 6, 7],
                [7, 8, 9, 10]]
            ]

b =[ [[2, 2, 2, 2],
     [2, 3, 2, 2],
     [2, 2, 2, 2]]]


with tf.Session() as sess:
    c=tf.reshape(a,[1,-1,4])
    d=tf.reshape(b,[1,-1,4])
    f=tf.concat(0,[d,c])
    print(tf.shape(c))
    print(c)
    print(sess.run(c))
    print(sess.run(d))
    print("ssssssssssss")
    print(sess.run(f))


