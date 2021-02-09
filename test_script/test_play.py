import numpy as np
import tensorflow as tf

#设置两个乘数，用占位符表示
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
#设置乘积
output = tf.multiply(input1, input2)
output2 = tf.add(input1, input2)
with tf.Session() as sess:
  #用feed_dict以字典的方式填充占位
 print(sess.run([output], feed_dict={input1:[8.],input2:[2.]}))
    # print(sess.run(w1))
    # print(sess.run(w2))