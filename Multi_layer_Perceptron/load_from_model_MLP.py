#!/usr/bin/env python3
# coding=utf-8
'''

读取模型示例代码
MLP

'''

_author_ = 'zixuwang'
_datetime_ = '2018-1-26'





import tensorflow as tf
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 读取数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


# 输入变量
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)



'''
***********************************************************************************************************
***********************************************************************************************************
                                                MLP
***********************************************************************************************************
***********************************************************************************************************

'''
# 需要读入的参数
# 在这里需要先把神经网络搭建一遍
W1 = tf.Variable(tf.zeros([784,300]))
b1 = tf.Variable(tf.zeros(300))
h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x,W1) + b1),keep_prob)

W2 = tf.Variable(tf.zeros([300,10]))
b2 = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(h1,W2) + b2)



'''
***********************************************************************************************************
***********************************************************************************************************
                                                MLP测试
***********************************************************************************************************
***********************************************************************************************************

'''



# 读取模型参数
saver = tf.train.Saver()
model_path = "/Users/apple/Documents/TensorFlow/Model/model_MLP/model_MLP.ckpt"
saver.restore(sess, model_path)
print('MLP_model loaded')

# 测试集精度验证
# result = sess.run(y, feed_dict=({x:mnist.test.images,keep_prob:1.0}))
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print('测试集上面的精度为：',accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))