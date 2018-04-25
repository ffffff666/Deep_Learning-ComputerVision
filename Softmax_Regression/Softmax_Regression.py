#!/usr/bin/env python3
# coding=utf-8
'''

    Softmax Regression手写体识别
做手写体识别，利用TF的MNIST数据集验证
本次Softmax不含隐层，只有两层结构
主要是为验证Softmax，并尝试TF基本语法

'''

_author_ = 'zixuwang'
_datetime_ = '2018-1-26'

'''
导入手写体识别的训练集、验证集、测试集
其中训练集55000、验证集5000、测试集10000
图像大小28*28＝784，需要变成一维向量
one-hote=True表示将lable变为［0，0，...1，0，0］
'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# 验证数据集的导入是否正确
# print('训练集：',mnist.train.images.shape,mnist.train.images.shape)
# print('验证集：',mnist.validation.images.shape,mnist.validation.images.shape)
# print('测试集：',mnist.test.images.shape,mnist.test.images.shape)


'''
***********************************************************************************************************
***********************************************************************************************************
                                                Softmax
***********************************************************************************************************
***********************************************************************************************************

'''

import tensorflow as tf

'''
Step 1:Inputdata
首先导入数据
placeholder是代表为输入数据占空
暂时不填入信息，调用Run函数的时候补全输入
要输入的数据是经一维变换后的784维行向量（实际上有不确定的样本－－None，所以是n*784维）
以及
实际输出的标签y_(n*10维)
'''
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])



'''
Step 2:FeedForward
定义算法公式，就是神经网络前向传播的途径
TF会自动帮忙计算反向传播的梯度公式
这里需要计算的是：y = Softmax（W*X + b）
Variable是将参数保留下来，便于更新
而不是同其他tensor一样一次流过
在此w和b全0初始化
W 维度 784*10
b 维度 10
'''
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros(10))

y = tf.nn.softmax(tf.matmul(x,W) + b)



'''
Step 3:Loss
计算损失，构建损失函数
本次使用cross-entropy Loss

==> loss = - Sum(y_ * log(y))

注意
reduce_sum、reduce_mean分别是求和、求平均
reduction_indices ＝ 1（或［1，0］）的意思是，将得到的结果按照行压缩求和、求平均
reduction_indices ＝ 0（或［0，1］）的意思是，将得到的结果按照列压缩求和、求平均
如果没有这个参数就意味着把矩阵弄成一个数字
具体见下图：
https://pic3.zhimg.com/v2-c92ac5c3a50e4bd3d60e29c2ddc4c5e9_r.jpg
'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))



'''
Step 4:Optimizer
选择优化器，并指定优化器优化loss
主要是用梯度下降法、随机梯度下降法、批处理梯度下降法
需要设置学习率（a=0.5）
TF会自动进行BP算法梯度更新
'''
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)





'''
Step 4:Training
开始训练，使用批处理梯度下降、每次选一个mini_batch，并feed给placeholder
(总共1000轮，每个batch包含100样本)
当然在一开始的时候需要调用TF全局参数初始化器
InteractiveSession是将这个session设置为默认session
'''
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

print('======> Start training:')
import time
start_time = time.time()
for i in range(10000):
    batch_x,batch_y = mnist.train.next_batch(100)
    train_step.run({x:batch_x,y_:batch_y})

time_used = time.time() - start_time
print('======> Training End!!!!')
print('训练耗时：%.1fs'%time_used)

#在此训练完成,将训练完成的参数保存到文件
saver = tf.train.Saver()  # 默认保存所有参数，本例中为：W和b
model_path = "/Users/apple/Documents/TensorFlow/Model/model_Softmax_Regression/model_Softmax_Regression.ckpt"
save_path = saver.save(sess, model_path)

'''
读取模型操作：
saver = tf.train.Saver()
saver.restore(sess, "/Users/apple/Documents/TensorFlow/Model/model_Softmax_Regression.ckpt")
result = sess.run(y, feed_dict={x: data})
'''




'''
Step 6:Correct_prediction from testSet
对模型在测试集上进行准确率的验证
tf.argmax(y,1)就是找到各个预测概率中最大的那个
tf.argmax(y_,1)就是找到真实的标记
tf.cast是把bool变为float32，然后才能求平均
'''
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print('测试集上面的精度为：',accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))