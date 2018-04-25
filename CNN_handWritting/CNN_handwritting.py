#!/usr/bin/env python3
# coding=utf-8
'''

        ［CNN手写体识别］
做手写体识别，利用TF的MNIST数据集验证
本次CNN包含两个卷积层、两个池化层和一个全连接层
主要是为测试CNN

'''

_author_ = 'zixuwang'
_datetime_ = '2018-1-26'

'''
导入手写体识别的训练集、验证集、测试集
其中训练集55000、验证集5000、测试集10000
图像大小28*28，由于使用卷积操作，不需要将图像进行一维扁平化处理
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
                                                CNN
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
x_image = tf.reshape(x,[-1,28,28,1])    # -1代表自动计算，这里是不知道输入图片总共有多少张；28*28是转变后二维图片的大小；1是灰度图片（占一个维度）
y_ = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32) #dropout时的保留率


'''
Step 2:FeedForward
定义算法公式，就是神经网络前向传播的途径
TF会自动帮忙计算反向传播的梯度公式
首先先进行一些函数的定义，方便后面使用
'''
# 由于卷积层使用了ReLU激活函数，进行初始化时需要随机初始化
# 这里用截断的正态分布sigma＝0.1初始化W，用常数0.1作为噪声初始化b
def weight_init(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_init(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# 然后定义卷积处理操作，tf.conv2d输入参数列表［input，filter，strides,padding......］：
#       input 是指输入的数据
#       filter是卷积核的参数：
#［5，5，1，32］代表5*5的滤波器，通道为1（灰度图为1，RGB为3），卷积核数量为32（提取32类特征，即32个重叠的卷积层）
#       strides是卷积模版移动的步长：［1，1，1，1］代表一个一个的移动，不漏下任何的点
#       padding是指边缘处理方式，SAME指的是自动填充使得输入输出尺寸相同

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# 然后定义池化层的处理操作,tf.max_pool（最大值池化）输入参数列表［input，ksize，strides，padding］
# 这里是用的是2*2最大值池化，将一个2*2的像素块降为1*1的像素块，取灰度最大的一个像素为代表
#       input是输入的数据
#       ksize是池化对应的映射操作，［1，2，2，1］是将一个2*2的像素块降为1*1的像素块
#       strides是池化模板移动的步长，［1，2，2，1］是以两个单位移动（2*2的像素块移动）
#            通过ksize和strides可以把原图片整体缩小4倍（行列各除以2）

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 第一个卷积层
# 卷积核：［5，5，1，32］
W_conv1 = weight_init([5,5,1,32])
b_conv1 = bias_init([32])   #第一个卷积层一共有32张不同的卷积（32个卷积核），需要32个偏置单元
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1) #数据经卷积层操作后做ReLU激活
h_pool1 = max_pool_2x2(h_conv1) #池化操作：n*28*28*32---->n*14*14*32


# 第二个卷积层
# 卷积核：［5，5，32，64］   因为从第一个卷积层出来的共有32个卷积核结果，第二层卷积核有64个
W_conv2 = weight_init([5,5,32,64])
b_conv2 = bias_init([64])   #第二个卷积层一共有64张不同的卷积（64个卷积核），需要64个偏置单元
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2) #数据经卷积层操作后做ReLU激活
h_pool2 = max_pool_2x2(h_conv2) #池化操作：n*14*14*64---->n*7*7*64

# 全连接层
# 使用Dropout
# 但需要把7*7*64的卷积输出变为一维向量，全连接层隐层结点数为1024个
W_fc1 = weight_init([7*7*64,1024])
b_fc1 = bias_init([1024])
h_pool2_input = tf.reshape(h_pool2,[-1,7*7*64]) #这里这个－1表示的是不知道有多少个输入样本，但特征还是一维的，即7*7*64
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_input,W_fc1) + b_fc1)  #ReLU
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob=keep_prob)   #dropout

# Softmax层
# 最后需要连接一个softmax层做输出
# 不采用激活函数，不采用dropout
W_fc2 = weight_init([1024,10])
b_fc2 = bias_init([10])
y = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)




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
选择Adam优化器，设置学习率（a=1e-4）
TF会自动进行BP算法梯度更新
'''
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)





'''
Step 4:Training
开始训练，使用批处理梯度下降、每次选一个mini_batch，并feed给placeholder
(总共20000轮，每个batch包含50样本)
当然在一开始的时候需要调用TF全局参数初始化器
InteractiveSession是将这个session设置为默认session
'''
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


print('======> Start training:')

for i in range(20000):
    batch = mnist.train.next_batch(50)

    #每100轮显示一下当前在批处理训练集上的精度
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print('step: %d     training accuracy: %g'%(i,train_accuracy))

    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

print('======> Training End!!!!')

#在此训练完成,将训练完成的参数保存到文件
# saver = tf.train.Saver()  # 默认保存所有参数，本例中为：W和b
# model_path = "/Users/apple/Documents/TensorFlow/Model/model_CNN_handwritting/model_CNN_handwritting.ckpt"
# save_path = saver.save(sess, model_path)

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
# correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print('测试集上面的精度为：',accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})) #keep_prob=1    (dropout_rate = 0)