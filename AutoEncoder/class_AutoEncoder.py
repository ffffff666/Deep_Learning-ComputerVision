#!/usr/bin/env python3
# coding=utf-8
'''

        ［自编码器］
从输入层到隐层到输出层共三层
期望：输入＝输出
隐层做特征的提取和抽象
争取用高阶特征重构样本，但不是完全照搬
所以隐层结点小于输入输出结点，且加入少量噪声，期望去除这些噪声

        x－－h－－y
          E：y＝x'

自编码器主要用在对网络权重的初始化，用来防止梯度弥散
但在现在的CNN、RNN、MLP中已经不再使用

'''

_author_ = 'zixuwang'
_datetime_ = '2018-1-27'



import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
import time


'''
***********************************************************************************************************
***********************************************************************************************************
                                                  AutoEncoder
***********************************************************************************************************
***********************************************************************************************************

'''


'''
        自编码器定义：带加性高斯噪声
    ＊＊＊＊＊        AGN        ＊＊＊＊＊
'''
class AdditiveGaussianNoise_AutoEncoder(object):

    #神经网络初始化函数定义，输入列表：［输入结点数，隐层结点数，激活函数，优化器，高斯噪声系数］
    def __init__(self,
                 num_input,
                 num_hidden,
                 transfer_function = tf.nn.softplus,
                 optimizer = tf.train.AdamOptimizer(),
                 scale = 0.1):

        # 神经网络的一些初始化操作
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.transfer_function = transfer_function
        self.scale = scale

        self.nn_weights =  self._initialize_weights()      #神经网络初始化权重
        self.x = tf.placeholder(tf.float32,shape=[None,self.num_input]) #神经网络原始输入
        self.input = self.x + self.scale * tf.random_uniform((num_input,))   #加入高斯噪声后的输入

        # 隐层结点经激活函数处理后的输出
        self.hidden = self.transfer_function(tf.matmul(self.input,self.nn_weights['w1']) + self.nn_weights['b1'])

        # 重构的输出，期望它等于输入
        self.reconstruction = tf.add(tf.matmul(self.hidden,self.nn_weights['w2']),self.nn_weights['b2'])


        # 定义损失函数
        # 这里使用平方损失函数： loss = 1 / 2*m  *  sum[(y-x)*(y-x)]
        self.loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x) , 2.0))

        # 定义优化器，最小化损失
        self.optimizer = optimizer.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    #网络权重初始化函数
    def _initialize_weights(self):
        nn_weights = dict()
        nn_weights['w1'] = tf.Variable(self.xavier_init( self.num_input,self.num_hidden) ) #w1权重初始化使用xavier初始化
        nn_weights['b1'] = tf.Variable(tf.zeros([self.num_hidden],dtype=tf.float32))
        nn_weights['w2'] = tf.Variable(tf.zeros([self.num_hidden,self.num_input],dtype=tf.float32))
        nn_weights['b2'] = tf.Variable(tf.zeros([self.num_input],dtype=tf.float32))
        return  nn_weights


    # 运行BP算法进行梯度下降，更新权重
    def run_Gradient_Descent(self,x):
        loss,opt = self.sess.run([self.loss,self.optimizer],
                                 feed_dict={self.x:x})
        return loss


    # 计算当前网络的损失
    def total_loss(self,x):
        return self.sess.run(self.loss,feed_dict={self.x:x})


    '''
    下面是其他一些保留的函数
    '''


    # 以下两个操作将一次feed forward拆分为两个子过程
    # 返回隐含层的输出结果，隐含层的主要功能是学习出数据中的高阶特征
    def hidden_output(self,x):
        return self.sess.run(self.hidden,feed_dict={self.x:x})

    # 从隐含层获取输出，传入重建层复原为原始数据
    def rebuild_input(self,hidden):
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})

    # 整体运行一次，得到复原后的数据
    def reconstruct(self,x):
        return self.sess.run(self.reconstruction,feed_dict={self.x:x})

    # 得到隐含层权重w1
    def get_hidden_weights(self):
        return self.sess.run(self.nn_weights['w1'])

    # 得到隐含层偏置单元b1
    def get_hidden_bias(self):
        return self.sess.run(self.nn_weights['b1'])

    # 在此训练完成,将训练完成的参数保存到文件
    def save_model(self):
        saver = tf.train.Saver()  # 默认保存所有参数，本例中为：W和b
        model_path = "/Users/apple/Documents/TensorFlow/Model/model_AutoEncoder/model_AutoEncoder.ckpt"
        save_path = saver.save(self.sess, model_path)

    #两个很重要的函数
    '''
    参数初始化
    采用xavier initialization初始化参数
    Xavier可以根据某一层网络的输入、输出结点数量自动调整最合适的分布
    DL中权重初始化很重要，太小更新缓慢，太大会导致梯度弥散
    因此需要初始化一个适中的权重
    创建的分布是：（－sqrt[6/(num_in+num_out)],+sqrt[6/(num_in+num_out)]）的均匀分布
    这个分布  均值为 0，  方差为 2/(num_in+num_out)
    '''

    def xavier_init(self,num_in, num_out, constant=1):
        low = - constant * np.sqrt(6.0 / (num_in + num_out))
        high = constant * np.sqrt(6.0 / (num_in + num_out))
        return tf.random_uniform((num_in, num_out), minval=low, maxval=high, dtype=tf.float32)


    '''
    数据预处理：将数据变为 零均值单位方差
    x = (x - avg) / sigma*sigma
    特别注意，在训练集验证集测试集上都要做预处理
    保证统一分布
    '''

    def pre_standard(self,x_train, x_test):
        preprocessor = prep.StandardScaler().fit(x_train)  # 在训练数据集上进行数据标准化
        # 返回标准化时使用的avg和sigma

        x_train = preprocessor.transform(x_train)  # 使用保存的avg和sigma标准化训练集和测试集（必须统一！！）
        x_test = preprocessor.transform(x_test)

        return x_train, x_test
