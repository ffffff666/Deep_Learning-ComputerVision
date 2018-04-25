#!/usr/bin/env python3
# coding=utf-8
'''

             ［VGGNet］
构建VGGNet-16卷积神经网络
随机生成图片进行测试
VGGNet包含13个卷积层、4个池化层（无LRN层）和两个全连接层，最后用一个Softmax全连接层做输出：
VGGNet中所有的卷积尺寸为3x3，步长为1
        所有的池化尺寸为2x2，步长为2（每次缩小一半）
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
   Layer1:  conv1-1  ［3，3，？，64］
            conv1-2  ［3，3，64，64］
            pool1:   最大池化  2x2 2x2

   Layer2:  conv2-1  ［3，3，64，128］
            conv2-2  ［3，3，64，128］
            pool2:   最大池化  2x2 2x2

   Layer3:  conv3-1  ［3，3，128，256］
            conv3-2  ［3，3，128，256］
            conv3-3  ［3，3，128，256］
            pool3:   最大池化  2x2 2x2

   Layer4:  conv4-1  ［3，3，256，512］
            conv4-2  ［3，3，256，512］
            conv4-3  ［3，3，256，512］
            pool4:   最大池化  2x2 2x2

   Layer5:  conv5-1  ［3，3，512，512］
            conv5-2  ［3，3，512，512］
            conv5-3  ［3，3，512，512］
            pool5:   最大池化  2x2 2x2


            local1:  全连接+ReLU 4096
            local2:  全连接+ReLU 4096
            logits： 预测输出    1000
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊

使用VGGNet做图像分类

'''

_author_ = 'zixuwang'
_datetime_ = '2018-1-31'


import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
import math
import time




'''
***********************************************************************************************************
***********************************************************************************************************
                                                VGGNet
***********************************************************************************************************
***********************************************************************************************************

'''
parameters = []

# 权重初始化方法是截断的正态分布
def weight_init(shape, stddev, name = 'weights'):
    weight = tf.Variable(tf.truncated_normal(shape,dtype=tf.float32,stddev=stddev),name=name)
    return weight
# bias初始化方法是零初始化
def bias_init(shape, const = 0.0, name = 'biases'):
    bias = tf.Variable(tf.constant(const,dtype=tf.float32,shape=shape),trainable=True,name=name)
    return bias

#显示每一层网络（conv or pooling）的名称和尺寸
def print_layer_detail(layer):
    print(layer.op.name,'  ',layer.get_shape().as_list())

# 由于需要频繁构造卷积层、池化层、以及全连接层，在此构造函数接口

#卷积层［输入tensor，输出通道数，名字，p,卷积核宽、高，步长］
def conv_layer(input,num_output,name,parameters,kernel_width=3,kernel_high=3,step_width=1,step_high=1):
    num_input = input.get_shape()[-1].value #得到输入通道数

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'weights',
                                 shape=[kernel_high,kernel_width,num_input,num_output],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())#使用Xavier初始化
        bias = bias_init([num_output])

        conv = tf.nn.conv2d(input,kernel,strides=[1,step_high,step_width,1],padding='SAME')

        output = tf.nn.relu(tf.nn.bias_add(conv,bias,name=scope))
        parameters += [kernel,bias]
        return output

#池化层［输入tensor，名字，池化核宽、高，步长，］
def pooling_layer(input,name,kernel_high=2,kernel_width=2,step_high=2,step_width=2):
    return tf.nn.max_pool(input,
                          ksize=[1,kernel_high,kernel_width,1],
                          strides=[1,step_high,step_width,1],
                          padding='SAME',
                          name=name)

# 全连接层［］
def fullyconnect_layer(input,num_hidden_layer,name,parameters):
    num_input = input.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        weights = tf.get_variable(scope+'weights',
                                  shape=[num_input,num_hidden_layer],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())  # 使用Xavier初始化

        bias = bias_init(shape=[num_hidden_layer],const=0.1)       #初始化0.1防止dead neuron

        # output = tf.nn.relu(tf.nn.bias_add(tf.matmul(input,weights),bias),name=scope)
        output = tf.nn.relu_layer(input,weights,bias,name=scope)
        parameters += [weights,bias]
        return output



'''
Step 2:FeedForward
定义算法公式，就是神经网络前向传播的途径
TF会自动帮忙计算反向传播的梯度公式
首先先进行一些函数的定义，方便后面使用
'''
def VGGNet(images,keep_prob):
    # 第一层:
    # Layer1:       conv1-1  ［3，3，？，64］
    #               conv1-2  ［3，3，64，64］
    #               pool1: 最大池化 2x2 2x2
    conv1_1 = conv_layer(images,num_output=64,name='conv1_1',parameters=parameters)
    conv1_2 = conv_layer(conv1_1,num_output=64,name='conv1_2',parameters=parameters)
    pool1 = pooling_layer(conv1_2,name='pool1')
    print_layer_detail(conv1_1)
    print_layer_detail(conv1_2)
    print_layer_detail(pool1)

    # 第二层:
    # Layer2:       conv2-1  ［3，3，64，128］
    #               conv2-2  ［3，3，128，128］
    #               pool2: 最大池化 2x2 2x2
    conv2_1 = conv_layer(pool1,num_output=128,name='conv2_1',parameters=parameters)
    conv2_2 = conv_layer(conv2_1,num_output=128,name='conv2_2',parameters=parameters)
    pool2 = pooling_layer(conv2_2,name='pool2')
    print_layer_detail(conv2_1)
    print_layer_detail(conv2_2)
    print_layer_detail(pool2)

    # 第三层:
    # Layer3:       conv3-1  ［3，3，128，256］
    #               conv3-2  ［3，3，256，256］
    #               conv3-3  ［3，3，256，256］
    #               pool3: 最大池化 2x2 2x2
    conv3_1 = conv_layer(pool2,num_output=256,name='conv3_1',parameters=parameters)
    conv3_2 = conv_layer(conv3_1,num_output=256,name='conv3_2',parameters=parameters)
    conv3_3 = conv_layer(conv3_2,num_output=256,name='conv3_3',parameters=parameters)
    pool3 = pooling_layer(conv3_3,name='pool3')
    print_layer_detail(conv3_1)
    print_layer_detail(conv3_2)
    print_layer_detail(conv3_3)
    print_layer_detail(pool3)


    # 第四层:
    # Layer4:       conv4-1  ［3，3，256，512］
    #               conv4-2  ［3，3，512，512］
    #               conv4-3  ［3，3，512，512］
    #               pool4: 最大池化 2x2 2x2
    conv4_1 = conv_layer(pool3,num_output=512,name='conv4_1',parameters=parameters)
    conv4_2 = conv_layer(conv4_1,num_output=512,name='conv4_2',parameters=parameters)
    conv4_3 = conv_layer(conv4_2,num_output=512,name='conv4_3',parameters=parameters)
    pool4 = pooling_layer(conv4_3,name='pool4')
    print_layer_detail(conv4_1)
    print_layer_detail(conv4_2)
    print_layer_detail(conv4_3)
    print_layer_detail(pool4)


    # 第五层:
    # Layer5:       conv5-1  ［3，3，512，512］
    #               conv5-2  ［3，3，512，512］
    #               conv5-3  ［3，3，512，512］
    #               pool5: 最大池化 2x2 2x2
    conv5_1 = conv_layer(pool4,num_output=512,name='conv5_1',parameters=parameters)
    conv5_2 = conv_layer(conv5_1,num_output=512,name='conv5_2',parameters=parameters)
    conv5_3 = conv_layer(conv5_2,num_output=512,name='conv5_3',parameters=parameters)
    pool5 = pooling_layer(conv5_3,name='pool5')
    print_layer_detail(conv5_1)
    print_layer_detail(conv5_2)
    print_layer_detail(conv5_3)
    print_layer_detail(pool5)


    # 第六层:  全连接层
    # Layer6:       local1:  ［?，4096］
    #               local2:  ［4096,4096］
    shape = pool5.get_shape()
    dim = shape[1].value * shape[2].value * shape[3].value      #dimension reduce
    reshape = tf.reshape(pool5,[-1,dim],name='reshape')

    local1 = fullyconnect_layer(reshape,num_hidden_layer=4096,name = 'local1',parameters=parameters)
    local1_drop = tf.nn.dropout(local1,keep_prob=keep_prob,name='local1_drop')

    local2 = fullyconnect_layer(local1_drop,num_hidden_layer=4096,name='local2',parameters=parameters)
    local2_drop = tf.nn.dropout(local2,keep_prob=keep_prob,name='local2_drop')

    print_layer_detail(local1)
    print_layer_detail(local2)


    # 最后一层输出层
    # 暂时不需要连接一个softmax，到计算损失的时候再用softmax
    # 因为不用softmax也可以比较出最大的那一个作为输出，得到分类结果
    # softmax只是用来计算loss
    # 不采用激活函数，不采用dropout
    logits = fullyconnect_layer(local2_drop,num_hidden_layer=1000,name='logits',parameters=parameters)
    softmax = tf.nn.softmax(logits)
    prediction = tf.argmax(softmax,1)


    return prediction,softmax,logits,parameters

# 本次测试用不到优化器、损失函数等



'''
定义计算时间消耗的函数
target为将要预测的目标
feed_dict主要用来传送dropout rate
'''
def time_TF_cost(session, target, feed_dict, info_string,total_batches = 30):
    num_steps_prepare = 10  #前10步在做初始化、catch命中等，不计算时间
    total_cost = 0.0    #总时间消耗
    total_cost_squared = 0.0    #总时间消耗的平方

    for i in range(num_steps_prepare + total_batches):
        start_time = time.time()
        session.run(target,feed_dict=feed_dict)
        time_used = time.time() - start_time
        if i >= num_steps_prepare:
            # if i % 10 == 0: #每10轮显示一次
            print(' step:  %d    time_used:  %.3f'%(i,time_used))
            total_cost += time_used
            total_cost_squared += time_used*time_used

    avg_cost = total_cost / total_batches
    stdd_cost = math.sqrt(total_cost_squared/total_batches - avg_cost*avg_cost)
    print('  ')
    print(' ')
    print('%s across %d steps.  Avg: %.3f  Stdd: %.3f'%(info_string,total_batches,avg_cost,stdd_cost))


batch_size = 32
keep_prob = tf.placeholder(tf.float32)  # dropout时的保留率

def run_all():
    with tf.Graph().as_default():
        batch_size = 32
        image_size = 224
        #生成随机图片
        images = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],dtype=tf.float32,stddev=1e-1))

        keep_prob = tf.placeholder(tf.float32)
        prediction,_,logits,parameters = VGGNet(images,keep_prob)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())


        print('begin')
        # time_TF_cost(sess,prediction,{keep_prob:1.0},'FeedForward')#正向传播计算耗时

        objective = tf.nn.l2_loss(logits)   #反向传播计算耗时（主要是求梯度）
        grad = tf.gradients(objective,parameters)
        time_TF_cost(sess,grad,{keep_prob:0.5},'BackPropagation')
        print('End')

run_all()