#!/usr/bin/env python3
# coding=utf-8
'''

             ［AlexNet］
构建AlexNet卷积神经网络
随机生成图片进行测试
AlexNet包含五个卷积层、三个池化层（两个LRN层）和两个全连接层，最后用一个Softmax全连接层做输出：

＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
    conv1：  卷积层+ReLU    5x5 4 96
    lan1:     LAN
    pool1:   最大池化       3x3 2x2
    conv2：  卷积层+ReLU    5x5 1 256
    lan2:     LAN
    pool2:   最大池化       3x3 2x2
    conv3：  卷积层+ReLU    3x3 1 384
    conv4：  卷积层+ReLU    3x3 1 384
    conv5：  卷积层+ReLU    3x3 1 256
    pool5:   最大池化       3x3 2x2
    local1:  全连接+ReLU
    local2:  全连接+ReLU
    logits： 预测输出
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊

使用AlexNet做图像分类

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
                                                AlexNet
***********************************************************************************************************
***********************************************************************************************************

'''
def AlexNet(images,keep_prob):
    parameters = []

    # 权重初始化方法是截断的正态分布
    def weight_init(shape, stddev=0.01, name = 'weights'):
        weight = tf.Variable(tf.truncated_normal(shape,dtype=tf.float32,stddev=stddev),name=name)
        return weight
    # bias初始化方法是零初始化
    def bias_init(shape, const = 0.0, name = 'biases'):
        bias = tf.Variable(tf.constant(const,dtype=tf.float32,shape=shape),trainable=True,name=name)
        return bias

    #显示每一层网络（conv or pooling）的名称和尺寸
    def print_layer_detail(layer):
        print(layer.op.name,'  ',layer.get_shape().as_list())


    '''
    Step 1:Inputdata
    首先导入数据
    placeholder是代表为输入数据占空
    暂时不填入信息，调用Run函数的时候补全输入
    在这里输入就不能是任意大小None了，必须是batch_size大小的数据
    数据的特征维度是24x24x3     即剪切后的24x24大小的二维图片，同时有RGB三个颜色通道
    '''




    '''
    Step 2:FeedForward
    定义算法公式，就是神经网络前向传播的途径
    TF会自动帮忙计算反向传播的梯度公式
    首先先进行一些函数的定义，方便后面使用
    '''
    # 第一个卷积层: 卷积－>LRN—>池化
    # 卷积核：［11，11，3，96］
    # 11*11的滤波器，通道为3（灰度图为1，RGB为3），卷积核数量为96（提取96类特征，即96个重叠的卷积层）
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_init(shape=[11,11,3,96],stddev=1e-1)  #权重初始化：标准差为0.1
        b_conv1 = bias_init(shape=[96])  #偏置单元全部初始化为0
        conv_kernel1 = tf.nn.conv2d(images,W_conv1,strides=[1,4,4,1],padding='SAME')  #卷积操作：4x4平移
        conv1 = tf.nn.relu(tf.nn.bias_add(conv_kernel1,b_conv1),name=scope)    #卷积后加偏置，ReLU激活
        parameters += [W_conv1,b_conv1]
    print_layer_detail(conv1)

    # lrn1 = tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='lrn1')    #LRN处理
    pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool1')
                        #3x3采样池化，2x2单位移动，增加数据丰富性,VALID是取样时不超出边框，不填充
    print_layer_detail(pool1)



    # 第二个卷积层: 卷积－>LRN—>池化
    # 卷积核：［5，5，96，256］
    # 5*5的滤波器，通道为96（上一个卷积输出96层卷积），卷积核数量为256（提取256类特征，即256个重叠的卷积层）
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_init(shape=[5,5,96,256],stddev=1e-1)     #权重初始化
        b_conv2 = bias_init(shape=[256])  #偏置单元全部初始化为0
        conv_kernel2 = tf.nn.conv2d(pool1,W_conv2,strides=[1,1,1,1],padding='SAME')   #卷积操作：一个不漏地平移
        conv2 = tf.nn.relu(tf.nn.bias_add(conv_kernel2,b_conv2),name=scope)    #卷积后加偏置，ReLU激活
        parameters += [W_conv2, b_conv2]

    print_layer_detail(conv2)

    # lrn2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75)    #LRN处理
    pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool2') #3x3采样池化，2x2单位移动
    print_layer_detail(pool2)


    # 第三个卷积层: 卷积
    # 卷积核：［3，3，256，384］
    # 3*3的滤波器，通道为256（上一个卷积输出256层卷积），卷积核数量为384（提取384类特征，即384个重叠的卷积层）
    with tf.name_scope('conv3') as scope:
        W_conv3 = weight_init(shape=[3,3,256,384],stddev=1e-1)
        b_conv3 = bias_init(shape=[384])
        conv_kernel3 = tf.nn.conv2d(pool2,W_conv3,strides=[1,1,1,1],padding='SAME')
        conv3 = tf.nn.relu(tf.nn.bias_add(conv_kernel3,b_conv3),name=scope)
        parameters += [W_conv3, b_conv3]

    print_layer_detail(conv3)

    # 第四个卷积层: 卷积
    # 卷积核：［3，3，384，384］
    # 3*3的滤波器，通道为384（上一个卷积输出384层卷积），卷积核数量为384（提取384类特征，即384个重叠的卷积层）
    with tf.name_scope('conv4') as scope:
        W_conv4 = weight_init(shape=[3,3,384,384],stddev=1e-1)
        b_conv4 = bias_init(shape=[384])
        conv_kernel4 = tf.nn.conv2d(conv3,W_conv4,strides=[1,1,1,1],padding='SAME')
        conv4 = tf.nn.relu(tf.nn.bias_add(conv_kernel4,b_conv4),name=scope)
        parameters += [W_conv4, b_conv4]

    print_layer_detail(conv4)


    # 第五个卷积层: 卷积-->池化
    # 卷积核：［3，3，384，256］
    # 3*3的滤波器，通道为384（上一个卷积输出384层卷积），卷积核数量为256（提取256类特征，即256个重叠的卷积层）
    with tf.name_scope('conv5') as scope:
        W_conv5 = weight_init(shape=[3,3,384,256],stddev=1e-1)
        b_conv5 = bias_init(shape=[256])
        conv_kernel5 = tf.nn.conv2d(conv4,W_conv5,strides=[1,1,1,1],padding='SAME')
        conv5 = tf.nn.relu(tf.nn.bias_add(conv_kernel5,b_conv5),name=scope)
        parameters += [W_conv5,b_conv5]

    print_layer_detail(conv5)

    pool5 = tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool5')
    print_layer_detail(pool5)


    # 第一个全连接层
    # 使用Dropout
    # 需要把卷积输出变为一维向量，但在这里不知道输出特征维度是多少
    # but...知道样本数量是batch_size，可以用reshape自动算出特征维度
    # 全连接层隐层结点数为4096个

    reshape = tf.reshape(pool5,shape=[batch_size,-1])
    dim = reshape.get_shape()[1].value  #得到特征的维度
    with tf.name_scope('local1') as scope:
        W_fc1 = weight_init(shape=[dim,4096],stddev=0.04)   #权重初始化
        b_fc1 = bias_init(shape=[4096])   #偏置单元全部初始化为0.1
        fc1 = tf.nn.relu(tf.matmul(reshape,W_fc1) + b_fc1)     #全连接ReLU输出,注意这里的输入是一维变换后的reshape
        local1 = tf.nn.dropout(fc1,keep_prob=keep_prob,name=scope) #dropout
        parameters += [W_fc1, b_fc1]

    print_layer_detail(local1)


    # 第二个全连接层
    # 使用Dropout
    # 全连接层隐层结点数为4096个

    with tf.name_scope('local2') as scope:
        W_fc2 = weight_init(shape=[4096,4096],stddev=0.04)   #权重初始化
        b_fc2 = bias_init(shape=[4096])   #偏置单元全部初始化为0.1
        fc2 = tf.nn.relu(tf.matmul(local1,W_fc2) + b_fc2)     #全连接ReLU输出,注意这里的输入是一维变换后的reshape
        local2 = tf.nn.dropout(fc2,keep_prob=keep_prob,name=scope) #dropout
        parameters += [W_fc2, b_fc2]

    print_layer_detail(local2)


    # 最后一层输出层
    # 暂时不需要连接一个softmax，到计算损失的时候再用softmax
    # 因为不用softmax也可以比较出最大的那一个作为输出，得到分类结果
    # softmax只是用来计算loss
    # 不采用激活函数，不采用dropout
    with tf.name_scope('logits') as scope:
        W_fc3 = weight_init(shape=[4096,10],stddev=1/4096.0) #不设置L2正则化
        b_fc3 = bias_init(shape=[10])  #全0初始化
        logits = tf.add(tf.matmul(local2,W_fc3),b_fc3,name=scope)  #全连接输出，不带激活函数，不使用dropout
        softmax = tf.nn.softmax(logits)
        prediction = tf.argmax(softmax,1)
        parameters += [W_fc3, b_fc3]


    return prediction,softmax,logits,parameters

# 本次测试用不到优化器、损失函数等



'''
定义计算时间消耗的函数
target为将要预测的目标
feed_dict主要用来传送dropout rate
'''
def time_TF_cost(session, target, feed_dict, info_string,total_batches = 100):
    num_steps_prepare = 10  #前10步在做初始化、catch命中等，不计算时间
    total_cost = 0.0    #总时间消耗
    total_cost_squared = 0.0    #总时间消耗的平方

    for i in range(num_steps_prepare + total_batches):
        start_time = time.time()
        session.run(target,feed_dict=feed_dict)
        time_used = time.time() - start_time
        if i >= num_steps_prepare:
            if i % 10 == 0: #每10轮显示一次
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
        prediction,_,logits,parameters = AlexNet(images,keep_prob)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())



        time_TF_cost(sess,prediction,{keep_prob:1.0},'FeedForward')#正向传播计算耗时

        objective = tf.nn.l2_loss(logits)   #反向传播计算耗时（主要是求梯度）
        grad = tf.gradients(objective,parameters)
        time_TF_cost(sess,grad,{keep_prob:0.5},'BackPropagation')
        print('End')

run_all()