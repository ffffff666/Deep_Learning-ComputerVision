#!/usr/bin/env python3
# coding=utf-8
'''

             ［VGGNet］
构建VGGNet-16卷积神经网络
做图像识别分类，利用CIFAR-10数据集验证
AlexNet包含13个卷积层、4个池化层（无LRN层）和两个全连接层，最后用一个Softmax全连接层做输出：
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

'''
导入CIFAR-10数据库
导入一些会用到的包
初始化参数：最大训练轮数和每次批处理的样本数
'''
import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time

max_step = 30
batch_size = 1  #随机梯度下降
data_dir = '/Users/apple/Documents/Python/DeepLearning/models/tutorials/image/cifar10/cifar-10-batches-bin'

# 使用cifar10类下载数据集，并解压、展开到其默认位置
# cifar10.maybe_download_and_extract()


# 再使用cifar10_input类的distorted_inputs方法产生训练使用的数据，返回已经封装好的tensor
# 每次执行都会返回一个batch_size的样本，并且已经对数据进行了数据增强，包括：
#   随机水平翻转、随机剪切一块24x24大小的图片、随机亮度以及对比度以及对数据标准化处理
# 通过上述操作得到更多样化的样本（带噪声），有利于提高泛化能力（通过这些操作可以把一张图片变为多张图片）
images_train,labels_train = cifar10_input.distorted_inputs(data_dir = data_dir,batch_size = batch_size)

# 测试数据就不需要那么多操作了，但还需要剪切和标准化操作
images_test,labels_test = cifar10_input.inputs(eval_data = True,data_dir = data_dir,batch_size = batch_size)



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
        return  output

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
Step 1:Inputdata
首先导入数据
placeholder是代表为输入数据占空
暂时不填入信息，调用Run函数的时候补全输入
在这里输入就不能是任意大小None了，必须是batch_size大小的数据
数据的特征维度是24x24x3     即剪切后的24x24大小的二维图片，同时有RGB三个颜色通道
'''

image_holder = tf.placeholder(tf.float32,[batch_size,24,24,3])
label_holder = tf.placeholder(tf.int32,[batch_size])
keep_prob = tf.placeholder(tf.float32) #dropout时的保留率


'''
Step 2:FeedForward
定义算法公式，就是神经网络前向传播的途径
TF会自动帮忙计算反向传播的梯度公式
首先先进行一些函数的定义，方便后面使用
'''

# 第一层:
# Layer1:       conv1-1  ［3，3，？，64］
#               conv1-2  ［3，3，64，64］
#               pool1: 最大池化 2x2 2x2
conv1_1 = conv_layer(image_holder,num_output=64,name='conv1_1',parameters=parameters)
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

def loss(logits,labels):
    labels = tf.cast(labels,tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits,
                                                                    labels=labels,
                                                                    name='cross_entropy_per_example')
    # 计算每一个样本的交叉熵损失 （对输出进行softmax）
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy') #计算总的平均交叉熵损失

    return cross_entropy_mean  #返回所有样本的总损失



loss = loss(logits,label_holder)    #将神经网络输出结果和真实标记传入loss函数得到总损失


'''
Step 4:Optimizer
选择优化器，并指定优化器优化loss
主要是用梯度下降法、随机梯度下降法、批处理梯度下降法
选择Adam优化器，设置学习率（a=1e-3）
TF会自动进行BP算法梯度更新
'''
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 利用tf.nn.in_top_k计算输出结果top k的准确率，这里就是用top 1
top_k_op = tf.nn.in_top_k(logits,label_holder,1)

# 上面代码等同于：
# correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

'''
Step 4:Training
开始训练，使用批处理梯度下降、每次选一个mini_batch，并feed给placeholder
(总共30轮，每个batch包含1样本)
当然在一开始的时候需要调用TF全局参数初始化器
InteractiveSession是将这个session设置为默认session
'''
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 这一步是启动图片数据增强的线程队列，一共使用16个线程来加速
# 如果不启动线程的话，后续的训练无法进行
tf.train.start_queue_runners()


print('======> Start training:')

for step in range(max_step):
    start_time = time.time() #记录每一轮训练的开始时间
    image_batch,label_batch = sess.run([images_train,labels_train])    #得到一个batch的数据
    _,loss_value = sess.run([train_step,loss],feed_dict={image_holder:image_batch,label_holder:label_batch,keep_prob:0.5})
                                                                    #传入变量，开始训练
    time_used = time.time() - start_time

    #每10轮显示一下当前在批处理训练集上的精度，以及耗时
    if step%10 == 0:
        examples_per_sec = batch_size/time_used     #计算每秒钟可以训练的样本数
        sec_per_batch = float(time_used)            #计算训练每一个batch的耗时
        format_str = ('step: %d,   loss = %.2f     [每秒可训练样本数：%.1f ，训练每批样本耗时：%.2fs]')
        print(format_str % (step,loss_value,examples_per_sec,sec_per_batch))

print('======> Training End!!!!')

#在此训练完成,将训练完成的参数保存到文件
saver = tf.train.Saver()  # 默认保存所有参数，本例中为：W和b
model_path = "/Users/apple/Documents/TensorFlow/Model/model_AlexNet/model_AlexNet.ckpt"
save_path = saver.save(sess, model_path)
print('CNN model saved!!!!')
'''
读取模型操作：
saver = tf.train.Saver()
saver.restore(sess, "/Users/apple/Documents/TensorFlow/Model/model_Softmax_Regression.ckpt")
result = sess.run(y, feed_dict={x: data})
'''




'''
Step 6:Correct_prediction from testSet
对模型在测试集上进行准确率的验证
在对测试集验证的时候也需要采用批处理的方法验证
'''
num_examples = 10000
import math
num_iter = int(math.ceil(num_examples/batch_size))  #计算总共需要取多少次
true_count = 0  #预测正确的次数
total_sample_count = batch_size * num_iter    #总共需要迭代验证的样本数（并不是10000）
step = 0

while step<num_iter:
    image_batch,label_batch = sess.run([images_test,labels_test])   #一次取一批测试样本
    prediction = sess.run([top_k_op],feed_dict={image_holder:image_batch,label_holder:label_batch,keep_prob:1.0})#计算精度
     #这里的精度，也就是top_k_op，是指这批样本中预测类别和真实类别相等的情况
     #返回的是一个矩阵［1,1,0,0...1,0,1］，为0则代表预测错误，矩阵维度等于批处理样本的数量

    true_count += np.sum(prediction)    #计算这批样本中预测正确的个数

    step += 1

precision = true_count/total_sample_count*100   #计算精度（准确率）
print('测试集上面的精度为：%.2f%%'%precision)