#!/usr/bin/env python3
# coding=utf-8
'''

             ［CNN图像分类］
图像识别分类，利用CIFAR-10数据集验证
本次CNN包含两个卷积层、两个池化层、两个LRN层和两个全连接层，最后用一个全连接层做输出：

＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
    conv1：  卷积层+ReLU
    pool1:   最大池化
    lan1:     LAN
    conv2：  卷积层+ReLU
    lan2:     LAN
    pool2:   最大池化
    local3:  全连接+ReLU
    local4:  全连接+ReLU
    logits： 预测输出
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊

使用CNN做图像分类

'''

_author_ = 'zixuwang'
_datetime_ = '2018-1-26'

'''
导入CIFAR-10数据库
导入一些会用到的包
初始化参数：最大训练轮数和每次批处理的样本数
'''
import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time

max_step = 3000
batch_size = 128
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
                                                CNN
***********************************************************************************************************
***********************************************************************************************************

'''

# 本次模型中需要设置一个惩罚因子防止过拟合，利用L2正则
# L1正则可以让参数稀疏，无用的特征被置0
# L2正则会让参数权重不过大，使得特征的权重都比较平均
# variable_with_weight_loss函数就是在初始化权重的时候就为权重添加了L2正则
# 在权重更新过程中L2正则也在更新
# w1是L2正则的系数（lambda）
# 权重初始化方法是截断的正态分布
def variable_with_weight_loss(shape,stddev,w1):
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var),w1,name='weight_loss')
        tf.add_to_collection('losses',weight_loss)
    return var

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
# 由于卷积层使用了ReLU激活函数，进行初始化时需要随机初始化



# 第一个卷积层: 卷积－>池化—>LRN
# 卷积核：［5，5，3，64］
# 5*5的滤波器，通道为3（灰度图为1，RGB为3），卷积核数量为64（提取64类特征，即64个重叠的卷积层）

W_conv1 = variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)  #权重初始化：标准差为0.05，不设置L2正则
b_conv1 = tf.Variable(tf.constant(0.0,shape=[64]))  #偏置单元全部初始化为0
conv_kernel1 = tf.nn.conv2d(image_holder,W_conv1,strides=[1,1,1,1],padding='SAME')  #卷积操作：一个不漏地平移
conv1 = tf.nn.relu(tf.nn.bias_add(conv_kernel1,b_conv1))    #卷积后加偏置，ReLU激活
pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')  #3x3采样池化，2x2单位移动，增加数据丰富性
lrn1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)    #LRN处理

# 第二个卷积层: 卷积－>LRN—>池化
# 卷积核：［5，5，64，64］
# 5*5的滤波器，通道为64（第一次卷积输出64层卷积），卷积核数量为64（提取64类特征，即64个重叠的卷积层）

W_conv2 = variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)     #权重初始化：不设置L2正则
b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]))  #偏置单元全部初始化为0.1
conv_kernel2 = tf.nn.conv2d(lrn1,W_conv2,strides=[1,1,1,1],padding='SAME')   #卷积操作：一个不漏地平移
conv2 = tf.nn.relu(tf.nn.bias_add(conv_kernel2,b_conv2))    #卷积后加偏置，ReLU激活
lrn2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75)    #LRN处理
pool2 = tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME') #3x3采样池化，2x2单位移动




# 第一个全连接层
# 使用Dropout，并且加入L2正则
# 需要把卷积输出变为一维向量，但在这里不知道输出特征维度是多少
# but...知道样本数量是batch_size，可以用reshape自动算出特征维度
# 全连接层隐层结点数为384个

reshape = tf.reshape(pool2,shape=[batch_size,-1])
dim = reshape.get_shape()[1].value  #得到特征的维度
W_fc3 = variable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)   #权重初始化：设置L2正则w1=0.004
b_fc3 = tf.Variable(tf.constant(0.1,shape=[384]))   #偏置单元全部初始化为0.1
local3 = tf.nn.relu(tf.matmul(reshape,W_fc3) + b_fc3)     #全连接ReLU输出,注意这里的输入是一维变换后的reshape



# 第二个全连接层
# 使用Dropout，并且加入L2正则
# 隐层结点减少一半，其余不变
# 全连接层隐层结点数为192个

W_fc4 = variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)   #权重初始化：设置L2正则w1=0.004
b_fc4 = tf.Variable(tf.constant(0.1,shape=[192]))   #偏置单元全部初始化为0.1
local4 = tf.nn.relu(tf.matmul(local3,W_fc4) + b_fc4)    #全连接ReLU输出

# 最后一层输出层
# 暂时不需要连接一个softmax，到计算损失的时候再用softmax
# 因为不用softmax也可以比较出最大的那一个作为输出，得到分类结果
# softmax只是用来计算loss
# 不采用激活函数，不采用dropout

W_fc5 = variable_with_weight_loss(shape=[192,10],stddev=1/192.0,w1=0.0) #不设置L2正则化
b_fc5 = tf.Variable(tf.constant(0.0,shape=[10]))    #全0初始化
logits = tf.add(tf.matmul(local4,W_fc5),b_fc5)  #全连接输出，不带激活函数，不使用dropout




'''
Step 3:Loss
计算损失，构建损失函数
本次使用cross-entropy Loss

==> loss = - Sum(y_ * log(y))

但这里的loss还需要加上L2正则的weight_loss
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
    tf.add_to_collection('losses',cross_entropy_mean)   #把交叉熵损失和L2正则损失加在一起

    return tf.add_n(tf.get_collection('losses'),name='total_loss')  #返回所有样本的总损失之和



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
(总共3000轮，每个batch包含128样本)
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
    _,loss_value = sess.run([train_step,loss],feed_dict={image_holder:image_batch,label_holder:label_batch})
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
model_path = "/Users/apple/Documents/TensorFlow/Model/model_CNN_imageClassify/model_CNN_imageClassify.ckpt"
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
    prediction = sess.run([top_k_op],feed_dict={image_holder:image_batch,label_holder:label_batch})#计算精度
     #这里的精度，也就是top_k_op，是指这批样本中预测类别和真实类别相等的情况
     #返回的是一个矩阵［1,1,0,0...1,0,1］，为0则代表预测错误，矩阵维度等于批处理样本的数量

    true_count += np.sum(prediction)    #计算这批样本中预测正确的个数

    step += 1

precision = true_count/total_sample_count*100   #计算精度（准确率）
print('测试集上面的精度为：%.2f%%'%precision)