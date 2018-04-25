#!/usr/bin/env python3
# coding=utf-8
'''

             ［使用VGGNet-19做风格迁移］
采用预训练好的VGGNet-19网络（使用ImageNet数据集）

风格迁移：
CNN可以提取图像高级特征以及抽象特征
风格迁移是用一张图片的风格绘制另一张图片
可以利用一个VGG卷积神经网络提取高阶特征
在很高的卷积层中，原始图像的类型应该相同（高阶特征很抽象，维度低，类似的图片在同一个聚类簇内）
在中间多层卷基层中，风格图片的风格特点（低阶抽象特征）应该相似，具体可以表现为同一层卷积中各通道输出类似
定义这两个损失函数作为优化目标，用原始图像和风格图片作为输入，训练同一个网络，得到迁移后的图片
属于迁移学习

'''

_author_ = 'zixuwang'
_datetime_ = '2018-2-2'


import time
import numpy as np
import scipy.io
import scipy.misc
import math
import tensorflow as tf

'''
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
                                            部分参数与定义
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
'''

ORIGINAL_IMAGE = './images/original_image.jpg'
STYLE_IMAGE = './images/style_image.jpg'
OUTPUT_IMAGE = './images/output_No350.png'  #使用这个可以读取上次训练中断的图片

# 图片尺寸：800*600 RGB
IMAGE_WIDTH = 800
IMAGE_HIGHT = 600
COLOR_CHANNELS = 3

# 设置随即噪声与原始图像的比例
NOISE_RATE = 0.6
# 设置总的训练轮数
total_step = 1000
# 设置原始图像与风格图像的比重
alpha = 1
beta = 500
# 加载VGG19模型以及设定其均值
VGG_Model = './model/VGG_19.mat'
MEAN_VALUES = np.array([123.68,116.779,103.939]).reshape((1,1,1,3))
# 设置需要用到的卷积层：保留原始图片信息的高层卷积层、保留风格图片细节的卷积层
ORIGINAL_LAYERS = [('conv4_2',1.)]
STYLE_LAYERS = [('conv1_1',0.2),('conv2_1',0.2),('conv3_1',0.2),('conv4_1',0.2),('conv5_1',0.2)]


'''
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
                                        重构VGGNet：主要函数设计
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
'''

# 读入图片
def load_image(path):
    image = scipy.misc.imread(path)
    image = np.reshape(image,(1,)+image.shape)  #由于风格迁移每次输入一张图片，所以第一个维度应该设置为1（额外增加一个维度）
    image = image - MEAN_VALUES #对图片做标准化
    return image

# 保存图片
def save_image(image, path):
    image = image + MEAN_VALUES #需要加入均值复原
    image = image[0]    #去除多余的维度（降1维）
    image = np.clip(image,0,255).astype('uint8')    #恢复彩色图片
    scipy.misc.imsave(path,image)

# 给原始图片加随机噪声
def add_noise(image, noise_ratio = NOISE_RATE):
    noise = np.random.uniform(-20,20,(1,IMAGE_HIGHT,IMAGE_WIDTH,COLOR_CHANNELS)).astype('float32')
    image = noise * noise_ratio + image * (1-noise_ratio)
    return image

# 建立卷积层和池化层
def build_net(type, input, weights_and_biases=None):
    if type == 'conv':
        return tf.nn.relu(tf.nn.conv2d(input,weights_and_biases[0],strides=[1,1,1,1],padding='SAME') + weights_and_biases[1])

    elif type == 'pool':
        return tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 返回网络某一卷积层的权重和偏置
def get_weight_bias(vgg_layers, i):
    weights = vgg_layers[i][0][0][0][0][0]
    weights = tf.constant(weights)

    biases = vgg_layers[i][0][0][0][0][1]
    biases = tf.constant(np.reshape(biases,(biases.size)))

    return weights,biases

# 提取.mat中模型，重构网络卷积部分
def build_vgg19(path):
    net = {}
    vgg_rawnet = scipy.io.loadmat(path)
    vgg_layers = vgg_rawnet['layers'][0]
    net['input'] = tf.Variable(np.zeros((1,IMAGE_HIGHT,IMAGE_WIDTH,COLOR_CHANNELS)).astype('float32'))
    net['conv1_1'] = build_net('conv', net['input'], get_weight_bias(vgg_layers, 0))
    net['conv1_2'] = build_net('conv', net['conv1_1'], get_weight_bias(vgg_layers, 2))
    net['pool1'] = build_net('pool', net['conv1_2'])

    net['conv2_1'] = build_net('conv', net['pool1'], get_weight_bias(vgg_layers, 5))
    net['conv2_2'] = build_net('conv', net['conv2_1'], get_weight_bias(vgg_layers, 7))
    net['pool2'] = build_net('pool', net['conv2_2'])

    net['conv3_1'] = build_net('conv', net['pool2'], get_weight_bias(vgg_layers, 10))
    net['conv3_2'] = build_net('conv', net['conv3_1'], get_weight_bias(vgg_layers, 12))
    net['conv3_3'] = build_net('conv', net['conv3_2'], get_weight_bias(vgg_layers, 14))
    net['conv3_4'] = build_net('conv', net['conv3_3'], get_weight_bias(vgg_layers, 16))
    net['pool3'] = build_net('pool', net['conv3_4'])

    net['conv4_1'] = build_net('conv', net['pool3'], get_weight_bias(vgg_layers, 19))
    net['conv4_2'] = build_net('conv', net['conv4_1'], get_weight_bias(vgg_layers, 21))
    net['conv4_3'] = build_net('conv', net['conv4_2'], get_weight_bias(vgg_layers, 23))
    net['conv4_4'] = build_net('conv', net['conv4_3'], get_weight_bias(vgg_layers, 25))
    net['pool4'] = build_net('pool', net['conv4_4'])

    net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(vgg_layers, 28))
    net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30))
    net['conv5_3'] = build_net('conv', net['conv5_2'], get_weight_bias(vgg_layers, 32))
    net['conv5_4'] = build_net('conv', net['conv5_3'], get_weight_bias(vgg_layers, 34))
    net['pool5'] = build_net('pool', net['conv5_4'])

    return net

'''
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
                                        风格迁移：定义损失函数
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
'''

# 计算网络保留原始图片结构的损失
def original_layer_loss(p,x):
    M = p.shape[1] * p.shape[2]
    N = p.shape[3]
    loss = (1. / (2*M*N)) * tf.reduce_sum(tf.pow((x - p) , 2))
    return loss

def total_original_layer_loss(session,net):
    layers = ORIGINAL_LAYERS
    total_loss = 0.0

    for layer_name,weight in layers:
        p = session.run(net[layer_name])    #使用原始图片做输入，在网络在跑出来的某一卷积层输出
        x = net[layer_name]                 #在使用带噪声的原始图片训练时每一轮中卷积层的输出
        # 这个损失的作用主要是让这个网络对带噪声的原始图片提取的高阶特征
        # 尽量与输入原始图片后卷积层提取的高层特征相同
        total_loss += original_layer_loss(p,x) * weight

    total_loss = total_loss / float(len(layers))
    return total_loss

# 计算gram矩阵
# gram矩阵是求同一层卷积的不同通道的相关度
def gram_matrix(x,area,depth):
    x1 = tf.reshape(x,(area,depth))
    gram = tf.matmul(tf.transpose(x1),x1)
    return gram


# 计算网络保留风格图片细节的损失
def style_loss(a,x):
    M = a.shape[1] * a.shape[2]
    N = a.shape[3]

    A = gram_matrix(a,M,N)
    X = gram_matrix(x,M,N)

    loss = (1. / (4 * M**2 * N**2))  * tf.reduce_sum(tf.pow((X-A) , 2))
    return loss

def total_style_loss(session,net):
    layers = STYLE_LAYERS
    total_loss = 0.0

    for layer_name,weight in layers:
        a = session.run(net[layer_name])    # 使用风格图片做输入，在网络在跑出来的某一卷积层输出
        x = net[layer_name]  # 在使用带噪声的原始图片训练时每一轮中卷积层的输出
        # 这个损失的作用主要是让这个网络对带噪声的原始图片提取的每一层特征
        # 尽量与输入风格图片后卷积层提取的每一层特征"相关性"尽量相同（即保留风格图片的风格细节）
        total_loss += style_loss(a,x)

    total_loss = total_loss/float(len(layers))
    return total_loss


'''
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
                                        风格迁移：主函数
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
'''
def main():
    net = build_vgg19(VGG_Model)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 读取原始图片和风格图片
    original_image = load_image(ORIGINAL_IMAGE)
    style_image = load_image(STYLE_IMAGE)

    #损失函数1:输入原始图片，计算当前网络保留原始图片高阶特征的损失
    sess.run(net['input'].assign(original_image))
    original_loss = total_original_layer_loss(sess,net)

    # 损失函数2:输入风格图片，计算当前网络保留风格图片风格细节的损失
    sess.run(net['input'].assign(style_image))
    style_loss = total_style_loss(sess,net)

    # 总损失
    total_loss = alpha * original_loss  +  beta * style_loss

    # 给原始图片加随机噪声
    input_image = add_noise(original_image)
    input_image = load_image(OUTPUT_IMAGE)
    # 定义优化器，定义训练目标
    optimizer = tf.train.AdamOptimizer(2.0)
    train_op = optimizer.minimize(total_loss)

    sess.run(tf.global_variables_initializer())
    sess.run(net['input'].assign(input_image))
    print('==========>training start!')
    for i in range(total_step):
        if i==0:
            start_time = time.time()
            sess.run(train_op)
            print('time cost: %.2fs'%(time.time()-start_time))
        else :
            sess.run(train_op)
        print('------step %d-------'%i)

        if i % 10 == 0:
            mixed_image = sess.run(net['input'])
            print(' loss: ',sess.run(total_loss)/100000000)

            save_path = './images/output___No%d.png'%(i)
            save_image(mixed_image,save_path)



main()

'''
开始并不理解这个是怎么做到的
后来发现输出的变换图片就是网络中的input
想了想没错，input设置为Variable，也算是一个结点，每次都采用梯度下降对这个输入的图片做优化
图片变得越来越符合条件，但网络权重是否更新？这点想不明白，我估计是不会改变的
因为特征的提取步骤应该是相似的，相应的各个权重也都是从模型中读入的，也是重构了
但并没有设置为节点，而是设置了初始值后就不管了，所以我想权重应该是不会改变的
改变的只是那个加了噪声的原始图片，变得越来越像风格图片和原始图片的混合体
也可以理解为这个混合体是一种两类损失的折中
'''
