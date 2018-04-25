import cifar10,cifar10_input
import tensorflow as tf
import matplotlib.pyplot as plt
import threading

dic = {
0: 'airplane',
1: 'automobile',
2: 'bird',
3: 'cat',
4: 'deer',
5: 'dog',
6: 'frog',
7: 'horse',
8: 'ship',
9: 'truck'
}




batch_size = 1
data_dir = '/Users/apple/Documents/Python/DeepLearning/models/tutorials/image/cifar10/cifar-10-batches-bin'

# 读取数据集
original_images_test, images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

# 输入变量
image_holder = tf.placeholder(tf.float32,[batch_size,24,24,3])
label_holder = tf.placeholder(tf.int32,[batch_size])

# 需要读入的参数
# 在这里需要先把神经网络搭建一遍

def weight_bias_init(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))


# 卷积层1
W_conv1 = weight_bias_init([5,5,3,64])
b_conv1 = weight_bias_init([64])
conv_kernel1 = tf.nn.conv2d(image_holder,W_conv1,strides=[1,1,1,1],padding='SAME')
conv1 = tf.nn.relu(tf.nn.bias_add(conv_kernel1,b_conv1))
pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
lrn1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)

# 卷积层2
W_conv2 = weight_bias_init([5,5,64,64])
b_conv2 = weight_bias_init([64])
conv_kernel2 = tf.nn.conv2d(lrn1,W_conv2,strides=[1,1,1,1],padding='SAME')
conv2 = tf.nn.relu(tf.nn.bias_add(conv_kernel2,b_conv2))
lrn2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
pool2 = tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

# 全连接层1
reshape = tf.reshape(pool2,[batch_size,-1])
dim = reshape.get_shape()[1].value
W_fc3 = weight_bias_init([dim,384])
b_fc3 = weight_bias_init([384])
local3 = tf.nn.relu(tf.matmul(reshape,W_fc3) + b_fc3)

# 全连接层2
W_fc4 = weight_bias_init([384,192])
b_fc4 = weight_bias_init([192])
local4 = tf.nn.relu(tf.matmul(local3,W_fc4) + b_fc4)

# 输出层
W_fc5 = weight_bias_init([192,10])
b_fc5 = weight_bias_init([10])
logits = tf.add(tf.matmul(local4,W_fc5),b_fc5)



sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 这一步是启动图片数据增强的线程队列，一共使用16个线程来加速
tf.train.start_queue_runners()

# 读取模型参数
saver = tf.train.Saver()
model_path = "/Users/apple/Documents/TensorFlow/Model/model_CNN_imageClassify/model_CNN_imageClassify.ckpt"
saver.restore(sess, model_path)
print('CNN Model Loaded')

# 测试集验证


def show_picture(X):
    plt.figure(figsize=(1, 1))
    plt.axis("off")
    plt.imshow(X)
    plt.show()

while True:
    print('')
    print('')
    origin_batch, image_batch, label_batch = sess.run([original_images_test, images_test, labels_test])   #一次取一个测试样本

    print("True Label is : {}".format(dic[label_batch[0]]))
    print('')

    predicted_label = tf.argmax(logits, axis=1).eval(feed_dict={image_holder:image_batch,label_holder:label_batch})
    print("Predicted Label is : {}".format(dic[predicted_label[0]]))

    show_picture(origin_batch[0])