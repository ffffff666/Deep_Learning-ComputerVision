［AlexNet］
=====
构建AlexNet卷积神经网络<br>
做图像识别分类，利用CIFAR-10数据集验证<br>
AlexNet包含五个卷积层、三个池化层（两个LRN层）和两个全连接层，最后用一个Softmax全连接层做输出：<br>
<br>
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊<br>
conv1：  卷积层+ReLU    5x5 4 96<br>
lan1:     LAN<br>
pool1:   最大池化       3x3 2x2<br>
conv2：  卷积层+ReLU    5x5 1 256<br>
lan2:     LAN<br>
pool2:   最大池化       3x3 2x2<br>
conv3：  卷积层+ReLU    3x3 1 384<br>
conv4：  卷积层+ReLU    3x3 1 384<br>
conv5：  卷积层+ReLU    3x3 1 256<br>
pool5:   最大池化       3x3 2x2<br>
local1:  全连接+ReLU<br>
local2:  全连接+ReLU<br>
logits： 预测输出<br>
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊<br>
<br>
使用AlexNet做图像分类<br>
<br>
<br>
但是，cifar10并不能做测试，因为图片太小了，需要用李飞飞教授的ImageNet数据集做测试了<br>
所以第一个代码是跑不了的<br>
我测试了一下单次正向传播和反向传播的耗时<br>
这个东西真正用的时候没有GPU也是gg了<br>
<br>
<br>
tips:<br>
foxmail：  zixuwang1997@foxmail.com<br>
gamil:     zixuwang1997@gmail.com<br>
others:    zixuwang@csu.edu.cn<br>
