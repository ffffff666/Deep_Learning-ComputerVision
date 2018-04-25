［VGGNet］
=====
<br>
<br>
构建VGGNet-16卷积神经网络<br>
做图像识别分类，利用CIFAR-10数据集验证<br>
AlexNet包含13个卷积层、4个池化层（无LRN层）和两个全连接层，最后用一个Softmax全连接层做输出：<br>
VGGNet中所有的卷积尺寸为3x3，步长为1<br>
所有的池化尺寸为2x2，步长为2（每次缩小一半）<br>
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊<br>
Layer1:  conv1-1  ［3，3，？，64］<br>
conv1-2  ［3，3，64，64］<br>
pool1:   最大池化  2x2 2x2<br>
<br>
Layer2:  conv2-1  ［3，3，64，128］<br>
conv2-2  ［3，3，64，128］<br>
pool2:   最大池化  2x2 2x2<br>
<br>
Layer3:  conv3-1  ［3，3，128，256］<br>
conv3-2  ［3，3，128，256］<br>
conv3-3  ［3，3，128，256］<br>
pool3:   最大池化  2x2 2x2<br>
<br>
Layer4:  conv4-1  ［3，3，256，512］<br>
conv4-2  ［3，3，256，512］<br>
conv4-3  ［3，3，256，512］<br>
pool4:   最大池化  2x2 2x2<br>
<br>
Layer5:  conv5-1  ［3，3，512，512］<br>
conv5-2  ［3，3，512，512］<br>
conv5-3  ［3，3，512，512］<br>
pool5:   最大池化  2x2 2x2<br>
<br>
<br>
local1:  全连接+ReLU 4096<br>
local2:  全连接+ReLU 4096<br>
logits： 预测输出    1000<br>
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊<br>
<br>
使用VGGNet做图像分类<br>
<br>
<br>
<br>
但是，cifar10并不能做测试，因为图片太小了，需要用李飞飞教授的ImageNet数据集做测试了<br>
所以第一个代码是跑不了的<br>
我测试了一下单次正向传播和反向传播的耗时<br>
这个东西真正用的时候没有GPU也是gg了<br>
<br>
tips:<br>
foxmail：  zixuwang1997@foxmail.com<br>
gamil:     zixuwang1997@gmail.com<br>
others:    zixuwang@csu.edu.cn<br>
