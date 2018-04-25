[AutoEncoder]
===========
<br>
从输入层到隐层到输出层共三层<br>
期望：输入＝输出<br>
隐层做特征的提取和抽象<br>
争取用高阶特征重构样本，但不是完全照搬<br>
所以隐层结点小于输入输出结点，且加入少量噪声，期望去除这些噪声<br>

        x－－h－－y<br>
          E：y＝x'<br>

自编码器主要用在对网络权重的初始化，用来防止梯度弥散<br>
但在现在的CNN、RNN、MLP中已经不再使用<br>
<br>
<br>
class_AutoEncoder.py是单独可以使用的自编码器，无验证输入输出的代码<br>
<br>
<br>
这个代码你没有GPU估计是跑不动的<br>
<br>
<br>
tips:<br>
foxmail：  zixuwang1997@foxmail.com<br>
gamil:     zixuwang1997@gmail.com<br>
others:    zixuwang@csu.edu.cn<br>
