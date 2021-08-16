# Swin Transformer
## 前言
SwinTransformer通过计算Local Attention，极大地降低了密集预测任务中Transformer的计算量，同时采用了一种Shift Window的策略，引入Local Windows间的联系，增强了其建模能力，并且在图像分类、目标检测以及分割等多个任务上都取得了很好的结果

论文：[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)

源码：https://github.com/microsoft/Swin-Transformer
## 整体结构
Swin Transformer共提出了4个网络框架，它们从小到大依次是Swin-T，Swin-S, Swin-B和Swin-L，本文以最简单的Swin-T作为示例来讲解，Swin-T的结构如图所示

![image](https://user-images.githubusercontent.com/65380826/129197601-d418fc3b-8af5-4165-b392-aed21b6dbffb.png)
### 1、输入层Patch Partition
Patch Partition部分把输入为（H，W，3）的RGB图像分割为一系列不重叠的patches，得到（H/4，W/4，48）

Linear Embedding部分把patch的维度从48转化为C，得到（H/4，H/4，C）

### 2、Patch Merging
经过Stage1后输出仍为（H/4，H/4，C）

Patch Merging操作首先会将临近2×2范围内的patch拼接起来，得到一个4C维度的feature，即（H/8，H/8，4C）

然后经过一个线形层将维度降低为2C，即（H/8，H/8，2C）

以此类推得到Stage3（H/16, W/16，4C）和 Stage4（H/32，W/32，8C）
### 3、Swin Transformer Block
Swin Transformer Block是该算法的核心点，它由窗口多头自注意层（window multi-head self-attention, W-MSA）和移位窗口多头自注意层（shifted-window multi-head self-attention, SW-MSA）组成，如下图所示。因此Swin Transformer的层数要为2的整数倍，一层提供给W-MSA，一层提供给SW-MSA。

<div align=center><img src="https://user-images.githubusercontent.com/65380826/129200279-75e7bfb6-03f7-400a-a53b-54709ca0bbca.png"></div>

#### （1）W-MSA
标准的Transformer结构或其变体都采用的是Global Self Attention，其会计算一个token和其他所有token的关系，其计算复杂度太高，不适合与密集预测等需要大量token的任务。
为了降低计算复杂度，SwinTransformer在局部Windows内部计算Self-Attention。传统的MSA和W-MSA的复杂度分别是：
<div align=center><img src="https://user-images.githubusercontent.com/65380826/129298998-f1ec888a-dc0e-455f-8046-82e9b6558ee2.png" width="350" height="80"></div>

式中，h，w为输入特征图的宽高，C为特征图的维度，M为滑动Windows的大小（论文中取7）。可见这种方法极大减轻了网络的计算量，将复杂度降低到了图像尺寸的线性比例。

#### （2）SW-MSA
单独的使用W-MSA得到的网络的建模能力是非常差的，因为它将每个窗口当做一个独立区域计算而忽略了窗口之间交互的必要性，基于这个动机，Swin Transformer提出了SW-MSA，从而实现了跨窗口通信
<div align=center><img src="https://user-images.githubusercontent.com/65380826/129300756-768d1454-ae5b-4425-a2e4-2442f152eee9.png" width="500" height="200"></div>

以上图为例，对于一个8×8大小的特征图，window大小M=4：
* 左边为使用W-MSA时的划分结果，在每个窗口内计算self-attention
* 右边是使用SW-MSA时的划分结果，使用了移动窗口的策略，windows从特征图（M/2,M/2）的位置（M=7时取整为3）开始移动，然后再计算对应窗口的self-attention

这样一来，部分windows的大小不足M×M，为了解决这种缺陷，引入了Batch Computation Approach机制。通过设定特殊的mask，在计算attention时，仅对一个window内的有效部分进行Attention，其余部分被mask掉，即可实现在原来计算Attention方法不变的情况下，对非规则的Window计算Attention。

<div align=center><img src="https://user-images.githubusercontent.com/65380826/129307078-a6a8726c-f496-4a13-9530-369aa082642d.png"></div>

对于cyclic shift四种不同的区域会产生4中不同的mask（从左至右依次是左上、右上、右下、左下四个区域的mask）
<div align=center><img src="https://user-images.githubusercontent.com/65380826/129316849-64857958-6c71-4879-89bb-4aa41175133f.png" width="400" height="100"></div>

> 灰色区域为需要计算相似度的部分，黑色区域则是无意义的相似度

以左下部分mask为例，具体推导过程如下所示：
<div align=center><img src="https://user-images.githubusercontent.com/65380826/129318027-c6db2dcb-e55c-42fb-b274-fd384e469feb.jpg" width="600" height="300"></div>

#### （3）Relative Position Bias
在计算Self-Attention的过程中，Swin Transformer也加入了相对位置编码的部分
<div align=center><img src="https://user-images.githubusercontent.com/65380826/129319982-3c6742fc-3469-4489-9d1f-9036b0db2ceb.png" width="600" height="50"></div>

这里采用的是相对位置编码，相对位置编码的范围是以窗口为单位的。

### 4、输出层
在stage-4完成计算后，得到Swin Transformer所提取特征，维度为（H/32，W/32，8C），然后用于图像分类、目标检测等领域。
> Swin Transformer可作为其他目标检测算法的backbone，然后跟各种不同的Neck和Head结构融合，从而实现目标检测的功能。 
## 结构变量
Swin Transformer共提出了4个不同尺寸的模型，它们的区别在于隐层节点的长度C，每个stage的层数(2的倍数)，多头自注意力机制的头的个数，具体值见下
```
Swin-T: hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24)
Swin-S: hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24)
Swin-B: hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32)
Swin-L: hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48)
```
## 参考
https://zhuanlan.zhihu.com/p/370766757

https://zhuanlan.zhihu.com/p/361366090
