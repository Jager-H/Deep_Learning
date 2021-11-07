# DETR
## 前言
第一个将 Transformer 成功整合为检测 pipeline 中心构建块的目标检测框架。基于Transformers的端到端目标检测，没有NMS后处理步骤、真正的没有anchor，且对标超越Faster RCNN。

论文：[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

代码：https://github.com/facebookresearch/detr

## 整体结构
<div align=center><img src="https://user-images.githubusercontent.com/65380826/129825728-d390a164-63ea-4a22-b42f-1c993f8655d5.png"></div>

### 1、Backbone
（1）Backbone部分是传统的CNN结构，输入image（H0，W0，3），输出feature map（H，W，C），其中H=H0/32，W=W0/32，C通常取2048。

（2）然后用1×1的卷积核维度压缩到d(源码取256)，（H，W，d）

（3）最后把宽高压缩成一个维度的feature map，即（H * W，d）

（4）feature map + positional encoding构成transformer encoder的输入（positional encoding维度与feature map相同）
### 2、Encoder + Decoder
<div align=center><img src="https://user-images.githubusercontent.com/65380826/129853406-6ac11e85-1b41-45ee-af4a-32dcfe3f72ec.png" width="500" heigth="600"></div>

transformer encoder和transformer decoder的结构与Transformer基本相同，源码计算过程如下：
输入 X：[H * W，d]，三个参数矩阵Wq、Wk、Wv：[d，d]，偏置b：[d]，通过X * transpose(W)+b 计算得到Q、K、V [H * W，d]。然后分组得到Qi、Ki、Vi [nheads，H * W，d/nheads]，其中nheads通常取8。

headi = Attention（Qi，Ki，Vi）= softmax（Qi * transpose（Ki）/sqrt（d/nheads））Vi -> [nheads，H * W，d/nheads]

MultiHead（Q，K，V）= concat（head0，head1，...head8）* transpose（W）+ b # [H * W，d]×[d，d]->[H * W，d]

> Encoder中，计算attention时，q,k是Image features+Spatial positional encoding作为输入，v则是Image features作为输入。
> 
> Decoder中，第一个attention的q,k是object queries+query_pos作为输入，v是object queries作为输入；第二个attention的q是第一个attention后面的输出+query_pos作为输入，k是encoder的输出+Spatial positional encoding作为输入，v则是encoder的输出作为输入；


* Object queries
> Object queries是 N 个learnable embedding，训练刚开始时可以随机初始化。在训练过程中，因为需要生成不同的boxes，object queries会被迫使变得不同来反映位置信息，所以也可以称为learnt positional encoding 
> 
> 此外，和原始的Transformer不同的是，DETR的Transformer Decoder是一次性处理全部的object queries，即一次性输出全部的predictions；而不像原始的Transformer是auto-regressive的，从左到右一个词一个词地输出。
> 
> N个object quries被decoder转换为N个输出。

输出结果为（N，channel），N为object queries的序列长度，通常取100，即一幅图最多可以预测100个目标。
### 3、Prediction Heads
最后通过一个Linear层输出class预测（N，num_classes+1），第num_classes+1维表示是否为背景

通过一个多层感知机MLP结构输出box预测（N，4），4表示预测框归一化之后的参数(center_x, center_y, height, width)，而不是基于偏移量offset的参数。
## 损失函数

## 参考
https://zhuanlan.zhihu.com/p/267156624
