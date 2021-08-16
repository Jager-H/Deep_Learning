# Vision Transformer(ViT)
## 前言
Transformer最初提出是针对NLP领域的，并且在NLP领域大获成功。ViT则是Transformer在CV领域（主要是图像分类）的应用，试图取代CNN在CV领域多年来的统治地位。

论文：[An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale](https://arxiv.org/abs/2010.11929)

代码：[pytorch版](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py)
## 整体结构
<div align=center><img src="https://user-images.githubusercontent.com/65380826/129018844-cf180426-fab8-4382-a40f-4ca4ca7a5751.png"></div>

### 输入部分
#### 1、Linear Projection of Flattened Patches（Embedding层）
（1）以大小为（224，224，3）的图像为例

（2）将图像分均分为大小为patch_size（16，16），因此一共有14×14个patches，因此得到输出结果为（14，14，768）

（3）然后Flatten得到（196，768）

> （4）然后将每个patch进行Linear线性映射成（196，128）【暂不确定】

#### 2、Patch + Position Embedding
（1）额外增加一个token用作class分类（1，768）

（2）与patch拼接得到（196+1，768），即（197，768）

（3）叠加Position Embedding（197，768）即可得到Transformer Encoder的输入（197，768）

> （4）貌似还有一个Dropout的操作
### Transformer Encoder
Transformer Encoder其实就是重复堆叠Encoder Block L次，用于提取特征，得到输出仍为（197，768）

* 这里的MLP类似于一个Feed Forward，第一个全连接层会把输入节点个数翻4倍，即（197，768×4），第二个全连接层会还原回原节点个数（197，768）。结构如下：
<div align=center><img src="https://user-images.githubusercontent.com/65380826/129024365-4f2424dc-237f-4d9e-b8e7-f2b4abc43810.png"></div>

### MLP Head
（1）先进行Layer Norm

（2）再Extract class token，即（197，768）中抽取出[class]token对应的（1，768）。

（3）MLP Head原论文在训练ImageNet21K时是由Linear+tanh激活函数+Linear组成。但是迁移到ImageNet1K上或者自己的数据上时，只用一个Linear即可。结果为（1，num_classes）

## 参考
https://blog.csdn.net/qq_37541097/article/details/118242600




