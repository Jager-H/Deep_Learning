# HaloNet
## 前言
CNN是一种参数量和感受野相关（parameter-dependent scaling）交互与输入内容无关的操作（content-independent interactions），而Self-Attention（SA）是一种参数量与感受野无关、交互与输入内容相关的操作，因此作者提出用SA的操作来替代CNN的操作，以此来解决CNN的缺点。基于此，作者提出了HaloNet，并在多个任务上做了实验，证明了其有效性。HaloNet在ImageNet上达到了84.9%的top-1准确率，在目标检测和实力分割任务中也取得了不错的效果。

论文：[Scaling Local Self-Attention for Parameter Efficient Visual Backbones](https://arxiv.org/pdf/2103.12731.pdf)

代码：https://github.com/lucidrains/halonet-pytorch （非官方，官方未开源）
## 核心算法
![image](https://user-images.githubusercontent.com/65380826/141220416-7597f5df-4c4e-42ab-afac-d4fc419d2bcc.png)
如上图所示，如果每次只考虑block内的信息，必然会导致信息的损失，因此在计算Local Self-Attention之前，作者先对每个block进行的haloing操作。也就是在每个block外，再用原始图片的信息padding一圈（这个操作叫做Halo），使得每个block的感受野能够适当变大，关注更多的信息。
> Halo是光环、日晕的意思，halo操作在原始的block上再外加一圈额外的信息，就类似在block之外再加了一层光环，起到了增加感受野的作用

结合代码理解：
```
attn = HaloAttention(
    dim = 3,           # 特征图维度C
    block_size = 8,    # 一幅图划分成块的大小 bs
    halo_size = 4,     # 块的感受野
    dim_head = 64,     # 每个头的维度 D/n_heads
    heads = 4          # 头的数量 n_heads
)
```
对于任意一张输入的特征图：[H，W，C]，先将其划分成为bs * bs大小的块，每个块就是一个token

每一块的大小为：[(H/bs) * (W/bs)，C]

于是得到q的输入：[(H/bs) * (W/bs)，bs * bs，C]

k、v的输入由halo操作得到，先对每一块block进行padding操作，halo后的大小为原来的四倍（长宽各扩充为2倍）

即：[(H/bs) * (W/bs)，C] -> [(2 * H/bs) * (2 * W/bs)，C]

所以k、v的输入：[(H/bs) * (W/bs)，bs * bs * 4，C]

于是通过Wq生成Q向量然后利用多头注意力机制：

[(H/bs) * (W/bs)，bs * bs，C] -> [(H/bs) * (W/bs)，bs * bs，D] -> [(H/bs) * (W/bs) * n_heads，bs * bs，D/n_heads]

K，V同理：[(H/bs) * (W/bs)，bs * bs * 4，C] -> [(H/bs) * (W/bs)，bs * bs * 4，D] -> [(H/bs) * (W/bs) * n_heads，bs * bs * 4，D/n_heads]

所以attention = softmax(Q * transpose(K)/sqrt(D/n_heads)) * V 
> 计算bs * bs个token 以及在对token进行halo操作之后的bs * bs * 4个token之间的相似性。

[(H/bs) * (W/bs) * n_heads，bs * bs，D/n_heads] * [(H/bs) * (W/bs) * n_heads，D/n_heads，bs * bs * 4] * [(H/bs) * (W/bs) * n_heads，bs * bs * 4，D/n_heads] = [(H/bs) * (W/bs) * n_heads，bs * bs，D/n_heads]

然后再映射回原来的维度C

[(H/bs) * (W/bs) * n_heads，bs * bs，D/n_heads] -> [(H/bs) * (W/bs)，bs * bs，D] -> [(H/bs) * (W/bs)，bs * bs，C] -> [H，W，C]

## 算法补充
![image](https://user-images.githubusercontent.com/65380826/141232949-6ce9bef4-d672-4993-955e-991f4ce94d75.png)
在论文中有一个Subsampling的操作，类似于CNN的pooling操作，使得输出的特征图的大小为原特征图的一半。当生成Q大小为[(H/bs) * (W/bs) * n_heads，bs * bs，D/n_heads]时，先下采样为[(H/bs/2) * (W/bs/2) * n_heads，bs * bs，D/n_heads]，这样最终的输出为[H/2，W/2，C]。这样的操作能够减少4倍的FLOPs，因为这里是对每个block的信息进行采样后计算，而不是对每个block内的信息进行计算，并且这样的操作也是不会影响模型精度的。

## 算法分析
Halo操作计算self-attention时，Q是感受野是每个块的大小，生成Q复杂度为（<img src="https://latex.codecogs.com/svg.image?hwC^{2}" title="hwC^{2}" />），K、V的感受野则是每个块的基础上扩充4倍之后的大小，生成K、V的复杂度为（<img src="https://latex.codecogs.com/svg.image?4hwC^{2}" title="4hwC^{2}" />），总复杂度为（<img src="https://latex.codecogs.com/svg.image?9hwC^{2}" title="9hwC^{2}" />），虽然K、V实现了增大感受野的功能，但缺点是复杂度明显提升了。
