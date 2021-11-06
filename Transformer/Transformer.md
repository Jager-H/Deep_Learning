# Transformer
## 前言
论文：[Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
* 宏观理解：模型看作黑箱，输入一段信号，输出另一段信号。黑箱由编码组件、解码组件以及之间的连接组成。
<div align=center>
<img src="https://user-images.githubusercontent.com/65380826/128860641-5b1900f9-1982-409d-8277-207362891118.png" width="780" heigth="505">
</div>

<div align=center>
<img src="https://user-images.githubusercontent.com/65380826/128861699-70fc3ea3-b4a4-4e64-9934-e5673da12e63.png" width="580" heigth="305">
</div>

## 整体结构
<div align=center>
<img src="https://user-images.githubusercontent.com/65380826/128860531-b72f0447-2aa1-4ba5-8165-a960f48851bb.png" width="380" heigth="505">
</div>

> 首先输入经过编码器Encoders，然后通过解码器Decoders得到输出。其中，Encoders和Decoders的数量保持一致，都为N（这是一个超参数，论文中取6）
> 
> 所有的encoders结构完全相同，但参数并不相同。所有decoders结构完全相同，参数也不相同。
<div align=center>
<img src="https://user-images.githubusercontent.com/65380826/128861594-f2f0409c-bb94-43c8-a830-9a5489e477e5.png" width="380" heigth="505">
</div>

### 输入部分
#### 1、Input Embedding
将每个单词（字）转化为一个512维的词向量，一句话sentence如果有3个单词（字），那就形成一个词列表（Size为3×512）

#### 2、Positional Encoder
<div align=center>
<img src="https://user-images.githubusercontent.com/65380826/128863645-3927a1a3-cdcf-48c8-8fe0-80a860440094.png" width="580" heigth="505">
</div>

* 偶数位置（0，2，4……）使用sin，奇数位置（1，3，5……）使用cos
#### 3、Input Embedding+Positional Encoder
每个词向量与位置编码向量相加便得到了transformer的输入
<div align=center>
<img src="https://user-images.githubusercontent.com/65380826/128863956-3bf39382-16db-4e83-9b66-5c74f7873411.png" width="580" heigth="505">
</div>

> 为什么需要位置编码？位置编码为什么有效？

> 位置编码与输入为什么相加？拼接是否可以？
### Encoder部分
#### 1、Muti-Head Attention
在self-attention中，每个单词有3个不同的向量，它们分别是Query向量，Key向量和Value向量，长度均是64。它们是通过3个不同的权值矩阵由嵌入向量乘以三个不同的权值矩阵得到，其中三个矩阵的尺寸也是相同的，均是512×64。

<div align=center>
<img src="https://user-images.githubusercontent.com/65380826/128866281-a6b5ac9b-c383-4bc3-b830-545bd3466cfe.png" width="580" heigth="505">
</div>

Attention公式为

![image](https://user-images.githubusercontent.com/65380826/128865050-12084aa5-4901-4b3b-af39-3c7a6673efe6.png)

* Q和K的相乘本质是计算不同token之间的相似度，而V则起到一个加权的作用。
并行计算时如下图所示
<div align=center>
<img src="https://user-images.githubusercontent.com/65380826/128866405-f75a57af-c32a-4cf7-9a7c-cd627e9c48ef.png" width="580" heigth="505">
</div>

所谓Muti-Head Attention，就是同一句话用多套参数得到不同的Attention，Attention用Zi表示。然后把所有的Zi拼接在一起再乘以权重矩阵W0
> “多头”的数量是分组得到的。假如输入1个句子里面有64个单词，那么就存在64个不同的tokens，输入大小为（64，512），如果将其分为8组（d_head=8），每一组大小为（8，512），对每组进行attention的计算。这样一来，模型就可以关注不同方面的信息。
<div align=center>
<img src="https://user-images.githubusercontent.com/65380826/128866941-aa0ca339-c1c7-4e31-9346-632497e5d5cf.png" width="580" heigth="505">
</div>

#### 2、Add & Normalize
Add就是Muti-Head Attention的输出结果加上其输入（相当于一个残差结构）

通常使用Layer Normalize而不用Batch Normalize
> 为什么使用Layer Normalize而不用Batch Normalize？
#### 3、Feed Forword Network
Feed Forword是一个简单的全连接层，共包含两层，第一层将输入的512维向量转成2048维，第二层再把2048维转回512维。
### Decoder部分
decoder分为Autoregressive decoder和Non-Autoregressive decoder
* AT decoder：输入与输出长度相等，如输入“机器学习”（长度为4），输出“机器学习”（长度为4）
* NAT decoder：输入与输出不相等，如输入“机器学习”（长度为4），输出“machining learning”（长度为2）

以AT decoder为例
<div align=center>
<img src="https://user-images.githubusercontent.com/65380826/128965368-bf712b0e-4981-46b4-9e82-b2a6f04a8738.png" width="580" heigth="505">
</div>


#### 1、Output Embedding+Positional Encoder
train阶段：将所有标签作为output

test（inference）阶段：先将begin这个special token作为output，然后将得到的输出作为下一个词的output

得到output之后与encoder一样都需要进行Embedding和Positional Encoder
#### 2、Masked Muti-Head Attention
为了使模型在train阶段和inference阶段接收的数据相同，由于inferece阶段计算当前词的attention的时候只能接收到之前词的信息，而train阶段会同时接收到所有词的信息，因此在train阶段需要用mask屏蔽掉后面词汇的信息（即只计算当前词的Q向量×之前词的K向量，而不是当前词的Q向量×所有词的K向量）。
#### 3、Cross attentionn
Cross attention是连接decoder与encoder之间的桥梁，也就是下图所示部分，输入来自decoder的输出的Q向量和encoder输出的K向量、V向量
<div align=center>
<img src="https://user-images.githubusercontent.com/65380826/128965308-a2e678fa-301a-45c0-bf72-0699d900c5ad.png" width="180" heigth="105">
</div>

* 在论文中，decoder中每一层都是利用encoder中最后一层的输出，但这不是必须的，可以设计其他的结构。
## 模型训练
在训练过程中，模型得到输入数据之后，会输出一个字典Output Vocabulary，也就是所有可能结果的概率分布distribution。数据的label也会编码成Groud truth，如图所示
<div align=center>
<img src="https://user-images.githubusercontent.com/65380826/128985115-2740ecfe-98c4-4f96-bcb0-4217549dbb56.png" width="580" heigth="505">
</div>

Output Vocabulary和Groud truth之间的损失函数采用Cross Entropy

* 值得注意的是像前文提到的mask机制，计算“begin”信号的attention时，需要屏蔽掉未来信息（“机器学习”），计算“机”信号的attention时，也需要屏蔽掉未来信息（“器学习”），以此类推。
<div align=center>
<img src="https://user-images.githubusercontent.com/65380826/128986135-6836a325-7471-4b54-b492-ff3ea923cce5.png" width="580" heigth="505">
</div>

## 参考
https://zhuanlan.zhihu.com/p/48508221
https://blog.csdn.net/longxinchen_ml/article/details/86533005
