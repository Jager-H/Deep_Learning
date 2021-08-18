# DETR
## 前言
第一个将 Transformer 成功整合为检测 pipeline 中心构建块的目标检测框架。基于Transformers的端到端目标检测，没有NMS后处理步骤、真正的没有anchor，且对标超越Faster RCNN。

论文：[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

代码：https://github.com/facebookresearch/detr

## 整体结构
<div align=center><img src="https://user-images.githubusercontent.com/65380826/129825728-d390a164-63ea-4a22-b42f-1c993f8655d5.png"></div>

### 1、Backbone
（1）Backbone部分是传统的CNN结构，输入image（H0，W0，3），输出feature map（H，W，C），其中H=H0/32，W=W0/32，C通常取2048。

（2）然后用1×1的卷积核维度压缩到d，（H，W，d）

（3）最后把宽高压缩成一个维度的feature map，即（H * W，d）
### 2、Encoder + Decoder
feature map + positional encoding构成transformer encoder的输入（positional encoding维度与feature map相同）

transformer encoder和transformer decoder的结构与Transformer基本相同相同，参考https://github.com/Jager-H/Deep_Learning/blob/main/Transformer/Transformer.md

### 3、Prediction Heads
