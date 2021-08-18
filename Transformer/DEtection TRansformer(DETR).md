# DETR
## 前言
第一个将 Transformer 成功整合为检测 pipeline 中心构建块的目标检测框架。基于Transformers的端到端目标检测，没有NMS后处理步骤、真正的没有anchor，且对标超越Faster RCNN。

论文：[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

代码：https://github.com/facebookresearch/detr

## 整体结构
<div align=center><img src="https://user-images.githubusercontent.com/65380826/129825728-d390a164-63ea-4a22-b42f-1c993f8655d5.png"></div>

### Backbone
