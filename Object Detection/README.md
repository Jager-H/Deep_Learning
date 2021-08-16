# 目标检测框架
## 前言

![image](https://user-images.githubusercontent.com/65380826/125915085-08a6780b-1d65-4fdf-9731-217eef616275.png)

* R-CNN：将CNN引入目标检测领域的开山之作，提议区域（proposal region）的生成方法为选择性搜索（selective search），所有的提议区域都要通过网络，因此检测速度慢，无法广泛应用于实际
* Fast R-CNN: 引入兴趣区域池化（Region of Interest pooling，RoI池化），从而不需要将所有提议区域送入网络，在一定程度上提高了检测速度
* Faster R-CNN: 生成提议区域的方法从选择性搜索变成了区域提议网络（Region Proposal Network，RPN），RPN能够学习到如何生成高质量的提议区域，从而在减少提议区域数量的情况下也能保证目标检测的精度https://blog.csdn.net/u011746554/article/details/74999010
* YOLOv1
* 
## R-CNN

论文：[Rich feature hierarchies for accurate object detection and semantic segmentation]

## YOLO系列
https://zhuanlan.zhihu.com/p/136382095
