# Faster-RCNN详解
## 前言

## 整体结构
![image](https://user-images.githubusercontent.com/65380826/127453396-ecf0f2af-6cb8-46be-9f86-12431f3230ae.png)
### 1、特征提取网络
图中提取特征的网络采用VGG16，最终输出Feature Map的大小为[1,512,M/16,N/16],可以换成其他网络（例如ZF、Resnet等）。
### 2、Region Proposal Networks(RPN)
![image](https://user-images.githubusercontent.com/65380826/127454439-7aaebb79-aabb-4e7c-baf5-c571b9d569ff.png)

* a.Feature Map经过3×3卷积之后，输出为[1,256,H,W] (H=M/16,W=N/16)
* b1.经过上面1×1卷积后，输出为[1,2×9,H,W] （这里是一个二分类，判断每个网格的9个Anchor里面是否含有物体，即foreground or background），
然后通过reshape输出为[1,2,9×W.H]，便于softmax分类，分类完之后再reshape成[1,18,H,W]
* b2.经过下面的1×1卷积后，输出为[1,4×9,H,W] （输出每个网格的9个Anchor的坐标的偏移量、宽高的缩放尺度）
* c.上面一共得到9×H×W个预测框，经过Proposal筛选后生成m个得分比较高的proposal并输出其左上角和右下角坐标，输出为[m,4+1]。细节如下图所示：
![image](https://user-images.githubusercontent.com/65380826/127457053-9f7125e8-dafd-4932-9171-182f4a34efce.png)
### 3、RoI pooling
![image](https://user-images.githubusercontent.com/65380826/127458099-ec44d4d9-c7c8-4f70-9637-2e6b08d3a2f5.png)

ROI pooling对每个proposal提取固定尺寸的特征图，输出为[m,512,7,7]。具体操作为：
* （1）根据输入image，将ROI映射到feature map对应位置（取整）；

* （2）将映射后的区域划分为相同大小的sections（如7*7）；

* （3）对每个sections进行max pooling操作；
### 4、全连接层
![image](https://user-images.githubusercontent.com/65380826/127458676-b6b42858-10cc-4b3f-941a-7a86101e51c5.png)

* bbox_pred的输出为[m,4×(num_classes+1)]
* cls_prob的输出为[m,num_classes+1] 
* 1代表background
## post-processing后处理
为了调整边界框，需要考虑概率最大的类别的 proposals. 忽略概率最大值为 background class 的proposals.
当得到最终的 objects 时，并忽略被预测为 background 的结果，采用 class-based NMS. 主要是通过对 objects 根据类别class 分组，然后根据概率排序，并对每个独立的分组采用 NMS 处理，最后再放在一起.
最终得到的 objects 列表，仍可继续通过设定概率阈值的方式，来限制每个类的 objects 数量。
## 参考
https://blog.csdn.net/u011746554/article/details/74999010
