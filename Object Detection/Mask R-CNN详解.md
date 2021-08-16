# Mask R-CNN详解
## 前言
Mask-RCNN
## 整体结构
![image](https://user-images.githubusercontent.com/65380826/127629383-abde3b54-99d9-4ef1-a084-ec5bec62790b.png)
### 1、特征提取网络
Mask R-CNN的特征提取网络为ResNet+FPN，通过输出图像[1,3,1024,1024]提取出5个不同尺度的特征图，分别为P2[1,256,256,256]、P3[1,256,128,128]、P4[1,256,64,64]、P5[1,256,32,32]、P6[1,256,16,16]
* 图中只画了4个会用在RoIAlign部分的特征图，缺少了P6，P6只用于RPN部分。
### 2、RPN
![image](https://user-images.githubusercontent.com/65380826/127631218-6183b5b5-e0d4-4fed-adb0-7b3178fc9905.png)

（1）RPN的输入是5个特征图[P2、P3、P4、P5、P6]，分三条线路先得到下面三个输出：
> * rpn_class_logits[1,261888,2]：代表是物体FG还是背景BG
> * rpn_probs[1,261888,2]：对rpn_class_logits进行softmax，得到是FG/BG的概率
> * rpn_bbox[1,261888,4]：预测框相对于anchor的偏移量
> 
> 其中261888=3×（256×256+128×128+64×64+32×32+16×16）
> 
![image](https://user-images.githubusercontent.com/65380826/127633572-658a78e8-edec-4657-9a00-cfacee2fd411.png)

（2）然后根据上面的三个输出产生1000个region proposal，即rpn_rois[1,1000,4]
### 3、RoIAlign
![20190429102311571](https://user-images.githubusercontent.com/65380826/127644249-ef24ed69-b175-4010-b997-33515fc7b393.png)

RoIAlign的输入是rpn_rios和四个特征图[P2、P3、P4、P5]，输出为[1,1000,256,7,7]具体操作为：
* a.根据输入image，将ROI映射到feature map对应位置（不取整）；

* b.将映射后的区域划分为相同大小的sections（如7 * 7）；

* c.再把每个section分成4个bin，bin的中心为采样点(通常是浮点数)，每个采样点的值利用双线性插值（整数）替换；

* d.然后对每个section内的四个采样点进行maxpooling or averagepooling，就可以得到最终的ROIAlign的结果
> 值得一提的是，这1000个region proposal需要在4种不同的特征图中切割出RoI区域，所以需要知道每个region proposal是在哪个特征图上进行切割。实际上，我们会选择最合适的尺度的特征图来切ROI。具体来说，我们通过一个公式来决定宽w和高h的ROI到底要从哪个Pi（i=2,3,4,5） 来切：
> 
> ![image](https://user-images.githubusercontent.com/65380826/127645652-fd871d39-7a06-4b05-aaf6-efa08556cd08.png)
> 
> 这里224表示用于预训练的ImageNet图片的大小。 k0 表示面积为 w×h=224×224 的ROI所应该在的层级。作者将 k0 设置为4，也就是说 w×h=24×224 的ROI应该从P4中切出来。假设ROI的scale小于224（比如说是112×112）， k=k0-1=4-1=3，就意味着要从更高分辨率的P3中产生。另外，k值会做取整处理，防止结果不是整数。



### 4、classifier + detection + mask
![image](https://user-images.githubusercontent.com/65380826/127643942-c8500de6-4faa-4875-83d7-86e751ff6a3d.png)
（1）classifier
得到三个输出：
> * mrcnn_class_logits (1000,81) 结果是这1000个ROI所属物体的类别的score
> * mrcnn_class (1000,81) 结果是这1000个ROI所属物体的类别的概率
> * mrcnn_bbox (1000,81,4) 结果是1000个ROI对每个类别所产生的bbox的偏移,这里不是真正的坐标,只是针对RPN结果的delta

（2）detection

输入是rpn_rois、mrcnn_class、mrcnn_bbox，输出是detection_boxes[16,6], 16是最终筛选出来的预测框，每个框含有的数据格式是[y1, x1, y2, x2, class_id, score]
> 先确定ROI类别：对输入的1000个ROIs,根据mrcnn_class (1000,81）找到probs的最大值
> 然后调整ROI坐标：认为mrcnn_bbox (1000,81,4)类别最大的对应的坐标值是ROI的坐标值
> 这时候的ROI在box的坐标值上已经调整,然后每个ROI也具有类别信息
> * 用最大类别的Box delta来调整ROI的box
> * 用最大类别的score值来去除一些ROI,同时直接去除背景ROI
> * 对剩下的ROI,按照类别做NMS，得到最终的detection_boxes[16,6]

（3）mask

输入是detection_boxes，输出[16,81,28,28]的sigmoid分割值。具体操作如下图：

![image](https://user-images.githubusercontent.com/65380826/127651530-abe02a4f-a3ad-4ae3-85b8-45356dbb4d1c.png)
## 参考
[1]https://blog.csdn.net/hnshahao/article/details/81231211

[2]https://zhuanlan.zhihu.com/p/37998710
