# CenterNet详解
## 前言
CenterNet是在2019年论文Objects as points中提出，相比yolo，ssd，faster_rcnn依靠大量anchor的检测网络，CenterNet是一种Anchor-free的目标检测网络，在速度和精度上都比较有优势。

论文：[Objects as Points](https://arxiv.org/abs/1904.07850)

官方源码：https://github.com/xingyizhou/CenterNet

训练自己的数据集：https://blog.csdn.net/weixin_43509263/article/details/100799415
## 整体结构
![image](https://user-images.githubusercontent.com/65380826/128345498-370d800d-0e9f-4f8a-814d-528050a77c0f.png)
### 1、特征提取网络
![image](https://user-images.githubusercontent.com/65380826/128345779-9b9970da-8a7e-4578-843b-3c4e5f0387d7.png)
除了检测任务外，CenterNet还可以用于肢体识别或者3D目标检测等，因此CenterNet论文中提出了三种backbone的网络结构， 三种backbone准确度和速度如下：
* Resnet-18 with up-convolutional layers : 28.1% COCOAP and 142 FPS
* DLA-34 : 37.4% COCOAP and 52 FPS
* Hourglass-104 : 45.1% COCOAP and 1.4 FPS

无论哪种特征提取网络，最终得到的特征图大小均是输入图像的4倍下采样
### 2、Heat maps
Heat maps的大小为[80,128,128]，80代表分类类别，每一个类别都有一张Heat maps，每一张Heat maps上，若某个坐标处有物体目标的中心点，即在该坐标处产生一个keypoint(用高斯圆表示）

如下图左边是缩放后送入网络的图片，尺寸为512x512，右边是生成的heatmap图，尺寸为128x128（网络最后预测的heatmap尺度为128x128。其步骤如下：

> 1.将目标的box缩放到128x128的尺度上，然后求box的中心点坐标并取整，设为point
> 
> 2.根据目标box大小计算高斯圆的半径，设为R
> 
> 3.在heatmap图上，以point为圆心，半径为R填充高斯函数计算值。(point点处为最大值，沿着半径向外按高斯函数递减)

![image](https://user-images.githubusercontent.com/65380826/128347410-8924927a-2824-4e6c-bdc1-ae54a755405d.png)

* 注意：由于两个目标都是猫，属于同一类别，所以在同一张heatmap上。若还有一只狗，则狗的keypoint在另外一张heatmap上

关于高斯圆的半径确定，主要还是依赖于目标box的宽高，其计算方法为下图所示。 实际情况中会取IOU=0.7，即下图中的overlap=0.7作为临界值，然后分别计算出三种情况的半径，取最小值作为高斯核的半径r
![image](https://user-images.githubusercontent.com/65380826/128348331-a6432fc9-118f-4c06-90d6-0a7ac8b06124.png)

### 3、Center offset
Center offset的大小为[2,128,128]，2代表x,y的偏移量。之所以需要计算偏移量，是因为特征图的坐标值是在原图在4倍下采样之后取整得到的，这必然引入一定的误差。例如原图的groud truth的中心为(211,385)，在特征图上对应(52,96)，存在偏移量为(0.75,0.25)。Center offset的引入能更精准地恢复出原图分辨率下的数据。
### 4、Boxes size
Boxes size的大小为[2,128,128]，2预测框的宽、高。
## 损失函数
![image](https://user-images.githubusercontent.com/65380826/128350503-3c61cf11-0dcd-46a4-9c88-ec2985c956ae.png)

损失函数包括三部分：
> 1.heatmap的loss,改进的Focal loss
> 
> ![image](https://user-images.githubusercontent.com/65380826/128350917-3f60d0c0-b8ec-4727-9009-05a39e3cfe96.png)
> 
> 2.目标中心点偏移值loss,L1 Loss
> 
> ![image](https://user-images.githubusercontent.com/65380826/128350976-c093768b-aa4f-43be-9693-6ab9bd956afc.png)
> 
> 3.目标长宽预测loss,L1 Loss
> 
> ![image](https://user-images.githubusercontent.com/65380826/128351065-8cd35af5-b87b-4ffa-a5d9-df507a721c07.png)
## 缺点
1.在实际训练中，如果在图像中，同一个类别中的某些物体的GT中心点，在下采样时会挤到一块，也就是两个物体在GT中的中心点重叠了，CenterNet对于这种情况也是无能为力的，也就是将这两个物体的当成一个物体来训练(因为只有一个中心点)。同理，在预测过程中，如果两个同类的物体在下采样后的中心点也重叠了，那么CenterNet也是只能检测出一个中心点，不过CenterNet对于这种情况的处理要比faster-rcnn强一些的，具体指标可以查看论文相关部分。

2.有一个需要注意的点，CenterNet在训练过程中，如果同一个类的不同物体的高斯分布点互相有重叠，那么则在重叠的范围内选取较大的高斯点。
## 参考
[1]https://www.jianshu.com/p/d5d7cd7ad200

[2]https://blog.csdn.net/weixin_42398658/article/details/117514336
