# CNN入门论文
## 前言

![image](https://user-images.githubusercontent.com/65380826/125913841-5aefc54a-1d64-4bdb-953f-1fd3ec13609f.png)

* LeNet：CNN的开山鼻祖，也是识别手写体的经典论文，非常适合CNN入门者阅读。

* AlexNet：ILSVRC-2012的大赛冠军，也是CNN的历史性突破，后续多种网络都是基于此进行改进的,非常值得一读。（使用了ReLU做激活函数；重叠池化；Dropout；数据增强）

* GoogLeNet(InceptionV1)：ILSVRC-2014冠军，InceptionV1通过增加网络的宽度减少的训练参数量，同时提高了网络对多种尺度的适应性。InceptionV2-V4都是在在V1的基础上作改进，使网络更深，参数更少

* VGG：ILSVRC-2014亚军，通过增加网络的深度提升网络的性能，证明更深的网络层数是提高精度的有效手段。

* ResNet：更深的网络极易导致梯度弥散，从而导致网络无法收敛，ResNet可以很好的缓解深层网络带来的梯度消失的问题，其残差块的设计结构不可谓不妙

* DenseNet：通过特征重用来大幅减少网络的参数量，又在一定程度上缓解了梯度消失问题

* SENet：主要是从特征通道之间的关联关系入手提升网络性能，能突出重要的特征，抑制不重要的特征（需要注意的是这不是一个完整的网络结构，而只是一个模块，能嵌套进其他的网络中）

* EfficientNet：之前所有的网络都是在网络的宽度或者深度、图像分辨率中某一个维度上进行改进从而提升CNN的性能，而该网络平衡网络的所有维度（宽度、深度和图像分辨率），最大限度地提高网络整体性能。
## 1、LeNet
论文原文链接：[Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

中文版参考： [LeNet论文翻译——中文版](http://www.cxyzjd.com/article/qq_34288751/113575650)

网络结构：

![ScreenClip](https://user-images.githubusercontent.com/65380826/125878680-1beff245-6a25-436a-baa3-e6a39e6d8091.png)


## 2、AlexNet
论文原文链接：[ImageNet Classification with Deep Convolutional Neural Networks](https://pan.baidu.com/s/1lp5Tqu-hNzz0ru6QAO1New)[百度网盘提取码：xcvx]

中文版参考：[AlexNet论文翻译——中文版](http://noahsnail.com/2017/07/18/2017-07-18-AlexNet%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)

网络结构：

![ScreenClip  1](https://user-images.githubusercontent.com/65380826/125878909-4b853b50-db2f-4e06-a00d-1a9da9e6f526.png)

直观一点：

![image](https://user-images.githubusercontent.com/65380826/125878969-7d95af17-6da3-4968-805e-38862ced849e.png)

## 3、Inception & GoogLeNet
论文原文链接：[Going Deeper with Convolutions](https://pan.baidu.com/s/13i6Euh4hjph0a1-TgWl0rw)[百度网盘提取码：wk3p]

中文版参考: [GoogLeNet论文翻译——中文版](http://noahsnail.com/2017/07/21/2017-07-21-GoogleNet%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)

网络结构： 
### InceptionV1

![image](https://user-images.githubusercontent.com/65380826/125879463-2e3474b2-e231-4ab4-bc7d-529e2a87c0ca.png)

### InceptionV2、V3、V4用到的模块

<img src="https://user-images.githubusercontent.com/65380826/125882098-8dc476fd-2528-4cd7-8e5b-ca333f4351e0.png" width="300px"><img src="https://user-images.githubusercontent.com/65380826/125880006-baa8f923-e880-4459-b623-449af589d91d.png" width="300px"><img src="https://user-images.githubusercontent.com/65380826/125880019-5cfd3c86-0c4d-4376-809f-a9e0374e7f9a.png" width="300px">

## 4、VGG
论文原文链接：[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://pan.baidu.com/s/1Eh8S9Ce0ho2OxcEmcvHfzA)[百度网盘提取码：6eue]

中文版参考: [VGG论文翻译——中文版](http://noahsnail.com/2017/08/17/2017-08-17-VGG%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)

网络结构： 

![image](https://user-images.githubusercontent.com/65380826/125882469-55510d38-0925-4920-8977-d2b035670614.png)

## 5、ResNet
论文原文链接：[Deep Residual Learning for Image Recognition](https://pan.baidu.com/s/1ZTsN3Yrvv3fKHCEWR3MX3Q)[百度网盘提取码：p34o]

中文版参考: [ResNet论文翻译——中文版](http://noahsnail.com/2017/07/31/2017-07-31-ResNet%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)

![ScreenClip  4](https://user-images.githubusercontent.com/65380826/125913496-ffdc177f-fdbd-4a83-80e6-2f2544adf8d3.png)

## 6、DenseNet
论文原文链接：[Densely Connected Convolutional Networks](https://pan.baidu.com/s/1HsaHA8sGGSwJ3fF4k7liSQ)[百度网盘提取码：g5pk]

中文版参考: [DenseNet论文翻译——中文版](https://alvin.red/2017/10/07/densenet/)

![ScreenClip  2](https://user-images.githubusercontent.com/65380826/125913344-ea7f0197-6a78-4fe3-a9fe-6a2f9bd7667d.png)

## 7、SENet
论文原文链接：[Squeeze-and-Excitation Networks](https://pan.baidu.com/s/1iTTe1GIGHdsDJz-BP6UYVw)[百度网盘提取码：snqd]

中文版参考: [SENet论文翻译——中文版](http://noahsnail.com/2017/11/20/2017-11-20-Squeeze-and-Excitation%20Networks%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)

![ScreenClip  3](https://user-images.githubusercontent.com/65380826/125913570-a325f7d1-0203-466b-98ec-323c74409a75.png)

## 8、EfficientNet
论文原文链接：[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://pan.baidu.com/s/1197Db0TcA4AnQcExzhbDCA)[百度网盘提取码：ww98]

中文版参考: [EfficientNet论文翻译——中文版](https://blog.csdn.net/weixin_42464187/article/details/100939130)

![image](https://user-images.githubusercontent.com/65380826/125913108-660287af-7a9f-4fc3-b21a-2a30d8557736.png)
