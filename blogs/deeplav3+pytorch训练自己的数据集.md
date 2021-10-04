# deeplav3+pytorch训练自己的数据集
## 一、下载项目代码
```
git clone https://github.com/jfzhang95/pytorch-deeplab-xception.git
```
## 二、配置环境
```
python==3.7
torch==1.9.1
tensorboardX==2.4
Pillow==7.0.0 #pillow版本不对会报错
opencv_python
pycocotools
tqdm
```
## 三、准备数据
### 1、数据集格式如下
```
- ImageSets
	- Segmentation
		- train.txt
		- trainval.txt
		- val.txt
- JPEGImages
- SegmentationClass
```
ImageSets/Segmentation文件夹内存放txt文件

![Screenshot from 2021-10-03 21-50-36](https://user-images.githubusercontent.com/65380826/135756618-875d39c9-480d-4c92-b1b5-c152f295065b.png)

其中train.txt如下所示

![Screenshot from 2021-10-03 21-51-15](https://user-images.githubusercontent.com/65380826/135756630-758f4202-44b9-4d74-bf0b-81d12be28c74.png)

JPEGImage文件夹内存放原图

![Screenshot from 2021-10-03 21-48-49](https://user-images.githubusercontent.com/65380826/135756468-0abd049a-2160-4170-9841-dfe816741930.png)

SegmentationClass存放原图的掩膜mask

![Screenshot from 2021-10-03 21-50-09](https://user-images.githubusercontent.com/65380826/135756583-ef05d730-9b56-4eaa-806c-5990742cab33.png)
### 2、数据集制作流程（Win10下制作）
#### （1）打开Anaconda安装labelme
```
pip install labelme==3.16.2
```

#### （2）
## 四、训练模型

## 五、测试模型
