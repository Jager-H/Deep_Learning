# pytorch-yolov4训练自己的数据集
## 一、下载项目代码
```
git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git
```
## 二、配置环境
根据requirements.txt的参数配置对应的环境
```
numpy==1.18.2
torch==1.4.0
tensorboardX==2.0
scikit_image==0.16.2
matplotlib==2.2.3
tqdm==4.43.0
easydict==1.9
Pillow==7.1.2
skimage
opencv_python
pycocotools
```
## 三、下载预训练权重文件
在该目录下新建文件夹weight，用于存放权重文件yolov4.weights 和 yolov4.conv.137.pth
## 四、测试模型
输入命令：
```
python demo.py -cfgfile ./cfg/yolov4.cfg -weightfile ./weight/yolov4.weights -imgfile ./data/dog.jpg
```
运行结果如下：

![image](https://user-images.githubusercontent.com/65380826/125452982-0ae03bf6-7a2b-46d0-abee-140395bd87f8.png)

在该目录下得到predictions.jpg

![image](https://user-images.githubusercontent.com/65380826/125453178-17ab55b4-e480-41c7-b737-e00a923c5e70.png)

## 五、制作数据集
1、我用的是作者的硬币数据集（打开Use_yolov4_to_train_your_own_data.md，翻到最下面找到作者自己制作的coins数据集），可以把自己的数据集按照这种形式存放，其中train.txt的的制作方法可参考这篇[博客](https://github.com/Jager-H/Jager-H.github.io/blob/main/blogs/%E5%81%9A%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E9%A1%B9%E7%9B%AE%E6%97%B6%E6%94%B6%E5%88%B0%E4%B8%80%E5%A0%86%E4%B8%AD%E6%96%87%E5%91%BD%E5%90%8D%E7%9A%84%E6%95%B0%E6%8D%AE%E9%9B%86%E6%80%8E%E4%B9%88%E5%A4%84%E7%90%86%EF%BC%9F.md)

![image](https://user-images.githubusercontent.com/65380826/125453320-b0eef5dd-566b-47c9-b1f4-f8e500237d19.png)

2、在data文件夹下新建JPEGImages，然后把下载好的数据集图片复制到该文件夹下（图片路径像这样./data/JPEGImages/P00524-151911.jpg）
3、把coins数据集内的train.txt复制到data文件夹下，可以忽略我的val.txt

![image](https://user-images.githubusercontent.com/65380826/125453436-299788de-0bb1-47b2-a2fb-bacf6c9a66d7.png)

4、在data文件下新建文件coins.names，内容如下:

![image](https://user-images.githubusercontent.com/65380826/125453477-6f97e586-14e1-4a49-a24a-f7ee627d1f17.png)

## 六、训练模型
1、在cfg.py里修改参数，Cfg.use_darknet_cfg一定要修改成Fasle，batch和subdivision根据自己电脑的配置情况修改

![image](https://user-images.githubusercontent.com/65380826/125453531-c3045a20-0896-40f4-8984-00ad25916154.png)

2、修改dataset.py代码，否则训练时可能会报错

![image](https://user-images.githubusercontent.com/65380826/125453558-b73828b4-34cc-4171-9665-1e9225382b6d.png)

3、输入训练命令：
```
python train.py -l 0.001 -g 0 -pretrained ./weight/yolov4.conv.137.pth -classes 3 -dir ./data/JPEGImages/ -train_label_path ./data/train.txt
```
训练结束后模型保存在文件夹./checkpoints里面

![image](https://user-images.githubusercontent.com/65380826/125453645-cee75da1-d389-4341-805e-7bf20bb0e89a.png)

##七、测试模型
1、修改models.py代码如下:

![image](https://user-images.githubusercontent.com/65380826/125453701-e08a8ff6-e909-48ab-a64f-829b0a3ba647.png)

2、输入测试命令：
```
python models.py 3 ./checkpoints/Yolov4_epoch100.pth data/JPEGImages/P00524-152409.jpg 608 608 data/coins.names
#python models.py num_classes weightfile imagepath namefile
```
测试结果保存在目录下的predictions.jpg

![predictions](https://user-images.githubusercontent.com/65380826/125454014-51891675-0e4e-411c-b731-277d972a2ca8.jpg)

