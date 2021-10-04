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
![image](https://user-images.githubusercontent.com/65380826/135787965-eeb24ff4-e122-40bb-9d64-fbe8e5c10001.png)

#### （2）打开labelme对数据集进行标注
直接输入labelme就可以打开

![image](https://user-images.githubusercontent.com/65380826/135788093-c8014e79-ef5e-4673-b5db-59b28959e203.png)

标注后保存为json文件保存在同一个文件夹（该文件夹与图片文件夹JPEGImages在同一目录下）

![image](https://user-images.githubusercontent.com/65380826/135788131-81d30b49-d7b6-46ea-bae3-ba5cd708ba4f.png)
#### （3）利用json文件批量转产生mask
找到json_to_dataset.py文件（通常在Anaconda文件所在位置下）

![image](https://user-images.githubusercontent.com/65380826/135788762-1d1e3b14-ded3-4fb4-ae48-67f3361c8e6e.png)
> 图中json_to_dataset_old.py为原来的json_to_dataset.py文件，新的json_to_dataset.py代码如下
```
import argparse
import base64
import json
import os
import os.path as osp

import PIL.Image
import yaml

from labelme.logger import logger
from labelme import utils

path = "E:/gesaer/gesaer_data/Json"
dirs = os.listdir(path)


def label(json_file, out_dir, label_name_to_value):
    data = json.load(open(json_file))
    if data['imageData']:
        imageData = data['imageData']
    else:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = utils.img_b64_to_arr(imageData)
    
    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name
    lbl_viz = utils.draw_label(lbl, img, label_names)

    PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
    utils.lblsave(osp.join(out_dir, 'label.png'), lbl)
    PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))

    with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
        for lbl_name in label_names:
            f.write(lbl_name + '\n')

    logger.warning('info.yaml is being replaced by label_names.txt')
    info = dict(label_names=label_names)
    with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
        yaml.safe_dump(info, f, default_flow_style=False)

    logger.info('Saved to: {}'.format(out_dir))
    


def main():
    logger.warning('This script is aimed to demonstrate how to convert the'
                   'JSON file to a single image dataset, and not to handle'
                   'multiple JSON files to generate a real-use dataset.')

    parser = argparse.ArgumentParser()
    parser.add_argument('json_file_dir')
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()
    label_name_to_value = {'_background_': 0}
    
    
    for json_file in dirs:
        if args.out is None:
            out_dir = osp.basename(json_file).replace('.', '_')
            out_dir = osp.join(osp.dirname(json_file), out_dir)
        else:
            out_dir = args.out
        if not osp.exists(out_dir):
            os.mkdir(out_dir)
    
        label(json_file, out_dir, label_name_to_value)
        

if __name__ == '__main__':
    main()
```

文件修改完成后，cd到json文件所在位置，运行
```
labelme_json_to_dataset outputdir
```
![image](https://user-images.githubusercontent.com/65380826/135789441-f8606fad-ff5c-40ff-8141-9e507d966adc.png)

然后在输出目录下找到mask相关文件

![image](https://user-images.githubusercontent.com/65380826/135789536-195f64ab-51b6-4851-b328-112043ac1a85.png)

#### （4）整理label.png文件
上述每一个文件下都包含五个文件，只有label.png是做语义分割时需要的mask图。新建label.py文件
```
import os
import shutil

inputdir = 'F:/dataset/Mask'
outputdir = 'F:/dataset/SegmentationClass'

for dir in os.listdir(inputdir):
    # 设置旧文件名（就是路径+文件名）
    oldname = inputdir + os.sep + dir + os.sep + 'label.png'  # os.sep添加系统分隔符

    # 设置新文件名
    newname = outputdir + os.sep + dir.split('_')[0] + '.png'

    shutil.copyfile(oldname, newname)  # 用os模块中的rename方法对文件改名
    print(oldname, '======>', newname)
```
于是得到了SegmentationClass文件下的mask图片
![image](https://user-images.githubusercontent.com/65380826/135790128-7fd54b20-4f49-436c-98ab-3f9f0c8ba05a.png)
#### （5）制作txt文件
txt.py代码如下：
```
from sklearn.model_selection import train_test_split
import os

imagedir = 'F:/dataset/SegmentationClass/'
outdir = 'F:/dataset/ImageSets/Segmentation/'

images = []
for file in os.listdir(imagedir):
    filename = file.split('.')[0]
    images.append(filename)

train, test = train_test_split(images, train_size=0.7, random_state=0)
val, test = train_test_split(test, train_size=0.2 / 0.3, random_state=0)

with open(outdir + "train.txt", 'w') as f:
    f.write('\n'.join(train))

with open(outdir + "val.txt", 'w') as f:
    f.write('\n'.join(val))

with open(outdir + "test.txt", 'w') as f:
    f.write('\n'.join(test))
```
最终得到ImageSets/Segmentation下的train.txt，test.txt，val.txt文件

![image](https://user-images.githubusercontent.com/65380826/135795791-65f91998-2ee0-4ed4-8a21-89b18045cbb5.png)
### 3、数据迁移
以上三个文件夹拷入到路径 ./deeplabv3/dataloaders/Mydataset 下
## 四、训练模型
### 1、修改文件
#### (1)在mypath.py中添加自己的数据集名称与路径
![1](https://user-images.githubusercontent.com/65380826/135804874-8acc6605-2d14-4c50-a1b7-a8502c8620eb.png)
#### (2)在train.py中添加自己的数据集名称
![2](https://user-images.githubusercontent.com/65380826/135804888-e6fd0efa-ac67-4bb1-b8c4-a4b2d2c227f5.png)
#### (3)在dataloaders/__init__.py中第一行添加自己的数据集名称,在make_data_loader函数内添加自己数据集的加载方式
![3](https://user-images.githubusercontent.com/65380826/135807171-8f8381af-5e75-4fe5-ae4f-eb412c006474.png)
#### (4)在dataloaders/utils.py中第24行添加代码(n_classes为分割类别,含背景),在76行左右添加代码，设置每一类别的颜色显示。
![5](https://user-images.githubusercontent.com/65380826/135807326-4b0c4cb6-4b02-48e7-a650-d8901d505797.png)

![4](https://user-images.githubusercontent.com/65380826/135807314-1a345668-d091-4881-b598-e5623de005b7.png)
#### (5)在dataloaders/datasets目录下添加文件(复制pascal.py为Mydataset.py)
![6](https://user-images.githubusercontent.com/65380826/135807340-38a613af-92ea-4f87-a4d0-cfd7c307fd66.png)
#### (6)修改Mydataset.py里面的类别数和数据集名称
![7](https://user-images.githubusercontent.com/65380826/135807354-133fe108-4878-47f1-9b14-5ca684bbb276.png)
### 2、开始训练
```
python train.py --backbone mobilenet --lr 0.007 --workers 1 --epochs 50 --batch-size 8 --gpu-ids 0 --checkname deeplab-mobilenet
```

* 1、第一次运行时会先下载mobilenet的权重文件,如果下载到一半因为网络问题中断,再次运行会报错
```
RuntimeError: unexpected EOF, expected 21636 more bytes. The file might be corrupted.
```
> 解决办法:卸载原来不完整的权重文件(文件路径在终端窗口有提示),重新运行即可.

* 2、如果Pillow版本太高,可能会报错
```
TypeError: an integer is required (got type tuple)
```
> 解决办法:降低Pillow安装包的版本

## 五、测试模型
### 1、新建测试文件testdemo.py
```
import argparse
import os
import numpy as np
import time

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid, save_image

def main():

    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--in-path', type=str,  default='/home/daiteng/PycharmProjects/deeplabv3/dataloaders/Mydataset/test',
                        help='image to test')
    # parser.add_argument('--out-path', type=str, required=True, help='mask image to save')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--ckpt', type=str, default='deeplab-resnet.pth',
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes','invoice'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    model_s_time = time.time()
    model = DeepLab(num_classes=args.num_classes,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    model_u_time = time.time()
    model_load_time = model_u_time-model_s_time
    print("model load time is {}".format(model_load_time))

    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])
    for name in os.listdir(args.in_path):
        s_time = time.time()
        image = Image.open(args.in_path+"/"+name).convert('RGB')

        # image = Image.open(args.in_path).convert('RGB')
        target = Image.open(args.in_path+"/"+name).convert('L')
        sample = {'image': image, 'label': target}
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

        model.eval()
        if args.cuda:
            tensor_in = tensor_in.cuda()
        with torch.no_grad():
            output = model(tensor_in)

        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()),
                                3, normalize=False, range=(0, 255))
        save_image(grid_image,args.in_path+"/"+"{}_mask.png".format(name[0:-4]))
        u_time = time.time()
        img_time = u_time-s_time
        print("image:{} time: {} ".format(name,img_time))
        # save_image(grid_image, args.out_path)
        # print("type(grid) is: ", type(grid_image))
        # print("grid_image.shape is: ", grid_image.shape)
    print("image save in in_path.")
if __name__ == "__main__":
   main()
```
### 2、运行测试命令
