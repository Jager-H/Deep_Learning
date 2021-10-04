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

## 四、训练模型

## 五、测试模型
