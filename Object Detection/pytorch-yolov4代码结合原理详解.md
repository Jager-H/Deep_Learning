# pytorch-yolov4代码结合原理详解
## 前言
yolov4算法在yolov3的基础上，广泛采用各种能提高检测精度的tricks，同时保证了检测速度。可以说在同等精度的模型下，它速度最快，同等速度的模型下，它精度最高，真正做到了精度与速度的完美结合。
yolov4算法的开源代码是C++语言实现的（AB大神Darknet版本的源码实现: https://github.com/AlexeyAB/darknet ），然而现在大部分人研究yolov4时是基于python语言的pytorch框架。
故本文针对pytorch版本实现的yolov4进行分析（[代码](https://github.com/Tianxiaomo/pytorch-YOLOv4)出自[Tianxiaomo](https://github.com/Tianxiaomo/pytorch-YOLOv4/commits?author=Tianxiaomo)），同时也记录一下阅读代码过程中的问题。主要结合原理分析以下三个部分：
* 模型的搭建
* 模型的训练
* 测试图片
## 1、模型的搭建 (models.py)
[下图](https://blog.csdn.net/bai666ai/article/details/107087696)是yolov4模型的结构图
![image](https://user-images.githubusercontent.com/65380826/126728326-e0d5d636-4fa3-41b3-8ac5-82220266ee3a.png)
* Backbone:CSPDarknet53。该部分主要用于提取图像特征，通常会选用特征提取能力强的模型，且模型不能太大，否则影响检测速度。
* Neck：SPP+PAN。该部分主要用于融合不同尺度的特征，以提高网络检测不同尺寸目标的能力。
* Head：YOLO Head。该部分主要是输出指定大小的特征。

### yolov4代码
```python
class Yolov4(nn.Module):
    def __init__(self, yolov4conv137weight=None, n_classes=80, inference=False):
        super().__init__()

        output_ch = (4 + 1 + n_classes) * 3  #对应（4个坐标信息+1个置信度信息+n个类别信息）*3个anchor

        # backbone
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()
        # neck
        self.neek = Neck(inference)
        # yolov4conv137
        if yolov4conv137weight:
            _model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.neek)
            pretrained_dict = torch.load(yolov4conv137weight)
            model_dict = _model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            _model.load_state_dict(model_dict)
        
        # head
        self.head = Yolov4Head(output_ch, n_classes, inference)
    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        x20, x13, x6 = self.neek(d5, d4, d3)

        output = self.head(x20, x13, x6)
        return output
```
>分析：class Yolov4()部分非常直观，包含backbone、neck、head三个部分，输入是images，输出是bboxes_pred，是一个包含三个大小分别为
[batchsize,output_ch,76,76],[batchsize,output_ch,38,38],[batchsize,output_ch,19,19]的矩阵的list，
其中batchsize是一次输入图像的个数，ouput_ch是输出的通道个数，含义在代码注释中。

>对于backbone而言，输入是一张[608,608,3]的图片，DownSample1()对应图片中的CBM+CSP1，DownSample2()对应CSP2，DownSample3()对应CSP8，这里输出为[76,76,256]，的DownSample4()对应下一个CSP8，
这里输出为[38,38,512]，DownSample5()对应CSP4，这里输出为[19,19,1024]。三个输出分别对应不同特征图的大小，分别用来检测小、中、大型物体。

>对于Neck而言，输入是backbone的三个输出，得到对应输出为[76,76,128],[38,38,256],[19,19,512]（即代码中的x20，x3，x16）

>对于Head而言，输入是Neck的三个输出，得到最终输出[76,76,output_ch],[38,38,output_ch],[38,38,output_ch]，以[76,76,output_ch]为例，其含义是把图像分成了76×76大小的grid cell（网格），
每个grid cell上生成3个bounding box（预测框），每个预测框包含其位置信息（<img src="https://latex.codecogs.com/svg.image?t_{x}" title="t_{x}" />，<img src="https://latex.codecogs.com/svg.image?t_{y}" title="t_{y}" />，<img src="https://latex.codecogs.com/svg.image?t_{w}" title="t_{w}" />，<img src="https://latex.codecogs.com/svg.image?t_{h}" title="t_{h}" />）,置信度（conf），分类结果（cls）。
#### Tricks
* (1)Mish激活函数y=x∗tanh(softplus(x))=x∗tanh(ln(1+exp(x)))
* 
* (2)CSP(Cross stage partial connections)
* 
* (3)MiWRC(Multi-input weighted residual connections)策略
## 2、模型的训练 (train.py)
模型的训练主要包含加载数据、加载模型、训练模型三个部分，最终会得到一个权重文件。其中数据在加载时会经过各种方法处理，以便训练出更好的模型。
训练模型时的损失函数的定义方法极为重要，直接决定着训练出模型的质量。

### 数据处理（dataset.py）


### 损失函数
代码如下：
```python
class Yolo_loss(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32] #划分grid cell时的步长[608/76,608/38,608/19]
        image_size = 608
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]#不同尺寸的先验框大小，根据训练集中标注真实框的宽高聚类得到的，可以修改
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]#负责选中anchors，前三个负责预测小目标，中间三个负责预测中目标，后三个负责预测大目标
        self.ignore_thre = 0.5 #iou阈值

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):... 

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):     #output_id对应三个特征图，ouput对应某一特征图的输出结果[batchsize,output_ch,fize,fsize]
            batchsize = output.shape[0]
            fsize = output.shape[2]                  #特征图大小
            n_ch = 5 + self.n_classes
            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)#将3个不同anchor的预测结果分开
            output = output.permute(0, 1, 3, 4, 2)  # 调整大小为shape[batchsize,self.n_anchors,fize,fsize,n_ch]

            # logistic activation for xy, obj, cls
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])
            # output解码过程
            pred = output[..., :4].clone()
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]
            #解码结束
            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)

            # loss calculation
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2],
                                              weight=tgt_scale * tgt_scale, reduction='sum')
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], reduction='sum') / 2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], reduction='sum')
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], reduction='sum')
            loss_l2 += F.mse_loss(input=output, target=target, reduction='sum')

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2
```
> class Yolo_loss()，从forward()开始看，输入是模型的输出结果bboxes_pred（对应变量xin）和数据的真实标签bboxes（对应变量labels）

> 这里讲一下logistic activation for xy, obj, cls（对中心坐标值、物体置信度、物体分类进行激活）和output解码过程
> + 首先，模型输出bounding box的中心坐标（<img src="https://latex.codecogs.com/svg.image?t_{x}" title="t_{x}" />，<img src="https://latex.codecogs.com/svg.image?t_{y}" title="t_{y}" />）是偏移量，而并不是在grid cell上的绝对位置，若对网络(6,6)中心的预测是(0.4,0.7)，那么中心位于76×76特征图上的(6.4,6.7)。若预测的x，y坐标大于1，比如(1.2,0.7)，则中心位于(7.2,6.7)。注意现在中心位于图像的第7排第8列单元格，这打破了YOLO背后的理论，因为如果假设原区域负责预测某个目标，目标的中心必须位于这个区域中，而不是位于此区域旁边的其他网格里。为解决这个问题，输出是通过一个sigmoid函数传递的，该函数在0到1的范围内缩放输出，有效地将中心保持在预测的网格中。
> + bounding box的宽高（<img src="https://latex.codecogs.com/svg.image?t_{w}" title="t_{w}" />，<img src="https://latex.codecogs.com/svg.image?t_{h}" title="t_{h}" />）是根据对应anchor尺度放缩之后的量，也不是实际宽高。
> + 物体置信度conf和分类结果cls也需要sigmod激活的原因是为了限制输出为(0,1)之间的值。
> + 图中（<img src="https://latex.codecogs.com/svg.image?b_{x}" title="b_{x}" />,<img src="https://latex.codecogs.com/svg.image?b_{y}" title="b_{y}" />,<img src="https://latex.codecogs.com/svg.image?b_{w}" title="b_{w}" />,<img src="https://latex.codecogs.com/svg.image?b_{h}" title="b_{h}" />）是解码之后的值对应代码中的pred[..., 0]、pred[..., 1]、pred[..., 2]、pred[..., 3]，（c_x,c_y,p_w,p_h）对应代码中的self.grid_x，self.grid_y，anchor_w，anchor_h
![image](https://user-images.githubusercontent.com/65380826/126755236-f5d97a4e-be3d-4eb2-9931-d42587042c68.png)

> 回到损失函数的公式，我们需要知道网络的回归框loss，置信度loss，以及分类loss。
> ![image](https://user-images.githubusercontent.com/65380826/126776556-656fe96a-988b-4c97-a8de-e0e09652f8f9.png)
> 
> ![image](https://user-images.githubusercontent.com/65380826/126776413-0bbc9bc6-23ab-42f5-b32f-741ebe64c5e0.png)
> 
> ![image](https://user-images.githubusercontent.com/65380826/126776575-156bd266-be27-4cbd-ad76-e09e97a1ff06.png)
> 
> ![image](https://user-images.githubusercontent.com/65380826/126776585-f4edd38c-4a3d-4614-8a90-564667836564.png)
> 
> ![image](https://user-images.githubusercontent.com/65380826/126776627-da08786b-1e14-4384-b4ee-8ba1dcdfa28f.png)
> 
> ![image](https://user-images.githubusercontent.com/65380826/126776646-edec4af8-ca2a-43ce-bc0c-7c80886fafb9.png)
> + 第1、2、3项回归框loss通过比对回归框的输出值与期望输出值，很显然网络输出值就是框的中心横纵坐标偏移值、宽高的尺度放缩量，那期望值是labels吗?并不是，我们的labels是标注值，包含真实框的左上角横纵坐标、右下角横纵坐标。所以我们希望通过对labels进行一定的处理从而得到与输出值对应的期望值。
> + 第4、5项置信度loss，直接把含有物体的预测框的置信度的期望值都设置为1
> + 第6项分类loss，网络输出的每个预测框分类结果的大小为类别数，根据物体的实际类别把对应的数据设置为1
> + 总的来说，我们需要将大小为[batchsize，N,5]的label处理成大小跟output匹配的期望值target[batchsize,self,n_anchor,fszie,fsize,n_ch]
> + 除此之外，还需要得到<img src="https://latex.codecogs.com/svg.image?I_{i,j}^{obj}" title="I_{i,j}^{obj}" />、<img src="https://latex.codecogs.com/svg.image?I_{i,j}^{noobj}" title="I_{i,j}^{noobj}" />，前者表示第(i,j)个grid cell里面的第a个预测框内是否有物体，如果有，则为1，反之为0，后者表示第(i,j)个grid cell的预测框里面是否不含物体，如果不含，则为1，反之为0.
> 代码中的tgt_mask是回归框loss及分类loss中的<img src="https://latex.codecogs.com/svg.image?I_{i,j}^{obj}" title="I_{i,j}^{obj}" />，obj_mask是置信度loss中的<img src="https://latex.codecogs.com/svg.image?I_{i,j}^{obj}" title="I_{i,j}^{obj}" />、<img src="https://latex.codecogs.com/svg.image?I_{i,j}^{noobj}" title="I_{i,j}^{noobj}" />两者之和（**这个地方存在一点问题，按理来说obj_mask应该是<img src="https://latex.codecogs.com/svg.image?I_{i,j}^{obj}&plus;\lambda_{noobj}*I_{i,j}^{noobj}" title="I_{i,j}^{obj}+\lambda_{noobj}*I_{i,j}^{noobj}" />，但是代码作者好像直接把前面的系数忽略了**）。tgt_scale则是<img src="https://latex.codecogs.com/svg.image?\sqrt{2-w_{i}*h_{i}}" title="\sqrt{2-w_{i}*h_{i}}" />。
> + 得到以上4个值之后就可以开始计算损失函数的值了，分别通过binary_cross_entropy和mse_loss计算。

> obj_mask, tgt_mask, tgt_scale, target，通过built_target()函数得到，接下来讲解如何得到这4个值
> 1. 计算label里面的真实框truth_box跟预设的3组（共9个）先验框ref_anchors之间的IOU，目的是确认哪种类型anchors适合用于预测该图像中的物体
> 2. 筛选出truth_box中每个框跟哪个ref_anchor的IOU最大，继而筛选出符合当前特征图下的真实框（因为真实框中有的框适合在76×76的特征图下预测，有的则适合在19×19的特征图下预测）
> 3. 计算模型得到的所有预测框跟truth_box之间的IOU，目的是判断某个预测框内是否含有物体
> 4. 筛选出含有物体的预测框，继而筛选出负责预测物体的预测框（预测框内有物体不代表负责预测该物体，因为每个grid cell会生成3个预测框，需要筛选出适合预测物体的预测框，即IOU大于某一阈值）
> 5. 以特征图76×76为例，一共有3×76×76个预测框，经过以上筛选后：
> > * obj_mask[batchsize,n_anchors,76,76]：把所有不含物体的预测框设为1加上把负责预测物体的预测框设为1，其他框设为0
> > * tgt_mask[batchsize,n_anchors,76,76,4+n_classes]：把所有含物体的预测框设为1，其他框设为0
> > * tgt_scale[batchsize,n_anchors,76,76,2]:按照上述公式计算
> > * target[batchsize,n_anchors,76,76,5+n_classes]：target[:,:,:,:,0]和target[:,:,:,:,1]为truth_box的
> > <img src="https://latex.codecogs.com/svg.image?\hat{b_{x}}" title="\hat{b_{x}}" />,<img src="https://latex.codecogs.com/svg.image?\hat{b_{y}}" title="\hat{b_{y}}" />译码成的<img src="https://latex.codecogs.com/svg.image?\hat{t_{x}}" title="\hat{t_{x}}" />，<img src="https://latex.codecogs.com/svg.image?\hat{t_{y}}" title="\hat{t_{y}}" />；target[:,:,:,:,2]和target[:,:,:,:,3]为truth_box的<img src="https://latex.codecogs.com/svg.image?\hat{b_{w}}" title="\hat{b_{w}}" />,<img src="https://latex.codecogs.com/svg.image?\hat{b_{h}}" title="\hat{b_{h}}" />译码成的<img src="https://latex.codecogs.com/svg.image?\hat{t_{w}}" title="\hat{t_{w}}" />,<img src="https://latex.codecogs.com/svg.image?\hat{t_{h}}" title="\hat{t_{h}}" />,target[:,:,:,:,4]为truth_box置信度，有预测框负责预测物体则对应置信度为1，target[:,:,:,:,5：5+num_classes],根据物体类别将对应地方设置为1。

> 上面提到了IOU(Intersection Over Union),IOU包含经典IOU、GIOU(Generalized Intersection over Union)、DIOU(Distance-IoU)、CIOU(Complete-IoU)
> ![image](https://user-images.githubusercontent.com/65380826/126856881-cffd5710-df15-4545-bea9-a135b1112748.png)
>  **值得一提的是，代码中的损失函数中回归框的损失不是按照CIOU-loss计算的，仍然是考虑预测框和真实框中心坐标、宽高之间的差距，即按照上述损失函数公式计算的。此外，在上述步骤2中计算iou时用的是CIOU，而在步骤3中计算iou是用的是经典IOU，这一点暂时还不清楚是为什么**

#### tricks
* CIOU loss
