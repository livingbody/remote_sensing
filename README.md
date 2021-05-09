# 常规赛：遥感影像地块分割-4月份第6名方案

# 1.比赛介绍
## 1. 1比赛页面传送门： [常规赛：遥感影像地块分割](https://aistudio.baidu.com/aistudio/competition/detail/63)


##  2.赛题介绍
本赛题由 2020 CCF BDCI 遥感影像地块分割 初赛赛题改编而来。遥感影像地块分割, 旨在对遥感影像进行像素级内容解析，对遥感影像中感兴趣的类别进行提取和分类，在城乡规划、防汛救灾等领域具有很高的实用价值，在工业界也受到了广泛关注。现有的遥感影像地块分割数据处理方法局限于特定的场景和特定的数据来源，且精度无法满足需求。因此在实际应用中，仍然大量依赖于人工处理，需要消耗大量的人力、物力、财力。本赛题旨在衡量遥感影像地块分割模型在多个类别（如建筑、道路、林地等）上的效果，利用人工智能技术，对多来源、多场景的异构遥感影像数据进行充分挖掘，打造高效、实用的算法，提高遥感影像的分析提取能力。
赛题任务
本赛题旨在对遥感影像进行像素级内容解析，并对遥感影像中感兴趣的类别进行提取和分类，以衡量遥感影像地块分割模型在多个类别（如建筑、道路、林地等）上的效果。

## 1.2数据说明
本赛题提供了多个地区已脱敏的遥感影像数据，各参赛选手可以基于这些数据构建自己的地块分割模型。

### 训练数据集
样例图片及其标注如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/8087a965609d48a19a5e60f0330fa9054d04097644de48ffa3d557e7a8ad64ad)
![](https://ai-studio-static-online.cdn.bcebos.com/d18664ecf0514cb686c95958d30bbf8a2f5efb0691bc4d66a5f6317ab511d6d0)

![](https://ai-studio-static-online.cdn.bcebos.com/e42f2c222f204094ac3a0ea8582ca331b0452fb2b1704eabaae379d499906977)
![](https://ai-studio-static-online.cdn.bcebos.com/d5260bd5a820486a85aeb2105adfb6fa10284bd94453459f892755bc43e10b8a)


训练数据集文件名称：train_and_label.zip

包含2个子文件，分别为：训练数据集（原始图片）文件、训练数据集（标注图片）文件，详细介绍如下：

* **训练数据集**（原始图片）文件名称：img_train

	包含66,653张分辨率为2m/pixel，尺寸为256 * 256的JPG图片，每张图片的名称形如T000123.jpg。
* **训练数据集**（标注图片）文件名称：lab_train

	包含66,653张分辨率为2m/pixel，尺寸为256 * 256的PNG图片，每张图片的名称形如T000123.png。
* **备注**： 全部PNG图片共包括4种分类，像素值分别为0、1、2、3。此外，像素值255为未标注区域，表示对应区域的所属类别并不确定，在评测中也不会考虑这部分区域。

### 测试数据集
测试数据集文件名称：img_test.zip，详细介绍如下：

包含4,609张分辨率为2m/pixel，尺寸为256 * 256的JPG图片，文件名称形如123.jpg。


## 1.2 提交内容及格式
* 以zip压缩包形式提交结果文件，文件命名为 result.zip；
* zip压缩包中的图片格式必须为单通道PNG；
* PNG文件数需要与测试数据集中的文件数相同，且zip压缩包文件名需要与测试数据集中的文件名一一对应；
* 单通道PNG图片中的像素值必须介于0~3之间，像素值不能为255。如果存在未标注区域，评测系统会自动忽略对应区域的提交结果。
### 提交示例
提交文件命名为：result.zip，zip文件的组织方式如下所示：

```
主目录                                                                        
├── 1.png         #每个结果文件命名为：测试数据集图片名称+.png                      
├── 2.png                                                              
├── 3.png                                                    
├── ...     
```

**备注**： 主目录中必须包含与测试数据集相同数目、名称相对应的单通道PNG图片，且每张单通道PNG图片中的像素值必须介于0~3之间，像素值不能为255。

## 2.环境准备
### 2.1 PaddleX安装
此次比赛，先后尝试了PaddleSeg以及PaddleX，最终为了速度还是使用了PaddleX。

```
!pip install paddlex -i https://mirror.baidu.com/pypi/simple
```
### 2.2 import必须和显卡环境配置
```
# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import matplotlib
import os
import paddlex as pdx

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

## 3.数据准备

### 3.1解压数据集
解压缩数据集，使用一次即可。-o是为了覆盖，防止有人数据集变化重复解压提示解压不了；-qa是为了静默，不需要日志。
```
# 使用一次即可，后续可以注释掉
!unzip -oqa data/data80164/train_and_label.zip
!unzip -oqa data/data80164/img_test.zip
```

### 3.2数据增强
数据增强采用Paddlex.seg自带的数据增强工具。实测Flip挺好使，提升成绩好高。

```
from paddlex.seg import transforms

# 定义训练和验证时的transforms
train_transforms = transforms.Compose([
    # 上下、左右翻转概率默认为0.5
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomBlur(prob=0.1),
    transforms.RandomRotate(rotate_range=10),
    transforms.Normalize()
])
eval_transforms = transforms.Compose([
    transforms.Normalize()
])
```
### 3.3数据集划分

```
#获取总数据列表
import numpy as np

datas = []
image_base = 'img_train'
annos_base = 'lab_train'

ids_ = [v.split('.')[0] for v in os.listdir(image_base)]

for id_ in ids_:
    img_pt0 = os.path.join(image_base, '{}.jpg'.format(id_))
    img_pt1 = os.path.join(annos_base, '{}.png'.format(id_))
    datas.append((img_pt0.replace('/home/aistudio/work/', ''), img_pt1.replace('/home/aistudio/work/', '')))
    if os.path.exists(img_pt0) and os.path.exists(img_pt1):
        pass
    else:
        raise "path invalid!"

print('total:', len(datas))
print(datas[0][0])
print(datas[0][1])

data_dir = '/home/aistudio/work/'
```

    total: 66652
    img_train/T068733.jpg
    lab_train/T068733.png



```
# train、valid数据及划分
import numpy as np

labels = ['建筑', '耕地', '林地',  '其他']

with open('labels.txt', 'w') as f:
    for v in labels:
        f.write(v+'\n')

np.random.seed(5)
np.random.shuffle(datas)

split_num = int(0.2*len(datas))

# 划分训练集和验证集
train_data = datas[:-split_num]
valid_data = datas[-split_num:]

with open('train_list.txt', 'w') as f:
    for img, lbl in train_data:
        f.write(img + ' ' + lbl + '\n')

with open('valid_list.txt', 'w') as f:
    for img, lbl in valid_data:
        f.write(img + ' ' + lbl + '\n')

print('train:', len(train_data))
print('valid:', len(valid_data))
```

    train: 53322
    valid: 13330


### 3.4PaddleX数据集定义
```
data_dir = './'

# # 定义训练和验证数据集
train_dataset = pdx.datasets.SegDataset(
    data_dir=data_dir,
    file_list='train_list.txt',
    label_list='labels.txt',
    transforms=train_transforms,
    shuffle=True)
    
eval_dataset = pdx.datasets.SegDataset(
    data_dir=data_dir,
    file_list='valid_list.txt',
    label_list='labels.txt',
    transforms=eval_transforms)
```

    2021-04-18 23:36:47 [INFO]	53322 samples in file train_list.txt
    2021-04-18 23:36:47 [INFO]	13330 samples in file valid_list.txt


## 4.模型训练
采用DeepLabv3p模型，骨干采用MobileNetV3_large_x1_0_ssld，原因无他，速度快，不过据悉Xception65精度更高，但是会特别慢。此外使用了 resume_checkpoint 继续进行训练。

```
import paddle

num_classes = len(train_dataset.labels)


model = pdx.seg.DeepLabv3p(
    num_classes=num_classes,  backbone='MobileNetV3_large_x1_0_ssld'
)


model.train(
    num_epochs=100,
    train_dataset=train_dataset,
    train_batch_size=200,
    eval_dataset=eval_dataset,
    learning_rate=0.002,
    save_interval_epochs=2,
    pretrain_weights='CITYSCAPES',
    save_dir='output/deeplabv3p_mobilenetv3_large_ssld/pretain',
    use_vdl=True)
```


​    


## 5.模型评估


```
model = pdx.load_model('./output/best_model')
model.evaluate(eval_dataset, batch_size=25, epoch_id=None, return_details=True)
```


## 6.模型预测


```
from tqdm import tqdm
import cv2

test_base = 'img_testA/'
out_base = 'result/'

if not os.path.exists(out_base):
    os.makedirs(out_base)


for im in tqdm(os.listdir(test_base)):
    if not im.endswith('.jpg'):
        continue
    pt = test_base + im
    result = model.predict(pt)
    cv2.imwrite(out_base+im.replace('jpg', 'png'), result['label_map'])
```

## 7.其他改进
本版本训练iou能够达到70左右，但是交分数大概在60分左右。。。。。。


```
# 生成提交文件
!zip -r result.zip result/
```

## 8.比赛页面传送门： [常规赛：遥感影像地块分割](https://aistudio.baidu.com/aistudio/competition/detail/63)

## 9.欢迎参加 [飞桨领航团实战速成营](https://aistudio.baidu.com/aistudio/education/group/info/16606) 一起学习 ，欢迎一键三连 Fork~
