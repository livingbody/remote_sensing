# 飞桨常规赛：遥感影像地块分割 - 4月第6名方案


# 1.比赛介绍
## 1.1比赛页面传送门： [常规赛：遥感影像地块分割](https://aistudio.baidu.com/aistudio/competition/detail/63)
## 1.2赛题概况
本赛题由 2020 CCF BDCI 遥感影像地块分割 初赛赛题改编而来。遥感影像地块分割, 旨在对遥感影像进行像素级内容解析，对遥感影像中感兴趣的类别进行提取和分类，在城乡规划、防汛救灾等领域具有很高的实用价值，在工业界也受到了广泛关注。现有的遥感影像地块分割数据处理方法局限于特定的场景和特定的数据来源，且精度无法满足需求。因此在实际应用中，仍然大量依赖于人工处理，需要消耗大量的人力、物力、财力。本赛题旨在衡量遥感影像地块分割模型在多个类别（如建筑、道路、林地等）上的效果，利用人工智能技术，对多来源、多场景的异构遥感影像数据进行充分挖掘，打造高效、实用的算法，提高遥感影像的分析提取能力。
赛题任务
本赛题旨在对遥感影像进行像素级内容解析，并对遥感影像中感兴趣的类别进行提取和分类，以衡量遥感影像地块分割模型在多个类别（如建筑、道路、林地等）上的效果。

## 1.3数据说明
本赛题提供了多个地区已脱敏的遥感影像数据，各参赛选手可以基于这些数据构建自己的地块分割模型。

### 1.3.1训练数据集
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

### 1.3.2测试数据集
测试数据集文件名称：img_test.zip，详细介绍如下：

包含4,609张分辨率为2m/pixel，尺寸为256 * 256的JPG图片，文件名称形如123.jpg。、
### 1.3.3数据增强工具
用什么数据增强，PaddleX自带的足矣！！！
## 1.4提交内容及格式
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


```
!pip install paddlex
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Requirement already satisfied: paddlex in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (1.3.9)
    Requirement already satisfied: pycocotools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (2.0.2)
    Requirement already satisfied: paddleslim==1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (1.1.1)
    Requirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.36.1)
    Requirement already satisfied: shapely>=1.7.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (1.7.1)
    Requirement already satisfied: flask-cors in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (3.0.8)
    Requirement already satisfied: paddlehub==2.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (2.1.0)
    Requirement already satisfied: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.4.4)
    Requirement already satisfied: psutil in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.7.2)
    Requirement already satisfied: visualdl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (2.1.1)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.1.1.26)
    Requirement already satisfied: xlwt in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (1.3.0)
    Requirement already satisfied: sklearn in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.0)
    Requirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.1.2)
    Requirement already satisfied: colorlog in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (4.1.0)
    Requirement already satisfied: paddlenlp>=2.0.0rc5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (2.0.0rc7)
    Requirement already satisfied: easydict in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (1.9)
    Requirement already satisfied: filelock in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.0.12)
    Requirement already satisfied: flask>=1.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (1.1.1)
    Requirement already satisfied: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (2.2.3)
    Requirement already satisfied: gunicorn>=19.10.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (20.0.4)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (1.20.2)
    Requirement already satisfied: packaging in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (20.9)
    Requirement already satisfied: paddle2onnx>=0.5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (0.5.1)
    Requirement already satisfied: pyzmq in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (18.1.1)
    Requirement already satisfied: Pillow in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (7.1.2)
    Requirement already satisfied: rarfile in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.1)
    Requirement already satisfied: gitpython in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.1.14)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.0->paddlehub==2.1.0->paddlex) (1.1.0)
    Requirement already satisfied: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.0->paddlehub==2.1.0->paddlex) (2.10.1)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.0->paddlehub==2.1.0->paddlex) (0.16.0)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.0->paddlehub==2.1.0->paddlex) (7.0)
    Requirement already satisfied: setuptools>=3.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gunicorn>=19.10.0->paddlehub==2.1.0->paddlex) (56.0.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.1.0->paddlehub==2.1.0->paddlex) (1.1.1)
    Requirement already satisfied: six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddle2onnx>=0.5.1->paddlehub==2.1.0->paddlex) (1.15.0)
    Requirement already satisfied: protobuf in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddle2onnx>=0.5.1->paddlehub==2.1.0->paddlex) (3.14.0)
    Requirement already satisfied: h5py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (2.9.0)
    Requirement already satisfied: jieba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.42.1)
    Requirement already satisfied: seqeval in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (1.2.2)
    Requirement already satisfied: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (0.8.53)
    Requirement already satisfied: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (3.8.2)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (2.22.0)
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.21.0)
    Requirement already satisfied: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (0.7.1.1)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.0.0)
    Requirement already satisfied: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (2.6.0)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.6.1)
    Requirement already satisfied: importlib-metadata in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.23)
    Requirement already satisfied: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (2.2.0)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->paddlex) (2.8.0)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->paddlex) (2019.3)
    Requirement already satisfied: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->paddlex) (0.18.0)
    Requirement already satisfied: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->paddlex) (3.9.9)
    Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gitpython->paddlehub==2.1.0->paddlex) (4.0.5)
    Requirement already satisfied: smmap<4,>=3.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gitdb<5,>=4.0.1->gitpython->paddlehub==2.1.0->paddlex) (3.0.5)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata->flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.6.0)
    Requirement already satisfied: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata->flake8>=3.7.9->visualdl>=2.0.0->paddlex) (7.2.0)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->paddlehub==2.1.0->paddlex) (0.10.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->paddlehub==2.1.0->paddlex) (2.4.2)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->paddlehub==2.1.0->paddlex) (1.1.0)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->paddlehub==2.1.0->paddlex) (2.8.0)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.3.0)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (2.0.1)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (16.7.9)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.3.4)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.4.10)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (0.10.0)
    Requirement already satisfied: cython>=0.27.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools->paddlex) (0.29)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (1.25.6)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (3.0.4)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (2.8)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (2019.9.11)
    Requirement already satisfied: scikit-learn>=0.21.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from seqeval->paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.24.1)
    Requirement already satisfied: scipy>=0.19.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (1.6.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (2.1.0)
    Requirement already satisfied: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.14.1)
    [33mWARNING: You are using pip version 21.0.1; however, version 21.1.1 is available.
    You should consider upgrading via the '/opt/conda/envs/python35-paddle120-env/bin/python -m pip install --upgrade pip' command.[0m


# 2.环境准备

## 2.1 PaddleX安装
此次比赛，先后尝试了PaddleSeg以及PaddleX，最终为了速度还是使用了PaddleX。


```
!pip install paddlex
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Requirement already satisfied: paddlex in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (1.3.9)
    Requirement already satisfied: paddleslim==1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (1.1.1)
    Requirement already satisfied: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.4.4)
    Requirement already satisfied: flask-cors in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (3.0.8)
    Requirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.1.2)
    Requirement already satisfied: psutil in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.7.2)
    Requirement already satisfied: xlwt in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (1.3.0)
    Requirement already satisfied: paddlehub==2.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (2.1.0)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.1.1.26)
    Requirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.36.1)
    Requirement already satisfied: pycocotools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (2.0.2)
    Requirement already satisfied: sklearn in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.0)
    Requirement already satisfied: visualdl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (2.1.1)
    Requirement already satisfied: shapely>=1.7.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (1.7.1)
    Requirement already satisfied: filelock in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.0.12)
    Requirement already satisfied: paddlenlp>=2.0.0rc5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (2.0.0rc7)
    Requirement already satisfied: easydict in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (1.9)
    Requirement already satisfied: rarfile in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.1)
    Requirement already satisfied: gitpython in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.1.14)
    Requirement already satisfied: pyzmq in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (18.1.1)
    Requirement already satisfied: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (2.2.3)
    Requirement already satisfied: paddle2onnx>=0.5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (0.5.1)
    Requirement already satisfied: gunicorn>=19.10.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (20.0.4)
    Requirement already satisfied: flask>=1.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (1.1.1)
    Requirement already satisfied: colorlog in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (4.1.0)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (1.20.2)
    Requirement already satisfied: packaging in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (20.9)
    Requirement already satisfied: Pillow in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (7.1.2)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.0->paddlehub==2.1.0->paddlex) (0.16.0)
    Requirement already satisfied: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.0->paddlehub==2.1.0->paddlex) (2.10.1)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.0->paddlehub==2.1.0->paddlex) (7.0)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.0->paddlehub==2.1.0->paddlex) (1.1.0)
    Requirement already satisfied: setuptools>=3.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gunicorn>=19.10.0->paddlehub==2.1.0->paddlex) (56.0.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.1.0->paddlehub==2.1.0->paddlex) (1.1.1)
    Requirement already satisfied: six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddle2onnx>=0.5.1->paddlehub==2.1.0->paddlex) (1.15.0)
    Requirement already satisfied: protobuf in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddle2onnx>=0.5.1->paddlehub==2.1.0->paddlex) (3.14.0)
    Requirement already satisfied: h5py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (2.9.0)
    Requirement already satisfied: seqeval in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (1.2.2)
    Requirement already satisfied: jieba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.42.1)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (2.22.0)
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.21.0)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.0.0)
    Requirement already satisfied: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (0.7.1.1)
    Requirement already satisfied: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (0.8.53)
    Requirement already satisfied: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (3.8.2)
    Requirement already satisfied: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (2.6.0)
    Requirement already satisfied: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (2.2.0)
    Requirement already satisfied: importlib-metadata in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.23)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.6.1)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->paddlex) (2.8.0)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->paddlex) (2019.3)
    Requirement already satisfied: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->paddlex) (3.9.9)
    Requirement already satisfied: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->paddlex) (0.18.0)
    Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gitpython->paddlehub==2.1.0->paddlex) (4.0.5)
    Requirement already satisfied: smmap<4,>=3.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gitdb<5,>=4.0.1->gitpython->paddlehub==2.1.0->paddlex) (3.0.5)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata->flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.6.0)
    Requirement already satisfied: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata->flake8>=3.7.9->visualdl>=2.0.0->paddlex) (7.2.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->paddlehub==2.1.0->paddlex) (2.4.2)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->paddlehub==2.1.0->paddlex) (0.10.0)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->paddlehub==2.1.0->paddlex) (2.8.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->paddlehub==2.1.0->paddlex) (1.1.0)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (16.7.9)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.4.10)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.3.4)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (0.10.0)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (2.0.1)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.3.0)
    Requirement already satisfied: cython>=0.27.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools->paddlex) (0.29)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (1.25.6)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (2019.9.11)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (2.8)
    Requirement already satisfied: scikit-learn>=0.21.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from seqeval->paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.24.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (2.1.0)
    Requirement already satisfied: scipy>=0.19.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (1.6.2)
    Requirement already satisfied: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.14.1)
    [33mWARNING: You are using pip version 21.0.1; however, version 21.1.1 is available.
    You should consider upgrading via the '/opt/conda/envs/python35-paddle120-env/bin/python -m pip install --upgrade pip' command.[0m


## 2.2 import必须和显卡环境配置


```
# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import matplotlib
import os
import paddlex as pdx

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):


# 3.数据准备

## 3.1解压数据集
解压缩数据集，使用一次即可。
* -o是为了覆盖，防止有人数据集变化重复解压提示解压不了
* -qa是为了静默，不需要日志


```
# !unzip -oqa data/data80164/train_and_label.zip
# !unzip -oqa data/data80164/img_test.zip
```

## 3.2数据增强
数据增强采用paddlex.seg.transforms自带的数据增强工具,其他第三方、手写的图像增强工具懒得看了。
* 实测Flip挺好使，提升成绩好高
* 采用数据增强的办法不是越多越好，需要和实际情况相结合


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

## 3.3数据集划分

### 3.3.1 获取总数据列表


```
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
# print(datas[0][0])
# print(datas[0][1])

data_dir = '/home/aistudio/work/'
```

    total: 66652


### 3.3.2 train、valid数据及划分


```
import numpy as np

labels = ['建筑', '耕地', '林地',  '其他']

with open('labels.txt', 'w') as f:
    for v in labels:
        f.write(v+'\n')

np.random.seed(5)
np.random.shuffle(datas)

split_num = int(0.05*len(datas))

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

    train: 63320
    valid: 3332


## 3.4PaddleX数据集定义


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

    2021-05-09 19:37:04 [INFO]	63320 samples in file train_list.txt
    2021-05-09 19:37:04 [INFO]	3332 samples in file valid_list.txt


# 4.模型训练
采用DeepLabv3p模型，骨干采用MobileNetV3_large_x1_0_ssld。<br>
原因无他，速度快，不过据悉Xception65精度更高，但是会特别慢。此外使用了 resume_checkpoint 继续进行训练。


```
num_classes = len(train_dataset.labels)

model = pdx.seg.DeepLabv3p(
    num_classes=num_classes,
    backbone='MobileNetV3_large_x1_0_ssld',
    pooling_crop_size=(256, 256))

model.train(
    num_epochs=100,
    train_dataset=train_dataset,
    train_batch_size=300,
    eval_dataset=eval_dataset,
    learning_rate=0.01,
    save_interval_epochs=1,
    pretrain_weights='CITYSCAPES',
    save_dir='output/deeplabv3p_mobilenetv3_large_ssld/pretain',
    use_vdl=True)
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:298: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/mobilenet_v3.py:231
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:298: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/segmentation/deeplabv3p.py:287
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:298: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/segmentation/deeplabv3p.py:315
    The behavior of expression A + B has been unified with elementwise_add(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_add(X, Y, axis=0) instead of A + B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/framework.py:687: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      elif dtype == np.bool:
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:298: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/segmentation/model_utils/loss.py:74
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    2021-05-09 19:37:05,642 - INFO - If regularizer of a Parameter has been set by 'fluid.ParamAttr' or 'fluid.WeightNormParamAttr' already. The Regularization[L2Decay, regularization_coeff=0.000040] in Optimizer will not take effect, and it will only be applied to other Parameters!
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle2onnx/onnx_helper/mapping.py:42: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      int(TensorProto.STRING): np.dtype(np.object)
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle2onnx/constant/dtypes.py:43: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      np.bool: core.VarDesc.VarType.BOOL,
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle2onnx/constant/dtypes.py:44: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      core.VarDesc.VarType.FP32: np.float,
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle2onnx/constant/dtypes.py:49: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      core.VarDesc.VarType.BOOL: np.bool


    2021-05-09 19:37:07 [INFO]	Connecting PaddleHub server to get pretrain weights...
    Download https://paddleseg.bj.bcebos.com/models/deeplabv3p_mobilenetv3_large_cityscapes.tar.gz
    [##################################################] 100.00%
    Decompress /home/aistudio/.paddlehub/tmp/tmp0o55si46/deeplabv3p_mobilenetv3_large_cityscapes.tar.gz
    [##################################################] 100.00%
    2021-05-09 19:37:13 [INFO]	Load pretrain weights from output/deeplabv3p_mobilenetv3_large_ssld/pretain/pretrain/DeepLabv3p_MobileNetV3_large_x1_0_ssld_CITYSCAPES.
    2021-05-09 19:37:13 [WARNING]	[SKIP] Shape of pretrained weight decoder/weights doesn't match.(Pretrained: (19, 128, 1, 1), Actual: (4, 128, 1, 1))
    2021-05-09 19:37:13 [WARNING]	[SKIP] Shape of pretrained weight decoder/merge/weights doesn't match.(Pretrained: (19, 120, 1, 1), Actual: (4, 120, 1, 1))
    2021-05-09 19:37:13 [INFO]	There are 273 varaibles in output/deeplabv3p_mobilenetv3_large_ssld/pretain/pretrain/DeepLabv3p_MobileNetV3_large_x1_0_ssld_CITYSCAPES are loaded.
    2021-05-09 19:37:25 [INFO]	[TRAIN] Epoch=1/100, Step=2/211, loss=2.102176, lr=0.01, time_each_step=5.82s, eta=36:3:56
    2021-05-09 19:37:29 [INFO]	[TRAIN] Epoch=1/100, Step=4/211, loss=1.484626, lr=0.009999, time_each_step=3.85s, eta=23:50:23
    2021-05-09 19:37:32 [INFO]	[TRAIN] Epoch=1/100, Step=6/211, loss=1.521728, lr=0.009998, time_each_step=3.15s, eta=19:30:30
    2021-05-09 19:37:37 [INFO]	[TRAIN] Epoch=1/100, Step=8/211, loss=1.609498, lr=0.009997, time_each_step=2.89s, eta=17:55:9
    2021-05-09 19:37:40 [INFO]	[TRAIN] Epoch=1/100, Step=10/211, loss=1.228453, lr=0.009996, time_each_step=2.69s, eta=16:37:36
    2021-05-09 19:37:43 [INFO]	[TRAIN] Epoch=1/100, Step=12/211, loss=1.280863, lr=0.009995, time_each_step=2.48s, eta=15:22:48
    2021-05-09 19:37:47 [INFO]	[TRAIN] Epoch=1/100, Step=14/211, loss=1.207555, lr=0.009994, time_each_step=2.42s, eta=14:58:42
    2021-05-09 19:37:51 [INFO]	[TRAIN] Epoch=1/100, Step=16/211, loss=1.144117, lr=0.009994, time_each_step=2.34s, eta=14:27:14
    2021-05-09 19:37:54 [INFO]	[TRAIN] Epoch=1/100, Step=18/211, loss=1.042588, lr=0.009993, time_each_step=2.25s, eta=13:55:0
    2021-05-09 19:37:59 [INFO]	[TRAIN] Epoch=1/100, Step=20/211, loss=1.153433, lr=0.009992, time_each_step=2.26s, eta=13:57:55
    2021-05-09 19:38:02 [INFO]	[TRAIN] Epoch=1/100, Step=22/211, loss=1.136962, lr=0.009991, time_each_step=1.85s, eta=11:25:35
    2021-05-09 19:38:05 [INFO]	[TRAIN] Epoch=1/100, Step=24/211, loss=0.991741, lr=0.00999, time_each_step=1.82s, eta=11:15:21
    2021-05-09 19:38:09 [INFO]	[TRAIN] Epoch=1/100, Step=26/211, loss=1.007917, lr=0.009989, time_each_step=1.84s, eta=11:22:9
    2021-05-09 19:38:13 [INFO]	[TRAIN] Epoch=1/100, Step=28/211, loss=0.988545, lr=0.009988, time_each_step=1.81s, eta=11:10:30
    2021-05-09 19:38:16 [INFO]	[TRAIN] Epoch=1/100, Step=30/211, loss=1.033619, lr=0.009988, time_each_step=1.8s, eta=11:9:8
    2021-05-09 19:38:20 [INFO]	[TRAIN] Epoch=1/100, Step=32/211, loss=0.996365, lr=0.009987, time_each_step=1.85s, eta=11:25:7
    2021-05-09 19:38:24 [INFO]	[TRAIN] Epoch=1/100, Step=34/211, loss=1.031332, lr=0.009986, time_each_step=1.84s, eta=11:23:19
    2021-05-09 19:38:28 [INFO]	[TRAIN] Epoch=1/100, Step=36/211, loss=0.980095, lr=0.009985, time_each_step=1.86s, eta=11:29:35
    2021-05-09 19:38:32 [INFO]	[TRAIN] Epoch=1/100, Step=38/211, loss=0.961618, lr=0.009984, time_each_step=1.88s, eta=11:39:22
    2021-05-09 19:38:35 [INFO]	[TRAIN] Epoch=1/100, Step=40/211, loss=0.929827, lr=0.009983, time_each_step=1.82s, eta=11:15:29
    2021-05-09 19:38:39 [INFO]	[TRAIN] Epoch=1/100, Step=42/211, loss=0.937995, lr=0.009983, time_each_step=1.86s, eta=11:30:11
    2021-05-09 19:38:42 [INFO]	[TRAIN] Epoch=1/100, Step=44/211, loss=0.958771, lr=0.009982, time_each_step=1.85s, eta=11:28:1
    2021-05-09 19:38:46 [INFO]	[TRAIN] Epoch=1/100, Step=46/211, loss=0.98883, lr=0.009981, time_each_step=1.82s, eta=11:14:38
    2021-05-09 19:38:50 [INFO]	[TRAIN] Epoch=1/100, Step=48/211, loss=0.905162, lr=0.00998, time_each_step=1.86s, eta=11:28:11
    2021-05-09 19:38:54 [INFO]	[TRAIN] Epoch=1/100, Step=50/211, loss=0.957791, lr=0.009979, time_each_step=1.85s, eta=11:27:42
    2021-05-09 19:38:57 [INFO]	[TRAIN] Epoch=1/100, Step=52/211, loss=0.842869, lr=0.009978, time_each_step=1.84s, eta=11:23:26
    2021-05-09 19:39:01 [INFO]	[TRAIN] Epoch=1/100, Step=54/211, loss=0.946181, lr=0.009977, time_each_step=1.84s, eta=11:23:29
    2021-05-09 19:39:05 [INFO]	[TRAIN] Epoch=1/100, Step=56/211, loss=0.865188, lr=0.009977, time_each_step=1.85s, eta=11:24:11
    2021-05-09 19:39:08 [INFO]	[TRAIN] Epoch=1/100, Step=58/211, loss=0.886755, lr=0.009976, time_each_step=1.84s, eta=11:22:5
    2021-05-09 19:39:11 [INFO]	[TRAIN] Epoch=1/100, Step=60/211, loss=0.911376, lr=0.009975, time_each_step=1.81s, eta=11:11:22
    2021-05-09 19:39:15 [INFO]	[TRAIN] Epoch=1/100, Step=62/211, loss=0.952829, lr=0.009974, time_each_step=1.79s, eta=11:2:56
    2021-05-09 19:39:19 [INFO]	[TRAIN] Epoch=1/100, Step=64/211, loss=0.897487, lr=0.009973, time_each_step=1.83s, eta=11:18:24
    2021-05-09 19:39:22 [INFO]	[TRAIN] Epoch=1/100, Step=66/211, loss=0.894006, lr=0.009972, time_each_step=1.84s, eta=11:21:45
    2021-05-09 19:39:26 [INFO]	[TRAIN] Epoch=1/100, Step=68/211, loss=0.901722, lr=0.009971, time_each_step=1.81s, eta=11:9:42
    2021-05-09 19:39:30 [INFO]	[TRAIN] Epoch=1/100, Step=70/211, loss=0.916442, lr=0.009971, time_each_step=1.83s, eta=11:17:0
    2021-05-09 19:39:34 [INFO]	[TRAIN] Epoch=1/100, Step=72/211, loss=0.882168, lr=0.00997, time_each_step=1.84s, eta=11:22:49
    2021-05-09 19:39:39 [INFO]	[TRAIN] Epoch=1/100, Step=74/211, loss=0.86822, lr=0.009969, time_each_step=1.89s, eta=11:39:8
    2021-05-09 19:39:43 [INFO]	[TRAIN] Epoch=1/100, Step=76/211, loss=0.972915, lr=0.009968, time_each_step=1.9s, eta=11:45:16
    2021-05-09 19:39:46 [INFO]	[TRAIN] Epoch=1/100, Step=78/211, loss=0.792731, lr=0.009967, time_each_step=1.89s, eta=11:41:7
    2021-05-09 19:39:50 [INFO]	[TRAIN] Epoch=1/100, Step=80/211, loss=0.861633, lr=0.009966, time_each_step=1.96s, eta=12:5:6
    2021-05-09 19:39:54 [INFO]	[TRAIN] Epoch=1/100, Step=82/211, loss=0.873994, lr=0.009965, time_each_step=1.92s, eta=11:51:31
    2021-05-09 19:39:57 [INFO]	[TRAIN] Epoch=1/100, Step=84/211, loss=0.897076, lr=0.009965, time_each_step=1.9s, eta=11:43:31
    2021-05-09 19:40:01 [INFO]	[TRAIN] Epoch=1/100, Step=86/211, loss=0.926882, lr=0.009964, time_each_step=1.92s, eta=11:52:2
    2021-05-09 19:40:05 [INFO]	[TRAIN] Epoch=1/100, Step=88/211, loss=0.803018, lr=0.009963, time_each_step=1.92s, eta=11:52:2
    2021-05-09 19:40:08 [INFO]	[TRAIN] Epoch=1/100, Step=90/211, loss=0.911031, lr=0.009962, time_each_step=1.91s, eta=11:45:38
    2021-05-09 19:40:12 [INFO]	[TRAIN] Epoch=1/100, Step=92/211, loss=0.802107, lr=0.009961, time_each_step=1.91s, eta=11:47:1
    2021-05-09 19:40:16 [INFO]	[TRAIN] Epoch=1/100, Step=94/211, loss=0.957028, lr=0.00996, time_each_step=1.85s, eta=11:26:29
    2021-05-09 19:40:19 [INFO]	[TRAIN] Epoch=1/100, Step=96/211, loss=0.962513, lr=0.009959, time_each_step=1.8s, eta=11:4:44
    2021-05-09 19:40:23 [INFO]	[TRAIN] Epoch=1/100, Step=98/211, loss=0.883863, lr=0.009959, time_each_step=1.85s, eta=11:25:37
    2021-05-09 19:40:27 [INFO]	[TRAIN] Epoch=1/100, Step=100/211, loss=0.921632, lr=0.009958, time_each_step=1.85s, eta=11:23:51
    2021-05-09 19:40:31 [INFO]	[TRAIN] Epoch=1/100, Step=102/211, loss=0.855094, lr=0.009957, time_each_step=1.88s, eta=11:37:4
    2021-05-09 19:40:35 [INFO]	[TRAIN] Epoch=1/100, Step=104/211, loss=0.85595, lr=0.009956, time_each_step=1.88s, eta=11:36:59
    2021-05-09 19:40:38 [INFO]	[TRAIN] Epoch=1/100, Step=106/211, loss=0.871709, lr=0.009955, time_each_step=1.88s, eta=11:36:16
    2021-05-09 19:40:42 [INFO]	[TRAIN] Epoch=1/100, Step=108/211, loss=0.876193, lr=0.009954, time_each_step=1.89s, eta=11:37:45
    2021-05-09 19:40:46 [INFO]	[TRAIN] Epoch=1/100, Step=110/211, loss=0.890553, lr=0.009953, time_each_step=1.9s, eta=11:41:56
    2021-05-09 19:40:50 [INFO]	[TRAIN] Epoch=1/100, Step=112/211, loss=0.880856, lr=0.009953, time_each_step=1.91s, eta=11:44:38
    2021-05-09 19:40:54 [INFO]	[TRAIN] Epoch=1/100, Step=114/211, loss=0.830736, lr=0.009952, time_each_step=1.92s, eta=11:49:55
    2021-05-09 19:40:58 [INFO]	[TRAIN] Epoch=1/100, Step=116/211, loss=0.885855, lr=0.009951, time_each_step=1.95s, eta=12:2:46
    2021-05-09 19:41:02 [INFO]	[TRAIN] Epoch=1/100, Step=118/211, loss=0.787732, lr=0.00995, time_each_step=1.92s, eta=11:48:59
    2021-05-09 19:41:06 [INFO]	[TRAIN] Epoch=1/100, Step=120/211, loss=0.81879, lr=0.009949, time_each_step=1.93s, eta=11:52:5
    2021-05-09 19:41:10 [INFO]	[TRAIN] Epoch=1/100, Step=122/211, loss=0.86447, lr=0.009948, time_each_step=1.94s, eta=11:57:17
    2021-05-09 19:41:14 [INFO]	[TRAIN] Epoch=1/100, Step=124/211, loss=0.76384, lr=0.009948, time_each_step=1.96s, eta=12:6:0
    2021-05-09 19:41:18 [INFO]	[TRAIN] Epoch=1/100, Step=126/211, loss=0.869496, lr=0.009947, time_each_step=1.99s, eta=12:14:52
    2021-05-09 19:41:21 [INFO]	[TRAIN] Epoch=1/100, Step=128/211, loss=0.830324, lr=0.009946, time_each_step=1.95s, eta=12:0:35
    2021-05-09 19:41:26 [INFO]	[TRAIN] Epoch=1/100, Step=130/211, loss=0.829597, lr=0.009945, time_each_step=1.98s, eta=12:10:34
    2021-05-09 19:41:29 [INFO]	[TRAIN] Epoch=1/100, Step=132/211, loss=0.897173, lr=0.009944, time_each_step=1.92s, eta=11:51:13
    2021-05-09 19:41:33 [INFO]	[TRAIN] Epoch=1/100, Step=134/211, loss=0.858942, lr=0.009943, time_each_step=1.93s, eta=11:54:26
    2021-05-09 19:41:37 [INFO]	[TRAIN] Epoch=1/100, Step=136/211, loss=0.808036, lr=0.009942, time_each_step=1.92s, eta=11:50:5
    2021-05-09 19:41:41 [INFO]	[TRAIN] Epoch=1/100, Step=138/211, loss=0.744489, lr=0.009942, time_each_step=1.95s, eta=12:0:1
    2021-05-09 19:41:44 [INFO]	[TRAIN] Epoch=1/100, Step=140/211, loss=0.821231, lr=0.009941, time_each_step=1.92s, eta=11:48:31
    2021-05-09 19:41:48 [INFO]	[TRAIN] Epoch=1/100, Step=142/211, loss=0.843361, lr=0.00994, time_each_step=1.91s, eta=11:45:3
    2021-05-09 19:41:52 [INFO]	[TRAIN] Epoch=1/100, Step=144/211, loss=0.875032, lr=0.009939, time_each_step=1.88s, eta=11:36:1
    2021-05-09 19:41:55 [INFO]	[TRAIN] Epoch=1/100, Step=146/211, loss=0.84197, lr=0.009938, time_each_step=1.86s, eta=11:25:32
    2021-05-09 19:41:59 [INFO]	[TRAIN] Epoch=1/100, Step=148/211, loss=0.808388, lr=0.009937, time_each_step=1.88s, eta=11:33:57
    2021-05-09 19:42:03 [INFO]	[TRAIN] Epoch=1/100, Step=150/211, loss=0.865812, lr=0.009936, time_each_step=1.86s, eta=11:24:49
    2021-05-09 19:42:07 [INFO]	[TRAIN] Epoch=1/100, Step=152/211, loss=0.8952, lr=0.009936, time_each_step=1.92s, eta=11:47:11
    2021-05-09 19:42:11 [INFO]	[TRAIN] Epoch=1/100, Step=154/211, loss=0.729363, lr=0.009935, time_each_step=1.91s, eta=11:44:34
    2021-05-09 19:42:15 [INFO]	[TRAIN] Epoch=1/100, Step=156/211, loss=0.759849, lr=0.009934, time_each_step=1.91s, eta=11:44:7
    2021-05-09 19:42:19 [INFO]	[TRAIN] Epoch=1/100, Step=158/211, loss=0.793309, lr=0.009933, time_each_step=1.89s, eta=11:38:12
    2021-05-09 19:42:23 [INFO]	[TRAIN] Epoch=1/100, Step=160/211, loss=0.869632, lr=0.009932, time_each_step=1.92s, eta=11:48:51
    2021-05-09 19:42:26 [INFO]	[TRAIN] Epoch=1/100, Step=162/211, loss=0.780676, lr=0.009931, time_each_step=1.9s, eta=11:42:32
    2021-05-09 19:42:30 [INFO]	[TRAIN] Epoch=1/100, Step=164/211, loss=0.904182, lr=0.00993, time_each_step=1.93s, eta=11:51:55
    2021-05-09 19:42:34 [INFO]	[TRAIN] Epoch=1/100, Step=166/211, loss=0.961774, lr=0.00993, time_each_step=1.93s, eta=11:52:6
    2021-05-09 19:42:38 [INFO]	[TRAIN] Epoch=1/100, Step=168/211, loss=0.845463, lr=0.009929, time_each_step=1.97s, eta=12:7:42
    2021-05-09 19:42:42 [INFO]	[TRAIN] Epoch=1/100, Step=170/211, loss=0.84423, lr=0.009928, time_each_step=1.96s, eta=12:4:35
    2021-05-09 19:42:46 [INFO]	[TRAIN] Epoch=1/100, Step=172/211, loss=0.821791, lr=0.009927, time_each_step=1.93s, eta=11:53:9
    2021-05-09 19:42:49 [INFO]	[TRAIN] Epoch=1/100, Step=174/211, loss=0.780094, lr=0.009926, time_each_step=1.9s, eta=11:41:19


# 5.模型评估


```
model = pdx.load_model('output/deeplabv3p_mobilenetv3_large_ssld/pretain/best_model')
model.evaluate(eval_dataset, batch_size=160, epoch_id=None, return_details=False)
```

# 6.模型预测


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

# 7.打包提交


```
# 生成提交文件
!zip -r result.zip result/
```

# 8.其他改进
## 8.1 val和提交分数不一样
本版本训练iou能够达到**70左右**，但是交分数大概在**60分左右**

## 8.2 一定要注意保存模型
**训练模型没保存**，这不又得重新跑跑，看能不能达到之前的分数，然我哭一会先。偷懒了以后不跑了就没最好模型了。

## 8.3 显存拉满
多看看下面监控图，计算机算，200的batch size，显存利用率60%，那么333的batch size就接近100%了吧，就用300吧，做人没必要满分了。什么？你说batch size 最好2的指数？我看对计算机来说随意。随意，随意，大家随意就好。

![](https://ai-studio-static-online.cdn.bcebos.com/ac3e13a3e7fc48c781b07c1640535a5da7ee99b74f6844b5aaf83614c3ee71c7)

![](https://ai-studio-static-online.cdn.bcebos.com/9a8f43e12f0a4730ad91eec6508991b9d9fad259318c41ac85b88b0497c6e3c8)


## 8.4 batch size小点好 千万不要爆
***batchsize=999999***

## 8.5 中间模型一定要删掉
切记切记中间模型一定要删掉，不要动不动项目空间100Gb，打开半小时，自己都要被自己蠢哭了，还怪系统慢。。。

## 8.6 另外一种查看显卡状态方法
终端下运行***nvidia-smi***命令
![](https://ai-studio-static-online.cdn.bcebos.com/c8d650eeaaf44e10b5f178302a314369d182acada6fb4e9d8dfe797a9b752c16)

## 8.7 图像尺寸技巧
图像要尽可能缩小尺寸，这样能节约显卡资源

## 8.8 来一张GPU工作图
![](https://ai-studio-static-online.cdn.bcebos.com/eaf5be4636224921b5e3d0b5db9532e4028d6148bccb45d68cb7eb0a71f7f092)



# 9.其他
感谢领航团实例分割课程的各位老师、班班、助教的辛苦付出，也感谢各位同学，要不是你们，我随意跑个成绩就结束了，就不认真了。

## 9.1 比赛页面传送门： [常规赛：遥感影像地块分割](https://aistudio.baidu.com/aistudio/competition/detail/63)

## 9.2欢迎参加 [飞桨领航团实战速成营](https://aistudio.baidu.com/aistudio/education/group/info/16606) 一起学习 ，欢迎一键三连 Fork~

## 9.3据说大家喜欢github 我也来一个【欢迎一键三连，给点更新的动力】
[https://github.com/livingbody/remote_sensing](https://github.com/livingbody/remote_sensing)


![](https://ai-studio-static-online.cdn.bcebos.com/880f41e845d44617bda0c5c9ca9a4dd615ee605f23014477b8220cbdf4120e8f)
