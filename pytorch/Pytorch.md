# Pytorch

# Pytorch nn.Module

## nn.Module的基本用法

nn.Module是所有神经网络的基类

使用时主要实现两个方法

1. `__init__`：初始化方法，可以用于定义神经网络的结构
2. `forward`：前向传播，用于定义神经网络的前向传播

一个简单的例子（非神经网络）：

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        print("初始化模型")        
        self.model = nn.Linear(1, 1)

    def forward(self, x): # 定义前向传播方式，参数可以任意指定
        print("模型开始进行前向传播")
        return self.model(x)
```

定义好模型就可以使用了，直接调用构造函数

```python
model = Model()

输出
初始化模型
```

定义好模型后，直接向model传入参数，就会执行`forward`方法，并返回forward方法的返回值

```python
model(torch.Tensor([1]))
模型开始进行前向传播

tensor([0.7308], grad_fn=<AddBackward0>)
```

## **nn.Module的其他常用方法**

1. 
1. `cuda(device=None)`: 如果要使用GPU进行模型运算，需要将“模型参数和缓存”转移到GPU中。**需要下载GPU版本的pytorch，并且安装CUDA**

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.cuda(device) # 需要安装cuda
model.to(device) # 推荐使用该方式
```

```python
Model(
  (model): Linear(in_features=1, out_features=1, bias=True)
)
```

1. `eval()`：如果模型训练完毕，需要进行评估，需要显式的告知模型。调用该方法即可。

```python
model.eval()
model.train(False) # 效果一样
```

```python
Model(
  (model): Linear(in_features=1, out_features=1, bias=True)
)
```

1. `modules()`: 获取模型中的所有Module类型的对象，返回一个迭代器

```python
for idx, m in enumerate(model.modules()):
    print(idx, '->', m)
```

```python
0 -> Model(
  (model): Linear(in_features=1, out_features=1, bias=True)
)
1 -> Linear(in_features=1, out_features=1, bias=True)
```

1. `paramters()`: 获取模型的所有参数，返回一个迭代器

```python
for param in model.parameters():
    print(param)
```

```python
Parameter containing:
tensor([[0.5900]], requires_grad=True)
Parameter containing:
tensor([0.1408], requires_grad=True)
```

# **Pytorch nn.Linear**

## **nn.Linear的基本定义**

nn.Linear定义一个神经网络的线性层，方法签名如下：

```python
torch.nn.Linear(in_features, # 输入的神经元个数
           out_features, # 输出神经元个数
           bias=True # 是否包含偏置
           )
```

Linear其实就是对输入 $X_{n \times i}$执行了一个线性变换，即：

$Y_{n \times o} = X_{n \times i}W_{i\times o} + b$

**其中$W$是模型要学习的参数， $W$ 的维度为$W_{i \times o}$, $b$ 是o维的向量偏置**，$n$为输入向量的行数（例如，你想一次输入10个样本，即batch_size为10，则n = 10 n=10n=10 ），$i$ 为输入神经元的个数（例如你的样本特征数为5，则 i = 5 i=5i=5 ），$o$为输出神经元的个数。
使用演示：

```python
from torch import nn
import torch

model = nn.Linear(2, 1) # 输入特征数为2，输出特征数为1

input = torch.Tensor([1, 2]) # 给一个样本，该样本有2个特征（这两个特征的值分别为1和2）
output = model(input)
output

tensor([-1.4166], grad_fn=<AddBackward0>)
```

我们的输入为`[1,2]`，输出了`[-1.4166]`。可以查看模型参数验证一下上述的式子：

```python
# 查看模型参数
for param in model.parameters():
    print(param)
    
Parameter containing:
tensor([[ 0.1098, -0.5404]], requires_grad=True)
Parameter containing:
tensor([-0.4456], requires_grad=True)

```

可以看到，模型有3个参数，分别为两个权重和一个偏执。计算可得：

$y = [1, 2] * [0.1098, -0.5404]^T - 0.4456 = -1.4166$

# PytorchVision Transforms

## Transforms的基本概念

transforms是torchvision下的一个模块，主要帮助用户方便的对图像数据进行处理

它要求数据是`(C, H, W)`的三维数组，其中字母含义为：

- `C`: Channel, 图片的通道，例如R、G、B
- `H,W`, Height, Weight，图片的宽高

## **使用PIL读取一张图片**

在使用Transforms前，先读取一张图片，用于后续使用

```python
from PIL import Image

image = Image.open("images/mary.jpg")
image
```

## **Transforms的常用方法**

**Transforms的常用方法有如下**：

1.`ToTensor()`: 将一个`PIL Image`或一个`numpy.ndarray`转为Tensor

```python
trans = transforms.ToTensor()
img_data = trans(image)
img_data.shape

torch.Size([3, 225, 225])
```

输出`[3, 255, 255]`表示有3个通道（R,G,B），每个通道有255x255个像素点

2.`Normalize(mean, std, inplace=False)`: 将tensor归一化为均值为`mean`，方差为`std`的数据

```python
# 将三个通道分别做归一化
# 第一个通道归一化为 均值为0，方差为1
# 第二个通道归一化为 均值为1，方差为2
# 第三个通道归一化为 均值为2，方差为3
img_data = transforms.Normalize(mean=(0, 1, 2), std=(1,2,3))(img_data)
img_data.shape

torch.Size([3, 225, 225])
```

## **Transforms的Compose方法**

一张图片可能需要执行很多次Transforms方法，所以Transform提供了Compose方法，方便用户一次将其全部处理完毕

```python
img_data = transforms.ToTensor()(image)
img_data = transforms.Normalize(mean=(0, 1, 2), std=(1,2,3))(img_data)

compose = transforms.Compose(
    [ # 将要对图片做的处理，全部一次性写全
          transforms.ToTensor(),
          transforms.Normalize(mean=(0, 1, 2), std=(1,2,3))
    ]
)
compose(image).equal(img_data)

True
```

# Pytorch中DataLoader和Dataset

## **DataLoader支持的两种数据集**

1. Map格式：即key,value形式，例如 {0: ‘张三’, 1: ‘李四’}
2. Iterator格式：例如数组，迭代器等

## **Iterator格式的DataLoader**

Python中，只要可以for循环的数据，都是Iterator格式的数据。

### **Python的Iterator格式数据简介**

```python
data = [0,1,2,3,4]

for item in data:
    print(item, end=' ')

0 1 2 3 4 
```

上例子中，`list`数据类型是一个迭代器，for循环本质是每次调用了next函数。即其“效果”等价于下面的代码

```python
data = [0,1,2,3,4]
data_iter = iter(data) # 返回一个迭代器

item = next(data_iter, None) # 获取迭代器的下一个值
while item is not None:
    print(item, end=' ')
    item = next(data_iter, None)
    
0 1 2 3 4 
```

### **Pytorch使用DataLoader**

```python
from torch.utils.data import DataLoader

data = [i for i in range(100)] # 定义数据集，需要是一个可迭代的对象

"""
定义dataloader，其接受三个重要的参数
- dataset: 数据集
- batch_size: 要将数据集切分为多少份
- shuffle: 是否对数据集进行随机排序
"""
dataloader = DataLoader(dataset=data, batch_size=6, shuffle=False) 

for i, item in enumerate(dataloader): # 迭代输出
    print(i, item)

0 tensor([0, 1, 2, 3, 4, 5])
1 tensor([ 6,  7,  8,  9, 10, 11])
... 省略
15 tensor([90, 91, 92, 93, 94, 95])
16 tensor([96, 97, 98, 99])
```

上面例子中，输入一个数据集`0~99`，通过`dataloader`将数据集分成100/6 =17份，每份6个数据，最后一份因为不满6个，所以只返回了4个。

### **使用自定义的IterableDataset**

```python
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

class MyDataset(IterableDataset):
    
    def __init__(self):
        print('init...')
    
    def __iter__(self):
        print('iter...') # 获取迭代器
        self.n = 1
        return self
    
    def __next__(self):
        print('next...') # 获取下一个元素
        x = self.n
        self.n += 1
        
        if x >= 100: # 当x到100时停止
            raise StopIteration
        return x

dataloader = DataLoader(MyDataset(), batch_size=5)

for i, item in enumerate(dataloader):
    print(i, item)

init...
iter...
next... next... next... next... next... 0 tensor([1, 2, 3, 4, 5])
next... next... next... next... next... 1 tensor([ 6,  7,  8,  9, 10])
... 省略
next... next... next... next... next... 18 tensor([91, 92, 93, 94, 95])
next... next... next... next... next... 19 tensor([96, 97, 98, 99])

```

从上面的例子可以看出，可迭代对象在初始化中会调用一次`__init__`方法，在获取迭代器时会调用一次`__iter__`方法，之后在获取元素时，每获取一个元素都会调用一次`__next__`方法

### **实战：自定义图片加载DataLoader**

任务：从data\faces文件夹中读取图片，并做一定处理，然后通过dataloader加载。

```python
数据集：
链接：https://pan.baidu.com/s/1UrygNmmfzcWdjb29JHEpFg 
提取码：toyd 
```

1. **定义ImageDataset**

```python
import os
from torch.utils.data import IterableDataset
import torchvision
import torchvision.transforms as transforms

class ImageDataset(IterableDataset):
    def __init__(self, filepath):
        fnames = [filepath + '/' + filename for filename in os.listdir(filepath)] # 读取所有图片的文件路径
        self.i = -1 # 记录当前读取到的图片的下标
        self.compose = compose = [  # 图片的transform
                            transforms.ToPILImage(),
                            transforms.Resize((64, 64)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                        ]
    
    def __len__(self):
        return len(fnames) # 假设文件夹没有其他无关文件
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.i += 1
        if self.i >= len(self.fnames):
            raise StopIteration
        img = torchvision.io.read_image(fnames[self.i]) # 读取第i个图片
        transform = transforms.Compose(self.compose) # 对图片进行处理
        return transform(img) # 返回处理后的图片
```

1. **实例化dataset和dataloader**

```python
dataset = ImageDataset('./data/faces')
print(next(iter(dataset)).shape)

dataloader = DataLoader(dataset, batch_size=16)
print(dataloader)

torch.Size([3, 64, 64])
<torch.utils.data.dataloader.DataLoader object at 0x000001C10B6048E0>
```

1. **使用dataloader**

```python
import matplotlib.pyplot as plt

grid_img = torchvision.utils.make_grid(next(iter(dataloader)), nrow=4)
plt.figure(figsize=(10,10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()
```

## **Map格式的DataLoader**

```python
dataset = {0: '张三', 1:'李四', 2:'王五', 3:'赵六'}

dataloader = DataLoader(dataset, batch_size=2)

for i, value in enumerate(dataloader):
    print(i, value)

0 ['张三', '李四']
1 ['王五', '赵六']
```

### **自定义Map类型的Dataset**

自定义Map类型的Dataset只需要定义类，并继承 `torch.utils.data.Dataset` 方法即可，但要实现两个重要方法：`__getitem__(self, index)` 和 `__len__(self)`

```python
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomerDataSet(Dataset):

    def __init__(self):
        super(CustomerDataSet, self).__init__()
        self.data_dict = ['张三', '李四', '王五', '赵六']

    def __getitem__(self, index):
        return self.data_dict[index]

    def __len__(self):
        return len(self.data_dict)
```

其实这和上面的`dict`有异曲同工之妙，Dataloader会根据你dataset的大小，然后传一个index (0<=index<len(dataset)) 给`getitem` 方法，然后你返回该index对应的数据即可。例如：

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(CustomerDataSet(), batch_size=2, shuffle=True)
for i, value in enumerate(dataloader):
    print(i, value)

0 ['李四', '张三']
1 ['王五', '赵六']
```

在上面例子中，Dataloader的执行过程为：

1. 调用 len(dataset) 方法，获取dataset的长度，这里为 4
2. 然后生成 index list，即 [0,1,2,3]
3. 因为传了shuffle=True，所以将index顺序打乱，结果为：[1,0,2,3]
4. 然后按照顺序调用getitem方法，即：getitem(1)、getitem(0)、getitem(2)、getitem(3)
5. 根据batch_size进行返回，第一次返回两个getitem(1)、getitem(0)，第二次返回getitem(2)、getitem(3)

# **Pytorch详解NLLLoss和CrossEntropyLoss**

## **NLLLoss**

在图片单标签分类时，输入m张图片，输出一个m*N的Tensor，其中N是分类个数。比如输入3张图片，分三类，最后的输出是一个3*3的Tensor，举个例子：

![Untitled](Pytorch%20c8e70d807fac42489e2e73e423f3df30/Untitled.png)

第123行分别是第123张图片的结果，假设第123列分别是猫、狗和猪的分类得分。

可以看出模型认为第123张都更可能是猫。

然后对每一行使用Softmax，这样可以得到每张图片的概率分布。

![Untitled](Pytorch%20c8e70d807fac42489e2e73e423f3df30/Untitled%201.png)

这里dim的意思是计算Softmax的维度，这里设置dim=1，可以看到每一行的加和为1。比如第一行0.6600+0.0570+0.2830=1。

![Untitled](Pytorch%20c8e70d807fac42489e2e73e423f3df30/Untitled%202.png)

如果设置dim=0，就是一列的和为1。比如第一列0.2212+0.3050+0.4738=1。

我们这里一张图片是一行，所以dim应该设置为1。

然后对Softmax的结果取自然对数：

![Untitled](Pytorch%20c8e70d807fac42489e2e73e423f3df30/Untitled%203.png)

Softmax后的数值都在0~1之间，所以ln之后值域是负无穷到0。
NLLLoss的结果就是把上面的输出与Label对应的那个值拿出来，再去掉负号，再求均值。
假设我们现在Target是[0,2,1]（第一张图片是猫，第二张是猪，第三张是狗）。第一行取第0个元素，第二行取第2个，第三行取第1个，去掉负号，结果是：[0.4155,1.0945,1.5285]。再求个均值，结果是：

![Untitled](Pytorch%20c8e70d807fac42489e2e73e423f3df30/Untitled%204.png)

下面使用NLLLoss函数验证一下：

![Untitled](Pytorch%20c8e70d807fac42489e2e73e423f3df30/Untitled%205.png)

## **CrossEntropyLoss**

CrossEntropyLoss就是把以上Softmax–Log–NLLLoss合并成一步，我们用刚刚随机出来的input直接验证一下结果是不是1.0128：

![Untitled](Pytorch%20c8e70d807fac42489e2e73e423f3df30/Untitled%206.png)

# Pytorch nn.Embedding

## 什么情况下使用Embedding

假设要对单词进行编码，就会使用到one-hot

1. 首先要建立一个字典，例如

| 单词 | 索引 |
| --- | --- |
| hello | 0 |
| i | 1 |
| am | 2 |
| … | … |
1. 建立好词典后，使用一个与词典一样大小的数组，将要编码的单词对应索引下表改为1即可，例如"am"的编码为`[0,0,1,0,0,...]`，该数组的大小与词典大小一致

one-hot的最大的缺点显而易见：词典有多大，一个单词对应的数组向量的维度就有多大。当然还有其他缺点，例如：词和词之间没有关系，做不到king - queen = man - women。

所以需要有一个方法可以降维，可以将单词编码成指定的维度，例如，将"am"(3)编码成5维向量[-0.972,0.371,-0.172,0.581,0.134]

Embedding就是干这个事的

## nn.Embedding的基本用法

nn.Embeddding接受两个重要参数：
例如，我们现在词典大小为20，现在要对hello, i, am，这三个单词进行编码，想将其编码为5维向量，则对应代码为：

1. num_embeddings：字典的大小。对应上面词典的大小，如果你的词典中包含5000个单词，那么这个字段就填5000
2. embedding_dim：要将单词编码成多少维的向量

```python
import torch
from torch import nn

embedding = nn.Embedding(20, 5)
embedding(torch.LongTensor([0,1,2]))

tensor([[ 0.4471,  0.3875, -1.0195, -1.1125,  1.3481],
        [-1.7230, -0.1964, -0.0420,  0.5782,  0.4514],
        [-0.0310, -1.9674, -1.1344, -1.6752,  1.0801]],
       grad_fn=<EmbeddingBackward0>)

```

使用nn.Embedding，将*0*(hello)编码成了`[ 0.4471, 0.3875, -1.0195, -1.1125, 1.3481]`

> **使用注意事项**：
> 
> 1. embedding只接受LongTensor类型的数据
> 2. embedding的数据不能大于等于词典大小，例如上面指定了词典大小为20，那么要编码的索引大小就不能>=20

## **nn.Embedding的其他常用参数**

`padding_idx`：填充索引，即，如果是这个所以，一律编码为0。有时我们的字典里会增加一项`unknown`代表未知的单词，这样我们就可以使用该参数，对所有unknown的单词都编码成0。

```python
embedding = nn.Embedding(20, 5, padding_idx=3) # 加上unknown对应的索引是3
embedding(torch.LongTensor([0,1,2,3,4]))

tensor([[ 0.3824, -0.6734, -2.1156,  1.7065,  1.2072],
        [-0.5977, -1.0876,  0.6169,  1.4566,  0.0325],
        [ 1.1299,  0.5794, -1.5166,  0.1036,  0.3793],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-1.0468,  1.3305, -2.1740, -0.5534, -0.1062]],
       grad_fn=<EmbeddingBackward0>)
```

可以看到，`3`索引的编码全为0

## **nn.Embedding的可学习性**

nn.Embedding中的参数并不是一成不变的，它也是会参与梯度下降的。也就是更新模型参数也会更新nn.Embedding的参数，或者说nn.Embedding的参数本身也是模型参数的一部分。

举个例子：

```python
embedding = nn.Embedding(20, 5, padding_idx=3) # 对3不进行编码
optimizer = torch.optim.SGD(embedding.parameters(), lr=0.1)
criteria = nn.MSELoss()

for i in range(1000):
    outputs = embedding(torch.LongTensor([0,1,2,3,4]))
    loss = criteria(outputs, torch.ones(5, 5))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

在上面例子中，我对`nn.Embedding`不断的计算损失和梯度下降，让其编码往1的方向靠近，训练结束后，我们再次尝试使用`embedding`进行编码：

```python
embedding(torch.LongTensor([0,1,2,3,4]))

tensor([[1.0004, 0.9999, 1.0000, 1.0000, 1.0000],
        [0.9999, 0.9996, 0.9997, 0.9999, 0.9996],
        [0.9999, 0.9996, 0.9994, 0.9993, 0.9991],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.9991, 0.9999, 0.9999, 0.9998, 0.9997]],
       grad_fn=<EmbeddingBackward0>)

```

可以看到，经过训练后，embedding的参数发生了变化，把它们都编码成了1。

# **torch.nn.utils.clip_grad_norm_ 的使用与原理**

## **clip_grad_norm_的原理**

本文是对[梯度剪裁: torch.nn.utils.clip_grad_norm_()](https://blog.csdn.net/Mikeyboi/article/details/119522689)文章的补充。所以可以先参考这篇文章

从上面文章可以看到，`clip_grad_norm`最后就是对所有的梯度乘以一个`clip_coef`，而且乘的前提是**clip_coef一定是小于1的**，所以，按照这个情况：**`clip_grad_norm`只解决梯度爆炸问题，不解决梯度消失问题**

## **clip_grad_norm_参数的选择（[调参](https://so.csdn.net/so/search?q=%E8%B0%83%E5%8F%82&spm=1001.2101.3001.7020)）**

从上面文章可以看到，clip_coef的公式为：

$clip\_coef = \frac{max\_norm}{total\_norm}$

**max_norm的取值：**

假定忽略clip_coef > 1的情况，则可以根据公式推断出：

1. clip_coef越小，则对梯度的裁剪越厉害，即，使梯度的值缩小的越多
2. max_norm越小，clip_coef越小，所以，max_norm越大，对于梯度爆炸的解决越柔和，max_norm越小，对梯度爆炸的解决越狠、

> max_norm可以取小数
> 

接下来看下**total_norm和norm_type的取值**：

从上面文章可以看到，**total_norm受梯度大小和norm_type的影响**，通过公式很难直观的感受到，这里我通过实验得出了以下结论（不严谨，欢迎勘误）：

1. **梯度越大，total_norm值越大，进而导致clip_coef的值越小，最终也会导致对梯度的裁剪越厉害**，很合理
2. **norm_type不管取多少，对于total_norm的影响不是太大（1和2的差距稍微大一点），所以可以直接取默认值2**
3. **norm_type越大，total_norm越小**（实验观察到的结论，数学不好，不会证明，所以本条不一定对）

实验过程如下：

首先我对源码进行了一些修改，将.grad去掉，增加了一些输出，方便进行实验：

```python
import numpy as np
import torch
from torch import nn

def clip_grad_norm_(parameters, max_norm, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == np.inf:
        total_norm = max(p.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.data.mul_(clip_coef)
            
    print("max_norm=%s, norm_type=%s, total_norm=%s, clip_coef=%s" % (max_norm, norm_type, total_norm, clip_coef))
```

只改变norm_type的情况下，各变量值的变化：

```python
for i in range(1, 5):
    clip_grad_norm_(torch.Tensor([125,75,45,15,5]), max_norm=1, norm_type=i)
    
clip_grad_norm_(torch.Tensor([125,75,45,15,5]), max_norm=1, norm_type=np.inf)

max_norm=1.0, norm_type=1.0, total_norm=265.0, clip_coef=0.0037735848914204344
max_norm=1.0, norm_type=2.0, total_norm=153.3786163330078, clip_coef=0.006519813631054263
max_norm=1.0, norm_type=3.0, total_norm=135.16899108886716, clip_coef=0.007398146457602848
max_norm=1.0, norm_type=4.0, total_norm=129.34915161132812, clip_coef=0.007731013151704421
max_norm=1.0, norm_type=inf, total_norm=tensor(125.), clip_coef=tensor(0.0080)

```

只改变梯度，各变量值的变化：

```python
for i in range(1, 5):
    clip_grad_norm_(torch.Tensor([125*i,75,45,15,5]), max_norm=1, norm_type=2)

max_norm=1.0, norm_type=2.0, total_norm=153.3786163330078, clip_coef=0.006519813631054263
max_norm=1.0, norm_type=2.0, total_norm=265.3299865722656, clip_coef=0.003768891745519864
max_norm=1.0, norm_type=2.0, total_norm=385.389404296875, clip_coef=0.0025947781289671814
max_norm=1.0, norm_type=2.0, total_norm=507.83856201171875, clip_coef=0.001969129705451148
```

## **clip_grad_norm_使用演示**

在上面文章还提到一个重要的事情：**`clip_grad_norm_`要放在`backward`和`step`之间**。接下来我会实际演示梯度在训练过程中的变化，并解释要这么做的原因：

首先定义一个测试模型：

```python
class TestModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(1,1, bias=False),
            nn.Sigmoid(),
            nn.Linear(1,1, bias=False),
            nn.Sigmoid(),
            nn.Linear(1,1, bias=False),
            nn.Sigmoid(),
            nn.Linear(1,1, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.model(x)

model = TestModel()
```

定义好模型后，固定一下模型参数：

```python
for param in model.parameters():
    param.data = torch.Tensor([[0.5]])
    print("param=%s" % (param.data.item()))
    

param=0.5
param=0.5
param=0.5
param=0.5
```

可以看目前四个线性层的权重参数都为0.5。之后对模型进行一轮训练，并进行反向传播：

```python
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1)
predict_y = model(torch.Tensor([0.1]))
loss = criterion(predict_y, torch.Tensor([1]))
model.zero_grad()
loss.backward()
```

反向传播过后，再次打印模型参数，可以看到反向传播后计算好的各个参数的梯度：

```python
for param in model.parameters():
    print("param=%s, grad=%s" % (param.data.item(), param.grad.item()))

param=0.5, grad=-3.959321111324243e-05
param=0.5, grad=-0.0016243279678747058
param=0.5, grad=-0.014529166743159294
param=0.5, grad=-0.11987950652837753
```

重点来了，各个参数的梯度如上图所示（越靠近输入的位置，梯度越小，虽然没有出现梯度爆炸，反而出现了梯度消失，但不影响本次实验），现在对其进行梯度裁剪：

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01208, norm_type=2)

tensor(0.1208)
```

在上面，我传入的max_norm=0.01208，而total_norm=0.1208，所以可得clip_coef=0.1，即**所有的梯度都会缩小一倍**，此时我们再打印一下梯度：

```python
for param in model.parameters():
    print("param=%s, grad=%s" % (param.data.item(), param.grad.item()))

param=0.5, grad=-3.960347839893075e-06
param=0.5, grad=-0.00016247491294052452
param=0.5, grad=-0.001453293371014297
param=0.5, grad=-0.01199105940759182
```

看到没，所有的梯度都减小了10倍。之后我们执行`step()`操作，其就会将进行`param=param-lr*grad`操作来进行参数更新。再次打印网络参数：

```python
optimizer.step()

for param in model.parameters():
    print("param=%s, grad=%s" % (param.data.item(), param.grad.item()))

param=0.5000039339065552, grad=-3.960347839893075e-06
param=0.5001624822616577, grad=-0.00016247491294052452
param=0.5014532804489136, grad=-0.001453293371014297
param=0.5119910836219788, grad=-0.01199105940759182
```

可以看到，在执行`step`后，执行了`param=param-grad`操作（我设置的lr为1）。同时，[grad](https://so.csdn.net/so/search?q=grad&spm=1001.2101.3001.7020)并没有清0，所以这也是为什么要显式的调用`zero_grad`的原因。

# 反卷积与nn.ConvTranspose2d

## 反卷积的作用

**传统的卷积通常是将大图片卷积成一张小图片，而反卷积就是反过来，将一张小图片变成大图片**。
但这有什么用呢？其实有用，例如，在生成网络(GAN)中，我们是给网络一个向量，然后生成一张图片

![Untitled](Pytorch%20c8e70d807fac42489e2e73e423f3df30/Untitled%207.png)

所以我们需要想办法把这个向量一直扩，最终扩到图片的的大小。

## 卷积中padding的几个概念

### No Padding

![b94bfe1dc1dd5e7a5ee76f709c2a2de4.gif](Pytorch%20c8e70d807fac42489e2e73e423f3df30/b94bfe1dc1dd5e7a5ee76f709c2a2de4.gif)

**No Padding就是padding为0，这样卷积之后图片尺寸就会缩小**，这个大家应该都知道

> 下面的图片都是 蓝色为输入图片，绿色为输出图片。
> 

### Half(Same) Padding

![ca8ccf05c4fd41f9b5daff2542595c5e.gif](Pytorch%20c8e70d807fac42489e2e73e423f3df30/ca8ccf05c4fd41f9b5daff2542595c5e.gif)

Half Padding也称为Same Padding，先说Same，Same指的就是输出的图片和输入图片的大小一致，而在stride为1的情况下，若想让输入输出尺寸一致，需要指定 $p=\lfloor k/2 \rfloor$，这就是 Half 的由来，即padding数为kerner_size的一半。

在 pytorch 中支持same padding，例如：

```python
inputs = torch.rand(1, 3, 32, 32)
outputs = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding='same')(inputs)
outputs.size()

torch.Size([1, 3, 32, 32])
```

### Full Padding

![91087fd7a45f439b918e24636a25b448.gif](Pytorch%20c8e70d807fac42489e2e73e423f3df30/91087fd7a45f439b918e24636a25b448.gif)

当  $p=k-1$ 时就达到了 Full Padding。为什么这么说呢？可以观察上图，$k=3$，$p=2$，此时在第一格卷积的时候，只有一个输入单位参与了卷积。假设$p=3了$，那么就会存在一些卷积操作根本没有输入单位参与，最终导致值为0，那跟没做一个样。

我们可以用pytorch做个验证，首先我们来一个Full Padding：

```python
inputs = torch.rand(1, 1, 2, 2)
outputs = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=2, bias=False)(inputs)
outputs

tensor([[[[-0.0302, -0.0356, -0.0145, -0.0203],
          [-0.0515, -0.2749, -0.0265, -0.1281],
          [ 0.0076, -0.1857, -0.1314, -0.0838],
          [ 0.0187,  0.2207,  0.1328, -0.2150]]]],
       grad_fn=<SlowConv2DBackward0>)

```

可以看到此时的输出都是正常的，我们将padding再增大，变为3：

```python
inputs = torch.rand(1, 1, 2, 2)
outputs = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=3, bias=False)(inputs)
outputs

tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0000,  0.1262,  0.2506,  0.1761,  0.3091,  0.0000],
          [ 0.0000,  0.3192,  0.6019,  0.5570,  0.3143,  0.0000],
          [ 0.0000,  0.1465,  0.0853, -0.1829, -0.1264,  0.0000],
          [ 0.0000, -0.0703, -0.2774, -0.3261, -0.1201,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]]],
       grad_fn=<SlowConv2DBackward0>)
```

可以看到最终输出图像周围多了一圈 0，这就是部分卷积没有输入图片参与，导致无效了计算。

## 反卷积

反卷积其实和卷积是一样的，只不是参数对应关系有点变化。例如：

![87bb0f8f620d4d0698fd049876fe7752.gif](Pytorch%20c8e70d807fac42489e2e73e423f3df30/87bb0f8f620d4d0698fd049876fe7752.gif)

这是一个padding=0的反卷积，这时候你肯定要问了，这padding分明是2嘛，你怎么说是0呢？请看下面

### 反卷积中的Padding参数

在传统卷积中，我们的 padding 范围为$[0, k-1]$，$p=0$被称为 No padding，$p=k-1$被称为 Full Padding。

而在反卷积中的 p'刚好相反，也就是$p' = k-1 - p$ 。也就是当我们传 $p'=0$
时，相当于在传统卷积中传了$p=k-1$，而传$p'=k-1$时，相当于在传统卷积中传了$p=0$。

我们可以用如下实验进行验证：

```python
inputs = torch.rand(1, 1, 32, 32)
# 定义反卷积，这里 p'=2, 为反卷积中的Full Padding
transposed_conv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, padding=2, bias=False)
# 定义卷积，这里p=0，为卷积中的No Padding
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0, bias=False)
# 让反卷积与卷积kernel参数保持一致，这里其实是将卷积核参数的转置赋给了反卷积
transposed_conv.load_state_dict(OrderedDict([('weight', torch.Tensor(np.array(conv.state_dict().get('weight'))[:, :, ::-1, ::-1].copy()))]))
# 进行前向传递
transposed_conv_outputs = transposed_conv(inputs)
conv_outputs = conv(inputs)

# 打印卷积输出和反卷积输出的size
print("transposed_conv_outputs.size", transposed_conv_outputs.size())
print("conv_outputs.size", conv_outputs.size())

# 查看它们输出的值是否一致。
#（因为上面将参数转为numpy，又转了回来，所以其实卷积和反卷积的参数是有误差的，
# 所以不能直接使用==，采用了这种方式，其实等价于==）
(transposed_conv_outputs - conv_outputs) < 0.01

transposed_conv_outputs.size:  torch.Size([1, 1, 30, 30])
conv_outputs.size:  torch.Size([1, 1, 30, 30])

tensor([[[[True, True, True, True, True, True, True, True, True, True, True,
		 .... //略

```

从上面例子可以看出来，反卷积和卷积其实是一样的，区别就几点：

1. 反卷积进行卷积时，使用的参数是kernel的转置，但这项其实我们不需要关心
2. 反卷积的padding参数 $p'$和传统卷积的参数 p的对应关系为$p'=k-1-p$
。换句话说，卷积中的no padding对应反卷积的full padding；卷积中的full padding对应反卷积中的no padding。
3. 从2中还可以看到一个事情，在反卷积中 $p'$不能无限大，最大值为$k-1-p$。（其实也不是哦）

题外话，不感兴趣去可以跳过，在上面第三点我们说了 p'的最大值为 $k-1-p$，但实际你用pytorch实验会发现，$p'$是可以大于这个值的。而这背后，相当于是对原始图像做了裁剪。

在pytorch的nn.Conv2d中，padding是不能为负数的，会报错，但有时可能你需要让padding为负数（应该没这种需求吧），此时就可以用反卷积来实现，例如：

```python
inputs = torch.ones(1, 1, 3, 3)
transposed_conv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=1, padding=1, bias=False)
print(transposed_conv.state_dict())
outputs = transposed_conv(inputs)
print(outputs)

OrderedDict([('weight', tensor([[[[0.7700]]]]))])
tensor([[[[0.7700]]]], grad_fn=<SlowConvTranspose2DBackward0>)

```

上述例子中，我们传给网络的是图片：
$\begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}$

但是我们传的$p'=1, k=1$，这样在传统卷积中相当于 $p=k-1-p'=-1$，相当于 Conv2d(padding=-1)，这样在做卷积时，其实是对图片[1] 在做卷积（因为把周围裁掉了一圈），所以最后输出的尺寸为$(1,1,1,1)$

这个题外话好像没啥实际用途，就当是更加理解反卷积中的padding参数吧。

### **反卷积的stride参数**

反卷积的stride这个名字有些歧义，感觉起的不怎么好，具体什么意思可以看下图：

![87bb0f8f620d4d0698fd049876fe7752.gif](Pytorch%20c8e70d807fac42489e2e73e423f3df30/87bb0f8f620d4d0698fd049876fe7752%201.gif)

![c30e9b90061d4cb5bd781b19e98ba042.gif](Pytorch%20c8e70d807fac42489e2e73e423f3df30/c30e9b90061d4cb5bd781b19e98ba042.gif)

左边是stride=1（称为No Stride）的反卷积，右边是stride=2 的反卷积。可以看到，他们的区别就是在原始图片的像素点中间填充了0。没错，**在反卷积中，stride参数就是表示往输入图片每两个像素点中间填充0，而填充的数量就是 stride - 1**。

例如，我们对32x32的图片进行反卷积，stride=3，那么它就会在每两个像素点中间填充两个0，原始图片的大小就会变成$32+31\times 2=94$。用代码实验一下：

```python

inputs = torch.ones(1, 1, 32, 32)
transposed_conv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, padding=2, stride=3, bias=False)
outputs = transposed_conv(inputs)
print(outputs.size())

torch.Size([1, 1, 92, 92])
```

我们来算一下，这里我使用了反卷积的Full Padding（相当于没有对原始图像的边缘进行padding），然后stride传了3，相当于在每两个像素点之间填充两个0，那么原始图像就会变成 94x94 的，然后kernal是3，所以最终的输出图片大小为 $94-3+1=92$.

### **反卷积的output_padding参数**

不知道你有没有发现，如果卷积和反卷积的参数一致，卷积会让 $A$ 尺寸变为 $B$尺寸，那么反卷积就会将 $B$尺寸变为 $A$尺寸。

举个例子：

```python
inputs = torch.rand(1, 1, 32, 32)
outputs = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=18, padding=3, stride=1)(inputs)
outputs.size()

torch.Size([1, 1, 21, 21])
```

我们这里将32x32的图片通过卷积变为了 21x21。此时我们将卷积变为反卷积（参数不变），输入图片大小变为 21x21：

```python
inputs = torch.rand(1, 1, 21, 21)
outputs = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=18, padding=3, stride=1)(inputs)
outputs.size()

torch.Size([1, 1, 32, 32])
```

看，反卷积将 21x21 的图片又变回了 32x32，这也就是为什么要叫反卷积。

但。。，真的是这样嘛，我们再看一个例子：

```python
inputs = torch.rand(1, 1, 7, 7)
outputs = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0, stride=2)(inputs)
outputs.size()

torch.Size([1, 1, 3, 3])

inputs = torch.rand(1, 1, 8, 8)
outputs = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0, stride=2)(inputs)
outputs.size()

torch.Size([1, 1, 3, 3])

inputs = torch.rand(1, 1, 3, 3)
outputs = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, padding=0, stride=2)(inputs)
outputs.size()

torch.Size([1, 1, 7, 7])
```

上面我们对7x7和8x8的图片都使用卷积操作，他们最后结果都是3x3，这样反卷积就会存在歧义，而反卷积默认选择了转换为7x7。原因可以见下图：

![373077c2fa114beebaaa560d1f1010d8.gif](Pytorch%20c8e70d807fac42489e2e73e423f3df30/373077c2fa114beebaaa560d1f1010d8.gif)

从这张图可以看到，8x8的图片其实最右边和最下边的一行是没有参与卷积运算的，这是因为stride为2，再走2步就超出图片范围了。所以7x7和8x8最终的结果都为3x3。

那么如果我们想让3x3的反卷积得8x8而不是7x7，那么我们就需要在输出图片边缘补充数据，具体补几行就是output_padding指定的。所以output_padding的作用就是：在输出图像右侧和下侧补值，用于弥补stride大于1带来的缺失。其中output_stadding必须小于stride

例如：

```python
inputs = torch.rand(1, 1, 3, 3)
outputs = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, padding=0, stride=2, output_padding=1)(inputs)
outputs

```

![Untitled](Pytorch%20c8e70d807fac42489e2e73e423f3df30/Untitled%208.png)

具体这个 0.2199 是什么我也不太清楚，我测试了发现并不是平均值

## 反卷积总结

1. **反卷积的作用是将原始图像进行扩大**
2. 反卷积与传统卷积的区别不大，主要区别有：
    - padding的对应关系变了，反卷积的padding参数 $p' = k-1-p$
    。其中 k kk 是kernel_size, p为传统卷积的padding值；
    - **stride参数的含义不一样，在反卷积中stride表示在输入图像中间填充0，每两个像素点之间填充的数量为 stride-1**
    - 除了上述的俩参数外，其他参数没啥区别
3. **如果卷积和反卷积的参数一致，卷积会让 AA 尺寸变为 BB 尺寸，那么反卷积就会将 BB 尺寸变为 AA 尺寸**
4. **output_padding的作用就是：在输出图像右侧和下侧补值，用于弥补stride大于1带来的缺失**。其中output_stadding必须小于stride