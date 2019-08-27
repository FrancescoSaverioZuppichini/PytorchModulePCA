
# PytorchModulePCA

PCA is a technique for dimensionality reduction.  It can be used to visualize CNN layers. As we know, CNN learns to map images features to something (e.g labels). By applying PCA on the last CNN layer we can see how well the network maps those features. For example, in the next image, we can see how similar images are close to each other meaning that the network correctly learn how to encode them.

![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModulePCA/master/images/2.png)

## Install

pip install git+https://github.com/FrancescoSaverioZuppichini/PytorchModulePCA.git

It needs the following packages

```
setuptools==41.0.1
torch==1.1.0
dataclasses==0.6
matplotlib==3.1.0
numpy==1.16.4
tqdm==4.32.1
scikit_learn==0.21.3
```

## Example
This example shows only how to use the API, the model is untrained so we can seee that most of the points of the same class are not close to each other.
```python
import matplotlib.pyplot as plt

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from PytorchModulePCA import PytorchModulePCA
from fastai.layers import simple_cnn

ds = MNIST(root='~/Documents/datasets/', download=True, transform=ToTensor())
dl = DataLoader(ds, num_workers=14, batch_size=128, shuffle=False)

model = simple_cnn((1, 16, 32, 10)).cuda() # a random model

last_conv_layer = model[2][0] # get the last conv layer

module_pca = PytorchModulePCA(model.eval(), last_conv_layer.eval(), dl)
module_pca(k=2, n_batches=4) # run only on 4 batches
module_pca.plot() # plot
plt.savefig('./images/example')
df = module_pca.state.to_df() # get the points as pandas df
print(df)

```
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModulePCA/master/images/example.png)
```
         points_0  points_1  y
indices                       
0        1.007328 -0.205802  5
1        0.736135 -1.251487  0
2       -0.287514  0.478662  4
3       -1.154645 -0.535809  1
4       -1.003071 -0.153210  9
5        0.357879 -0.255997  2
...
```

## Getting started
It follows an more detailed tutorial.
The code can be run using this [notebook](https://github.com/FrancescoSaverioZuppichini/PytorchModulePCA/blob/master/main.ipynb)

First we need to load `PytorchModulePCA` and some others packages

```python
import matplotlib.pyplot as plt
from PytorchModulePCA import PytorchModulePCA
```


```python
%matplotlib notebook
plt.rcParams['figure.figsize'] = [10, 10]
```


```python
TRAIN = False
```

Then we need some data to work with, let's use the CIFAR10 dataset

## Dataset


```python
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale, RandomHorizontalFlip, RandomVerticalFlip, Normalize

from torchvision.datasets import MNIST, CIFAR10
from fastai.vision import *
from torch.utils.data import DataLoader

train_tr = Compose([RandomHorizontalFlip(), RandomVerticalFlip(), ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
tr = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
train_ds = CIFAR10(root='~/Documents/datasets/', download=True, transform=train_tr)
train_dl = DataLoader(train_ds, num_workers=14, batch_size=128, shuffle=True)

val_ds = CIFAR10(root='~/Documents/datasets/',  download=True, train=False, transform=tr)
val_dl = DataLoader(val_ds, num_workers=14, batch_size=128, shuffle=False)

data = ImageDataBunch(train_dl, val_dl)
```

After, we need a model to visualise

## Model
Let's use resnet18 

![alt](https://hackernoon.com/hn-images/1*uJ0IrP9JXYE2hHAzMjorQA.png)



```python
from PytorchModulePCA.utils import device 
from torchvision.models import resnet18

model = resnet18(False).to(device())

last_conv_layer = model.layer4[-1].conv2
```

## Not trained

This is how PCA in the last conv layer looks like on a untrained model. We need to unnormalize the images to properly visualise them


```python
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:model[0][2]
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    
un_normalize = UnNormalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
```


```python
module_pca = PytorchModulePCA(model.eval(), last_conv_layer.eval(), data.valid_dl)
module_pca(k=2, n_batches=None)
module_pca = module_pca.reduce(to=200)
module_pca.plot()
plt.savefig("./images/7.png") 
module_pca.annotate(zoom=0.6, transform=un_normalize)
plt.savefig("./images/8.png") 
```
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModulePCA/master/images/7.png)
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModulePCA/master/images/8.png)

### Train
A quick random train. We are going to use fastai


```python
model = resnet18(True)

learn = Learner(data, model, path='./', loss_func=CrossEntropyFlat())
learn.metrics=[accuracy]
```


```python
if TRAIN:
    learn.fit(10, lr=1e-03)
    learn.fit(5, lr=1e-04)
    learn.save('learn', return_path=True)
```


```python
learn.load('./learn')
last_conv_layer = learn.model.layer4[-1].conv2
learn.validate(metrics=[accuracy])
```

## Compute PCA on the last conv layer
`PytorchModulePCA` will run PCA on each batch and it stores only the points, the labels and the indeces of the dataset in RAM



```python
plt.rcParams['figure.figsize'] = [10, 10]
```


```python
last_conv_layer = learn.model.layer4[-1].conv2
module_pca = PytorchModulePCA(learn.model.eval(), last_conv_layer.eval(), data.valid_dl)
module_pca(k=2)
```


```python
module_pca.plot()
plt.savefig("./images/0.png") 
```
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModulePCA/master/images/0.png)

Yeah, it is a mess! We have too many points

### Reduce
We can reduce the number of points by calling `.reduce`. By default it uses **kmeans** to properly select the new points.


```python
module_pca = module_pca.reduce(to=200)
module_pca.plot()
plt.savefig("./images/1.png") 
module_pca.annotate(zoom=0.6, transform=un_normalize)
plt.savefig("./images/2.png") 
```
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModulePCA/master/images/1.png)
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModulePCA/master/images/2.png)

## 3D


```python
module_pca3d = PytorchModulePCA(learn.model, last_conv_layer, learn.data.valid_dl)
module_pca3d(k=3)
module_pca3d.plot()
plt.savefig("./images/3.png") 
```
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModulePCA/master/images/3.png)

### Reduce


```python
reduced_module_pca3d = module_pca3d.reduce(to=200)
reduced_module_pca3d.plot()
plt.savefig("./images/4.png") 
reduced_module_pca3d.annotate(zoom=0.6, transform=un_normalize)
plt.savefig("./images/5.png") 
```
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModulePCA/master/images/4.png)
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModulePCA/master/images/5.png)
