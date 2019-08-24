
# PytorchModulePCA

## Install

pip install git+https://github.com/FrancescoSaverioZuppichini/PytorchModulePCA.git


## Getting started

First we need to load `PytorchModulePCA`


```python
%load_ext autoreload
%autoreload 2
```


```python
import matplotlib.pyplot as plt
from PytorchModulePCA import PytorchModulePCA
```


```python
%matplotlib notebook
plt.rcParams['figure.figsize'] = [9, 9]
```


```python
TRAIN = False
```

Then we need some data to work with, let's use the MNIST dataset

## Dataset


```python
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from torchvision.datasets import MNIST
from fastai.vision import *
from torch.utils.data import DataLoader

tr = Compose([Grayscale(), ToTensor()])

train_ds = MNIST(root='~/Documents/datasets/', download=True, transform=tr)
train_dl = DataLoader(train_ds, num_workers=14, batch_size=128, shuffle=True)

val_ds = MNIST(root='~/Documents/datasets/', train=False, transform=tr)
val_dl = DataLoader(val_ds, num_workers=14, batch_size=128, shuffle=False)

data = ImageDataBunch(train_dl, val_dl)
```

After, we need a model to visualise

## Model


```python
from PytorchModulePCA.utils import device 
model = simple_cnn((1,16,32,64)).to(device())

learn = Learner(data, model, path='./', loss_func=CrossEntropyFlat())
learn.metrics=[accuracy]
```

### Train


```python
if TRAIN:
    learn.fit(25)
    learn.save('learn', return_path=True)
```


```python
learn.load('./learn')

learn.validate(metrics=[accuracy])
```

## Compute PCA on the last conv layer
`PytorchModulePCA` will run PCA on each batch and it stores only the points, the labels and the indeces of the dataset in RAM


```python
module_pca = PytorchModulePCA(learn.model, learn.model[3][0], learn.data.valid_dl)
module_pca(k=2)
module_pca.plot()
plt.savefig("./images/0.png") 
module_pca.annotate(zoom=0.8)
plt.savefig("./images/1.png") 
```
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModulePCA/master/images/0.png)
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModulePCA/master/images/1.png)
Yeah, it is a mess! We have too many points

### Reduce
We can reduce the number of points by calling `.reduce`. By default it uses **kmeans** to properly select the new points.


```python
reduced_module_pca = module_pca.reduce(to=100)
reduced_module_pca.plot()
plt.savefig("./images/2.png") 
reduced_module_pca.annotate(zoom=0.8)
plt.savefig("./images/3.png") 
```
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModulePCA/master/images/2.png)
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModulePCA/master/images/3.png)
## 3D


```python
module_pca3d = PytorchModulePCA(learn.model, learn.model[3][0], learn.data.valid_dl)
module_pca3d(k=3)
module_pca3d.plot()
plt.savefig("./images/4.png") 
```
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModulePCA/master/images/4.png)

### Reduce


```python
reduced_module_pca3d = module_pca3d.reduce(to=100)
reduced_module_pca3d.plot()
plt.savefig("./images/5.png") 
reduced_module_pca3d.annotate()
plt.savefig("./images/6.png") 
```
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModulePCA/master/images/5.png)
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModulePCA/master/images/6.png)