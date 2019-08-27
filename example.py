import matplotlib.pyplot as plt

from torchvision.transforms import ToTensor
from torchvision.models import resnet18
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
