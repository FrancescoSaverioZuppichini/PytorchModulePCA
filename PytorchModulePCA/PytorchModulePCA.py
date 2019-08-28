import torch

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dataclasses import dataclass
from PytorchStorage import ForwardModuleStorage
from sklearn.cluster import KMeans
from tqdm import tqdm_notebook as tqdm
from collections import OrderedDict
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D

from .utils import pca, ImageAnnotations3D, tensor2numpy, device


@dataclass
class State():
    points: torch.tensor = torch.empty(0)
    y: torch.tensor = torch.empty(0).long()
    indices: torch.tensor = torch.empty(0).long()

    def __repr__(self):
        return f"points={self.points.shape}"

    def numpy(self):
        return self.points.numpy(), self.y.numpy(), self.indices.numpy()

    def to_df(self):
        points, y, indices = self.numpy()
        dims = self.points.shape[-1]

        data = { f"points_{k}" : points[:,k] for k in range(dims)}
        data['y'] = y
        data['indices'] = indices

        df = pd.DataFrame(data=data)
        df = df.set_index('indices', drop=True)

        return df

class PytorchModulePCA():
    reducers = ['kmeans']
    plot_dimensions = [2, 3]
    """
    Apply and visualize PCA with k-features of a specific CNN-layer. 
    It computes the PCA values batch-wise to reduce memory usage and increase performance.
    """

    def __init__(self, module, layer, dataloader, device=device()):
        self.module, self.layer = module, layer
        self.storage = ForwardModuleStorage(module, [layer])
        self.dataloader = dataloader
        self.device = device
        self.state = State()

    def points(self, dataloader, k=2, n_batches=None):
        """
        Batch-wise PCA. It returns the pca points, the labels and the inputs as Pytorch Tensors.
        """
        for i, (x, y) in enumerate(dataloader):
            y, x = y.to(self.device), x.to(self.device)
            self.storage(x)  # run input into the storage
            with torch.no_grad():
                features = self.storage[self.layer][0]
                flat_features = features.view(features.shape[0], -1)
                pca_features = pca(flat_features, k=k)
                del self.storage.state[self.layer]  # reinit storage -> save memory
                self.storage.state[self.layer] = []
                if n_batches is not None and i == n_batches: break
                yield pca_features, y, x

    def before_store(self, points, y, x):
        """
        Called before the points, labels are stored into the .state
        """
        return points, y, x

    def after_store(self, points, y, x):
        """
        Called after the points, labels are stored into the .state
        """
        return points, y, x

    def __call__(self, *args, **kwargs):
        bar = tqdm(self.points(self.dataloader, *args, **kwargs))
        for points, y, x in bar:
            # store points and labels by bringing them to the cpu to save GPU memory
            points, y, x = self.before_store(points, y, x)
            self.state.points = torch.cat([self.state.points, points.cpu()])
            self.state.y = torch.cat([self.state.y, y.cpu()])
            points, y, x = self.after_store(points, y, x)
            bar.set_description(f"Stored {self.state.points.shape[0]} points")

        self.state.indices = torch.arange(len(self.state.points))
        return self

    def reduce(self, to=100, using='kmeans'):
        """
        Reduce the number of .state.points using different methods.
        """
        if using not in self.reducers: raise ValueError(f"Parameter 'using' must be one of {self.reducers}")
        points, y, indices = self.state.points, self.state.y, self.state.indices

        bar = tqdm(total=1)
        bar.set_description(f"Reducing {self.state.points.shape[0]} points to {to} using {using}")

        if using == 'kmeans':
            kmeans = KMeans(n_clusters=to)
            kmeans.fit(self.state.points.numpy(), y=self.state.y.numpy())
            # update points, labels and indices using the position of the clusters
            points = [self.state.points.numpy()[np.where(kmeans.labels_ == i)][0] for i in range(kmeans.n_clusters)]
            y = [self.state.y.numpy()[np.where(kmeans.labels_ == i)][0] for i in range(kmeans.n_clusters)]
            indices = [self.state.indices.numpy()[np.where(kmeans.labels_ == i)][0] for i in range(kmeans.n_clusters)]
        # creates a new ModulePCA with the reduced points
        reduced_module_pca = PytorchModulePCA(self.module, self.layer, self.dataloader)
        reduced_module_pca.state = State(torch.from_numpy(np.array(points)),
                                         torch.from_numpy(np.array(y)),
                                         torch.from_numpy(np.array(indices)))

        bar.update(1)

        return reduced_module_pca

    def _scatter(self):
        """
        Creates a scatter plot using self.fig, self.ax
        """
        points, y, _ = self.state.numpy()
        for i, label in enumerate(np.unique(y).tolist()):
            if points.shape[-1] == 2:
                self.ax.scatter(points[y == label, 0], points[y == label, 1], label=label, alpha=0.5)
            elif points.shape[-1] == 3:
                self.ax.scatter(points[y == label, 0], points[y == label, 1], points[y == label, 2], label=label,
                                alpha=0.5)

    def _legend(self):
        """
        Remove duplicates name in the legend
        """
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())

    def plot(self, *args, **kwargs):
        self.check_plot_dimensions()
        if self.state.points.shape[-1] == 2:
            self._plot2d(*args, **kwargs)
        elif self.state.points.shape[-1] == 3:
            self._plot3d(*args, **kwargs)

    def _plot2d(self):
        self.fig, self.ax = plt.figure(), plt.subplot(111)
        title = f"{self.state.points.shape[0]} points"
        # add subtitle with more info
        self._scatter()
        self._legend()
        plt.title(title)

        return self

    def _plot3d(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        title = f"{self.state.points.shape[0]} points"
        self._scatter()
        self._legend()

        plt.title(title)

        return self

    def _annotate2d(self, zoom=1, transform=None):
        self.fig, self.ax = plt.figure(), plt.subplot(111)
        self._scatter()
        for point, l, i in zip(*self.state.numpy()):
            x, y = point[0], point[1]
            img = self.dataloader.dataset[i][0]
            if transform is not None: img = transform(img)
            img_np = img.permute(1, 2, 0).numpy().squeeze()
            ab = AnnotationBbox(OffsetImage(img_np, zoom=zoom), (x, y), frameon=False)
            self.ax.add_artist(ab)

    def _annotate3d(self, zoom=1, transform=None):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection=Axes3D.name, picker=True)
        points = self.state.points.numpy()
        imgs = []
        # store the images from the dataset
        for i in self.state.indices.numpy():
            img = self.dataloader.dataset[i][0]
            if transform is not None: img = transform(img)
            img_np = img.permute(1, 2, 0).numpy().squeeze()
            imgs.append(img_np)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        self._scatter()

        ax2 = self.fig.add_subplot(111, frame_on=False)
        ax2.axis("off")
        ax2.axis([0, 1, 0, 1])

        ia = ImageAnnotations3D(np.c_[x, y, z], imgs, self.ax, ax2, zoom=zoom)

    def annotate(self, *args, **kwargs):
        self.check_plot_dimensions()
        if self.state.points.shape[-1] == 2:
            self._annotate2d(*args, **kwargs)
        elif self.state.points.shape[-1] == 3:
            self._annotate3d(*args, **kwargs)

        return self

    def check_plot_dimensions(self):
        k = self.state.points.shape[-1]
        if k not in self.plot_dimensions: raise ValueError(f"Cannot visualise a {k} dimension vector")
