import torch

from mpl_toolkits.mplot3d import proj3d
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import Axes3D


def pca(x, k=2):
    """
    From http://agnesmustar.com/2017/11/01/principal-component-analysis-pca-implemented-pytorch/
    """
    # preprocess the data
    X_mean = torch.mean(x, 0)
    x = x - X_mean.expand_as(x)
    # svd
    U, S, V = torch.svd(torch.t(x))
    return torch.mm(x, U[:, :k])


class ImageAnnotations3D():
    """
    From https://stackoverflow.com/questions/48180327/matplotlib-3d-scatter-plot-with-images-as-annotations. Yes matplotlib is shit
    """

    def __init__(self, xyz, imgs, ax3d, ax2d, zoom=1):
        self.xyz = xyz
        self.imgs = imgs
        self.ax3d = ax3d
        self.ax2d = ax2d
        self.zoom = zoom
        self.annot = []
        for s, im in zip(self.xyz, self.imgs):
            x, y = self.proj(s)
            self.annot.append(self.image(im, [x, y]))
        self.lim = self.ax3d.get_w_lims()
        self.rot = self.ax3d.get_proj()
        self.cid = self.ax3d.figure.canvas.mpl_connect("draw_event", self.update)

        self.funcmap = {"button_press_event": self.ax3d._button_press,
                        "motion_notify_event": self.ax3d._on_move,
                        "button_release_event": self.ax3d._button_release}

        self.cfs = [self.ax3d.figure.canvas.mpl_connect(kind, self.cb) \
                    for kind in self.funcmap.keys()]

    def cb(self, event):
        event.inaxes = self.ax3d
        self.funcmap[event.name](event)

    def proj(self, X):
        """ From a 3D point in axes ax1,
            calculate position in 2D in ax2 """
        x, y, z = X
        x2, y2, _ = proj3d.proj_transform(x, y, z, self.ax3d.get_proj())
        tr = self.ax3d.transData.transform((x2, y2))
        return self.ax2d.transData.inverted().transform(tr)

    def image(self, arr, xy):
        """ Place an image (arr) as annotation at position xy """
        im = offsetbox.OffsetImage(arr, zoom=self.zoom)
        im.image.axes = self.ax3d
        ab = offsetbox.AnnotationBbox(im, xy,
                                      xycoords='data', boxcoords="offset points", pad=0.1)
        self.ax2d.add_artist(ab)
        return ab

    def update(self, event):
        if np.any(self.ax3d.get_w_lims() != self.lim) or \
                np.any(self.ax3d.get_proj() != self.rot):
            self.lim = self.ax3d.get_w_lims()
            self.rot = self.ax3d.get_proj()
            for s, ab in zip(self.xyz, self.annot):
                ab.xy = self.proj(s)


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tensor2numpy(*args):
    return [e.cpu().numpy() for e in args]
