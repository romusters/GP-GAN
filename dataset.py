import os
import glob
import math
import numpy

import warnings
warnings.simplefilter('ignore')

from skimage import img_as_ubyte, img_as_float
from skimage.io import imread
from skimage.transform import resize

from chainer.dataset import dataset_mixin

from fuel.datasets.hdf5 import H5PYDataset

from color_utils import ColorJitter, to_pil_image

class H5pyDataset(dataset_mixin.DatasetMixin):
    def __init__(self, path, which_set='train', load_size=None, crop_size=None, dtype=numpy.float32, ratio=0.5):
        self._dtype = dtype
        self._load_size = load_size
        self._crop_size = crop_size
        self._size = int(self._crop_size * ratio)
        self._data_set = H5PYDataset(path, which_sets=(which_set,))

        self._jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, gamma=0.25, hue=0.5)

    def __len__(self):
        return self._data_set.num_examples

    def get_example(self, i):
        handle = self._data_set.open()
        data = self._data_set.get_data(handle, slice(i, i+1))
        self._data_set.close(handle)

        im = numpy.squeeze(data[0])

        w, h, _ = im.shape
        min_size = min(w, h)
        ratio = self._load_size/min_size
        rw, rh = int(math.ceil(w*ratio)), int(math.ceil(h*ratio))
        im = resize(im, (rw, rh), order=1, mode='constant')

        sx, sy = numpy.random.random_integers(0, rw-self._crop_size), numpy.random.random_integers(0, rh-self._crop_size)
        im = im[sx:sx+self._crop_size, sy:sy+self._crop_size,:].copy()
        gt = im[sx:sx+self._crop_size, sy:sy+self._crop_size,:].copy()

        # random style
        sx, sy = numpy.random.random_integers(0, self._crop_size-self._size),\
                 numpy.random.random_integers(0, self._crop_size-self._size)

        exchange_area = to_pil_image(img_as_ubyte(im[sx:sx+self._size, sy:sy+self._size]))
        exchange_area = img_as_float(numpy.asarray(self._jitter(exchange_area)))
        im[sx:sx + self._size, sy:sy + self._size] = exchange_area

        mask = numpy.zeros((self._crop_size, self._crop_size, 1))
        mask[sx:sx+self._size, sy:sy+self._size] = 1

        im = numpy.concatenate((im, mask), axis=-1)

        im = numpy.asarray(numpy.transpose(im*2-1, (2, 0, 1)), dtype=self._dtype)
        gt = numpy.asarray(numpy.transpose(gt*2-1, (2, 0, 1)), dtype=self._dtype)

        return im, gt

class BlendingDataset(dataset_mixin.DatasetMixin):
    def __init__(self, total_examples, folders, root, ratio, load_size, crop_size, dtype=numpy.float32):
        imgs_per_folder = {folder:glob.glob(os.path.join(root, folder, '*')) for folder in folders}
        self._len = total_examples

        self._dtype = dtype
        self._load_size = load_size
        self._crop_size = crop_size
        self._size = int(self._crop_size*ratio)
        self._sx = self._crop_size//2 - self._size//2

        self._imgs = []
        for _ in range(self._len):
            folder = numpy.random.choice(folders)
            obj_path, bg_path = numpy.random.choice(imgs_per_folder[folder], 2, replace=False)
            self._imgs.append((obj_path, bg_path))

    def __len__(self):
        return self._len

    def _crop(self, im, rw, rh, sx, sy):
        im = resize(im, (rw, rh), order=1, preserve_range=False, mode='constant')
        im = im[sx:sx+self._crop_size, sy:sy+self._crop_size,:]*2 - 1
        im = numpy.transpose(im, (2, 0, 1)).astype(self._dtype)

        return im

    def get_example(self, i):
        obj_path, bg_path = self._imgs[i]
        obj = imread(obj_path)
        bg  = imread(bg_path)

        w, h, _ = obj.shape
        min_size = min(w, h)
        ratio = self._load_size/min_size
        rw, rh = int(math.ceil(w*ratio)), int(math.ceil(h*ratio)) 
        sx, sy = numpy.random.random_integers(0, rw-self._crop_size), numpy.random.random_integers(0, rh-self._crop_size)              

        obj_croped = self._crop(obj, rw, rh, sx, sy)
        bg_croped  = self._crop(bg,  rw, rh, sx, sy)

        copy_paste = bg_croped.copy()
        copy_paste[:, self._sx:self._sx+self._size, self._sx:self._sx+self._size] = obj_croped[:, self._sx:self._sx+self._size, self._sx:self._sx+self._size]

        mask = numpy.zeros((1, self._crop_size, self._crop_size))
        mask[:, sx:sx + self._size, sy:sy + self._size] = 1
        mask = mask*2 - 1

        copy_paste = numpy.concatenate((copy_paste, mask), axis=0).astype(numpy.float32)

        return copy_paste, bg_croped