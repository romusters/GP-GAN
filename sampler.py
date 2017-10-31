import os

import numpy as np
from skimage.io import imsave

import chainer

from utils import make_grid


def save_img(var, path):
    img = chainer.cuda.to_cpu(var.data)
    img = make_grid(img[:,:3,:,:])
    img = np.asarray(np.transpose(np.clip((img + 1) * 127.5, 0, 255), (1, 2, 0)), dtype=np.uint8)
    imsave(path, img)

def sampler(G, dst, inputv, name):
    @chainer.training.make_extension()
    def make_image(trainer):
        if trainer.updater.iteration == 10:
            save_img(inputv, os.path.join(dst, name.format(0)))

        save_img(G(inputv, test=True), os.path.join(dst, name.format(trainer.updater.iteration)))

    return make_image