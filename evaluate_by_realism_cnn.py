from __future__ import print_function
import os
import glob
import pickle
import argparse

import chainer
from chainer import Variable, serializers

from model import RealismCNN

from utils import im_preprocess_vgg

import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Evaluate image blending algorithm by Realism CNN')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model_path', default='models/realismCNN_all_iter3.npz', help='Path for pretrained model')
    parser.add_argument('--data_root', required=True, help='Root folder for test dataset')
    parser.add_argument('--batch_size', type=int, default=10, help='Batchsize of 1 iteration')
    parser.add_argument('--load_size', type=int, default=224, help='Scale image to load_size')
    args = parser.parse_args()

    args.result_path = args.data_root+'.pkl'

    print('Predict realism for images in {} ...'.format(args.data_root))

    model = RealismCNN()
    print('Load pretrained model from {} ...'.format(args.model_path))
    serializers.load_npz(args.model_path, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()                                   # Copy the model to the GPU
    
    img_list = glob.glob(os.path.join(args.data_root, '*.png'))
    img_list = [os.path.basename(im) for im in img_list]
    dataset = chainer.datasets.ImageDataset(paths=img_list, root=args.data_root)
    data_iterator = chainer.iterators.SerialIterator(dataset, args.batch_size, repeat=False, shuffle=False)

    scores = []
    for idx, batch in enumerate(data_iterator):
        print('Processing batch {}->{}/{} ...'.format(idx*args.batch_size+1, min(len(dataset), (idx+1)*args.batch_size), len(dataset)))
        batch = [im_preprocess_vgg(np.transpose(im, [1, 2, 0]), args.load_size) for im in batch]
        batch = Variable(chainer.dataset.concat_examples(batch, args.gpu), volatile='on')
        result = chainer.cuda.to_cpu(model(batch, dropout=False).data)
        
        scores.append(result)

    scores = np.squeeze(np.vstack(scores))
    scores = scores[:, 1]
    print(np.mean(scores), np.std(scores))

    with open(args.result_path, 'wb') as f:
        pickle.dump(scores, f)

if __name__ == '__main__':
    main()