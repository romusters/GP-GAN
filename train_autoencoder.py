from __future__ import print_function

import os
import random
import argparse

import chainer
from chainer import training, Variable
from chainer.training import extensions

from model import EncoderDecoder, init_bn, init_conv
from dataset import H5pyDataset, BlendingDataset
from updater import EncoderDecoderUpdater
from sampler import sampler

def make_optimizer(model, alpha, beta1):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
    optimizer.setup(model)
    return optimizer

def main():
    parser = argparse.ArgumentParser(description='Train AutoEncoder')
    parser.add_argument('--nef', type=int, default=64, help='# of base filters in encoder')
    parser.add_argument('--ndf', type=int, default=64, help='# of base filters in decoder')
    parser.add_argument('--nc',  type=int, default=3,  help='# of output channels in decoder')
    parser.add_argument('--nBottleneck',  type=int, default=100, help='# of output channels in encoder')

    parser.add_argument('--lr', type=float, default=0.001,  help='Learning rate for AutoEncoder, default=0.001')
    parser.add_argument('--beta1', type=float, default=0.5,   help='Beta for Adam, default=0.5')

    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--n_epoch', type=int, default=25, help='# of epochs to train for')

    parser.add_argument('--data_root', help='Path to dataset')
    parser.add_argument('--load_size', type=int, default=64, help='Scale image to load_size')
    parser.add_argument('--image_size', type=int, default=64, help='The height/width of the input image to network')

    # Transient Attributes
    parser.add_argument('--trans_attr_data_root', help='Path to dataset trans_attr')
    parser.add_argument('--ratio', type=float, default=0.5, help='Ratio for center square size v.s. image_size')

    parser.add_argument('--experiment', default='encoder_decoder_generative_result', help='Where to store samples and models')
    parser.add_argument('--test_folder', default='samples', help='Where to store test results')
    parser.add_argument('--workers', type=int, default=10, help='# of data loading workers')
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size')
    parser.add_argument('--test_size', type=int, default=64, help='Batch size for testing')

    parser.add_argument('--manual_seed', type=int, default=5, help='Manul seed')

    parser.add_argument('--resume', default='', help='Resume the training from snapshot')    
    parser.add_argument('--snapshot_interval', type=int, default=1, help='Interval of snapshot (epochs)')
    parser.add_argument('--print_interval', type=int, default=1, help='Interval of printing log to console (iteration)')
    parser.add_argument('--plot_interval', type=int, default=10, help='Interval of plot (iteration)')
    args = parser.parse_args()
    
    random.seed(args.manual_seed)

    print('Input arguments:')
    for key, value in vars(args).items():
        print('\t{}: {}'.format(key, value))
    print('')

    # Set up AutoEncoder
    print('Create & Init models ...')
    autoencoder = EncoderDecoder(args.nef, args.ndf, args.nc, args.nBottleneck,
                       image_size=args.image_size, conv_init=init_conv, bn_init=init_bn)
    if args.gpu >= 0:
        print('\tCopy models to gpu {} ...'.format(args.gpu))
        chainer.cuda.get_device_from_id(args.gpu).use()  # Make a specified GPU current
        autoencoder.to_gpu()                             # Copy the model to the GPU
    print('Init models done ...\n')
    # Setup an optimizer
    optimizer = make_optimizer(autoencoder, args.lr, args.beta1)

    ########################################################################################################################
    # Setup dataset & iterator
    print('Load images from {} ...'.format(args.data_root))
    trainset = H5pyDataset(args.data_root, load_size=args.load_size, crop_size=args.image_size)
    print('\tTrainset contains {} image files'.format(len(trainset)))
    train_iter = chainer.iterators.MultiprocessIterator(trainset, args.batch_size, n_processes=args.workers,
                                                        n_prefetch=args.workers)

    testset = H5pyDataset(args.data_root, which_set='test', load_size=args.load_size, crop_size=args.image_size)
    print('\tTestset contains {} image files'.format(len(testset)))
    print('')
    # testset: Transient Attributes
    print('Load images from {} ...'.format(args.trans_attr_data_root))
    folders = sorted([folder for folder in os.listdir(args.trans_attr_data_root)
                      if os.path.isdir(os.path.join(args.trans_attr_data_root, folder))])
    print('\t{} folders in total...'.format(len(folders)))
    trans_attr_testset = BlendingDataset(args.test_size, folders, args.trans_attr_data_root,
                                         args.ratio, args.load_size, args.image_size)
    print('\tTestset[TransAttr] contains {} image files'.format(len(trans_attr_testset)))
    print('')
    ########################################################################################################################

    # Set up a trainer
    updater = EncoderDecoderUpdater(
        models=autoencoder,
        args=args,
        iterator=train_iter,
        optimizer={'main': optimizer},
        device=args.gpu
    )
    trainer = training.Trainer(updater, (args.n_epoch, 'epoch'), out=args.experiment)

    # Snapshot
    snapshot_interval = (args.snapshot_interval, 'epoch')
    trainer.extend(
        extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        autoencoder, 'model_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    
    # Display
    print_interval = (args.print_interval, 'iteration')
    trainer.extend(extensions.LogReport(trigger=print_interval))
    trainer.extend(extensions.PrintReport([
        'iteration', 'main/loss'
    ]), trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=args.print_interval))

    trainer.extend(extensions.dump_graph('main/loss', out_name='TrainGraph.dot'))

    # Plot
    plot_interval = (args.plot_interval, 'iteration')

    trainer.extend(
        extensions.PlotReport(['main/loss'], 'iteration', file_name='loss.png', trigger=plot_interval), trigger=plot_interval)
    
    # Test
    path = os.path.join(args.experiment, args.test_folder)
    if not os.path.isdir(path):
        os.makedirs(path)
    print('Saving samples to {} ...\n'.format(path))

    train_batch = [trainset[idx] for idx in range(args.test_size)]
    train_v = Variable(chainer.dataset.concat_examples(train_batch, args.gpu), volatile='on')
    trainer.extend(sampler(autoencoder, path, train_v, 'samples_train_{}.png'), trigger=plot_interval)

    test_batch = [testset[idx] for idx in range(args.test_size)]
    test_v = Variable(chainer.dataset.concat_examples(test_batch, args.gpu), volatile='on')
    trainer.extend(sampler(autoencoder, path, test_v, 'samples_test_{}.png'), trigger=plot_interval)

    trans_attr_test_batch = [trans_attr_testset[idx][0] for idx in range(args.test_size)]
    trans_attr_test_v = Variable(chainer.dataset.concat_examples(trans_attr_test_batch, args.gpu), volatile='on')
    trainer.extend(sampler(autoencoder, path, trans_attr_test_v, 'samples_test_trans_attr_{}.png'), trigger=plot_interval)
    
    if args.resume:
        # Resume from a snapshot
        print('Resume from {} ... \n'.format(args.resume))
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    print('Training start ...\n')
    trainer.run()

if __name__ == '__main__':
    main()