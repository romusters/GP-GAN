import os
import pickle
import argparse

from chainer import cuda, serializers

from skimage import img_as_float
from skimage.io import imread, imsave

from gp_gan import gp_gan
from model import EncoderDecoder

def main():
    parser = argparse.ArgumentParser(description='GP-AutoEncoder --- random test')
    parser.add_argument('--nef', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nc',  type=int, default=3)
    parser.add_argument('--nBottleneck',  type=int, default=100)

    parser.add_argument('--image_size', type=int, default=64, help='The height / width of the input image to network')

    parser.add_argument('--color_weight', type=float, default=1, help='Color weight')

    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--autoencoder_path', help='Path for pretrained autoencoder model')
    parser.add_argument('--list_path', default='random_test_list.txt', help='File path for random test list')
    parser.add_argument('--selected_images', default='selected_idx.pkl', help='Pickle file for selected images')
    parser.add_argument('--data_root', default='DataBase/TransientAttributes', help='Dataset root')
    parser.add_argument('--result_folder', default='autoencoder_result', help='Name for folder storing results')
    args = parser.parse_args()

    args.result_folder = os.path.join('results', args.result_folder)

    print('Input arguments:')
    for key, value in vars(args).items():
        print('\t{}: {}'.format(key, value))
    print('')

    # Init CNN model
    autoencoder = EncoderDecoder(args.nef, args.ndf, args.nc, args.nBottleneck, image_size=args.image_size)
    print('Load pretrained G model from {} ...'.format(args.autoencoder_path))
    serializers.load_npz(args.autoencoder_path, autoencoder)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()  # Make a specified GPU current
        autoencoder.to_gpu()                     # Copy the model to the GPU
    
    # Init image list
    print('Load images from {} ...'.format(args.list_path))
    with open(args.list_path) as f:
        test_list = [[os.path.join(args.data_root, p) for p in line.strip().split(';')] for line in f]
    print('\t {} images in total ...\n'.format(len(test_list)))
    
    # Init result folder
    if not os.path.isdir(args.result_folder):
        os.makedirs(args.result_folder)
    print('Result will save to {} ...\n'.format(args.result_folder))

    if args.selected_images:
        with open(args.selected_images, 'rb') as f:
            idx_list = pickle.load(f)
        idx_list = [idx-1 for idx in idx_list]
        total_size = len(idx_list)
    else:
        total_size = len(test_list)
        idx_list = range(total_size)

    for count, idx in enumerate(idx_list):
        print('Processing {}/{} ...'.format(count+1, total_size))
        
        # load image
        obj = img_as_float(imread(test_list[idx][0]))
        bg  = img_as_float(imread(test_list[idx][1]))
        mask = imread(test_list[idx][2]).astype(obj.dtype)

        ############################ Poisson GAN Image Editing ###########################
        blended_im = gp_gan(obj, bg, mask, autoencoder, args.image_size, args.gpu,
                            color_weight=args.color_weight, supervised=True)

        imsave(os.path.join(args.result_folder, '{}_gp_autoencoder.png'.format(idx+1)), blended_im)

if __name__ == '__main__':
    main()