import gzip
import os

import numpy as np
from PIL import Image

###########################################################################################
# script to generate moving mnist video dataset (frame by frame) as described in
# [1] arXiv:1502.04681 - Unsupervised Learning of Video Representations Using LSTMs
#     Srivastava et al
# by James Chan
# forked from tencia/moving_mnist.py
# saves in npy or jpg (individual frames) format
###########################################################################################


def arr_from_img(im, shift=0):
    w, h = im.size
    arr = im.getdata()
    c = np.product(arr.size) // (w*h)
    return np.asarray(arr, dtype=np.float32).reshape((h, w, c)).transpose(2, 1, 0) / 255. - shift


def get_picture_array(X, index, shift=0):
    ch, w, h = X.shape[1], X.shape[2], X.shape[3]
    ret = ((X[index]+shift)*255.).reshape(ch, w, h).transpose(2, 1, 0)
    if ch == 1:
        ret = ret.reshape(h, w)
    return ret.clip(0, 255).astype(np.uint8)


def load_dataset(path):
    """loads mnist from web on demand"""
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        import sys
        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve
        print("Downloading %s" % filename)
        urlretrieve(source+filename, filename)

    filename = 'train-images-idx3-ubyte.gz'
    if not os.path.exists(path+filename):
        download(filename)
    with gzip.open(path+filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2) / 255.


def generate_moving_mnist(mnist, shape=(64, 64), seq_len=30, seqs=10000, num_sz=28, nums_per_image=2):
    """generates and returns video frames in uint8 array"""
    width, height = shape
    lims = np.array([width-num_sz, height-num_sz])
    data = np.empty((seq_len*seqs, 1, width, height), dtype=np.uint8)
    for seq_idx in range(seqs):
        # randomly generate direc/speed/position, calculate velocity vector
        direcs = np.pi * (np.random.rand(nums_per_image)*2 - 1)
        angle = np.array([np.cos(direcs), np.sin(direcs)])
        speeds = np.random.randint(5, size=nums_per_image)+2
        veloc = np.multiply(angle, speeds).T

        mnist_images = [Image.fromarray(get_picture_array(mnist, r, shift=0)).resize((num_sz, num_sz), Image.ANTIALIAS)
                        for r in np.random.randint(0, mnist.shape[0], nums_per_image)]
        position = np.random.uniform(size=(nums_per_image, 2)) * lims
        for frame_idx in range(seq_len):
            canvases = [Image.new('L', (width, height))
                        for _ in range(nums_per_image)]
            canvas = np.zeros((1, width, height), dtype=np.float32)
            for i, canv in enumerate(canvases):
                canv.paste(mnist_images[i], tuple(position[i].astype(int)))
                canvas += arr_from_img(canv, shift=0)
            # bounce off wall if a we hit one
            for i in range(nums_per_image):
                for j in range(2):
                    newp = position[i][j]+veloc[i][j]
                    if newp < -2 or newp > lims[j]+2:
                        veloc[i][j] *= -1
            position += veloc
            # copy additive canvas to data array
            data[seq_idx*seq_len +
                 frame_idx] = (canvas * 255).astype(np.uint8).clip(0, 255)
        if seqs >= 10 and (seq_idx+1) % (seqs//10) == 0:
            print(((seq_idx+1) * 100 // seqs), "%")
    return data


def generate_seq(datadir="./", dest="MovingMNIST/", filetype='jpg', frame_size=64, seq_len=10, seqs=10, num_sz=28, nums_per_image=2):
    """generate moving mnist video dataset (frame by frame)"""
    print("Start Generate %d sequences" % seqs)
    data = generate_moving_mnist(mnist=load_dataset(datadir), shape=(frame_size, frame_size),
                                 seq_len=seq_len, seqs=seqs, num_sz=num_sz, nums_per_image=nums_per_image)
    dest = datadir + (dest if filetype == "jpg" else "MovingMNIST")
    print("Saving...", end="")
    if filetype == 'np':
        np.save(dest+str(seqs)+".npy", data)
    elif filetype == 'jpg':
        if os.path.isdir(dest):
            from shutil import rmtree
            rmtree(dest)
        os.mkdir(dest)
        for i in range(seqs):
            vdir = dest+"/v%d/" % i
            os.mkdir(vdir)
            for j in range(seq_len):
                picture_array = get_picture_array(data, i*seq_len+j, shift=0)
                Image.fromarray(picture_array).save(vdir+"%d.jpg" % j)
    else:
        raise ValueError("Filetype Error")
    print("Done")


if __name__ == '__main__':
    generate_seq(seq_len=20, seqs=10)
