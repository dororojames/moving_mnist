import os
import random
import shutil
import sys

import numpy as np
from PIL import Image

###########################################################################################
# script to generate moving mnist video dataset (frame by frame) as described in
# [1] arXiv:1502.04681 - Unsupervised Learning of Video Representations Using LSTMs
#     Srivastava et al
# by James Chan
# forked from tencia/moving_mnist.py
# saves in hdf5, npz, or jpg (individual frames) format
###########################################################################################

# helper functions
_PATH = os.path.dirname(os.path.abspath(__file__))+"/"


def arr_from_img(im, shift=0):
    w, h = im.size
    arr = im.getdata()
    c = np.product(arr.size) // (w*h)
    return np.asarray(arr, dtype=np.float32).reshape((h, w, c)).transpose(2, 1, 0) / 255. - shift


def get_picture_array(X, index, shift=0):
    ch, w, h = X.shape[1], X.shape[2], X.shape[3]
    ret = ((X[index]+shift)*255.).reshape(ch, w,
                                          h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
    if ch == 1:
        ret = ret.reshape(h, w)
    return ret

# loads mnist from web on demand


def load_dataset():
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
        return data / np.float32(255)
    return load_mnist_images(_PATH+'train-images-idx3-ubyte.gz')

# generates and returns video frames in uint8 array


def generate_moving_mnist(shape=(64, 64), seq_len=30, seqs=10000, num_sz=28, nums_per_image=2):
    mnist = load_dataset()
    width, height = shape
    lims = np.array([width-num_sz, height-num_sz])
    dataset = np.empty((seq_len*seqs, 1, width, height), dtype=np.uint8)
    for seq_idx in range(seqs):
        # randomly generate direc/speed/position, calculate velocity vector
        direcs = np.pi * (np.random.rand(nums_per_image)*2 - 1)
        speeds = np.random.randint(5, size=nums_per_image)+2
        veloc = np.multiply(
            np.array([np.cos(direcs), np.sin(direcs)]), speeds).T
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
            dataset[seq_idx*seq_len +
                    frame_idx] = (canvas * 255).astype(np.uint8).clip(0, 255)
        if (seq_idx+1) % (seqs//10) == 0:
            print(((seq_idx+1) * 100 // seqs), "%")
    return dataset


def main(dest=None, filetype='jpg', frame_size=64, seq_len=10, seqs=10, num_sz=28, nums_per_image=2):
    dat = generate_moving_mnist(shape=(frame_size, frame_size), seq_len=seq_len, seqs=seqs,
                                num_sz=num_sz, nums_per_image=nums_per_image)
    dest = _PATH + (dest if dest !=
                    None else "dataset/" if filetype == "jpg" else "data")
    print("Saving...", end="")
    if filetype == 'np':
        np.save(dest+str(seqs)+".npy", dat)
    elif filetype == 'jpg':
        shutil.rmtree(dest)
        os.mkdir(dest)
        for i in range(seqs):
            vdir = dest+"/v{}/".format(i)
            os.mkdir(vdir)
            for j in range(seq_len):
                Image.fromarray(get_picture_array(
                    dat, i*seq_len+j, shift=0)).save(vdir + "{}.jpg".format(j))
    else:
        raise TypeError("Filetype Error")
    print("Done")


def split_train_test(inputframes=10, predframes=10, testrate=10,  datadir=_PATH+"dataset/"):
    if not os.path.isdir(datadir):
        raise FileNotFoundError("Data Floder not exist")
    vdirs = sorted(os.listdir(datadir))
    video_len = len(vdirs)
    if video_len == 0:
        raise FileNotFoundError("No video in the {}".format(datadir))
    data = []
    frames = len(os.listdir(datadir+vdirs[0]))
    print("Input {} frames, pred {} frames".format(inputframes, predframes))
    if frames-inputframes-predframes < 0:
        raise ValueError(
            "inputframes+predframes out of range(Expected less then {})".format(frames))
    # combine imgs
    for vid in range(video_len):
        for i in range(frames-inputframes-predframes+1):
            im = "v{}_{}_{}".format(vid, i, i+inputframes)
            la = "v{}_{}_{}".format(
                vid, i+inputframes, i+inputframes+predframes)
            data.append([im, la])
    random.shuffle(data)
    datasize = len(data)
    print("Data size: {}".format(datasize))
    # save csv
    split_point = datasize * (100-testrate)//100
    with open(_PATH+"train_img.csv", "w") as train_img:
        with open(_PATH+"train_label.csv", "w") as train_label:
            for d in data[:split_point]:
                train_img.write(d[0]+"\n")
                train_label.write(d[1]+"\n")
    with open(_PATH+"test_img.csv", "w") as test_img:
        with open(_PATH+"test_label.csv", "w") as test_label:
            for d in data[split_point:]:
                test_img.write(d[0]+"\n")
                test_label.write(d[1]+"\n")
    print("Saved dataindex")


if __name__ == '__main__':
    main(filetype="jpg", seq_len=24, seqs=100)
    split_train_test(testrate=20, predframes=10)
