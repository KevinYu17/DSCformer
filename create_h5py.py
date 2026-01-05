import h5py
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

random.seed(3407)

img_folder = '../data/img/intensity'
label_folder = '../data/lbs'
img_range_folder = '../data/img/range'

img_files = [file for file in os.listdir(img_folder) if not file.startswith('.DS_Store')]
label_files = [file for file in os.listdir(label_folder) if not file.startswith('.DS_Store')]
random.shuffle(img_files)  # if shuffle is banned, use label_files below
len_files = len(img_files)

assert len(img_files) == len(label_files), "The number of images and labels must match."

hdf5_file = h5py.File('../FIND_fused.h5', 'w')
height, width, channels = 256, 256, 6  # channels need to change


train_img_dataset = hdf5_file.create_dataset('train_image', (int(0.8*len_files), height, width, channels), dtype='float')
train_label_dataset = hdf5_file.create_dataset('train_label', (int(0.8*len_files), height, width), dtype='float')

val_img_dataset = hdf5_file.create_dataset('val_image', ((len_files - int(0.8*len_files)), height, width, channels), dtype='float')
val_label_dataset = hdf5_file.create_dataset('val_label', ((len_files - int(0.8*len_files)), height, width), dtype='float')

ite = tqdm(range(len_files), ncols=70)
for i in ite:

    img_path = os.path.join(img_folder, img_files[i])
    label_path = os.path.join(label_folder, img_files[i].replace('.png', '.bmp'))  # if shuffle isn't used, use label_files, and use assert next row
    # assert img_files[i] == label_files[i], "img doesn't match label"
    img_range_path = os.path.join(img_range_folder, img_files[i])

    img = np.array(Image.open(img_path), dtype=float)
    label = np.array(Image.open(label_path), dtype=float)
    img_range = np.array(Image.open(img_range_path))
    img = np.concatenate((img, img_range), axis=2)

    if i < int(0.8*len_files):
        train_img_dataset[i] = img
        if np.max(label) == 0:
            train_label_dataset[i] = label
        else:
            train_label_dataset[i] = label / np.max(label)
        # assert np.max(label / np.max(label)) <= 1
    else:
        val_img_dataset[i-int(0.8*len_files)] = img
        if np.max(label) == 0:
            val_label_dataset[i - int(0.8 * len_files)] = label
        else:
            val_label_dataset[i-int(0.8*len_files)] = label / np.max(label)
        # assert np.max(label / np.max(label)) <= 1

hdf5_file.close()
