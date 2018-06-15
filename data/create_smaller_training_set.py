import os
from random import shuffle
from shutil import copyfile
from glob import glob

small_training_set_dir = os.path.join(os.getcwd(), 'train2014_small')
large_training_set_dir = os.path.join(os.getcwd(), 'train2014')
if not os.path.isdir(small_training_set_dir):
    os.makedirs(small_training_set_dir)

existing_images = glob(os.path.join(small_training_set_dir, '*')) # should just be an empty list
for existing_image in existing_images:
    os.remove(existing_image)


all_images = glob(os.path.join(large_training_set_dir, '*'))
shuffle(all_images)
for image_path in all_images[-50:]: # just need to train on 50 images for proof tht CPU and GPU training works
    copyfile(image_path, os.path.join(small_training_set_dir, os.path.basename(image_path)))
