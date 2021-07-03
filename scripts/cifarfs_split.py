#!/usr/bin/env python

import argparse
import csv
import os
import shutil

# define some global args, source dir and dest_dir variables for convenience
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='path to the INat dataset')
parser.add_argument('--split', type=str, default='../split/CIFARFS/', help='path to the CIFAR100 dataset')
args = parser.parse_args()

# where data is located
source_dir = args.data + 'data/'
# new location for data
dest_dir = args.data + 'images/'


def main():
    if not os.path.isdir(args.split):
        os.makedirs(args.split)

    # create split csv files
    create_split_csv('train')
    create_split_csv('val')
    create_split_csv('test')

    # -----------------------------------------------------
    # move all images into a single folder
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)


    for direct in os.listdir(source_dir):
        for images in os.listdir(source_dir + direct):
            shutil.copyfile(source_dir + direct + "/" + images, dest_dir + images)

def create_split_csv(split):
    """
    Converts the split file in the original CIFARFS r2d2 Bertinetto repo to
    .csv split files where first column is image name and second is image
    class name
    """
    split_file = open(args.data + '/splits/bertinetto/' + split + '.txt', 'r')
    classes = split_file.read().split('\n')

    with open(args.split + split + '.csv', 'w', newline='') as file:

        # write first line
        writer = csv.writer(file)
        writer.writerow(["filename", "label"])

        for directory in os.listdir(source_dir):
            if directory in classes:
                for image in os.listdir(source_dir + directory):
                    # write image + class name
                    writer.writerow([image, directory])

if __name__ == "__main__":
    main()
