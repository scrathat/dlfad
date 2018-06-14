import csv
import os
import pickle
import random

import argparse
import cv2
import lmdb
import numpy as np
from tqdm import tqdm


def augment_brightness(image):
        image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        random_bright = .25 + np.random.uniform()
        # print(random_bright)
        image1[:, :, 2] = image1[:, :, 2] * random_bright
        image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
        return image1


def image_blur(img):
        # Blur image with random kernel
        kernel_size = random.randint(1, 5)
        if kernel_size % 2 != 1:
                kernel_size += 1
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def add_random_shadow(image):
        top_y = 320 * np.random.uniform()
        top_x = 0
        bot_x = 160
        bot_y = 320 * np.random.uniform()
        image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        shadow_mask = 0 * image_hls[:, :, 1]
        X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
        Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]

        shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) *
                     (Y_m - top_y) >= 0)] = 1
        # random_bright = .25+.7*np.random.uniform()
        if np.random.randint(2) == 1:
                random_bright = .5
                cond1 = shadow_mask == 1
                cond0 = shadow_mask == 0
                if np.random.randint(2) == 1:
                        image_hls[:, :, 1][cond1] = image_hls[:, :, 1][
                                                            cond1] * random_bright
                else:
                        image_hls[:, :, 1][cond0] = image_hls[:, :, 1][
                                                            cond0] * random_bright
        image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

        return image


# randomly rotate to simulate camera jitter
def image_rotate(img):
        # Rotate image randomly to simulate camera jitter
        rotate = random.uniform(-1, 1)
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rotate,
                                    1)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        return img


def fliph_image(img, angle):
        """
        Returns a horizontally flipped image
        """
        angle = -angle
        return cv2.flip(img, 1), angle


def do_rotate(image, min=5, max=15, orientation='rand'):
        rows, cols, ch = image.shape

        # Randomly select a rotation angle from the range passed.
        random_rot = np.random.randint(min, max + 1)

        if orientation == 'rand':
                rotation_angle = random.choice([-random_rot, random_rot])
        elif orientation == 'left':
                rotation_angle = random_rot
        elif orientation == 'right':
                rotation_angle = -random_rot
        else:
                raise ValueError(
                        "Orientation is optional and can only be 'left' or 'right'.")

        M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2),
                                    rotation_angle, 1)
        return cv2.warpAffine(image, M,
                              (image.shape[1], image.shape[0])), -rotation_angle


def main():
        parser = argparse.ArgumentParser(
                description='Create LMDB from training data files')
        parser.add_argument(
                '-i',
                '--indir',
                type=str,
                nargs='?',
                default='training.data.files/',
                help='Input folder')
        parser.add_argument(
                '-o',
                '--outdir',
                type=str,
                nargs='?',
                default='training.data/',
                help='Output folder')
        args = parser.parse_args()

        csv_file_name = os.path.join(args.indir, 'training.data.csv')
        lmdb_name = args.outdir
        map_size = int(8e9) * 2

        with open(csv_file_name, newline='') as csvFile,\
                    lmdb.open(lmdb_name, map_size=map_size).begin(write=True) as txn:
                lines_count = sum(1 for line in open(csv_file_name))
                for row in tqdm(csv.DictReader(csvFile), total=(lines_count - 1)):
                        filename = row['filename']
                        img = cv2.imread(os.path.join(args.indir, filename))
                        img_jpg = cv2.imencode('.jpg', img)[1]
                        angle = float(row['angle'])
                        txn.put(filename.encode(),
                                pickle.dumps({'image': img_jpg, 'angle': angle}))

                        filename += '+flipped'
                        img_flipped, angle_flipped = fliph_image(img, angle)
                        img_jpg = cv2.imencode('.jpg', img_flipped)[1]
                        txn.put(filename.encode(),
                                pickle.dumps({'image': img_jpg, 'angle': angle_flipped}))


if __name__ == '__main__':
        main()
