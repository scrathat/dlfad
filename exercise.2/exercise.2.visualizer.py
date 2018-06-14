import itertools
import pickle

import argparse
import cv2
import lmdb


def main():
        parser = argparse.ArgumentParser(
                description='Visualize LMDB training data')
        parser.add_argument(
                '-i',
                '--indir',
                type=str,
                nargs='?',
                default='training.data/',
                help='Input folder')
        args = parser.parse_args()

        with lmdb.open(args.indir).begin().cursor() as cursor:
                head = itertools.islice(cursor, 10)
                for key, value in head:
                        filename = key.decode()
                        data = pickle.loads(value)
                        print('filename: ', filename)
                        print('angle: ', data['angle'])
                        cv2.namedWindow('Training Data')
                        cv2.imshow('Training Data',
                                   cv2.imdecode(data['image'],
                                                cv2.IMREAD_COLOR))
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()


if __name__ == '__main__':
        main()
