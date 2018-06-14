from __future__ import print_function

import argparse
import functools
import glob
import imghdr
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import rosbag
from cv_bridge import CvBridge, CvBridgeError
from tqdm import tqdm


def check_format(data):
        img_fmt = imghdr.what(None, h=data)
        return 'jpg' if img_fmt == 'jpeg' else img_fmt


def write_image(bridge, out_dir, msg, fmt='png'):
        results = {}
        image_filename = os.path.join(out_dir,
                                      str(msg.header.stamp.to_nsec())
                                      +
                                      '.' + fmt)
        try:
                if hasattr(msg, 'format') and 'compressed' in msg.format:
                        buf = np.ndarray(
                                shape=(1, len(msg.data)),
                                dtype=np.uint8,
                                buffer=msg.data)
                        cv_image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
                        if cv_image.shape[2] != 3:
                                print('Invalid image %s' % image_filename)
                                return results
                        results['height'] = cv_image.shape[0]
                        results['width'] = cv_image.shape[1]

                        # Avoid re-encoding if we don't have to
                        if check_format(msg.data) == fmt:
                                buf.tofile(image_filename)
                        else:
                                cv2.imwrite(image_filename, cv_image)
                else:
                        cv_image = bridge.imgmsg_to_cv2(msg, 'bgr8')
                        cv2.imwrite(image_filename, cv_image)
        except CvBridgeError as e:
                print(e)
        results['filename'] = image_filename
        return results


def process_camera_topic(bridge, d, msg, out_dir, topic):
        match = re.search(r'(left|center|right)', topic)
        if match:
                path = os.path.join(out_dir, match.group(0))
                r = write_image(bridge, path, msg, fmt='jpg')
                r['filename'] = os.path.relpath(r['filename'], out_dir)
                d['timestamp'].append(msg.header.stamp.to_nsec())
                d['width'].append(
                        r['width'] if 'width' in r else msg.width)
                d['height'].append(
                        r['height'] if 'height' in r else msg.height)
                d['frame_id'].append(msg.header.frame_id)
                d['filename'].append(r['filename'])
        else:
                print('Unexpected topic: {}'.format(topic))


def process_bag(out_dir,
                bag_name,
                filter_topics,
                camera_topics, camera_dict,
                steering_topics, steering_dict):
        bridge = CvBridge()
        with rosbag.Bag(bag_name, 'r') as bag:
                for topic, msg, _ in tqdm(
                            bag.read_messages(topics=filter_topics),
                            total=bag.get_message_count(filter_topics)):
                        if topic in camera_topics:
                                process_camera_topic(bridge,
                                                     camera_dict,
                                                     msg,
                                                     out_dir,
                                                     topic)
                        elif topic in steering_topics:
                                steering_dict['timestamp'].append(
                                        msg.header.stamp.to_nsec())
                                steering_dict['angle'].append(
                                        msg.steering_wheel_angle)
                                steering_dict['torque'].append(
                                        msg.steering_wheel_torque)
                                steering_dict['speed'].append(
                                        msg.speed)


def main():
        parser = argparse.ArgumentParser(
                description='Read rosbag files'
                            ' and dump images and csv while doing so.')
        parser.add_argument(
                '-i',
                '--indir',
                type=str,
                nargs='?',
                default='training.data.bags/',
                help='Input folder')
        parser.add_argument(
                '-o',
                '--outdir',
                type=str,
                nargs='?',
                default='training.data.files/',
                help='Output folder')
        args = parser.parse_args()

        camera_topics = []
        for pos in ['left', 'center', 'right']:
                os.makedirs(os.path.join(args.outdir, pos))
                camera_topics += ['/{}_camera/image_color'.format(pos)]
                camera_topics += ['/{}_camera/image_color/compressed'.format(pos)]
        steering_topics = ['/vehicle/steering_report']
        filter_topics = steering_topics + camera_topics

        camera_dict = defaultdict(list)
        steering_dict = defaultdict(list)

        for bag_name in glob.glob('{}*.bag'.format(args.indir)):
                print('Processing {}'.format(bag_name))
                sys.stdout.flush()
                process_bag(args.outdir,
                            bag_name,
                            filter_topics,
                            camera_topics, camera_dict,
                            steering_topics, steering_dict)

        camera_df = pd.DataFrame(data=camera_dict,
                                 columns=['timestamp',
                                          'width',
                                          'height',
                                          'frame_id',
                                          'filename'])
        camera_df.to_csv(os.path.join(args.outdir, '.camera.csv'),
                         index=False)

        steering_df = pd.DataFrame(data=steering_dict,
                                   columns=['timestamp',
                                            'angle',
                                            'torque',
                                            'speed'])
        steering_df.to_csv(os.path.join(args.outdir, '.steering.csv'),
                           index=False)

        for i in [camera_df, steering_df]:
                i['timestamp'] = pd.to_datetime(i['timestamp'])
                i.set_index(['timestamp'], inplace=True)
                i.index.rename('index', inplace=True)

        merged = functools.reduce(lambda left, right:
                                  pd.merge(left,
                                           right,
                                           how='outer',
                                           left_index=True,
                                           right_index=True),
                                  [camera_df, steering_df])
        merged.interpolate(method='time', inplace=True)

        filtered_cols = ['timestamp', 'width', 'height',
                         'frame_id', 'filename',
                         'angle', 'torque', 'speed']

        # back to only camera rows
        filtered = merged.loc[camera_df.index]
        filtered.fillna(0.0, inplace=True)
        # add back original timestamp integer col
        filtered['timestamp'] = filtered.index.astype('int')
        # cast back to int
        filtered['width'] = filtered['width'].astype('int')
        # cast back to int
        filtered['height'] = filtered['height'].astype('int')
        # filter and reorder columns for final output
        filtered = filtered[filtered_cols]

        filtered.to_csv(os.path.join(args.outdir, 'training.data.csv'),
                        header=True)


if __name__ == '__main__':
        main()
