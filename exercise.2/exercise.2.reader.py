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

LEFT_CAMERA_TOPIC = '/left_camera/image_color'
CENTER_CAMERA_TOPIC = '/center_camera/image_color'
RIGHT_CAMERA_TOPIC = '/right_camera/image_color'
LEFT_CAMERA_COMPRESSED_TOPIC = LEFT_CAMERA_TOPIC + '/compressed'
CENTER_CAMERA_COMPRESSED_TOPIC = CENTER_CAMERA_TOPIC + '/compressed'
RIGHT_CAMERA_COMPRESSED_TOPIC = RIGHT_CAMERA_TOPIC + '/compressed'
CAMERA_TOPICS = [
    LEFT_CAMERA_TOPIC, CENTER_CAMERA_TOPIC, RIGHT_CAMERA_TOPIC,
    LEFT_CAMERA_COMPRESSED_TOPIC, CENTER_CAMERA_COMPRESSED_TOPIC,
    RIGHT_CAMERA_COMPRESSED_TOPIC
]
STEERING_TOPIC = '/vehicle/steering_report'


def check_format(data):
    img_fmt = imghdr.what(None, h=data)
    return 'jpg' if img_fmt == 'jpeg' else img_fmt


def write_image(bridge, outdir, msg, fmt='png'):
    results = {}
    image_filename = os.path.join(outdir,
                                  str(msg.header.stamp.to_nsec()) + '.' + fmt)
    try:
        if hasattr(msg, 'format') and 'compressed' in msg.format:
            buf = np.ndarray(
                shape=(1, len(msg.data)), dtype=np.uint8, buffer=msg.data)
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


def camera2dict(msg, write_results, camera_dict):
    camera_dict['timestamp'].append(msg.header.stamp.to_nsec())
    camera_dict['width'].append(write_results['width']
                                if 'width' in write_results else msg.width)
    camera_dict['height'].append(write_results['height']
                                 if 'height' in write_results else msg.height)
    camera_dict['frame_id'].append(msg.header.frame_id)
    camera_dict['filename'].append(write_results['filename'])


def steering2dict(msg, steering_dict):
    steering_dict['timestamp'].append(msg.header.stamp.to_nsec())
    steering_dict['angle'].append(msg.steering_wheel_angle)
    steering_dict['torque'].append(msg.steering_wheel_torque)
    steering_dict['speed'].append(msg.speed)


def main():
    parser = argparse.ArgumentParser(
        description='Read rosbag files into lmdb'
        ' and dump images and csv while doing so.')
    parser.add_argument(
        '-i',
        '--indir',
        type=str,
        nargs='?',
        default='data/',
        help='Input folder')
    parser.add_argument(
        '-o',
        '--outdir',
        type=str,
        nargs='?',
        default='out/',
        help='Output folder')
    args = parser.parse_args()

    filter_topics = [STEERING_TOPIC] + CAMERA_TOPICS
    camera_cols = ['timestamp', 'width', 'height', 'frame_id', 'filename']
    camera_dict = defaultdict(list)
    steering_cols = ['timestamp', 'angle', 'torque', 'speed']
    steering_dict = defaultdict(list)
    bridge = CvBridge()
    for bag_name in glob.glob('{}*.bag'.format(args.indir)):
        print('Processing {}'.format(bag_name))
        sys.stdout.flush()
        with rosbag.Bag(bag_name, 'r') as bag:
            for topic, msg, _ in bag.read_messages(topics=filter_topics):
                if topic in CAMERA_TOPICS:
                    match = re.search(r'(left|center|right)', topic)
                    if match:
                        outdir = args.outdir + match.group(0)
                        results = write_image(bridge, outdir, msg, fmt='jpg')
                        results['filename'] = os.path.relpath(
                            results['filename'], args.outdir)
                        camera2dict(msg, results, camera_dict)
                    else:
                        print('Unexpected topic: {}'.format(topic))
                elif topic == STEERING_TOPIC:
                    steering2dict(msg, steering_dict)

    camera_csv_path = os.path.join(args.outdir, 'camera.csv')
    camera_df = pd.DataFrame(data=camera_dict, columns=camera_cols)
    camera_df.to_csv(camera_csv_path, index=False)

    steering_csv_path = os.path.join(args.outdir, 'steering.csv')
    steering_df = pd.DataFrame(data=steering_dict, columns=steering_cols)
    steering_df.to_csv(steering_csv_path, index=False)

    camera_df['timestamp'] = pd.to_datetime(camera_df['timestamp'])
    camera_df.set_index(['timestamp'], inplace=True)
    camera_df.index.rename('index', inplace=True)
    steering_df['timestamp'] = pd.to_datetime(steering_df['timestamp'])
    steering_df.set_index(['timestamp'], inplace=True)
    steering_df.index.rename('index', inplace=True)

    merged = functools.reduce(lambda left, right:
                              pd.merge(left,
                                       right,
                                       how='outer',
                                       left_index=True,
                                       right_index=True),
                              [camera_df, steering_df])
    merged.interpolate(method='time', inplace=True)

    filtered_cols = [
        'timestamp', 'width', 'height', 'frame_id', 'filename', 'angle',
        'torque', 'speed'
    ]
    filtered = merged.loc[camera_df.index]  # back to only camera rows
    filtered.fillna(0.0, inplace=True)
    filtered['timestamp'] = filtered.index.astype(
        'int')  # add back original timestamp integer col
    filtered['width'] = filtered['width'].astype('int')  # cast back to int
    filtered['height'] = filtered['height'].astype('int')  # cast back to int
    filtered = filtered[
        filtered_cols]  # filter and reorder columns for final output

    interpolated_csv_path = os.path.join(args.outdir, 'interpolated.csv')
    filtered.to_csv(interpolated_csv_path, header=True)


if __name__ == '__main__':
    main()
