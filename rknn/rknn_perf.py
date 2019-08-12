#!/usr/bin/env python3.6
import argparse
import os

from rknn.api import RKNN
import cv2
import matplotlib.pyplot as plt


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('-r', '--rknn', type=str, help='The model rknn file')
    return parser.parse_args()


def init_rknn(rknn_path):
    rknn = RKNN()
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('starting init runtime ...')
    rknn.init_runtime(target='rk1808')
    return rknn


def perf(rknn, image_path):
    assert os.path.exists(image_path), '{:s} not exist'.format(image_path)
    image = cv2.imread(args.image_path)
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    print('starting eval perf ...')
    rknn.eval_perf(inputs=[image], is_print=True)
    print('done')


if __name__ == '__main__':
    args = init_args()
    rknn = init_rknn(args.rknn)
    perf(rknn, args.image_path)
    rknn.release()