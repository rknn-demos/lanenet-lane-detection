#!/usr/bin/env python3.6
import argparse
import os.path as ops
import time

from rknn.api import RKNN
import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import global_config
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess

CFG = global_config.cfg


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


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


def run_test(rknn, image_path):
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)

    postprocessor = lanenet_postprocess.LaneNetPostProcessor()

    print('starting inference ...')
    instance_seg_image, binary_seg_image = rknn.inference(inputs=[image])
    binary_seg_image = binary_seg_image.reshape(1, 256, 512)
    instance_seg_image = instance_seg_image.reshape(1, 256, 512, 4)

    print("Binary :" + str(binary_seg_image.shape))
    print(binary_seg_image)
    print("Instance :" + str(instance_seg_image.shape))
    print(instance_seg_image)
    print('done')

    postprocess_result = postprocessor.postprocess(
        binary_seg_result=binary_seg_image[0],
        instance_seg_result=instance_seg_image[0],
        source_image=image_vis
    )
    mask_image = postprocess_result['mask_image']

    for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
        instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
    embedding_image = np.array(instance_seg_image[0], np.uint8)

    plt.figure('mask_image')
    plt.imshow(mask_image[:, :, (2, 1, 0)])
    plt.figure('src_image')
    plt.imshow(image_vis[:, :, (2, 1, 0)])
    plt.figure('instance_image')
    plt.imshow(embedding_image[:, :, (2, 1, 0)])
    plt.figure('binary_image')
    plt.imshow(binary_seg_image[0] * 255, cmap='gray')
    plt.show()


if __name__ == '__main__':
    args = init_args()

    rknn = init_rknn(args.rknn)
    run_test(rknn, args.image_path)
    rknn.release()
