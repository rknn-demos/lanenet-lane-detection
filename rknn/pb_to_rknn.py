from rknn.api import RKNN
import argparse


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='The model pb file')
    parser.add_argument('-o', '--output', type=str, help='The output rknn file')
    return parser.parse_args()


def to_rknn(pb_path, rknn_path):
    rknn = RKNN(verbose=True)
    rknn.config(channel_mean_value='127.5 127.5 127.5 127.5', reorder_channel='2 1 0')
    rknn.load_tensorflow(tf_pb=pb_path,
                         inputs=['input_tensor'],
                         outputs=[
                             'lanenet_model/vgg_backend/binary_seg/ArgMax',
                             #'lanenet_model/vgg_backend/binary_seg/Softmax',
                             'lanenet_model/vgg_backend/instance_seg/pix_embedding_conv/pix_embedding_conv'],
                         input_size_list=[[256, 512, 3]])

    rknn.build(do_quantization=False, dataset='./dataset.txt')
    rknn.export_rknn(rknn_path)


if __name__ == '__main__':
    args = init_args()
    to_rknn(args.input, args.output)
