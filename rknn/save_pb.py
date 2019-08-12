#!/usr/bin/env python3.6
import tensorflow as tf
from tensorflow.python.framework import graph_util
from lanenet_model import lanenet

weights_path = '../model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt'

if __name__ == '__main__':
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    sess = tf.Session()
    saver = tf.train.Saver()

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=tf.get_default_graph().as_graph_def(),  # 等于:sess.graph_def
            output_node_names=['lanenet_model/vgg_backend/binary_seg/ArgMax', 'lanenet_model/vgg_backend/instance_seg/pix_embedding_conv/pix_embedding_conv'])  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile("./lanenet.pb", "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
            print('done')
