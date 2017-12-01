import tensorflow as tf
from deep_ldr2hdr_utils import *
import numpy as np

# Create some wrappers for simplicity

def conv2d(input_, output_channels, k_h=5, k_w=5, pool_method=None, padding='SAME', name="conv2d"):
    # Conv2D wrapper
    # input: [batch, in_height, in_width, in_channels]
    # k_h k_w: filter_height, filter_width
    # w or filter:  [filter_height, filter_width, in_channels, out_channels],
    # pool_method could be ['stride', 'max', 'avg'], None as default: no pooling
    strides = 1
    if pool_method == 'stride':
        strides = 2

    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_channels],
                            initializer=tf.truncated_normal_initializer(stddev=1))
        biases = tf.get_variable('biases', [output_channels], initializer=tf.truncated_normal_initializer())
        conv = tf.nn.bias_add(tf.nn.conv2d(input_, w, strides=[1, strides, strides, 1], padding=padding), biases)
        return conv


def maxpool2d(x, k=2, name='max_pool'):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def avgpool2d(x, k=2, name='avg_pool'):
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


def deconv2d(input_, output_channels, output_imshape=[], k_h=5, k_w=5, method='resize', padding='SAME', name="deconv2d"):
    """deConv2D wrapper
    input_:  [batch, in_height, in_width, in_channels]
    output_shape:  [output_height, output_width],
    strides = 2 to upsample
    """
    with tf.variable_scope(name):
        batch_size, in_height, in_width, in_channels = input_.get_shape().as_list()
        biases = tf.get_variable('biases', output_channels, initializer=tf.truncated_normal_initializer())

        if method == 'upsample':
            '''deconv method : checkerboard issue'''
            w = tf.get_variable('w', [k_h, k_w, output_channels, in_channels],
                                initializer=tf.truncated_normal_initializer(stddev=1))
            output_shape = [batch_size, output_imshape[0], output_imshape[1], output_channels]
            strides = int(output_imshape[0] / in_height)
            deconv = tf.nn.bias_add(tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, strides, strides, 1], padding=padding, data_format='NHWC'), biases)  # deconv
        elif method == 'resize':
            '''resize-conv method http://distill.pub/2016/deconv-checkerboard/'''
            w = tf.get_variable('w', [k_h, k_w, in_channels, output_channels],
                                initializer=tf.truncated_normal_initializer(stddev=1))
            im_resized = tf.image.resize_images(input_, [output_imshape[0], output_imshape[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            deconv = tf.nn.bias_add(tf.nn.conv2d(im_resized, w, strides=[1, 1, 1, 1], padding=padding), biases)  # resize-conv
        return deconv


def fc2d(input_, fc_dim, name='fc2d'):
    # input_:  [batch, in_height, in_width, in_channels]
    with tf.variable_scope(name):
        batch_size = input_.get_shape()[0].value
        in_height = input_.get_shape()[1].value
        in_width = input_.get_shape()[2].value
        in_channels = input_.get_shape()[3].value
        in_dim = in_height * in_width * in_channels

        w = tf.get_variable('w', [in_dim, fc_dim],
                            initializer=tf.truncated_normal_initializer(stddev=1))

        fc = tf.reshape(input_, [-1, in_dim])
        fc = tf.matmul(fc, w)

        b = tf.get_variable('biases', fc_dim, initializer=tf.truncated_normal_initializer())
        fc = tf.add(fc, b)
        fc = tf.reshape(fc, [-1, 1, 1, fc_dim])
        return fc


def dfc2d(input_, out_height, out_width, out_channels, name='dfc2d'):
    # de-fully connected
    # input_:  [batch, 1, 1, fc_dim]
    with tf.variable_scope(name):
        batch_size = input_.get_shape()[0].value
        fc_dim = input_.get_shape()[-1].value
        input_ = tf.reshape(input_, [-1, fc_dim])

        out_dim = out_height * out_width * out_channels

        w = tf.get_variable('w', [fc_dim, out_dim],
                            initializer=tf.truncated_normal_initializer(stddev=1))
        fc = tf.matmul(input_, w)

        b = tf.get_variable('biases', out_dim, initializer=tf.truncated_normal_initializer())
        fc = tf.add(fc, b)
        fc = tf.reshape(fc, [-1, out_height, out_width, out_channels])
        return fc


def batch_norm(x, isTraining, name="batch_norm"):
    bn = tf.contrib.slim.batch_norm(x, decay=0.9, center=True, scale=True,
                                    updates_collections=None,
                                    is_training=isTraining,
                                    reuse=None,
                                    trainable=True,
                                    scope=name)
    return bn
