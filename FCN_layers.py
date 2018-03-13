from __future__ import print_function, absolute_import, division

import tensorflow as tf
import numpy as np

def conv_layer(layer_name, input_tensor, filter_height, filter_width, in_channels, out_channels,
               stride_height, stride_width, padding='SAME', activation_function=tf.nn.relu,
               filter_weights=None, bias_weights=None):
    """

    :param layer_name:
    :param input_tensor:
    :param filter_height:
    :param filter_width:
    :param in_channels:
    :param out_channels:
    :param stride_height:
    :param stride_width:
    :param padding:
    :param activation_function:
    :param filter_init:
    :param bias_init:
    :return:
    """
    with tf.variable_scope(layer_name):
        if filter_weights is None:
            filter_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)
        else:
            filter_init = tf.constant_initializer(value=filter_weights, dtype=tf.float32)

        filter = tf.get_variable(name='filter',
                                 shape=[filter_height, filter_width, in_channels, out_channels],
                                 initializer=filter_init,
                                 dtype=tf.float32)

        if bias_weights is None:
            bias_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
        else:
            bias_init = tf.constant_initializer(value=bias_weights, dtype=tf.float32)

        bias = tf.get_variable(name='bias',
                               shape=[out_channels],
                               initializer=bias_init,
                               dtype=tf.float32)

        conv = tf.nn.conv2d(input=input_tensor,
                            filter=filter,
                            strides=[1, stride_height, stride_width, 1],
                            padding=padding)

        conv_add = tf.nn.bias_add(value=conv, bias=bias)

        layer_out = activation_function(conv_add, name='layer_out')

    return layer_out

def pool_layer(layer_name, input_tensor, kernel_height, kernel_width,
               stride_height, stride_width, padding='SAME', pool_function=tf.nn.max_pool):
    """

    :param layer_name:
    :param input_tensor:
    :param kernel_height:
    :param kernel_width:
    :param stride_height:
    :param stride_width:
    :param padding:
    :param pool_function:
    :return:
    """
    with tf.variable_scope(layer_name):
        layer_out = pool_function(value=input_tensor,
                                  ksize=[1, kernel_height, kernel_width, 1],
                                  strides=[1, stride_height, stride_width, 1],
                                  padding=padding,
                                  name='layer_out')

    return layer_out

def dropout_layer(layer_name, input_tensor, keep_prob):
    """

    :param layer_name:
    :param input_tensor:
    :param keep_prob:
    :return:
    """
    with tf.variable_scope(layer_name):
        layer_out = tf.nn.dropout(x=input_tensor,
                                  keep_prob=keep_prob,
                                  name='layer_out')

    return layer_out

def deconv_layer(layer_name, input_tensor, filter_height, filter_width, out_channels, in_channels,
                 output_shape, stride_height, stride_width, padding='SAME'):
    """

    :param layer_name:
    :param input_tensor:
    :param filter_height:
    :param filter_width:
    :param out_channels:
    :param in_channels:
    :param output_shape:
    :param stride_height:
    :param stride_width:
    :param padding:
    :return:
    """
    with tf.variable_scope(layer_name):
        filter_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)
        filter = tf.get_variable(name='filter',
                                 shape=[filter_height, filter_width, out_channels, in_channels],
                                 initializer=filter_init,
                                 dtype=tf.float32)

        bias_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
        bias = tf.get_variable(name='bias',
                               shape=[out_channels],
                               initializer=bias_init,
                               dtype=tf.float32)


        deconv = tf.nn.conv2d_transpose(value=input_tensor,
                                        filter=filter,
                                        output_shape=output_shape,
                                        strides=[1, stride_height, stride_width, 1],
                                        padding=padding)

        layer_out = tf.nn.bias_add(value=deconv, bias=bias, name='layer_out')

    return layer_out
