from __future__ import print_function, absolute_import, division

import tensorflow as tf
import numpy as np

from FCN_layers import conv_layer, pool_layer, dropout_layer, deconv_layer

# Global parameters
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_OF_CLASS = 10
KEEP_PROB = 0.5
WEIGHTS = np.load('vgg19_weights.npy', encoding='bytes').item()
LEARNING_INIT = 0.001

def input(image_height, image_width):
    """
    initialize placeholder for images and annotations
    :param image_height:
    :param image_width:
    :return:
    """
    with tf.name_scope('input'):
        in_imgs = tf.placeholder(dtype=tf.float32,
                                 shape=[None, image_height, image_width, 3],
                                 name='input_images')
        in_ants = tf.placeholder(dtype=tf.int32,
                                 shape=[None, image_height, image_width, 1],
                                 name='input_annotations')

    return in_imgs, in_ants

def inference(image, num_of_class, weights, keep_prob):
    """
    main inference process
    :param image:
    :param num_of_class:
    :param weights_file:
    :param keep_prob:
    :return:
    """
    with tf.variable_scope('inference'):
        # downsampling inference
        # Conv1
        conv1_1 = conv_layer('conv1_1', image, 3, 3, 3, 64, 1, 1, 'SAME', tf.nn.relu,
                             weights['conv1_1'][0], weights['conv1_1'][1])
        conv1_2 = conv_layer('conv1_2', conv1_1, 3, 3, 64, 64, 1, 1, 'SAME', tf.nn.relu,
                             weights['conv1_2'][0],weights['conv1_1'][1])
        pool1 = pool_layer('pool1', conv1_2, 2, 2, 2, 2, 'SAME', tf.nn.avg_pool)

        # Conv2
        conv2_1 = conv_layer('conv2_1', pool1, 3, 3, 64, 128, 1, 1, 'SAME', tf.nn.relu,
                             weights['conv2_1'][0], weights['conv2_1'][1])
        conv2_2 = conv_layer('conv2_2', conv2_1, 3, 3, 128, 128, 1, 1, 'SAME', tf.nn.relu,
                             weights['conv2_2'][0], weights['conv2_2'][1])
        pool2 = pool_layer('pool2', conv2_2, 2, 2, 2, 2, 'SAME', tf.nn.avg_pool)

        # Conv3
        conv3_1 = conv_layer('conv3_1', pool2, 3, 3, 128, 256, 1, 1, 'SAME', tf.nn.relu,
                             weights['conv3_1'][0], weights['conv3_1'][1])
        conv3_2 = conv_layer('conv3_2', conv3_1, 3, 3, 256, 256, 1, 1, 'SAME', tf.nn.relu,
                             weights['conv3_2'][0], weights['conv3_2'][1])
        conv3_3 = conv_layer('conv3_3', conv3_2, 3, 3, 256, 256, 1, 1, 'SAME', tf.nn.relu,
                             weights['conv3_3'][0], weights['conv3_3'][1])
        conv3_4 = conv_layer('conv3_4', conv3_3, 3, 3, 256, 256, 1, 1, 'SAME', tf.nn.relu,
                             weights['conv3_4'][0], weights['conv3_4'][1])
        pool3 = pool_layer('pool3', conv3_4, 2, 2, 2, 2, 'SAME', tf.nn.avg_pool)

        # Conv4
        conv4_1 = conv_layer('conv4_1', pool3, 3, 3, 256, 512, 1, 1, 'SAME', tf.nn.relu,
                             weights['conv4_1'][0], weights['conv4_1'][1])
        conv4_2 = conv_layer('conv4_2', conv4_1, 3, 3, 512, 512, 1, 1, 'SAME', tf.nn.relu,
                             weights['conv4_2'][0], weights['conv4_2'][1])
        conv4_3 = conv_layer('conv4_3', conv4_2, 3, 3, 512, 512, 1, 1, 'SAME', tf.nn.relu,
                             weights['conv4_3'][0], weights['conv4_3'][1])
        conv4_4 = conv_layer('conv4_4', conv4_3, 3, 3, 512, 512, 1, 1, 'SAME', tf.nn.relu,
                             weights['conv4_4'][0], weights['conv4_4'][1])
        pool4 = pool_layer('pool4', conv4_4, 2, 2, 2, 2, 'SAME', tf.nn.avg_pool)

        # Conv5
        conv5_1 = conv_layer('conv5_1', pool4, 3, 3, 512, 512, 1, 1, 'SAME', tf.nn.relu,
                             weights['conv5_1'][0], weights['conv5_1'][1])
        conv5_2 = conv_layer('conv5_2', conv5_1, 3, 3, 512, 512, 1, 1, 'SAME', tf.nn.relu,
                             weights['conv5_2'][0], weights['conv5_2'][1])
        conv5_3 = conv_layer('conv5_3', conv5_2, 3, 3, 512, 512, 1, 1, 'SAME', tf.nn.relu,
                             weights['conv5_3'][0], weights['conv5_3'][1])
        conv5_4 = conv_layer('conv5_4', conv5_3, 3, 3, 512, 512, 1, 1, 'SAME', tf.nn.relu,
                             weights['conv5_4'][0], weights['conv5_4'][1])
        pool5 = pool_layer('pool5', conv5_4, 2, 2, 2, 2, 'SAME', tf.nn.max_pool)

        # Conv6
        conv6 = conv_layer('conv6', pool5, 7, 7, 512, 4096, 1, 1, 'SAME', tf.nn.relu)
        drop6 = dropout_layer('drop6', conv6, keep_prob)

        # Conv7
        conv7 = conv_layer('conv7', drop6, 1, 1, 4096, 4096, 1, 1, 'SAME', tf.nn.relu)
        drop7 = dropout_layer('drop7', conv7, keep_prob)

        # Conv8
        conv8 = conv_layer('conv8', drop7, 1, 1, 4096, num_of_class, 1, 1, 'SAME', tf.identity)

        # upsampling inference
        # Deconv1 -  32x
        deconv1_shape = tf.shape(pool4)
        deconv1_out_channels = pool4.get_shape()[3].value
        deconv1 = deconv_layer('deconv1', conv8, 4, 4, deconv1_out_channels, num_of_class,
                               deconv1_shape, 2, 2, 'SAME')
        fuse1 = tf.add(deconv1, pool4, name='fuse1')

        # Deconv2 - 16x
        deconv2_shape = tf.shape(pool3)
        deconv2_out_channels = pool3.get_shape()[3].value
        deconv2 = deconv_layer('deconv2', fuse1, 4, 4, deconv2_out_channels, deconv1_out_channels,
                               deconv2_shape, 2, 2, 'SAME')
        fuse2 = tf.add(deconv2, pool3, name='fuse2')

        # Deconv3 - 8x
        deconv3_shape = tf.shape(image)
        deconv3_shape = tf.stack([deconv3_shape[0], deconv3_shape[1], deconv3_shape[2], num_of_class])
        deconv3 = deconv_layer('deconv3', fuse2, 16, 16, num_of_class, deconv2_out_channels,
                               deconv3_shape, 8, 8, 'SAME')

        prediction = tf.expand_dims(tf.argmax(deconv3, dimension=3, name='pred_annotations'), dim=3)

    return deconv3, prediction

def loss(logits, annotation):
    with tf.name_scope('loss'):
        # change annotation into 0 and 1
        labels = tf.squeeze(annotation, squeeze_dims=[3])
        # labels_one_hot = tf.one_hot(labels, depth=2)
        # predictions = tf.argmax(logits, dimension=3)

        # softmax Cross_entropy
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=labels,
                                                                name='cross_entropy')

        # Sigmoid Cross_entropy
        # losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)

        # Hinge Loss
        # losses = tf.losses.hinge_loss(labels=labels_one_hot, logits=logits)

        # Huber Loss
        # losses = tf.losses.huber_loss(labels=labels_one_hot,
        #                               predictions=logits)

        # Log Loss
        # losses = tf.losses.log_loss(labels=labels, predictions=predictions)

        loss_op = tf.reduce_mean(losses)

    return loss_op

def train(loss_op, learning_init, var_list):
    with tf.name_scope('train'):
        # step counter
        global_step = tf.Variable(initial_value=0, name='global_step', dtype=tf.int32, trainable=False)

        # optimizer
        leanring_rate = tf.train.exponential_decay(learning_init, global_step, 100, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=leanring_rate)

        grad_and_vars = optimizer.compute_gradients(loss=loss_op, var_list=var_list)

        train_op = optimizer.apply_gradients(grads_and_vars=grad_and_vars, global_step=global_step)

    return train_op

def evaluate(prediction, annotation):
    with tf.name_scope('evaluate'):
        correctness = tf.equal(tf.cast(prediction, tf.int32), annotation)
        accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))
    return accuracy

def statistics(prediction, annotation):
    with tf.name_scope('statistics'):
        predict_values = tf.cast(prediction, tf.float32)
        annotation_values = tf.cast(annotation, tf.float32)

        true_positive = tf.count_nonzero(predict_values * annotation_values, dtype=tf.float32)
        true_negative = tf.count_nonzero((predict_values - 1) * (annotation_values - 1), dtype=tf.float32)
        false_positive = tf.count_nonzero(predict_values * (annotation_values - 1), dtype=tf.float32)
        false_negative = tf.count_nonzero((predict_values - 1) * annotation_values, dtype=tf.float32)

        precesion = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f_score = 2 * precesion * recall / (precesion + recall)
        mcc = (true_positive * true_negative - false_positive * false_negative) / tf.sqrt(
            (true_positive + false_positive) * (true_positive + false_negative) *
            (true_negative + false_positive) * (true_negative + false_negative))
    return precesion, recall, f_score, mcc

if __name__ == '__main__':

    writer = tf.summary.FileWriter(logdir='logs', graph=tf.get_default_graph())

    with tf.Session() as sess:
        img_holder, ant_holder = input(IMAGE_HEIGHT, IMAGE_WIDTH)
        logits, predictions = inference(img_holder, NUM_OF_CLASS, WEIGHTS, KEEP_PROB)

        trainable_var_list = tf.trainable_variables()

        loss_op = loss(logits, ant_holder)

        train_op = train(loss_op, LEARNING_INIT, trainable_var_list)

        writer.add_graph(sess.graph)
        writer.flush()

