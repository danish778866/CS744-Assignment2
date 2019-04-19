from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def alexnet_part_conv(builder, images):
    net = builder.conv('conv1', images, 64, [11, 11], stride=4, padding='VALID',
                       weight_stddev=0.01, weight_decay=0.0005)
    net = builder.max_pool('pool1', net, 3, stride=2, activation=tf.nn.relu)

    net = builder.conv('conv2', net, 192, [5, 5], stride=1, padding='SAME',
                       weight_stddev=0.01, bias_mean=1, weight_decay=0.0005,
                       activation=tf.nn.relu)
    net = builder.max_pool('pool2', net, 3, stride=2)

    net = builder.conv('conv3', net, 384, [3, 3], stride=1, padding='SAME',
                       weight_stddev=0.03, weight_decay=0.0005)
    net = builder.conv('conv4', net, 256, [3, 3], stride=1, padding='SAME',
                       weight_stddev=0.03, weight_decay=0.0005, activation=tf.nn.relu)

    net = builder.conv('conv5', net, 256, [3, 3], stride=1, padding='SAME',
                       weight_stddev=0.03, weight_decay=0.0005)
    net = builder.max_pool('pool3', net, 3, stride=2, activation=tf.nn.relu)
    return net


def alexnet_loss(logits, labels, scope=None):
    """Build objective function"""
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                                                                   #name='cross_entropy_batch')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    tf.add_to_collection(tf.GraphKeys.LOSSES, cross_entropy_mean)
    # return tf.identity(cross_entropy_mean, name='total_loss')

    total_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope)
                          + [cross_entropy_mean], name='total_loss')
    tf.add_to_collection(tf.GraphKeys.LOSSES, total_loss)
    return total_loss


def alexnet_inference(builder, images, labels, num_classes, scope=None):
    """Internal use"""
    net = alexnet_part_conv(builder, images)

    net = builder.fc('fc4096a', net, 4096,
                     weight_stddev=0.01, bias_mean=1.0, weight_decay=0.0005,
                     activation=tf.nn.relu)

    net = builder.dropout('dropout1', net, 0.75)

    net = builder.fc('fc4096b', net, 4096,
                     weight_stddev=0.01, bias_mean=1.0, weight_decay=0.0005,
                     activation=tf.nn.relu)

    net = builder.dropout('dropout2', net, 0.75)

    net = builder.fc('fc1000', net, num_classes,
                     weight_stddev=0.01, bias_mean=-7, weight_decay=0.0005)

    # save the unscaled logits for training
    logits = net

    with tf.name_scope('probs'):
        net = tf.nn.softmax(net)

    return net, logits, alexnet_loss(logits, labels, scope)


def alexnet_eval(probs, labels):
    """Evaluate, returns number of correct images"""
    with tf.name_scope('evaluation'):
        correct = tf.nn.in_top_k(probs, labels, k=1)
        return tf.reduce_sum(tf.cast(correct, tf.int32))
