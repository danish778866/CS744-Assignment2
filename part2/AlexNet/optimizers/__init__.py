from __future__ import absolute_import, division, print_function

import tensorflow as tf

from .momentumhybrid import HybridMomentumOptimizer  # noqa: F401


def exp_decay(start, tgtFactor, num_stairs, total_num_steps):
    decay_step = total_num_steps / (num_stairs - 1)
    decay_rate = (1 / tgtFactor) ** (1 / (num_stairs - 1))
    global_step = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0]
    return tf.train.exponential_decay(start, global_step, decay_step, decay_rate,
                                      staircase=True)


def lparam(learning_rate, momentum):
    return {
        'learning_rate': learning_rate,
        'momentum': momentum
    }
