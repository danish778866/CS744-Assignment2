from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops

from ..utils import tfhelper


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tfhelper.scalar_summary(var.name + '/mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tfhelper.scalar_summary(var.name + '/stddev', stddev)
        tfhelper.scalar_summary(var.name + '/max', tf.reduce_max(var))
        tfhelper.scalar_summary(var.name + '/min', tf.reduce_min(var))
        tfhelper.histogram_summary(var.name + '/histogram', var)


class DataSet(object):
    pass


class PartitionedTensor(object):
    """Output of partitioned layers, actually a list of tensors from each partition"""
    def __init__(self, tensors, pscope):
        super(PartitionedTensor, self).__init__()
        self._tensors = tensors
        self._pscope = pscope

    def __getitem__(self, key):
        return self._tensors[key]

    def current_partition(self):
        """Get the tensor on current device"""
        idx = self._pscope._current_idx
        return self._tensors[idx]


class PartitionedLayerScope(object):
    """A scope where layers are partitioned on multiple devices"""
    def __init__(self, builder, devices, colocate_variables):
        super(PartitionedLayerScope, self).__init__()
        self.builder = builder
        self.devices = devices
        self.colocate_variables = colocate_variables
        self._current_idx = -1
        self._iter = None

    def on_devices(self):
        self._current_idx = 0
        for dev in self.devices:
            with self.builder.device(dev, colocate_variables=self.colocate_variables):
                yield dev, self._current_idx
                self._current_idx += 1
        self._current_idx = -1


class ModelBuilder(object):
    """model config"""

    def __init__(self, param_dev=None):
        super(ModelBuilder, self).__init__()
        self._parameter_device = param_dev
        self._variable_scope_stack = ''
        self._pscope = None

    @contextmanager
    def parallel(self, devices, colocate_variables=False):
        old_pscope = self._pscope
        self._pscope = PartitionedLayerScope(self, devices, colocate_variables)
        yield self._pscope
        self._pscope = old_pscope

    @contextmanager
    def device(self, device, colocate_variables=False):
        if colocate_variables:
            old_parameter_device = self._parameter_device
            self.set_variable_device(device)
        with tf.device(device):
            yield
        if colocate_variables:
            self.set_variable_device(old_parameter_device)

    def set_variable_device(self, dev):
        self._parameter_device = dev

    def variable_device(self):
        return self._parameter_device

    def create_variable(self, name, shape, dtype, initializer):
        """Create a variable"""
        with tf.device(self.variable_device()):
            var = tf.get_variable(self._variable_scope_stack + name,
                                  shape, dtype=dtype, initializer=initializer)
            variable_summaries(var)
            return var

    def ensure_global_step(self):
        """Create global step"""
        l = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)
        if l:
            return l[0]
        with tf.device(self.variable_device()):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            tf.add_to_collection(tf.GraphKeys.GLOBAL_STEP, global_step)
        return global_step

    def _merge_inputs(self, inputs, axis=1):
        """Merge inputs which potentially is a list.
           Concat them along axis if it is a list.
           If the inputs is PartitionedTensor,
           only return the tensor on corresponding device when we are in the same scope,
           otherwise return the merged input"""
        if isinstance(inputs, PartitionedTensor):
            if self._pscope == inputs._pscope:
                # we are in the same scope
                return inputs.current_partition()
            else:
                inputs = inputs._tensors

        if not isinstance(inputs, list):
            return inputs

        if len(inputs) == 1:
            return inputs[0]
        return tf.concat(axis, inputs, name='merged_input')

    def conv(self, scope, inputs, num_outputs, filter_size, stride=1, padding='SAME',
             weight_stddev=0.01, bias_mean=0.5, weight_decay=None, activation=None,
             concat_axis=1):
        """Convolutional layer"""
        if self._pscope is None:
            return self._conv(scope, inputs, num_outputs, filter_size, stride, padding,
                              weight_stddev, bias_mean, weight_decay, activation, concat_axis)

        tensors = []
        num_outputs_part = num_outputs / len(self._pscope.devices)
        for _, idx in self._pscope.on_devices():
            scope_part = '{}part{}'.format(scope, idx)
            output = self._conv(scope_part, inputs, num_outputs_part, filter_size, stride,
                                padding, weight_stddev, bias_mean, weight_decay, activation,
                                concat_axis)
            tensors.append(output)
        return PartitionedTensor(tensors, self._pscope)

    def _conv(self, scope, inputs, num_outputs, filter_size, stride, padding, weight_stddev,
              bias_mean, weight_decay, activation, concat_axis):
        # Create variables
        weight_initializer = None
        if weight_initializer is None:
            weight_initializer = tf.random_normal_initializer(stddev=weight_stddev)

        bias_initializer = None
        if bias_initializer is None:
            bias_initializer = tf.constant_initializer(bias_mean)

        with tf.variable_scope(scope):
            inputs = self._merge_inputs(inputs, concat_axis)
            channels = inputs.get_shape()[-1]
            filters = self.create_variable('weights', filter_size + [channels, num_outputs],
                                           dtype=tf.float32,
                                           initializer=weight_initializer)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, filters)

            biases = self.create_variable('biases', [num_outputs],
                                          dtype=tf.float32,
                                          initializer=bias_initializer)
            tf.add_to_collection(tf.GraphKeys.BIASES, biases)

            if weight_decay is not None:
                # add weight_decay loss
                wl = tf.multiply(tf.nn.l2_loss(filters), weight_decay, name='weight_loss')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, wl)

            if activation is None:
                activation = tf.identity

            # Actual computation op
            conv = tf.nn.conv2d(inputs, filters, [1, stride, stride, 1], padding=padding)
            bias = tf.nn.bias_add(conv, biases)
            acts = activation(bias)
            # add summary
            tfhelper.histogram_summary(bias.name + '/pre_activations', bias)
            tfhelper.histogram_summary(acts.name + '/activations', bias)
            return acts

    def max_pool(self, scope, inputs, patch_size, stride=1, activation=None, concat_axis=1):
        """Max pooling"""
        if self._pscope is None:
            return self._max_pool(scope, inputs, patch_size, stride, activation, concat_axis)

        tensors = []
        for _, idx in self._pscope.on_devices():
            scope_part = '{}part{}'.format(scope, idx)
            output = self._max_pool(scope_part, inputs, patch_size, stride, activation,
                                    concat_axis)
            tensors.append(output)
        return PartitionedTensor(tensors, self._pscope)

    def _max_pool(self, scope, inputs, patch_size, stride, activation, concat_axis):
        if activation is None:
            activation = tf.identity

        with tf.name_scope(scope):
            inputs = self._merge_inputs(inputs, concat_axis)
            pool = tf.nn.max_pool(inputs, ksize=[1, patch_size, patch_size, 1],
                                  strides=[1, stride, stride, 1], padding='VALID')
            if activation is not None:
                pool = activation(pool)
        return pool

    def fc(self, scope, inputs, num_outputs, weight_stddev=0.01, bias_mean=0.0,
           weight_decay=None, activation=None, concat_axis=1):
        """Fully connected"""
        if self._pscope is None:
            return self._fc(scope, inputs, num_outputs, weight_stddev, bias_mean, weight_decay,
                            activation, concat_axis)

        tensors = []
        num_outputs_part = num_outputs / len(self._pscope.devices)
        for _, idx in self._pscope.on_devices():
            scope_part = '{}part{}'.format(scope, idx)
            output = self._fc(scope_part, inputs, num_outputs_part, weight_stddev, bias_mean,
                              weight_decay, activation, concat_axis)
            tensors.append(output)
        return PartitionedTensor(tensors, self._pscope)

    def _fc(self, scope, inputs, num_outputs, weight_stddev, bias_mean, weight_decay, activation,
            concat_axis):
        # Create variables
        weight_initializer = None
        if weight_initializer is None:
            weight_initializer = tf.random_normal_initializer(stddev=weight_stddev)

        bias_initializer = None
        if bias_initializer is None:
            bias_initializer = tf.constant_initializer(bias_mean)

        with tf.variable_scope(scope):
            inputs = self._merge_inputs(inputs, concat_axis)
            # Calculate shapes
            inputs_shape = inputs.get_shape().as_list()
            n_inputs = np.prod(inputs_shape[1:])
            weights_shape = [n_inputs, num_outputs]

            weights = self.create_variable('weights', weights_shape,
                                           dtype=tf.float32,
                                           initializer=weight_initializer)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)

            biases = self.create_variable('biases', [num_outputs],
                                          dtype=tf.float32,
                                          initializer=bias_initializer)
            tf.add_to_collection(tf.GraphKeys.BIASES, biases)

            if weight_decay is not None:
                # add weight_decay loss
                wl = tf.multiply(tf.nn.l2_loss(weights), weight_decay, name='weight_loss')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, wl)

            # reshape inputs if necessary
            if len(inputs_shape) > 2:
                inputs = array_ops.reshape(inputs, [-1, n_inputs])

            outputs = tf.matmul(inputs, weights)
            outputs = tf.nn.bias_add(outputs, biases)
            tfhelper.histogram_summary(outputs.name + '/pre_activations', outputs)

            if activation is not None:
                outputs = activation(outputs)
            tfhelper.histogram_summary(outputs.name + '/activations', outputs)

        return outputs

    def dropout(self, scope, inputs, keep_prob, concat_axis=1):
        """Dropout"""
        if self._pscope is None:
            return self._dropout(scope, inputs, keep_prob, concat_axis)

        tensors = []
        for _, idx in self._pscope.on_devices():
            scope_part = '{}part{}'.format(scope, idx)
            output = self._dropout(scope_part, inputs, keep_prob, concat_axis)
            tensors.append(output)
        return PartitionedTensor(tensors, self._pscope)

    def _dropout(self, scope, inputs, keep_prob, concat_axis):
        with tf.name_scope(scope):
            inputs = self._merge_inputs(inputs, concat_axis)
            return tf.nn.dropout(inputs, keep_prob)

    def average_gradients(self, replica_grads):
        """Calculate the average gradient for each shared variable across all replicas.
        Note that this function provides a synchronization point across all replicas.
        Args:
            replica_grads: List of lists of (gradient, variable) tuples. The outer list
                is over individual replicas. The inner list is over the gradient
                calculation for each replica.
        Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all replicas.
        """
        average_grads = []
        for grad_and_vars in zip(*replica_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_replica0, var0_replica0), ... , (grad0_replicaN, var0_replicaN))

            # Average over the 'replica' dimension.
            grads = [g for g, _ in grad_and_vars]
            #grad = tf.pack(grads)
            grad = tf.stack(grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across replicas. So .. we will just return the first replica's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            average_grads.append((grad, v))
        return average_grads
