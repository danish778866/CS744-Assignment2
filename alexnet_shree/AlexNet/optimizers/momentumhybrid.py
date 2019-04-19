"""Momentum for TensorFlow. Support per layer hyper parameter"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import re

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer, training_ops


class HybridMomentumOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Momentum algorithm suitable for hybrid parallelism.
    @@__init__
    """

    def __init__(self, layer_params, use_locking=False, name="HybridMomentum", use_nesterov=False):
        """Construct a new Momentum optimizer.
        Args:
          param_Map: A map of structure
              {
                  'var_name_reg': {
                       'learning_rate': learning_rate,
                       'momentum': momentum
                   }
                   'default': {
                       default value
                   }
               }
          where
          learning_rate: An `Output` or a floating point value.  The learning rate.
          momentum: An `Output` or a floating point value.  The momentum.

          use_locking: If `True` use locks for update operations.
          name: Optional name prefix for the operations created when applying
            gradients.  Defaults to "Momentum".
          use_nesterov: Optional whether use nesterov mementum
        """
        super(HybridMomentumOptimizer, self).__init__(use_locking, name)
        self._use_nesterov = use_nesterov
        self._layer_params = layer_params

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "velocity", self._name)

    def _prepare(self):
        if 'default' not in self._layer_params:
            self._layer_params['default'] = {
                'learning_rate': 0.02,
                'momentum': 0.9
            }
        for var_name in self._layer_params:
            lr = self._layer_params[var_name]['learning_rate']
            mom = self._layer_params[var_name]['momentum']

            lr = ops.convert_to_tensor(lr)
            mom = ops.convert_to_tensor(mom)
            self._layer_params[var_name]['learning_rate'] = lr
            self._layer_params[var_name]['momentum'] = mom

    def _params_for_var(self, var):
        name = var.op.name
        selected = 'default'
        if name in self._layer_params:
            selected = name
        else:
            for pattern, params in self._layer_params.items():
                if re.search(pattern, name):
                    selected = pattern
                    break
        if selected == 'default':
            print('WARNING: default parameter used for var {}'.format(name))
        return (self._layer_params[selected]['learning_rate'],
                self._layer_params[selected]['momentum'],
                self._use_locking,
                self._use_nesterov)

    def _apply_dense(self, grad, var):
        vec = self.get_slot(var, "velocity")
        lr, mom, locking, nesterov = self._params_for_var(var)
        return training_ops.apply_momentum(
            var, vec,
            math_ops.cast(lr, var.dtype.base_dtype),
            grad,
            math_ops.cast(mom, var.dtype.base_dtype),
            use_locking=locking,
            use_nesterov=nesterov).op

    def _apply_sparse(self, grad, var):
        vec = self.get_slot(var, "velocity")
        lr, mom, locking, nesterov = self._params_for_var(var)
        return training_ops.sparse_apply_momentum(
            var, vec,
            math_ops.cast(lr, var.dtype.base_dtype),
            grad.values, grad.indices,
            math_ops.cast(mom, var.dtype.base_dtype),
            use_locking=locking,
            use_nesterov=nesterov).op
