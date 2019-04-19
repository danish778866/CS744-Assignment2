from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pdb
import tensorflow as tf

from .common import ModelBuilder
from .alexnetcommon import alexnet_inference, alexnet_part_conv, alexnet_loss, alexnet_eval
from ..optimizers.momentumhybrid import HybridMomentumOptimizer


def original(images, labels, num_classes, total_num_examples, devices=None, is_train=True):
    """Build inference"""
    # pdb.set_trace()
    if devices is None:
        devices = [None]

    def configure_optimizer(global_step, total_num_steps):
        """Return a configured optimizer"""
        def exp_decay(start, tgtFactor, num_stairs):
            decay_step = total_num_steps / (num_stairs - 1)
            decay_rate = (1 / tgtFactor) ** (1 / (num_stairs - 1))
            return tf.train.exponential_decay(start, global_step, decay_step, decay_rate,
                                              staircase=True)

        def lparam(learning_rate, momentum):
            return {
                'learning_rate': learning_rate,
                'momentum': momentum
            }

        return HybridMomentumOptimizer({
            'weights': lparam(exp_decay(0.001, 250, 4), 0.9),
            'biases': lparam(exp_decay(0.002, 10, 2), 0.9),
        })

    def train(total_loss, global_step, total_num_steps):
        """Build train operations"""
        # Compute gradients
        with tf.control_dependencies([total_loss]):
            opt = configure_optimizer(global_step, total_num_steps)
            grads = opt.compute_gradients(total_loss)
        print("No of grads", len(grads))
        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        with tf.control_dependencies([apply_gradient_op]):
            return tf.no_op(name='train')

    with tf.device(devices[0]):
        builder = ModelBuilder()
	print('num_classes: ' + str(num_classes))
        net, logits, total_loss = alexnet_inference(builder, images, labels, num_classes)
        # pdb.set_trace()

        if not is_train:
            return alexnet_eval(net, labels)

        global_step = builder.ensure_global_step()
	print('total_num_examples: ' + str(total_num_examples))
        train_op = train(total_loss, global_step, total_num_examples)
    return net, logits, total_loss, train_op, global_step


def distribute(images, labels, num_classes, total_num_examples, devices, is_train=True):
    # You can refer to the "original" function above, it is for the single-node version.

    # 2. Configure your optimizer using HybridMomentumOptimizer.
    def configure_optimizer(global_step, total_num_steps):
        """Return a configured optimizer"""
        def exp_decay(start, tgtFactor, num_stairs):
            decay_step = total_num_steps / (num_stairs - 1)
            decay_rate = (1 / tgtFactor) ** (1 / (num_stairs - 1))
            return tf.train.exponential_decay(start, global_step, decay_step, decay_rate,
                                              staircase=True)

        def lparam(learning_rate, momentum):
            return {
                'learning_rate': learning_rate,
                'momentum': momentum
            }

        return HybridMomentumOptimizer({
            'weights': lparam(exp_decay(0.001, 250, 4), 0.9),
            'biases': lparam(exp_decay(0.002, 10, 2), 0.9),
        })

    # 3. Construct graph replica by splitting the original tensors into sub tensors. (hint: take a look at tf.split )
    def split_tensor(value, no_of_splits, axis):
        return tf.split(value, num_or_size_splits=no_of_splits,axis=axis,name='split')


    print("devices",len(devices)," ",devices)
    no_of_devices = len(devices)

    splits = split_tensor(images, no_of_splits=no_of_devices - 1, axis=0)
    label_splits = split_tensor(labels, no_of_splits=no_of_devices - 1, axis=0)

    print("splits are ",splits)





# 1. Create global steps on the parameter server node. You can use the same method that the single-machine program uses.

    print("Entering ",devices[no_of_devices-1])
    with tf.device(devices[no_of_devices-1]):
        builder = ModelBuilder()
        global_step = builder.ensure_global_step()




    # 4. For each worker node, create replica by calling alexnet_inference and computing gradients.
    #    Reuse the variable for the next replica. For more information on how to reuse variables in TensorFlow,
    #    read how TensorFlow Variables work, and considering using tf.variable_scope.
    total_grads = []
    optimizer1 = configure_optimizer(global_step, total_num_examples)
    with tf.variable_scope("scope", reuse=tf.AUTO_REUSE) as scope:

        for i in range(0,no_of_devices-1):
            with tf.device(devices[i]):
                print('num_classes: ' + str(num_classes))
                    # reuse = tf.AUTO_REUSE
                    # net = tf.get_variable("net", [384,1000])
                # pdb.set_trace()
                with tf.name_scope("scope_name"+str(i)) as name_scope:
                    net, logits, total_loss = alexnet_inference(builder, splits[i], label_splits[i], num_classes, scope=scope.name)
                    tf.add_to_collection("nets",net)
                    tf.add_to_collection("nets",logits)
                    tf.add_to_collection("nets",total_loss)
                    # pdb.set_trace()


                    grads = (optimizer1.compute_gradients(total_loss))
                    total_grads.append(grads)

    print('total_num_examples: ' + str(total_num_examples))


    # 5. On the parameter server node, apply gradients.

    with tf.device(devices[no_of_devices-1]):

        for grad in total_grads:
            apply_gradient_op = optimizer1.apply_gradients(grad, global_step=global_step)



        nets = []
        logits = []
        total_losses = []
        # pdb.set_trace()
        for i in range(0,no_of_devices-1):
            net = tf.get_collection("nets", scope="scope/scope_name"+str(i)+"/probs")
            logit = tf.get_collection("nets", scope="scope/scope_name"+str(i)+"/fc")
            total_loss = tf.get_collection("nets",scope="scope/scope_name"+str(i)+"/total_loss")

            nets.append(net)
            logits.append(logit)
            total_losses.append(total_loss)
        print(nets)
        print(logits)
        print(total_loss)
        net_avg = tf.reduce_mean(nets,0)[0]
        logit_avg = tf.reduce_mean(logits,0)[0]
        total_loss_avg = tf.reduce_mean(total_losses,0)[0]
        

        with tf.control_dependencies([apply_gradient_op]):
            train_op =  tf.no_op(name='train')

        
    if not is_train:
        return alexnet_eval(net, labels)

    # 6. return required values.
    return net_avg, logit_avg, total_loss_avg, train_op, global_step
    # return nets, logits, total_loss, train_op, global_step


