#! /usr/bin/env python
from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
from datetime import datetime
from timeit import default_timer
from operator import attrgetter

import numpy as np

import tensorflow as tf

from .. import datasets
from ..nets import alexnetmodes
from ..utils import tfhelper
from ..utils import misc


net_configs = {
    'single': (attrgetter('original'),
             ['/job:worker/task:0'],
             'grpc://localhost:2222', 1, 96),
    'cluster': (attrgetter('distribute'),
                  ['/job:worker/task:0', '/job:worker/task:1',
                   '/job:ps/task:0'],
                  'grpc://localhost:2222',
                  2, 48),
    'cluster2': (attrgetter('distribute'),
                  ['/job:worker/task:0', '/job:worker/task:1',
                   '/job:worker/task:2',
                   '/job:ps/task:0'],
                  'grpc://localhost:2222',
                  3, 32)
}

benchmarks = {
    'fake_data': datasets.fake_data,
    'flowers': datasets.flowers_data
}

def evaluate(net_configname, batch_size, devices=None, target=None,
             tb_dir=None, train_dir=None, benchmark_name=None):
    """Evaluation"""
    with tf.Graph().as_default():
        if tb_dir is None:
            tb_dir = '/tmp/workspace/tflogs'
        if train_dir is None:
            train_dir = './model'
        if benchmark_name is None:
            benchmark_name = 'fake_data'

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        input_data = benchmarks[benchmark_name]

        config = net_configs[net_configname]
        if devices is None:
            devices = config[1]
        if target is None:
            target = config[2]
        batch_size = config[3] * batch_size

        images, labels, num_classes = input_data(batch_size, None, is_train=False)

        correct_op = alexnetmodes.distribute(images, labels, num_classes, None, devices, is_train=False)

        saver = tf.train.Saver(tf.trainable_variables())

        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.Session(target, config=config) as sess:
            # create the summary write to write down the graph, but without any summaries
            tf.summary.FileWriter(tb_dir, sess.graph)

            sess.run(tfhelper.initialize_op())
            ckpt = tf.train.get_checkpoint_state(train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                print('Restore from {}'.format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return
            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            total_sample_count = 0
            true_count = 0  # Counts the number of correct predictions.
            try:
                while not coord.should_stop():
                    predictions = sess.run([correct_op])
                    total_sample_count += batch_size
                    true_count += np.sum(predictions)
                    precision = true_count / total_sample_count
                    print('{}: precision @ {} examples = {:.3f}'.format(datetime.now(),
                                                                        total_sample_count,
                                                                        precision))
            except tf.errors.OutOfRangeError:
                pass

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


def train(net_configname, batch_size, devices=None, target=None,
          batch_num=None, tb_dir=None, train_dir=None, benchmark_name=None):
    with tf.Graph().as_default():
        if tb_dir is None:
            tb_dir = '/tmp/workspace/tflogs'
        if train_dir is None:
            train_dir = './model'
        if benchmark_name is None:
            benchmark_name = 'fake_data'

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        input_data = benchmarks[benchmark_name]
        print("Input Data: ", input_data)

        config = net_configs[net_configname]

        if devices is None:
            devices = config[1]
        if target is None:
            target = config[2]
        batch_size = config[3] * batch_size
        if batch_num is None:
            batch_num = config[4]

        with tf.device(devices[-1]):
            images, labels, num_classes = input_data(batch_size, batch_num)

        print('Input batch shape: images: {} labels: {}'.format(images.get_shape(),
                                                                labels.get_shape()))

        if net_configname == "single":
            (net, logprob, total_loss,train_op, global_step) = alexnetmodes.original(images, labels, num_classes,batch_num * batch_size, devices)
        else:
            (net, logprob, total_loss,train_op, global_step) = alexnetmodes.distribute(images, labels, num_classes,batch_num * batch_size, devices)

        tfhelper.scalar_summary('total_loss', total_loss)
        summary_op = tfhelper.merge_all_summaries()

        # model saver
        saver = tf.train.Saver(tf.trainable_variables())

        # print some information
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            print(qr.name)

        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.Session(target, config=config) as sess:
            sess.run(tfhelper.initialize_op())
            coord = tf.train.Coordinator()
            queue_threads = tf.train.start_queue_runners(sess, coord)

            print('{} threads started for queue'.format(len(queue_threads)))

            summary_writer = tf.summary.FileWriter(tb_dir, sess.graph)

            speeds = []
            for step in range(batch_num):
                if coord.should_stop():
                    break

                # disable run time tracing, which is super slow
                if step % 4 == 1000:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    start_time = default_timer()
                    _, loss_value, summary = sess.run([train_op, total_loss, summary_op],
                                                      options=run_options,
                                                      run_metadata=run_metadata)
                    duration = default_timer() - start_time
                    summary_writer.add_run_metadata(run_metadata, 'step{}'.format(step), step)
                    summary_writer.add_summary(summary, step)
                else:
                    start_time = default_timer()
                    _, loss_value, summary = sess.run([train_op, total_loss, summary_op])
                    duration = default_timer() - start_time
                    summary_writer.add_summary(summary, step)

                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                speeds.append(examples_per_sec)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))
                sys.stdout.flush()

                # Save the model checkpoint periodically.
                if step % 100 == 0 or (step + 1) == batch_num:
                #if (step+1) % 2 == 0 or step % 100 == 0 or (step + 1) == batch_num:
                    checkpoint_path = os.path.join(train_dir, 'model-{}.ckpt'.format(step))
                    saver.save(sess, checkpoint_path, write_meta_graph=True)

            # When done, ask the threads to stop.
            coord.request_stop()
            # And wait for them to actually do it.
            coord.join(queue_threads)

            print('Average %.1f examples/sec' % np.average(speeds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train_hybrid.sh')
    parser.add_argument("--mode", help='the name of the parallel method',
                        choices=['single', 'cluster', 'cluster2'],
                        default='single')
    parser.add_argument("--work_dir",
                        help='directory for saving files, defaults to /tmp/workspace/tflogs',
                        default='/tmp/workspace/tflogs')
    parser.add_argument("--log_dir",
                        help="""directory for tensorboard logs, defaults to WORK_DIR/tf in train
                                mode and WORK_DIR/tfeval in evaluation mode""")
    parser.add_argument("--model_dir",
                        help='directory for model checkpoints, defaults to WORK_DIR/model')
    parser.add_argument("--dataset", help='dataset to use', choices=['fake_data', 'flowers'],
                        default='fake_data')
    parser.add_argument("--batch_num", help='total batch number', type=int)
    parser.add_argument("--batch_size", help='batch size', type=int,
                        default=128)
    parser.add_argument('--redirect_outerr',
                        help="""whether to redirect stdout to WORK_DIR/out.log
                                and stderr to WORK_DIR/err.log""",
                        action="store_true")
    parser.add_argument('--eval', help='evaluation or train',
                        action="store_true")
    args = parser.parse_args()
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    if args.log_dir is None:
        if args.eval:
            args.log_dir = os.path.join(args.work_dir, 'tfeval')
        else:
            args.log_dir = os.path.join(args.work_dir, 'tf')
    if args.model_dir is None:
        args.model_dir = os.path.join(args.work_dir, 'model')

    out_file = os.path.join(args.work_dir, 'out.log')
    err_file = os.path.join(args.work_dir, 'err.log')
    if args.redirect_outerr:
        with open(out_file, 'w') as f, misc.stdout_redirected(f):
            with open(err_file, 'w') as f, misc.stdout_redirected(f, stdout=sys.stderr):
                if args.eval:
                    evaluate(args.mode, tb_dir=args.log_dir, train_dir=args.model_dir,
                             benchmark_name=args.dataset, batch_size=args.batch_size)
                else:
                    train(args.mode, tb_dir=args.log_dir, train_dir=args.model_dir,
                          benchmark_name=args.dataset, batch_num=args.batch_num,
                          batch_size=args.batch_size)
    else:
        if args.eval:
            evaluate(args.mode, tb_dir=args.log_dir, train_dir=args.model_dir,
                     benchmark_name=args.dataset, batch_size=args.batch_size)
        else:
            train(args.mode, tb_dir=args.log_dir, train_dir=args.model_dir,
                  benchmark_name=args.dataset, batch_num=args.batch_num, batch_size=args.batch_size)
