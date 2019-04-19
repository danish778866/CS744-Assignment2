import tensorflow as tf
import time
import datetime
from timeit import default_timer
from tensorflow.examples.tutorials.mnist import input_data

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

clusterSpec_single = tf.train.ClusterSpec({
    "worker" : [
        "localhost:2222"
    ]
})

clusterSpec_cluster = tf.train.ClusterSpec({
    "ps" : [
        "node0:2222"
    ],
    "worker" : [
        "node0:2223",
        "node1:2222"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps" : [
        "node0:2222"
    ],
    "worker" : [
        "node0:2223",
        "node1:2222",
        "node2:2222",
    ]
})

clusterSpec = {
    "single": clusterSpec_single,
    "cluster": clusterSpec_cluster,
    "cluster2": clusterSpec_cluster2
}

clusterinfo = clusterSpec[FLAGS.deploy_mode]
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    is_chief = (FLAGS.task_index == 0)
    worker_device = "/job:%s/task:%d" % (FLAGS.job_name, FLAGS.task_index)
    with tf.device(tf.train.replica_device_setter(worker_device=worker_device, cluster=clusterinfo)):
        x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
        y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        # Construct model
        pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        learning_rate = 0.01
        training_epochs = 25
        batch_size = 256
        display_step = 1
        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        opt = optimizer.minimize(cost, global_step=global_step)

    stop_hook = tf.train.StopAtStepHook(last_step=6000)
    hooks = [stop_hook]
    # Monitored Training Session
    with tf.train.MonitoredTrainingSession(master = server.target, 
          is_chief=is_chief,
          max_wait_secs = 15,
          stop_grace_period_secs=5,
          hooks=hooks
          ) as mon_sess:
        print("Start Time:" + str(datetime.datetime.now()))
        start_time = default_timer() 
        while not mon_sess.should_stop():
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c, gs = mon_sess.run([opt, cost, global_step], feed_dict={x: batch_xs, y: batch_ys})
            print('Cost:', c, 'Step:', gs, 'Worker: ',FLAGS.task_index)
            duration = default_timer() - start_time
            print('Time Elapsed: ' + str(duration))
            if not mon_sess.should_stop():
        	print("Accuracy after ", gs, " steps:", accuracy.eval(session=mon_sess, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    duration = default_timer() - start_time
    print("Duration: " + str(duration))
    print("End Time:" + str(datetime.datetime.now()))


     
