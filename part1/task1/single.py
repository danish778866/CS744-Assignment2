import tensorflow as tf
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

    learning_rate = 0.01
    training_epochs = 25
    batch_size = 100
    display_step = 1
    # tf Graph Input
    with tf.name_scope("Input"):
        with tf.name_scope("x"):
            x = tf.placeholder(tf.float32, [None, 784], name="x") # mnist data image of shape 28*28=784
        with tf.name_scope("y"):
            y = tf.placeholder(tf.float32, [None, 10], name="y") # 0-9 digits recognition => 10 classes
    
    # Set model weights
    with tf.name_scope("Weights"):
        with tf.name_scope("W"):
            W = tf.Variable(tf.zeros([784, 10]))
        with tf.name_scope("b"):
            b = tf.Variable(tf.zeros([10]))
    
    # Construct model
    with tf.name_scope("Model"):
        pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
    
    # Minimize error using cross entropy
    with tf.name_scope("Loss"):
        cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    # Start training
    with tf.Session() as sess:
    
        # Run the initializer
        sess.run(init)
        tf.summary.scalar("loss", cost)
        #tf.summary.scalar("x", x)
        tf.summary.histogram("pred", pred)
        tf.summary.histogram("W", W)
        tf.summary.histogram("b", b)
        tf.summary.tensor_summary("Input_y", y)
        
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("/users/danish/summaries/train", sess.graph)
        
    
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                summary_writer.add_summary(summary, i)
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    
        print("Optimization Finished!")
    
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
