# tensorflo-gpu 1.3.0,softmax classification
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  #mnist.train，55000；mnist.validation，5000；mnist.test,10000

learning_rate=0.01
epoch=10
batch_size=100

##graph
x=tf.placeholder("float",[None,784],name='x_input')
W=tf.Variable(tf.random_normal([784,10]),name='Weight')
b=tf.Variable(tf.zeros([1,10]),name='bias')
y_pred=tf.nn.softmax(tf.add(tf.matmul(x,W),b),name='y_pred')    #softmax

# entropy loss
y=tf.placeholder("float",[None,10],name='y_true')
cost=-tf.reduce_sum(y * tf.log(y_pred))
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# compute accuracy
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

##session
init=tf.global_variables_initializer()
with tf.Session() as sess:
    writer=tf.summary.FileWriter('./graph/mnist_simple',sess.graph)  #tensorboard
    sess.run(init)  #初始化所有变量

    total_batch=int(mnist.train.images.shape[0]/batch_size) #每个epoch的batch数目
    for i in range(epoch*total_batch):  #训练
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})

    writer.close()

    print('accuracy on train:', sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels}))
    print('accuracy on validation:', sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels}))
    print('accuracy on test:',sess.run(accuracy,feed_dict={x: mnist.test.images, y: mnist.test.labels}))