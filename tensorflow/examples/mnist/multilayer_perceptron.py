from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess=tf.InteractiveSession()

#训练阶段的dropout：训练阶段保持激活的概率，可以改变keep_prop_train的值看看随机失活对结果的影响
keep_prop_train=0.75 

in_units=784
h1_units=300
W1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros(h1_units))
W2=tf.Variable(tf.zeros([h1_units,10]))
b2=tf.Variable(tf.zeros(10))

x=tf.placeholder(tf.float32,[None,784])
keep_prob=tf.placeholder(tf.float32)

hidden1=tf.nn.relu(tf.matmul(x,W1)+b1)
hidden1_drop=tf.nn.dropout(hidden1,keep_prob)

y_pred=tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)

y=tf.placeholder(tf.float32,[None,10])

cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_pred),reduction_indices=[1]))
optimizer=tf.train.AdagradOptimizer(0.3).minimize(cost)

tf.global_variables_initializer().run()
for i in range(6000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    optimizer.run({x:batch_xs,y:batch_ys,keep_prob:keep_prop_train})

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print('accuracy on test:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0}))
