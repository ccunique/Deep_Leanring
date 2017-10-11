#tensorflow-gpu 1.3.0
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')  #padding='SAME'，进行padding使卷积前后图片尺寸不变（宽、高）

def max_pool_2X2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

learing_rate=1e-4
epoch=20    #在训练集上运行20次
batch_size=100


##graph
x=tf.placeholder(tf.float32, [None, 784]) #数据集预先处理好了,图片和标签都是float的
x_image=tf.reshape(x, [-1, 28, 28, 1])

#conv1,输出32个特征图，每个feature map大小保持为28X28,池化后输出为14X14
W_conv1=weight_variable([5,5,1,32])     #卷积核大小，5X5X1,32个卷积核
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1=max_pool_2X2(h_conv1)

#conv2,输出64个特征图，每个feature map大小与con1输出相同：14X14，池化后输出为7X7
W_conv2=weight_variable([5,5,32,64])    #64个卷积核
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2X2(h_conv2)

#fc1,隐节点为1024 (上下结构全连接...)
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#dropout
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#fc2: Softmax
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#cost, cross_entropy
y=tf.placeholder(tf.float32,shape=[None,10])
#注意：cost算的一个batch中所有图片的交叉熵之和，*表示的点对点对应相乘，不是向量/矩阵乘法;也可以用每个batch的交叉熵均值
# cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_conv),reduction_indices=[1]))
cost=-tf.reduce_sum(y*tf.log(y_conv))

#train,optimizer
optimizer=tf.train.AdamOptimizer(learing_rate).minimize(cost)

#准确率
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

##session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch=int(mnist.train.images.shape[0]/batch_size)     #每个epoch有total_batch个批次
    for i in range(epoch*total_batch):      #epoch轮总共的batch数目
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        if i%200==0:    #各层W,b均已初始化，i=0的正确率即W,b为初始值时的正确率
            # 评测时keep_prob设为1，保留全部数据来追求最好的性能
            train_accuracy=accuracy.eval(feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0})
            print('batch: %d,training accuracy: %g'%(i,train_accuracy))
        # 训练时keep_prob设为0.5，随机丢弃一部分节点数据减轻过拟合。每过一个batch更新一次W,b
        optimizer.run(feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.5})

    print('test_accuracy:%g' % accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0}))

