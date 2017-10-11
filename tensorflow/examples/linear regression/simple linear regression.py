#y=kx+b，单特征
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#prepare data
n_samples=100
xs=np.linspace(-3,3,n_samples)
ys=np.sin(xs)+np.random.uniform(-0.5,0.5,n_samples)
plt.scatter(xs,ys)

#graph
X=tf.placeholder(tf.float32,name='X')
Y=tf.placeholder(tf.float32,name='Y')

W=tf.Variable(tf.random_normal([1]),name='W')
b=tf.Variable(tf.random_normal([1]),name='b')

Y_pred=tf.add(tf.multiply(X,W),b)

loss=tf.square(Y-Y_pred,name='loss')

learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#session
epoch=10
with tf.Session() as sess:
    # 写入日志,tensorboard
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    #初始化所有变量
    sess.run(tf.global_variables_initializer())

    for i in range(epoch):
        total_l = 0
        for x,y in zip(xs,ys):
            _,l=sess.run([optimizer,loss],feed_dict={X:x,Y:y})  #通过feed_dict喂数据
            total_l +=l

        #每2轮输出一次结果
        if i%2==0:
            print('epoch {0}: loss={1}'.format(i,total_l/n_samples))

    #关闭日志
    writer.close()

    #取出变量的值
    W,b=sess.run([W,b])

print('W:',W[0])
print('b:',b[0])

plt.plot(xs,ys,'bo',label='Real data')
plt.plot(xs,xs*W+b,'r',label='linear regression')
plt.legend()
