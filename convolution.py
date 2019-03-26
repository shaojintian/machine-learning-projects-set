import tensorflow as tf


#1
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)


#placeholder

x=tf.placeholder(tf.float32,[None,784])

y_=tf.placeholder(tf.float32,[None,10])


x_image=tf.reshape(x,[-1,28,28,1])


#first convolution layer

def weight_variable(shape):
	initial=tf.truncated_normal(shape,stddev=0.01)

	return tf.Variable(initial)


def bias_variable(shape):
	initial=tf.constant(0.1,shape=shape)

	return tf.Variable(initial)


def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


W_conv1=weight_variable([5,5,1,32])

b_conv1=bias_variable([32])
#activate function
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)

h_pool1=max_pool_2x2(h_conv1)

#second convolution layer 


W_conv2=weight_variable([5,5,32,64])

b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)

h_pool2=max_pool_2x2(h_conv2)


#full connected layer 1


W_fc1=weight_variable([7*7*64,1024])

b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])

h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)


h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)


#full connected layer 2

W_fc2=weight_variable([1024,10])

b_fc2=bias_variable([10])

y_conv=tf.matmul(h_fc1_drop,W_fc2)+b_fc2

#loss function

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))


#optimizer

optimizer=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)


#accuracy
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#求平均值




#start session


session=tf.InteractiveSession()

session.run(tf.global_variables_initializer())

#train 20,000 times ,50 images in one  batch
 

for i in range(1000):
	batch=mnist.train.next_batch(50)
	if i % 100 == 0:
		train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
		print("step %d is %g accuracy"%(i,train_accuracy))
	optimizer.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

print("test accuracy is %d" %(accuracy.eval(feed_dict={x:mnist.train.images,y_:mnist.train.labels,keep_prob:1.0})))


































































































































