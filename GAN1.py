import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def save_model(saver,sess,logdir,step):
    model_name='model'
    checkpoint_path = os.path.join(logdir,model_name)
    saver.save(sess,checkpoint_path,global_setp=step)
    print('checkpoint has been created')

def xavier_init(size):
    in_dim=size[0]
    xavier_stddev=1./tf.sqrt(in_dim/2.)
    return tf.random.normal(shape=size,stddev=xavier_stddev)

X_true=tf.compat.v1.placeholder(tf.float32,shape=[None,784])#the true samples
D_W1=tf.Variable(xavier_init([784,128]))
D_b1=tf.Variable(tf.zeros(shape=[128]))

D_W2=tf.compat.v1.Variable(xavier_init([128,1]))
D_b2=tf.compat.v1.Variable(tf.zeros(shape=[1]))

theta_D=[D_W1, D_W2, D_b1, D_b2] #需要更新的参数

Z=tf.compat.v1.placeholder(tf.float32,shape=[None,100])#the input of G network

G_W1=tf.Variable(xavier_init([100,128]))
G_b1=tf.Variable(tf.zeros([128]))

G_W2=tf.Variable(xavier_init([128,784]))
G_b2=tf.Variable(tf.zeros([784]))#why 784? cause the mnist image is 28*28

theta_G=[G_W1, G_W2, G_b1, G_b2] #parameters need to update

def generate_sampel_Z(m,n):
    return np.random.uniform(-1.,1.,size=[m,n])#random generate numbers between (-1 1]

def generator(z):
    G_h1=tf.nn.relu(tf.matmul(z,G_W1)+G_b1)
    G_h2=tf.matmul(G_h1,G_W2)+G_b2
    G_prob=tf.nn.sigmoid(G_h2)

    return G_prob

def discriminator(x):
    D_h1=tf.nn.relu(tf.matmul(x,D_W1)+D_b1)
    D_h2=tf.matmul(D_h1,D_W2)+D_b2
    D_prob=tf.nn.sigmoid(D_h2)
    return D_h2,D_prob

def plot(samples):
    fig=plt.figure(figsize=(4,4))
    gs = gridspec.GridSpec(4,4)
    gs.update(wspace=0.05,hspace=0.05)

    for i,sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    plt.show(block=False)
    return fig

G_sample=generator(Z)
D_logit_real, D_real = discriminator(X_true) #判别对于真实输入X_true，判别器的判别结果
D_logit_fake, D_fake = discriminator(generator(Z)) #判别对于G生成的数据，判别器的判别结果

D_loss=-tf.reduce_mean(tf.log(1e-10+D_real)+tf.log(1.-D_fake))
#Also, as per the paper’s suggestion, it’s better to maximize tf.reduce_mean(tf.log(D_fake)) instead of minimizing tf.reduce(1 - tf.log(D_fake)) in the algorithm above.
#G_loss=tf.reduce_mean(tf.log(1.-D_fake))
G_loss=-tf.reduce_mean(tf.log(1e-10+D_fake))

D_trainer=tf.train.AdamOptimizer().minimize(D_loss,var_list=theta_D)
G_trainer=tf.train.AdamOptimizer().minimize(G_loss,var_list=theta_G)

def sample_Z(m,n):
    #归一化
    return np.random.uniform(-1.,1.,size=[m,n])

mb_size=128
Z_dim=100
mnist=input_data.read_data_sets('/MNIST_data',one_hot=True)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    i=0
    for it in range(1000000):
        if it%1000==0:
            samples=sess.run(G_sample,feed_dict={Z:sample_Z(16,Z_dim)})
            fig=plot(samples)
            fname='{}.png'.format(str(i).zfill(5))
            plt.savefig(fname,bbox_inches='tight')
            print('saved image'+fname)
            i+=1
            plt.close()

        X_mb,_=mnist.train.next_batch(mb_size)
        _,D_loss_curr=sess.run([D_trainer,D_loss],feed_dict={X_true:X_mb,Z:sample_Z(mb_size,Z_dim)})
        _,G_loss_curr=sess.run([G_trainer,G_loss],feed_dict={Z:sample_Z(mb_size,Z_dim)})

        if it%1000==0:
            print('Iterations:{},D loss:{:.4},G loss:{:.4}'.format(it,D_loss_curr,G_loss_curr))

