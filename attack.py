# coding: utf-8

from __future__ import print_function
from six.moves import xrange
import os, math
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
from tensorflow.examples.tutorials.mnist import input_data
from visualize import *
import svhn_data, cifar10_data
from dataset import make_dataset
import argparse
from keras_contrib.layers import InstanceNormalization

def lrelu(x, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(leak*x, x)

def rescale(mat):
    return np.transpose(np.cast[np.float32]((mat)/255.0),(3,0,1,2))

parser = argparse.ArgumentParser('')
parser.add_argument('--data1', type=str, default='mnist')
parser.add_argument('--data2', type=str, default='svhn')
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--logdir', type=str, default='./log/')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
batch_size = 100
device = '/gpu:0'

log_dir = args.logdir
ckpt_dir = args.logdir
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

mnist = input_data.read_data_sets('./data/mnist', validation_size = 0)
trainx = mnist.train._images
mnist_trainY = mnist.train._labels.astype(np.int32)
testx = mnist.test._images
mnist_testY = mnist.test._labels.astype(np.int32)
mnist_trainX = np.reshape(trainx, (-1, 28, 28, 1))
mnist_testX = np.reshape(testx, (-1, 28, 28, 1))
# npad = ((0, 0), (2, 2), (2, 2), (0, 0))
# mnist_trainX = np.pad(mnist_trainX, pad_width=npad, mode='constant', constant_values=0)
# mnist_testX = np.pad(mnist_testX, pad_width=npad, mode='constant', constant_values=0)

trainx, svhn_trainY = svhn_data.load('./data/svhn','train')
testx, svhn_testY = svhn_data.load('./data/svhn','test')
svhn_trainX = rescale(trainx)
svhn_testX = rescale(testx)
sigma1 = 100
sigma2 = 100
sigma3 = 100

def classifier(images, inp, is_training=True, num_classes=10, reuse=False, scope=None):

    with tf.variable_scope('clf', reuse=reuse):

        #dp0 = ly.dropout(images, 0.7, is_training=is_training)
        conv1 = ly.conv2d(inp(images), 64, [3, 3], activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':is_training}, scope='conv1')
        conv2 = ly.conv2d(conv1, 64, [3, 3], activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':is_training}, scope='conv2')
        conv3 = ly.conv2d(conv2, 64, [3, 3], activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':is_training}, scope='conv3')
        pool1 = ly.max_pool2d(conv3, [2, 2], 2, scope='pool1')
        dp1 = ly.dropout(pool1, 0.5, is_training=is_training)

        conv4 = ly.conv2d(dp1, 64, [3, 3], activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':is_training}, scope='conv4')
        conv5 = ly.conv2d(conv4, 64, [3, 3], activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':is_training}, scope='conv5')
        conv6 = ly.conv2d(conv5, 64, [3, 3], activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':is_training}, scope='conv6')
        pool2 = ly.max_pool2d(conv6, [2, 2], 2, scope='pool2')
        dp2 = ly.dropout(pool2, 0.5, is_training=is_training)

        conv7 = ly.conv2d(dp2, 64, [3, 3], activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':is_training}, scope='conv7')
        conv8 = ly.conv2d(conv7, 64, [3, 3], activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':is_training}, scope='conv8')
        conv9 = ly.conv2d(conv8, 64, [3, 3], activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':is_training}, scope='conv9')
        gap = tf.reduce_mean(conv9, [1,2])

        logits = ly.fully_connected(gap, num_classes, activation_fn=None, scope='fc', normalizer_fn=None)

    #with tf.variable_scope('clf', reuse=reuse):
    #with tf.variable_scope(scope, reuse=reuse2):
    return logits, dp2

def small_classifier(z, is_training=True, num_classes=10, reuse=False):
    with tf.variable_scope('small_clf', reuse=reuse):
        conv7 = ly.conv2d(z, 64, [3, 3], activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':is_training}, scope='conv1')
        conv8 = ly.conv2d(conv7, 64, [3, 3], activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':is_training}, scope='conv2')
        conv9 = ly.conv2d(conv8, 64, [3, 3], activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':is_training}, scope='conv3')
        gap = tf.reduce_mean(conv9, [1,2])
        logits = ly.fully_connected(gap, num_classes, activation_fn=None, scope='fc', normalizer_fn=None)
    return logits

def critic(z, scope='critic', reuse=False, is_training=True):
    with tf.variable_scope(scope, reuse=reuse):
        conv1 = ly.conv2d(z, 64, [3, 3], activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':is_training}, scope='conv1')
        conv2 = ly.conv2d(conv1, 64, [3, 3], activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':is_training}, scope='conv2')
        conv3 = ly.conv2d(conv2, 64, [3, 3], activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':is_training}, scope='conv3')
        gap = tf.reduce_mean(conv3, [1,2])
        logits = ly.fully_connected(gap, 1, activation_fn=None, scope='fc', normalizer_fn=None)
    return logits

def build_graph():
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3))
    feature1 = tf.placeholder(dtype=tf.float32, shape=(None, 8, 8, 64))
    y1 = tf.placeholder(dtype=tf.int32, shape=(None,))
    x2 = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1))
    feature2 = tf.placeholder(dtype=tf.float32, shape=(None, 8, 8, 64))

    epsilon = tf.get_variable('eps', shape=(100, 8, 8, 64), dtype=tf.float32)
    #y2 = tf.placeholder(dtype=tf.int32, shape=(None,))

    inp = InstanceNormalization(-1)
    y1_logits, _ = classifier(x1, inp=inp)

    y1_test_logits, f1 = classifier(x1, inp=inp, reuse=True, is_training=False)
    y1_test = tf.argmax(y1_test_logits, axis=1)
    y2_test_logits, f2 = classifier(tf.tile(tf.image.resize_images(x2, [32,32]), [1,1,1,3]), inp=inp, reuse=True, is_training=False)
    y2_test = tf.argmax(y2_test_logits, axis=1)
    loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y1, logits=y1_logits))
    theta_clf = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='clf')

    f1_logits_critic = critic(feature1, reuse=False)
    f2_logits_critic = critic(feature2, reuse=True)
    f1a_logits_critic = critic(feature1 + epsilon[:tf.shape(feature1)[0]], reuse=True)
    loss2_critic = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(f1_logits_critic), logits=f1_logits_critic) \
                 + tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(f2_logits_critic), logits=f2_logits_critic))

    f1_logits_clf = small_classifier(feature1, reuse=False)
    f2_logits_clf = small_classifier(feature2, reuse=True)
    f1a_logits_clf = small_classifier(feature1 + epsilon[:tf.shape(feature1)[0]], reuse=True)
    f2_softmax_clf = tf.nn.softmax(f2_logits_clf)
    f2_softmax_clf_mean = tf.reduce_mean(f2_softmax_clf, 0)
    loss2_clf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y1, logits=f1_logits_clf))\
              + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=f2_softmax_clf, logits=f2_logits_clf))\
              + 1e-2 * tf.reduce_sum(f2_softmax_clf_mean * tf.log(f2_softmax_clf_mean))
    f1_test = tf.argmax(small_classifier(feature1, reuse=True, is_training=False), axis=1)
    f2_test = tf.argmax(small_classifier(feature2, reuse=True, is_training=False), axis=1)
    theta_clf_1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='small_clf')
    theta_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

    loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(f1_logits_clf), logits=f1a_logits_clf)) \
          + 1.0 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(f1a_logits_critic), logits=f1a_logits_critic)) \
          + tf.maximum(1 - tf.reduce_sum(epsilon)/batch_size, 0)
    loss1_sum = tf.summary.scalar("loss1", loss1)
    loss2_sum = tf.summary.scalar("loss2", loss2_clf+loss2_critic)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="clf")):
        counter1 = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        # rate = tf.train.exponential_decay(0.15, step, 1, 0.9999)
        opt = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5, beta2=0.999).minimize(loss1, var_list=theta_clf, global_step=counter1)

    init = tf.initialize_variables(theta_critic + theta_clf_1)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="small_clf") + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="critic")):
        counter2 = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        opt_2 = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5, beta2=0.999).minimize(loss2_clf+loss2_critic, var_list=theta_critic + theta_clf_1, global_step=counter2)

    reset = tf.assign(epsilon, tf.zeros_like(epsilon))
    opt_3 = tf.train.RMSPropOptimizer(learning_rate=1e-2).minimize(loss3, var_list=[epsilon])

    return x1, f1, y1, x2, f2, opt, loss1, y1_test, y2_test, feature1, feature2, opt_2, loss2_critic, loss2_clf, f1_test, f2_test, reset, opt_3, loss3, epsilon, init

def main():
    max_iter_step = 60000
    mnist_trainset = make_dataset(mnist_trainX, mnist_trainY)
    svhn_trainset = make_dataset(svhn_trainX, svhn_trainY)


    with tf.device(device):
        x1, f1, y1, x2, f2, opt, loss1, y1_test, y2_test, feature1, feature2, opt_2, loss2_critic, loss2_clf, f1_test, f2_test, reset, opt_3, loss3, epsilon, init = build_graph()
    merged_all = tf.summary.merge_all()
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = False
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        F1 = np.zeros([svhn_trainY.shape[0], 8,8,64])
        Y1 = np.zeros(svhn_trainY.shape)
        F2 = np.zeros([mnist_trainY.shape[0], 8,8,64])
        Y2 = np.zeros(mnist_trainY.shape)
        print("Pretraining...")
        for i in range(20000):
            svhn_bx, svhn_by = svhn_trainset.next_batch(batch_size)
            mnist_bx, _ = mnist_trainset.next_batch(batch_size)
            [  _, loss1_] = sess.run([opt, loss1], feed_dict={x1:svhn_bx, y1:svhn_by, x2:mnist_bx})
        
            if i % 100 == 99:
                print("Pretraining ite %d, loss1: %f" % (i, loss1_))
        
            if i % 1000 == 999:
                acc1_ = []
                for j in range(26):
                    y1_test_ = sess.run([y1_test], feed_dict={x1: svhn_testX[j*1000:(j+1)*1000]})
                    acc1_.append(np.mean(np.equal(svhn_testY[j*1000:(j+1)*1000], y1_test_).astype(np.float32)))
                acc2_ = []
                for j in range(10):
                    y2_test_ = sess.run([y2_test], feed_dict={x2: mnist_testX[j*1000:(j+1)*1000]})
                    acc2_.append(np.mean(np.equal(mnist_testY[j*1000:(j+1)*1000], y2_test_).astype(np.float32)))
                print("--------->Pretraining ite %d, acc1: %f, acc2: %f" % (i, np.mean(acc1_),  np.mean(acc2_)))
        for i in range(int(math.ceil(svhn_trainset._num_examples/batch_size))):
            start = i * batch_size
            end = min((i+1)*batch_size, svhn_trainset._num_examples)
            F1[start:end] = sess.run(f1, feed_dict={x1:svhn_trainset._images[start:end]})
            Y1[start:end] = svhn_trainset._labels[start:end]
        for i in range(int(math.ceil(mnist_trainset._num_examples/batch_size))):
            start = i * batch_size
            end = min((i+1)*batch_size, mnist_trainset._num_examples)
            F2[start:end] = sess.run(f2, feed_dict={x2:mnist_trainset._images[start:end]})
            Y2[start:end] = mnist_trainset._labels[start:end]
        np.savez('features.npz', F1=F1, Y1=Y1, F2=F2, Y2=Y2)

#         npz = np.load('features.npz')
#         F1 = npz['F1']
#         F2 = npz['F2']
#         Y1 = npz['Y1']
#         Y2 = npz['Y2']
        print("Trainging...")
        F1_set = make_dataset(F1, Y1)
        F2_set = make_dataset(F2, Y2)
        for ite in range(100):
            sess.run(init)
            for i in range(2000):
                f1_, y1_ = F1_set.next_batch(batch_size)
                f2_, y2_ = F2_set.next_batch(batch_size)
                [_, loss2_clf_, loss2_critic_] = sess.run([opt_2, loss2_clf, loss2_critic], feed_dict={feature1:f1_, y1:y1_, feature2:f2_})
                if i % 100 == 99:
                    [f1_test_, f2_test_] = sess.run([f1_test, f2_test], feed_dict={feature1:f1_, feature2:f2_})
                    print("Training epoch %d, ite %d, loss2_clf: %f, loss2_critic: %f, acc1: %f, acc2: %f" % (ite, i, loss2_clf_, loss2_critic_,
                            np.mean(np.equal(y1_, f1_test_).astype(np.float32)),
                            np.mean(np.equal(y2_, f2_test_).astype(np.float32))))

            tmp = np.zeros(F1_set._images.shape)
            for i in range(int(math.ceil(F1_set._num_examples/batch_size))):
                print("Training adversarial samples at batch %d" % i)
                start = i * batch_size
                end = min((i+1)*batch_size, svhn_trainset._num_examples)
                f1_ = F1_set._images[start:end]
                sess.run(reset)
                for j in range(100):
                    [_, loss3_] = sess.run([opt_3, loss3], feed_dict={feature1: f1_})
                    #print(i, j, loss3_)
                tmp[start:end] = epsilon.eval()[:end-start] + f1_

            F1_set._images = tmp

main()
