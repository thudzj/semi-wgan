
# coding: utf-8

# In[1]:

from __future__ import print_function
from six.moves import xrange
#import tensorflow.contrib.slim as slim
import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import mixture
from visualize import *
import svhn_data, cifar10_data
from dataset import make_dataset
import argparse


# In[2]:

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(leak*x, x)

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def rescale(mat):
    return np.transpose(np.cast[np.float32]((-127.5 + mat)/127.5),(3,0,1,2))


# In[3]:

parser = argparse.ArgumentParser('')
parser.add_argument('--data', type=str, default='mnist')
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--logdir', type=str, default='')
parser.add_argument('--d', type=int, default=5)
parser.add_argument('--g', type=int, default=1)
parser.add_argument('--count', type=int, default=10)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--wy', type=float, default=1)
parser.add_argument('--wz', type=float, default=0.5)
parser.add_argument('--sigma', type=float, default=10)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
batch_size = 100
z_dim = 128
device = '/gpu:0'
s = 32
Citers = args.d

image_dir = args.data
log_dir = './log_cwgan' + '/' + image_dir + '/' + args.logdir
ckpt_dir = './ckpt_cwgan' + '/' + image_dir + '/' + args.logdir
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if image_dir == 'mnist':
    channel = 1
    mnist = input_data.read_data_sets('./data/mnist', validation_size = 0)
    trainx = mnist.train._images
    trainy = mnist.train._labels.astype(np.int32)
    testx = mnist.test._images
    testy = mnist.test._labels.astype(np.int32)
    trainx = 2 * trainx - 1
    testx = 2 * testx - 1
    trainx = np.reshape(trainx, (-1, 28, 28, channel))
    testx = np.reshape(testx, (-1, 28, 28, channel))
    npad = ((0, 0), (2, 2), (2, 2), (0, 0))
    trainx = np.pad(trainx, pad_width=npad, mode='constant', constant_values=-1)
    testx = np.pad(testx, pad_width=npad, mode='constant', constant_values=-1)
elif image_dir == 'svhn':
    channel = 3
    trainx, trainy = svhn_data.load('./data/svhn','train')
    testx, testy = svhn_data.load('./data/svhn','test')
    trainx = rescale(trainx)
    testx = rescale(testx)
else:
    channel = 3
    trainx, trainy = cifar10_data.load("./data/cifar10", subset='train')
    testx, testy = cifar10_data.load("./data/cifar10", subset='test')
    trainx = np.transpose(trainx, [0, 2, 3, 1])
    testx = np.transpose(testx, [0, 2, 3, 1])

print(trainx.shape)
print(np.max(trainx), np.min(trainx))
trainy_one_hot = trainy.copy()
trainy = dense_to_one_hot(trainy)
testy = dense_to_one_hot(testy)
#assert(np.max(trainx) == 1.0 and np.min(trainx) == -1.0)
# select labeled data
rng = np.random.RandomState(1)
inds = rng.permutation(trainx.shape[0])
trainx = trainx[inds]
trainy = trainy[inds]
txs = []
tys = []
for j in range(10):
    txs.append(trainx[trainy_one_hot==j][:args.count])
    tys.append(trainy[trainy_one_hot==j][:args.count])
txs = np.concatenate(txs, axis=0)
tys = np.concatenate(tys, axis=0)

def generator_conv(y, z, reuse=False):
    with tf.variable_scope('generator') as scope:
        if reuse:
            scope.reuse_variables()
        train = ly.fully_connected(
            tf.concat([z,y], 1), 4 * 4 * 512, activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm)
        train = tf.reshape(train, (-1, 4, 4, 512))
        train = ly.conv2d_transpose(train, 256, 5, stride=2,
                                    activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME')
        train = ly.conv2d_transpose(train, 128, 5, stride=2,
                                    activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME')
        train = ly.conv2d_transpose(train, channel, 5, stride=2,
                                    activation_fn=tf.nn.tanh, padding='SAME')
    return train

def critic_conv(x, reuse=False):
    with tf.variable_scope('critic') as scope:
        if reuse:
            scope.reuse_variables()
        img = ly.conv2d(x, num_outputs=128, kernel_size=5,
                        stride=2, activation_fn=lrelu)
        img = ly.conv2d(img, num_outputs=256, kernel_size=5,
                        stride=2, activation_fn=lrelu)
        img = ly.conv2d(img, num_outputs=512, kernel_size=5,
                        stride=2, activation_fn=lrelu)
        logit = ly.fully_connected(tf.reshape(img, [batch_size, -1]), 1, activation_fn=None)
    return logit

def critic_mlp(z, reuse=False):
    with tf.variable_scope('critic_mlp') as scope:
        if reuse:
            scope.reuse_variables()
        img = ly.fully_connected(z, 100, activation_fn=lrelu)
        img = ly.fully_connected(img, 100,activation_fn=lrelu)
        img = ly.fully_connected(img, 100,activation_fn=lrelu)
        logit = ly.fully_connected(img, 1, activation_fn=None)
    return logit

def encoder_z(x, reuse=False, flag = False):
    with tf.variable_scope('encoder_z') as scope:
        if reuse:
            scope.reuse_variables()
        if flag:
            img = ly.conv2d(tf.nn.dropout(x, 0.5), num_outputs=64, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        else:
            img = ly.conv2d(x, num_outputs=64, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        img = ly.conv2d(img, num_outputs=128, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        img = ly.conv2d(img, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        img = ly.conv2d(img, num_outputs=512, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        logit = ly.fully_connected(tf.reshape(img, [batch_size, -1]), z_dim, activation_fn=None)
    return logit

def encoder_y(x, reuse=False, flag = False):
    with tf.variable_scope('encoder_y') as scope:
        if reuse:
            scope.reuse_variables()
        if flag:
            img = ly.conv2d(tf.nn.dropout(x, 0.5), num_outputs=64, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        else:
            img = ly.conv2d(x, num_outputs=64, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        img = ly.conv2d(img, num_outputs=128, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        img = ly.conv2d(img, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        img = ly.conv2d(img, num_outputs=512, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        logit = ly.fully_connected(tf.reshape(img, [batch_size, -1]), 10, activation_fn=None)
    return logit

def build_graph():
    generator = generator_conv
    critic = critic_conv
    critic_z = critic_mlp

    real_data = tf.placeholder(dtype=tf.float32, shape=(batch_size, 32, 32, channel))
    real_label = tf.placeholder(dtype=tf.float32, shape=(batch_size, 10))
    unlabeled_data = tf.placeholder(dtype=tf.float32, shape=(batch_size, 32, 32, channel))
    z = tf.placeholder(dtype=tf.float32, shape=(batch_size, z_dim))
    y = tf.placeholder(dtype=tf.float32, shape=(batch_size, 10))

    true_z = encoder_z(real_data)
    true_y = encoder_y(real_data)

    unlabeled_z = encoder_z(unlabeled_data, reuse=True)
    unlabeled_y = encoder_y(unlabeled_data, reuse=True)

    fake_x = generator(y, z)
    fake_z = encoder_z(fake_x, reuse=True)
    fake_y = encoder_y(fake_x, reuse=True)

    true_logit = critic(unlabeled_data) + critic_z(z)
    fake_logit_1 = critic(fake_x, reuse=True)
    fake_logit_2 = critic_z(unlabeled_z, reuse=True)

    c_loss = tf.reduce_mean(fake_logit_1 + fake_logit_2 - true_logit)

    alpha = tf.random_uniform(
        shape=[batch_size, 1, 1, 1],
        minval=0.,
        maxval=1.
    )
    interpolates = unlabeled_data + (alpha * (fake_x - unlabeled_data))
    gradients = tf.gradients(critic(interpolates, reuse=True), [interpolates])
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients[0]), reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    c_loss += (args.sigma)*gradient_penalty

    alpha1 = tf.random_uniform(
        shape=[batch_size, 1],
        minval=0.,
        maxval=1.
    )
    interpolates1 = z + (alpha1 * (unlabeled_z - z))
    gradients1 = tf.gradients(critic_z(interpolates1, reuse=True), [interpolates1])
    slopes1 = tf.sqrt(tf.reduce_sum(tf.square(gradients1[0]), reduction_indices=[1]))
    gradient_penalty1 = tf.reduce_mean((slopes1-1.)**2)
    c_loss += (args.sigma)*gradient_penalty1

    recon_y = tf.losses.softmax_cross_entropy(y, fake_y)
    recon_z = tf.losses.mean_squared_error(z, fake_z)
    labeled_loss = tf.losses.softmax_cross_entropy(real_label, true_y)
    unlabeled_loss = tf.losses.softmax_cross_entropy(tf.nn.softmax(unlabeled_y), unlabeled_y) - 0.1 * tf.reduce_sum(tf.log(tf.reduce_mean(tf.nn.softmax(unlabeled_y), 0))) / 300.0

    e_loss_y = recon_y * args.wy + unlabeled_loss * args.beta + labeled_loss
    e_loss_z = tf.reduce_mean(-fake_logit_2) + recon_z * args.wz
    g_loss = tf.reduce_mean(-fake_logit_1) + args.wy * recon_y + args.wz * recon_z

    e_loss_z_sum = tf.summary.scalar("e_loss_z", e_loss_z)
    e_loss_y_sum = tf.summary.scalar("e_loss_y", e_loss_y)
    c_loss_z_sum = tf.summary.scalar("c_loss", c_loss)
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    img_sum = tf.summary.image("img", fake_x, max_outputs=10)

    theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_mlp')
    theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    theta_e_z = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder_z')
    theta_e_y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder_y')

    counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_c = ly.optimize_loss(loss=c_loss, learning_rate= None,
                    optimizer=tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9),
                    variables=theta_c, global_step=counter_c,
                    summaries = 'gradient_norm')

    counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_g = ly.optimize_loss(loss=g_loss, learning_rate=None,
                    optimizer=tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9),
                    variables=theta_g, global_step=counter_g,
                    summaries = 'gradient_norm')

    counter_e_z = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_e_z = ly.optimize_loss(loss=e_loss_z, learning_rate= None,
                    optimizer=tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9),
                    variables=theta_e_z, global_step=counter_e_z,
                    summaries = 'gradient_norm')

    counter_e_y = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_e_y = ly.optimize_loss(loss=e_loss_y, learning_rate= None,
                    optimizer=tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9),
                    variables=theta_e_y, global_step=counter_e_y,
                    summaries = 'gradient_norm')

    return real_data, real_label, unlabeled_data, y, z, opt_c, opt_e_z, opt_e_y, opt_g, fake_x, c_loss, e_loss_z, e_loss_y, g_loss, true_y


# In[9]:

def main():
    max_iter_step = 60000
    labeledset = make_dataset(txs, tys)
    trainset = make_dataset(trainx, trainy)
    testset = make_dataset(testx, testy)
    with tf.device(device):
        real_data, real_label, unlabeled_data, y, z, opt_c, opt_e_z, opt_e_y, opt_g, fake_x, c_loss, e_loss_z, e_loss_y, g_loss, true_y = build_graph()
    merged_all = tf.summary.merge_all()
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    with tf.Session(config=config) as sess:
        def next_y():
            return np.random.multinomial(1, [1/10.]*10, size=batch_size)
        def next_z():
            return np.random.normal(0, 1, [batch_size, z_dim]).astype(np.float32)
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        print("Training semi-supervised wgan...")
        for i in range(max_iter_step):
            citers = Citers
            for j in range(citers):
                unlabeled_img, _ = trainset.next_batch(batch_size)
                if i % 100 == 99 and j == 0:
                    train_img, train_label = labeledset.next_batch(batch_size)
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, merged, loss_c_ = sess.run([opt_c, merged_all, c_loss], feed_dict={real_data: train_img, real_label: train_label, y: next_y(), z: next_z(), unlabeled_data: unlabeled_img}, options=run_options, run_metadata=run_metadata)
                    summary_writer.add_summary(merged, i)
                    summary_writer.add_run_metadata(run_metadata, 'critic_metadata {}'.format(i), i)
                else:
                    _ = sess.run([opt_c], feed_dict={unlabeled_data: unlabeled_img, y: next_y(), z: next_z()})

            train_img, train_label = labeledset.next_batch(batch_size)
            unlabeled_img, _ = trainset.next_batch(batch_size)
            bz = next_z()
            by = next_y()
            if i % 100 == 99:
                _, _, _, merged, loss_e_y_, loss_e_z_, loss_g_ = sess.run([opt_e_y, opt_e_z, opt_g, merged_all, e_loss_y, e_loss_z, g_loss], feed_dict={real_data: train_img, real_label: train_label, y: by, z: bz, unlabeled_data: unlabeled_img}, options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged, i)
                summary_writer.add_run_metadata(run_metadata, 'generator_and_encoder_metadata {}'.format(i), i)
            else:
                _, _, _ = sess.run([opt_e_y, opt_e_z, opt_g], feed_dict={real_data: train_img, real_label: train_label, y: by, z: bz, unlabeled_data: unlabeled_img})

            if i % 100 == 99:
                print("Training ite %d, c_loss: %f, e_loss_z: %f, e_loss_y: %f, g_loss: %f" % (i, loss_c_, loss_e_z_, loss_e_y_, loss_g_))
                batch_y = []
                batch_z = []
                tmp = np.random.normal(0, 1, [10, z_dim]).astype(np.float32)
                for j in range(10):
                    batch_z.append(tmp)
                    tmpy = np.zeros((10 ,10))
                    tmpy[:, j] = 1
                    batch_y.append(tmpy)
                batch_z = np.concatenate(batch_z, 0)
                batch_y = np.concatenate(batch_y, 0)

                bx = sess.run(fake_x, feed_dict={y: batch_y, z: batch_z})
                fig = plt.figure(image_dir + '.semi-wgan')
                grid_show(fig, (bx + 1) / 2, [32, 32, channel])
                if not os.path.exists('./logs/{}/{}'.format(image_dir, args.logdir)):
                    os.makedirs('./logs/{}/{}'.format(image_dir, args.logdir))
                fig.savefig('./logs/{}/{}/{}.png'.format(image_dir, args.logdir, (i-99)/100))

            if i % 1000 == 999:
                saver.save(sess, os.path.join(
                    ckpt_dir, "model.ckpt"), global_step=i)
                testset._index_in_epoch = 0
                preds = np.zeros((testset.num_examples / batch_size * batch_size))
                gts = np.zeros((testset.num_examples / batch_size * batch_size))
                for j in range(testset.num_examples / batch_size):
                    test_img, test_label = testset.next_batch(batch_size)
                    by = sess.run(true_y, feed_dict={real_data: test_img})
                    preds[j*batch_size:(j+1)*batch_size] = np.argmax(by, 1)
                    gts[j*batch_size:(j+1)*batch_size] = np.argmax(test_label, 1)
                acc = np.sum(preds == gts) / float(gts.shape[0])
                print("Training ite %d, testing acc: %f" % (i, acc))




# In[10]:

main()
