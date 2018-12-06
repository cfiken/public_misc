# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)
test_data = mnist.test.images
test_label = mnist.test.labels

num_inputs = 784 # 28x28
num_outputs = 10
num_units = 128
num_layers = 6
batch_size = 64
num_epocs= 200000
learning_rate = 0.001
dropout_rate = 0.1

tf.reset_default_graph()

class Model:
    
    def __init__(self):
        # placeholder
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, num_inputs], name='inputs')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, num_outputs], name='outputs')
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        outputs = self.build(self.x)
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(outputs + 10e-8), 1))
        y_label = tf.argmax(self.y, 1)
        outputs_label = tf.argmax(outputs, 1)
        correct = tf.equal(y_label, outputs_label)
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
        

    def build(self, x):
        layer = tf.layers.dropout(x, rate=dropout_rate, training=self.is_training)
        for l in range(num_layers):
            layer = tf.layers.dense(layer, num_units, tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name='layer_{}'.format(l))
            layer = tf.layers.dropout(layer, rate=dropout_rate , training=self.is_training, name='layer_dropout_{}'.format(l))

        out = tf.layers.dense(layer,
                                   num_outputs, 
                                   tf.nn.softmax, 
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                                   name='out_layer')
        return out

model = Model()
global_step = tf.train.get_or_create_global_step()
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(model.loss, global_step=global_step)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        train_data, train_label = mnist.train.next_batch(batch_size)
        _, loss_value = sess.run([train_op, model.loss], feed_dict={
            model.x: train_data,
            model.y: train_label,
            model.is_training: True
        })
        step = sess.run(global_step)
        if step % 500 == 0:
            print('{} step loss: {}'.format(step, loss_value))
    saver.save(sess, 'ckpt/model.ckpt', step)

# ## モデルのインスタンスを使う

tf.reset_default_graph()
model = Model()
saver = tf.train.Saver()
with tf.Session() as sess:
    ckpt_path = tf.train.latest_checkpoint('ckpt/')
    saver.restore(sess, ckpt_path)
    res = sess.run(model.accuracy, feed_dict={
        model.x: test_data,
        model.y: test_label,
        model.is_training: False
    })
print('accuracy: ', res)

# ## `.meta` を使う

tf.reset_default_graph()
with tf.Session() as sess:
    ckpt_path = tf.train.latest_checkpoint('ckpt/')
    saver = tf.train.import_meta_graph(ckpt_path + '.meta')
    saver.restore(sess, ckpt_path)
    res = sess.run('accuracy:0', feed_dict={
        'inputs:0': test_data,
        'outputs:0': test_label,
        'is_training:0': False
    })
print('accuracy: ', res)

# ## `.meta` と `get_tensor_by_name` 使う

tf.reset_default_graph()
with tf.Session() as sess:
    ckpt_path = tf.train.latest_checkpoint('ckpt/')
    saver = tf.train.import_meta_graph(ckpt_path + '.meta')
    saver.restore(sess, ckpt_path)
    
    is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')
    inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
    outputs = tf.get_default_graph().get_tensor_by_name('outputs:0')
    accuracy = tf.get_default_graph().get_tensor_by_name('accuracy:0')
    res = sess.run(accuracy, feed_dict={
        inputs: test_data,
        outputs: test_label,
        is_training: False
    })
print('accuracy: ', res)


