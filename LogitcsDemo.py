#!/usr/bin/env python
#coding=utf-8
##tf2逻辑回归
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

#1模型的工具和数据y = 3(weight)x + 0.217(bias)
#2模型的定义
input_x = np.random.rand(10)
input_y = 3 * input_x + 0.217
weight = tf.Variable(1., dtype=tf.float32, name="weight")
bias = tf.Variable(1., dtype=tf.float32, name="bias")

def model(xs):
    return tf.multiply(xs, weight) + bias
#3梯度函数的更新
opt = tf.optimizers.Adam(1e-1)
for xs, ys in zip(input_x, input_y):
    xs = np.reshape(xs, [1])
    ys = np.reshape(ys, [1])
    with tf.GradientTape() as tape:
        # _loss = tf.reduce_mean(tf.pow((model(xs) - ys), 2)) / (2 * 1000)
        #损失函数的定义
        _loss = lambda: tf.losses.MeanSquaredError()(model(xs), ys)
    # grads = tape.gradient(_loss, [weight, bias])
    # opt.apply_gradients(zip(grads, [weight, bias]))
    opt.minimize(_loss, [weight, bias])
    print("train loss is: ", _loss().numpy())
print(weight)
print(bias)


