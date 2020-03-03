#!/usr/bin/env python
# coding: utf-8
# 模型模块


import tensorflow as tf


# Function:
#       模型结构
# Args:
#       images: 一个batch的所有图片
#       batch_size: 一个batch的图片数量
#       n_classes: 类别的数量
# Returns:
#       softmax_linear: 模型最后的输出
def inference(images, batch_size, n_classes):
    # conv1, shape=[kernel_size, kerbel_size, channels, kernel_numbers]
    with tf.variable_scope("conv1") as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,
                                                                              dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                shape=[16], 
                                dtype=tf.float32, 
                                initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, 'conv1')
    
    # pool1 & norml1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, 
                               ksize=[1, 3, 3, 1], 
                               strides=[1, 2, 2, 1], 
                               padding='SAME', 
                               name='pooling1')
        norm1 = tf.nn.lrn(pool1, 
                          depth_radius=4, 
                          bias=1.0, 
                          alpha=0.001/9.0, 
                          beta=0.75, 
                          name='norm1')
    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights', 
                                  shape=[3, 3, 16, 16], 
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,
                                                                              dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[16], 
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, 
                            weights, 
                            strides=[1, 1, 1, 1], 
                            padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
    
    # pool2 & norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = tf.nn.max_pool(conv2, 
                               ksize=[1, 3, 3, 1], 
                               strides=[1, 2, 2, 1], 
                               padding='SAME', 
                               name='pooling2')
        norm2 = tf.nn.lrn(pool2, 
                          depth_radius=4, 
                          bias=1.0, 
                          alpha=0.001/9.0, 
                          beta=0.75, 
                          name='norm2')
    
    # full-connect1
    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(norm2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights', 
                                  shape=[dim, 128], 
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, 
                                                                              dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128], 
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name='fc1')
        
    # full-connect2
    with tf.variable_scope('fc2') as scope:
        weights = tf.get_variable('weights', 
                                  shape=[128, 128], 
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, 
                                                                              dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128], 
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name='fc2')
    
    # softmax
    with tf.variable_scope("softmax_linear") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, 
                                                                              dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name="softmax_linear")
        # softmax_linear = tf.nn.softmax(softmax_linear)  # softmax激活函数
    
    return softmax_linear


# Function:
#       计算训练过程中的损失
# Args:
#       logits: 函数inference()的输出，即一个batch的模型softmax层输出结果，代表对猫和狗的预测概率
#       labels: 一个bathc的标签的Ground Truth
# Returns:
#       loss: 一个batch的损失
def losses(logits, labels):  # logits为inference的返回值，labels为ground truth
    with tf.variable_scope("loss") as scope:
        # 使用sparse就不需要one-hot编码
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels,
                                                                       name="entropy_per_example")
        loss = tf.reduce_mean(cross_entropy, name="loss")
        tf.summary.scalar(scope.name + "loss", loss)
        return loss
    
    
# Function:
#       训练过程
# Args:
#       loss: losses()的输出，即一个batch的损失
#       learning_rate: 学习率
# Returns:
#       train_op: 损失
def training(loss, learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# Function:
#       评估模型，计算正确率
# Args:
#       logits: inference()的输出，即模型的softmax层输出
#       labels: 一个batch的标签的ground truth
# Returns:
#       accuracy: 一个batch上的正确率
def evaluation(logits, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy

