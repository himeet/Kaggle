#!/usr/bin/env python
# coding: utf-8
# 处理输入图像模块


import tensorflow as tf
import numpy as np
import random
import os


# Function:
#       根据路径获取训练数据集和验证数据集图片及其标签
# Args:
#       file_dir: 图片的路径，以“/”结束
#       is_shuffle: 是否shuffle
#       val_percent: 验证数据集所占的比例
# Returns:
#       train_imgs: 训练集图片路径+文件名 np.array
#       train_labels: 训练集标签 np.array
#       val_imgs:  验证集图片路径+文件名 np.array
#       val_labels: 验证集标签 np.array
def get_train_and_val_files(file_dir, is_shuffle, val_percent):
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []

    # 载入数据路径并写入标签值，用0表示猫，1表示狗
    for file in os.listdir(file_dir):
        name = file.split('.')[0]
        if name == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        elif name == 'dog':
            dogs.append(file_dir + file)
            label_dogs.append(1)
        else:
            print('Training dataset error.')

    # 划分训练集和验证集
    n_val = int(len(np.hstack((cats, dogs))) * val_percent)
    n_val_cats = n_val // 2
    n_val_dogs = n_val - n_val_cats

    val_cats_index_list = random.sample(range(len(cats)), n_val_cats)
    val_dogs_index_list = random.sample(range(len(dogs)), n_val_dogs)

    train_cats_index_list = list(set(range(len(cats))) - set(val_cats_index_list))
    train_dogs_index_list = list(set(range(len(dogs))) - set(val_dogs_index_list))

    val_imgs = np.array([])
    val_labels = np.array([])
    for i in range(len(val_cats_index_list)):
        val_imgs = np.append(val_imgs, cats[val_cats_index_list[i]])
        val_labels = np.append(val_labels, label_cats[val_cats_index_list[i]])
    for i in range(len(val_dogs_index_list)):
        val_imgs = np.append(val_imgs, dogs[val_dogs_index_list[i]])
        val_labels = np.append(val_labels, label_dogs[val_dogs_index_list[i]])

    train_imgs = np.array([])
    train_labels = np.array([])
    for i in range(len(train_cats_index_list)):
        train_imgs = np.append(train_imgs, cats[train_cats_index_list[i]])
        train_labels = np.append(train_labels, int(label_cats[train_cats_index_list[i]]))
    for i in range(len(train_dogs_index_list)):
        train_imgs = np.append(train_imgs, dogs[train_dogs_index_list[i]])
        train_labels = np.append(train_labels, int(label_dogs[train_dogs_index_list[i]]))

    return train_imgs, train_labels, val_imgs, val_labels


# Function:
#       使用队列生成大小相同的image batch和label batch
# Args:
#       image_list: 图片np.array
#       label_list: 标签np.array
#       image_width: 生成的batch中要求的图片的宽
#       image_height: 生成的batch中要求的图片的高
#       batch_size: 一个batch中的图片数量
#       capacity： 队列中最多可以容纳的个数
# Returns:
#       image_batch: 一个batch的图片
#       label_batch: 一个batch的标签
def get_batch(image_list, label_list, image_width, image_height, batch_size, capacity):
    # 将list转换成tf能够识别的格式
    image_list = tf.cast(image_list, tf.string)
    label_list = tf.cast(label_list, tf.int32)

    # 生成输入队列
    input_queue = tf.train.slice_input_producer([image_list, label_list])

    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    label = input_queue[1]

    # 统一图像大小
    # image = tf.image.resize_images(image, [image_height, image_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.resize_image_with_crop_or_pad(image, image_width, image_height)
    # image = tf.cast(image, tf.float32)  # 为了conv使用float32而不用int32

    # 标准化图像数据
    image = tf.image.per_image_standardization(image)
    # image = image / 255

    image_batch, label_batch = tf.train.batch([image, label],  # train.shuffle_batch
                                              batch_size=batch_size,
                                              num_threads=64,  # 线程数64
                                              capacity=capacity)
    # label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch

