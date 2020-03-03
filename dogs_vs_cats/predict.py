#!/usr/bin/env python
# coding: utf-8
# 使用模型进行预测模块


import os
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import common
import model


# Function:
#       根据路径获取测试集图片
# Args:
#       file_dir: 图片的路径，以“/”结束
# Returns:
#       images: 测试图片路径+文件名 py.list
def get_test_files(file_dir):
    images = []
    for file in os.listdir(file_dir):
        images.append(file_dir + file)
    return images


# Function:
#       根据路径和图片名获得一张图片内容
# Args:
#       file_dir: 图片的路径，以“/”结束
# Returns:
#       image: 一张图片 np.array
def get_one_image_content(image):
    image = Image.open(image)
    image = image.resize([208, 208])
    image = np.array(image)
    return image


# Function:
#       预测一张图片所属的类别
# Args:
#       image: 一张图片 np.array
# Returns:
#       max_index: 0或者1，0代表猫，1代表狗 int
def predict_one_image(image):

    with tf.Graph().as_default():
        image_tensor = tf.cast(image, tf.float32)
        image_tensor = tf.reshape(image_tensor, [1, common.IMG_W, common.IMG_W, 3])
        logit = model.inference(image_tensor, 1, common.N_CLASSES)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[common.IMG_W, common.IMG_W, 3])
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # print('Reading checkpoints...')
            ckpt = tf.train.get_checkpoint_state(common.LOGS_TRAIN_DIR)
            # 提取训练好的模型 官网有步骤
            if ckpt and ckpt.model_checkpoint_path:
                # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                # print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            prediction = sess.run(logit, feed_dict={x: image})
            max_index = np.argmax(prediction)

            return max_index


if __name__ == '__main__':
    image_list = get_test_files(common.TEST_DATASET_DIR)
    result_list = []
    for i in range(len(image_list)):
        print('预测第%d个图片' % i)
        image = get_one_image_content(image_list[i])
        predict = predict_one_image(image)
        img_num = image_list[i].split('/')[-1].split('.')[0]
        result_list.append([img_num, predict])
    result_list = sorted(result_list)

    column_name = ['id', 'label']
    data_df = pd.DataFrame(columns=column_name, data=result_list)
    data_df.to_csv(common.TEST_RESULT_DIR + 'predict.csv', index=False)


