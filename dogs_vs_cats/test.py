#!/usr/bin/env python
# coding: utf-8
# 测试模块


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import input_data


def test_input_data_get_train_and_val_files():
    file_dir = './dataset/train/'
    train_imgs, train_labels, val_imgs, val_labels = input_data.get_train_and_val_files(file_dir, False, 0.3)
    print(type(train_imgs[0]))
    print(type(train_labels[0]))
    print(val_imgs.shape)
    print(val_labels)
    #print(image_list)
    #print(label_list)


def test_input_data_get_batch():
    batch_size = 8
    image_list, label_list = input_data.get_files('./dataset/train/')
    image_batch, label_batch = input_data.get_batch(image_list, label_list, 200, 200, batch_size, 256)

    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()  # 创建一个线程管理器
        threads = tf.train.start_queue_runners(coord=coord)  # 使用start_queue_runners 启动队列填充
        try:
            while not coord.should_stop() and i < 1:
                img, label = sess.run([image_batch, label_batch])
                print(img.shape)
                print(label)
                for j in np.arange(batch_size):
                    print("下图的label: %d" % label[j])
                    plt.imshow(img[j, :, :, :])
                    plt.show()
                i += 1
        except tf.errors.OutOfRangeError:
            print('Done.')


def test_input_data_divide_train_val():
    train_dir = './dataset/train/'
    img_list, l_list = input_data.get_train_files(train_dir, False)
    train_image_arr,train_label_arr, val_image_arr,val_label_arr = input_data.divide_train_val(img_list, l_list, 0.3)
    print(train_image_arr[0:10])
    print(train_label_arr[0:10])
    print(val_image_arr[0:10])
    print(val_label_arr[0:10])


def main():
    #test_input_data_get_files()
    # test_input_data_get_batch()
    test_input_data_get_train_and_val_files()
    #t_imgs, t_labels, v_imgs, v_lables = test_input_data_divide_train_val()

    #print(t_imgs, '\n', t_labels, '\n', v_imgs, '\n',v_lables)


if __name__ == '__main__':
    main()