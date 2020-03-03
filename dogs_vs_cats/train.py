#!/usr/bin/env python
# coding: utf-8
# 训练模型模块


import os
import time
import numpy as np
import tensorflow as tf
import input_data
import model
import common


# 开始训练
def run_training():
    # 记录训练开始时间
    time_start = time.time()

    train_data, train_labels, val_data, val_labels = input_data.get_train_and_val_files(common.TRAIN_DATASET_DIR,
                                                                                        False,
                                                                                        common.VALIDATION_PERCENTAGE)
    train_batch, train_label_batch = input_data.get_batch(train_data,
                                                          train_labels,
                                                          common.IMG_W,
                                                          common.IMG_H,
                                                          common.BATCH_SIZE,
                                                          common.CAPACITY)

    val_batch, val_label_batch = input_data.get_batch(val_data,
                                                      val_labels,
                                                      common.IMG_W,
                                                      common.IMG_H,
                                                      common.BATCH_SIZE,
                                                      common.CAPACITY)

    x = tf.placeholder(tf.float32, shape=[common.BATCH_SIZE, common.IMG_W, common.IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[common.BATCH_SIZE])

    train_logits = model.inference(train_batch, common.BATCH_SIZE, common.N_CLASSES)
    loss = model.losses(train_logits, train_label_batch)
    train_op = model.training(loss, common.LEARNING_RATE)
    acc = model.evaluation(train_logits, train_label_batch)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(common.LOGS_TRAIN_DIR, sess.graph)
        val_writer = tf.summary.FileWriter(common.LOGS_VAL_DIR, sess.graph)

        try:
            for step in np.arange(common.MAX_STEP):
                if coord.should_stop():
                    break

                tra_images, tra_labels = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                                feed_dict={x: tra_images, y_: tra_labels})

                if step % 100 == 0:
                    print("Step %d, train loss = %.4f, train accuracy = %.4f%%" % (step, tra_loss, tra_acc*100.0))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)
                if step % 200 == 0 or (step + 1) == common.MAX_STEP:
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    val_loss, val_acc = sess.run([loss, acc],
                                                 feed_dict={x: val_images, y_: val_labels})
                    print("Step %d, val loss = %.4f, val accuracy = %.4f%%" % (step, val_loss, val_acc * 100.0))
                    summary_str = sess.run(summary_op)
                    val_writer.add_summary(summary_str, step)
                if step % 2000 == 0 or (step + 1) == common.MAX_STEP:
                    checkpoint_path = os.path.join(common.LOGS_TRAIN_DIR, "model.ckpt")
                    saver.save(sess, checkpoint_path, global_step=step)
        except tf.errors.OutOfRangeError:
            print("Done training -- epoch limit reached.")
        finally:
            coord.request_stop()
        coord.join(threads)

    # 计算训练用时
    time_end = time.time()
    time_cost = time_end - time_start
    print('Total training time is: ', time_cost, 's')


if __name__ == '__main__':
    run_training()
