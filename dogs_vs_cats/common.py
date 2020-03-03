#!/usr/bin/env python
# coding: utf-8
# 常量模块


'''
模型训练相关参数
'''
VALIDATION_PERCENTAGE = 0.3  # 验证集所占的比例
N_CLASSES = 2
IMG_W = 208  # 输入模型的图片的宽
IMG_H = 208  # 输入模型的图片的宽
BATCH_SIZE = 32
CAPACITY = 2000  # 队列中的最大容量
MAX_STEP = 15000  # 最大训练步数
LEARNING_RATE = 0.0001


'''
路径相关参数
'''
TRAIN_DATASET_DIR = './dataset/train/'  # 训练数据集存放路径
TEST_DATASET_DIR = './dataset/test/'  # 测试数据集存放路径
LOGS_TRAIN_DIR = './logs/train_1/'  # 训练logs及训练好的model存放路径
LOGS_VAL_DIR = './logs/val_1/'  # 验证logs存放路径
TEST_RESULT_DIR = './result/predict_1/'  # 测试集结果csv文件存放路径
