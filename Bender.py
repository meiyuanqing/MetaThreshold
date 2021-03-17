#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/3/16
Time: 17:59
File: Bender.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

Use Bender method to derive metric threshold.

"""

import os
import csv
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

working_dir = "F:\\NJU\\MTmeta\\experiments\\training_data\\"
result_dir = "F:\\NJU\\MTmeta\\experiments\\thresholds\\"

os.chdir(working_dir)


# print(os.getcwd())

# reads a metric from a csv file, uses bender method to derive a threshold
# input: a csv file;            output: a dict
def bender_threshold():
    with open(working_dir + 'List.txt') as l:
        lines = l.readlines()
    print("the number of list files is ", len(lines))

    for line in lines:
        file = line.replace("\n", "")

        print("the currenting file is ", file)
        # project_file: a csv file of a version source code from a project；threshold_file：stores threshold values；
        # f3:用于存储每个项目的logit回归系数，用于后续断点回归的running variable
        with open(working_dir + file, 'r', encoding="ISO-8859-1") as project_file, \
             open(result_dir + "Bender_thresholds.csv", 'a+', encoding="utf-8", newline='') as threshold_file:
            reader = csv.reader(project_file)
            writer = csv.writer(threshold_file)
            fieldnames = next(reader)  # 获取数据的第一行，作为后续要转为字典的键名生成器，next方法获取
            # 对每个项目文件查看一下，把属于metric度量的字段整理到metricData
            metricData = fieldnames[8:75]        # 对fieldnames切片取出所有要处理的度量,一共68个
        #     # 先写入columns_name
        #     if os.path.getsize(result_dir + "metricThresholds.csv") == 0:
        #         writer.writerow(["fileName", "metric", "k-fold", "corr", "corr_std", "B_0", "B_1", "B_0_pValue",
        #                          "B_1_pValue", "cov11", "cov12", "cov22", "VARLThreshold", "VARL_variance",])
        #
        #     # 读入一个项目
        #     df = pd.read_csv(file)
        #     # 依次遍历每一个度量
        #     for metric in metricData:
        #         print("the current file is ", file)
        #         print("the current metric is ", metric)
        #         # 若度量中存在undef或undefined数据，由于使得每个度量值的个数不同，故舍去该度量的值
        #         undef = 0
        #         undefined = 0
        #         for x in df[metric]:
        #             if x == 'undef':
        #                 undef = 1
        #             if x == 'undefined':
        #                 undefined = 1
        break

    print("bender method to derive metric threshold")


if __name__ == "__main__":
    bender_threshold()
    print("__name__ is ", __name__)
    print("the script is running")
