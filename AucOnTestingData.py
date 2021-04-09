#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/4/9
Time: 15:46
File: AucOnTestingData.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

Performance metric: AUC
(1)tests the unsupervised and supervised deriving method's threshold respectively on testing data;
(2)tests the 9 methods' meta-analysis value of threshold on testing data.

"""
import time


def auc_testing(working_dir="F:\\NJU\\MTmeta\\experiments\\pooled\\PoolingThresholds\\",
                result_dir="F:\\NJU\\MTmeta\\experiments\\pooled\\",
                training_list="List.txt"):

    import os
    import csv
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    # from sklearn import metrics
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix

    # 显示所有列
    pd.set_option('display.max_columns', None)

    # 显示所有行
    pd.set_option('display.max_rows', None)

    # the item of row of dataframe
    pd.set_option('display.width', 5000)

    working_directory = working_dir
    result_directory = result_dir
    os.chdir(working_directory)



if __name__ == '__main__':

    s_time = time.time()
    auc_testing()
    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ". This is end of AucOnTestingData.py!\n",
          "The execution time of AucOnTestingData.py script is ", execution_time)