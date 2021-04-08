#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/3/16
Time: 17:59
File: Bender.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

Use Bender method to derive metric threshold in each system.


Reference:
[1]  Bender, R. Quantitative risk assessment in epidemiological studies investigating threshold effects.
     Biometrical Journal, 41 (1999), 305-319.（计算VARL的SE（标准误）的参考文献P310）
"""

import os
import csv
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

working_dir = "F:\\NJU\\MTmeta\\experiments\\supervised\\trainingData\\"
result_dir = "F:\\NJU\\MTmeta\\experiments\\supervised\\"

os.chdir(working_dir)
print(os.getcwd())


# reads a metric from a csv file, uses bender method to derive a threshold
# input: a csv file;            output: a dict
def bender_threshold():
    with open(working_dir + 'List.txt') as l:
        lines = l.readlines()
    print("the number of list files is ", len(lines))

    for line in lines:
        file = line.replace("\n", "")

        # if file != "camel-1.4.0.csv":
        #     continue
        print("the current file is ", file)
        print("the current repr file is ", repr(file))
        # project_file: a csv file of a version source code from a project；Bender_threshold：stores threshold values；
        with open(working_dir + file, 'r', encoding="ISO-8859-1") as project_file, \
             open(result_dir + "BenderThreshold\\Bender_thresholds.csv", 'a+', encoding="utf-8", newline='') \
                        as Bender_threshold, \
             open(result_dir + "BenderThreshold\\deletedList.csv", 'a+', encoding="utf-8", newline='') as deletedList:
            reader = csv.reader(project_file)
            writer = csv.writer(Bender_threshold)
            writer_deletedList = csv.writer(deletedList)
            # receives the first line of a file and convert to dict generator
            fieldnames = next(reader)
            # exclude the non metric fields (12 items) and metric values including undef and undefined (17 items)
            non_metric = ["relName", "className", "bug"]

            # metric_data stores the metric fields (102 items)
            def fun_1(m):
                return m if m not in non_metric else None

            metric_data = filter(fun_1, fieldnames)
            # print("the metric_data are ", metric_data)
            # print("the metric_data are ", [xx for xx in metric_data])
            # print("the 2 metric_data are ", [xx for xx in metric_data])
            # list_xx = set([xx for xx in metric_data])
            # print("the metric_data are ", len(list_xx))

            if os.path.getsize(result_dir + "BenderThreshold\\Bender_thresholds.csv") == 0:
                writer.writerow(["fileName", "metric", "corr", "B_0", "B_1", "BaseProbability_1", "B_0_pValue",
                                 "B_1_pValue", "cov11", "cov12", "cov22", "VARL_threshold", "VARL_variance"])

            if os.path.getsize(result_dir + "BenderThreshold\\deletedList.csv") == 0:
                writer_deletedList.writerow(["fileName", "metric", "B_0_pValue", "B_0"])

            # read a csv file of a system
            df = pd.read_csv(file)
            # drop all rows that have any NaN values,删除表中含有任何NaN的行
            df.dropna(axis=0, how='any', inplace=True)
            # iterate through each metric in turns
            i = 0
            for metric in metric_data:

                i += 1
                print("the No. of metric is ", i, "the current metric is ", metric)

                # 由于bug中存储的是缺陷个数,转化为二进制存储,若item>2,则可预测bug为3个以上的阈值,其他类推
                df['bugBinary'] = df.bug.apply(lambda item: 1 if item > 0 else 0)
                df['intercept'] = 1.0

                # 通过 statsmodels.api 逻辑回归分类; 指定作为训练变量的列，不含目标列`bug`
                logit = sm.Logit(df['bugBinary'], df.loc[:, [metric, 'intercept']])
                # 拟合模型,disp=1 用于显示结果
                result = logit.fit(method='bfgs', disp=0)
                print(result.summary())

                pValueLogit = result.pvalues
                if pValueLogit[0] > 0.05:  # 自变量前的系数
                    writer_deletedList.writerow([file, metric, pValueLogit[0], B[0]])
                    continue

                # 求VARL作为阈值 VARL.threshold = (log(Porbability[1]/Porbability[2])-B[1])/B[2]
                valueOfbugBinary = df["bugBinary"].value_counts()  # 0 和 1 的各自的个数
                print("the value of valueOfbugBinary[0] is ", valueOfbugBinary[0])
                print("the value of valueOfbugBinary[1] is ", valueOfbugBinary[1])

                # 用缺陷为大于0的模块数占所有模块之比
                BaseProbability_1 = valueOfbugBinary[1] / (valueOfbugBinary[0] + valueOfbugBinary[1])
                B = result.params  # logit回归系数
                if B[0] == 0:  # 自变量前的系数
                    writer_deletedList.writerow([file, metric, pValueLogit[0], B[0]])
                    continue

                # 计算VARL阈值及标准差
                VARLThreshold = (np.log(BaseProbability_1 / (1 - BaseProbability_1)) - B[1]) / B[0]
                # 计算LOGIT回归系数矩阵的协方差矩阵,因为计算VARL的标准差要用到,见参考文献[1]
                cov = result.cov_params()
                cov11 = cov.iloc[0, 0]
                cov12 = cov.iloc[0, 1]
                cov22 = cov.iloc[1, 1]
                VARLThreshold_se = ((cov.iloc[0, 0] + 2 * VARLThreshold * cov.iloc[0, 1]
                                     + VARLThreshold * VARLThreshold * cov.iloc[1, 1]) ** 0.5) / B[0]
                VARL_variance = VARLThreshold_se ** 2

                # 判断每个度量与bug之间的关系
                CorrDf = df.loc[:, [metric, 'bug']].corr('spearman')
                # 当每个度量与bug之间的相关系数大于零,则正相关,当前的VARL阈值为最大值;当度量值大于该阈值,则预测为有缺陷.
                if CorrDf[metric][1] < 0:
                    df['predictBinary_1'] = df[metric].apply(lambda m: 1 if m <= VARLThreshold else 0)
                else:
                    df['predictBinary_1'] = df[metric].apply(lambda m: 1 if m >= VARLThreshold else 0)

                writer.writerow([file, metric, CorrDf[metric][1], B[0], B[1], BaseProbability_1, pValueLogit[0],
                                 pValueLogit[1], cov11, cov12, cov22, VARLThreshold, VARL_variance])


        # break

    print("bender method to derive metric threshold")


if __name__ == "__main__":
    s_time = time.time()
    bender_threshold()
    e_time = time.time()
    execution_time = e_time - s_time
    print("__name__ is ", __name__, "! the execution time of Bender.py script is ", execution_time)
