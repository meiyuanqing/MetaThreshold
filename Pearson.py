#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/4/10
Time: 10:52
File: Pearson.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

Gets the effective size and it's variance for Pearson, then for meta-analysis.
主要步骤：
    (1)计算Spearman相关系数Spearman_value；
    (2)由于Pearson相关系数需要度量与缺陷变量满足正态分布，
       计算近似Pearson相关系数：Pearson_value = 2 * np.sin(np.pi * Spearman_value / 6)
    (3)通过Fisher变换：Fisher_Z = 0.5 * np.log((1 + Pearson_value) / (1 - Pearson_value))
    (4)计算Fisher_Z的方差： Fisher_Z_variance = 1 / (Sample_size - 3), Sample_size为第i系统上样本数；
    (5)然后对Fisher_Z做随机效应元分析，最后通过Fisher反向变换，得出Pearson的元分析值，其符号为方向，即正号为正相关，负号为负相关。

"""
import time


def pearson_effect(working_dir="F:\\NJU\\MTmeta\\experiments\\supervised\\trainingData\\",
                   result_dir="F:\\NJU\\MTmeta\\experiments\\supervised\\PearsonEffect\\",
                   training_list="List.txt"):
    import os
    import csv
    import numpy as np
    import pandas as pd

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    working_directory = working_dir
    result_directory = result_dir
    os.chdir(working_directory)

    with open(working_dir + training_list) as l:
        lines = l.readlines()

    for line in lines:
        file = line.replace("\n", "")
        print('the file is ', file)

        # 分别处理每一个项目: f1取出要被处理的项目;
        #                  f2:用于存储每一个项目的Spearman,pearson,FisherZ和variance
        #                  deletedList: 用于存储项目中某个度量样本数小于3，和pearson等于1的度量。
        with open(working_directory + file, 'r', encoding="ISO-8859-1") as f1, \
                open(result_directory + "Pearson_effects.csv", 'a+', encoding="utf-8", newline='') as f2, \
                open(result_directory + "Pearson_effects_deletedList.csv", 'a+', encoding="utf-8") as deletedList:

            reader = csv.reader(f1)
            writer = csv.writer(f2)
            writer_deletedList = csv.writer(deletedList)
            # receives the first line of a file and convert to dict generator
            fieldnames = next(reader)
            # exclude the non metric fields
            non_metric = ["relName", "className", "bug"]

            # metric_data stores the metric fields (102 items)
            def fun_1(m):
                return m if m not in non_metric else None

            metric_data = filter(fun_1, fieldnames)

            df = pd.read_csv(file)
            # drop all rows that have any NaN values,删除表中含有任何NaN的行,并重新设置行号
            df = df.dropna(axis=0, how='any', inplace=False).reset_index(drop=True)

            if os.path.getsize(result_directory + "Pearson_effects.csv") == 0:
                writer.writerow(["fileName", "metric", "Sample_size", "Spearman_metric_bug", "Pearson_metric_bug",
                                 "Fisher_Z", "Fisher_Z_variance"])

            if os.path.getsize(result_directory + "Pearson_effects_deletedList.csv") == 0:
                writer_deletedList.writerow(["fileName", "metric", "Sample_size", "Spearman_metric_bug",
                                             "Pearson_metric_bug", "Fisher_Z", "Fisher_Z_variance"])

            for metric in metric_data:
                print("the current file is ", file)
                print("the current metric is ", metric)

                # 由于bug中存储的是缺陷个数,转化为二进制存储,若x>2,则可预测bug为3个以上的阈值,其他类推
                df['bugBinary'] = df.bug.apply(lambda x: 1 if x > 0 else 0)

                # 判断每个度量与bug之间的关系,因为该关系会影响到断点回归时,相关系数大于零,则LATE估计值大于零,反之,则LATE估计值小于零
                Spearman_metric_bug = df.loc[:, [metric, 'bug']].corr('spearman')

                Spearman_value = Spearman_metric_bug[metric][1]
                Pearson_value = 2 * np.sin(np.pi * Spearman_value / 6)

                Sample_size = len(df[metric])

                Fisher_Z = 0.5 * np.log((1 + Pearson_value) / (1 - Pearson_value))
                Fisher_Z_variance = 1 / (Sample_size - 3)

                if (Sample_size <= 3) or (Pearson_value == 1):
                    writer_deletedList.writerow([file, metric, Sample_size, Spearman_value, Pearson_value, Fisher_Z,
                                                 Fisher_Z_variance])
                else:
                    writer.writerow([file, metric, Sample_size, Spearman_value, Pearson_value, Fisher_Z,
                                     Fisher_Z_variance])
        # break


if __name__ == '__main__':

    s_time = time.time()
    pearson_effect()
    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ". This is end of Pearson.py!\n",
          "The execution time of Pearson.py script is ", execution_time)
