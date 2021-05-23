#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/5/23
Time: 11:55
File: DescriptiveTraining.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

Descriptive statistics for the training data sets and test data sets
i.e., Number of classes and Number of bugs of versions of each project: Max. Min. Mean Std.

"""
import time


def Descriptive_data(t_dir="F:\\NJU\\MTmeta\\experiments\\trainings\\",
                     d_dir="F:\\NJU\\MTmeta\\experiments\\descriptive\\",
                     training_list="List.txt",
                     project_list="project_List.txt"):
    import os
    import csv
    from scipy.stats import norm  # norm.cdf() the cumulative normal distribution function in Python
    from scipy import stats  # 根据卡方分布计算p值: p_value=1.0-stats.chi2.cdf(chisquare,freedom_degree)
    import numpy as np
    import pandas as pd

    os.chdir(d_dir)
    print(os.getcwd())

    with open(t_dir + training_list) as l:
        lines = l.readlines()

    for line in lines:
        file = line.replace("\n", "")
        print('the file is ', file)

        df = pd.read_csv(t_dir + file)
        Number_of_classes = len(df)
        Number_of_bugs = df["bug"].sum()
        print("\tNumber_of_classes's len is ", Number_of_classes, "\tNumber_of_bugs's len is ", Number_of_bugs)

        with open(d_dir + "descriptive_training.csv", 'a+', encoding="utf-8", newline='') as f:
            writer_f = csv.writer(f)
            if os.path.getsize(d_dir + "descriptive_training.csv") == 0:
                writer_f.writerow(["project_names", "Number_of_classes_max", "Number_of_classes_min",
                                   "Number_of_classes_mean", "Number_of_classes_std", "Number_of_bugs_max",
                                   "Number_of_bugs_min", "Number_of_bugs_mean", "Number_of_bugs_std"])
            # writer_f.writerow([])


if __name__ == '__main__':
    s_time = time.time()

    Descriptive_data()

    e_time = time.time()
    execution_time = e_time - s_time
    print("The __name__ is ", __name__, ".\tThis is end of DescriptiveTraining.py!",
          "\tThe execution time is ", execution_time)
