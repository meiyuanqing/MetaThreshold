#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/4/8
Time: 21:47
File: PooledMeta.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

Deriving Pooled methods threshold by meta-analysis: four unsupervised and five supervised methods.
Four unsupervised methods: Alves, Vale, Ferreira, Oliveira;
Five supervised methods: Bender, ROC, BPP, MFM, GM.
"""
import time


def pooled_meta(t_dir="F:\\NJU\\MTmeta\\experiments\\pooled\\PoolingThresholds\\",
                m_dir="F:\\NJU\\MTmeta\\experiments\\pooled\\"):
    import os
    import csv
    from scipy.stats import norm  # norm.cdf() the cumulative normal distribution function in Python
    from scipy import stats  # 根据卡方分布计算p值: p_value=1.0-stats.chi2.cdf(chisquare,freedom_degree)
    import numpy as np
    import pandas as pd

    # 输入：两个匿名数组，effect_size中存放每个study的effect size，variance存放对应的方差
    # 输出：fixed effect model固定效应元分析后的结果，包括
    #      (1)fixedMean：固定效应元分析后得到的效应平均值；(2) fixedStdError：固定效应元分析的效应平均值对应的标准错
    def fixed_effect_meta_analysis(effect_size, variance):
        fixed_weight = []
        sum_Wi = 0
        sum_WiYi = 0
        d = {}  # return a dict
        study_number = len(variance)
        for i in range(study_number):
            if variance[i] == 0:
                continue
            fixed_weight.append(1 / variance[i])
            sum_Wi = sum_Wi + fixed_weight[i]
            sum_WiYi = sum_WiYi + effect_size[i] * fixed_weight[i]
        fixedMean = sum_WiYi / sum_Wi  # 固定模型元分析后得到的效应平均值
        fixedStdError = (1 / sum_Wi) ** 0.5  # 固定模型元分析的效应平均值对应的标准错
        d['fixedMean'] = fixedMean
        d['fixedStdError'] = fixedStdError
        return d

    # 输入：两个匿名数组，effect_size中存放每个study的effect size，variance存放对应的方差
    # 输出：random effect model随机效应元分析后的结果，包括：
    #      (1) randomMean：随机模型元分析后得到的效应平均值； (2) randomStdError：随机模型元分析的效应平均值对应的标准错
    def random_effect_meta_analysis(effect_size, variance):

        sum_Wi = 0
        sum_WiWi = 0
        sum_WiYi = 0  # Sum(Wi*Yi), where i ranges from 1 to k, and k is the number of studies
        sum_WiYiYi = 0  # Sum(Wi*Yi*Yi), where i ranges from 1 to k, and k is the number of studies

        sum_Wistar = 0
        sum_WistarYi = 0
        d = {}  # return a dict

        study_number = len(variance)
        fixed_weight = [0 for i in range(study_number)]  # 固定模型对应的权值
        random_weight = [0 for i in range(study_number)]  # 随机模型对应的权值

        for i in range(study_number):
            if variance[i] == 0:
                continue
            fixed_weight[i] = 1 / variance[i]
            sum_Wi = sum_Wi + fixed_weight[i]
            sum_WiWi = sum_WiWi + fixed_weight[i] * fixed_weight[i]
            sum_WiYi = sum_WiYi + effect_size[i] * fixed_weight[i]
            sum_WiYiYi = sum_WiYiYi + fixed_weight[i] * effect_size[i] * effect_size[i]

        Q = sum_WiYiYi - sum_WiYi * sum_WiYi / sum_Wi
        df = study_number - 1
        C = sum_Wi - sum_WiWi / sum_Wi

        # 当元分析过程中只有一个study研究时，没有研究间效应，故研究间的方差为零
        if study_number == 1:
            T2 = 0
        else:
            T2 = (Q - df) / C  # sample estimate of tau square

        if T2 < 0:
            T2 = (- 1) * T2  # 20190719，若T2小于0，取相反数

        for i in range(study_number):
            random_weight[i] = 1 / (variance[i] + T2)  # random_weight 随机模型对应的权值

        for i in range(study_number):
            sum_Wistar = sum_Wistar + random_weight[i]
            sum_WistarYi = sum_WistarYi + random_weight[i] * effect_size[i]

        randomMean = sum_WistarYi / sum_Wistar  # 随机模型元分析后得到的效应平均值
        randomStdError = (1 / sum_Wistar) ** 0.5  # 随机模型元分析的效应平均值对应的标准错
        # 当元分析过程中只有一个study研究时，没有研究间异质性，故异质性为零
        if study_number == 1:
            I2 = 0
        else:
            I2 = ((Q - df) / Q) * 100  # Higgins et al. (2003) proposed using a statistic, I2,
            # the proportion of the observed variance reflects real differences in effect size

        pValue_Q = 1.0 - stats.chi2.cdf(Q, df)  # pValue_Q = 1.0 - stats.chi2.cdf(chisquare, freedom_degree)

        d["C"] = C
        d["mean"] = randomMean
        d["stdError"] = randomStdError
        d["LL_CI"] = randomMean - 1.96 * randomStdError  # The 95% lower limits for the summary effect
        d["UL_CI"] = randomMean + 1.96 * randomStdError  # The 95% upper limits for the summary effect
        d["ZValue"] = randomMean / randomStdError  # a Z-value to test the null hypothesis that the mean effect is zero
        d["pValue_Z"] = 2 * (1 - norm.cdf(randomMean / randomStdError))  # norm.cdf() 返回标准正态累积分布函数值
        d["Q"] = Q
        d["df"] = df
        d["pValue_Q"] = pValue_Q
        d["I2"] = I2
        d["tau"] = T2 ** 0.5
        d["LL_ndPred"] = randomMean - 1.96 * (T2 ** 0.5)  # tau、randomMean 已知情况下的新出现的study的effctsize所落的区间
        d["UL_ndPred"] = randomMean + 1.96 * (T2 ** 0.5)  # tau、randomMean 已知情况下的新出现的study的effctsize所落的区间
        # tau、randomMean 未知情况（估计）下的新出现的study的effctsize所落的区间
        # stats.t.ppf(0.975,df)返回学生t分布单尾alpha=0.025区间点(双尾是alpha=0.05)的函数，它是stats.t.cdf()累积分布函数的逆函数
        d["LL_tdPred"] = randomMean - stats.t.ppf(0.975, df) * ((T2 + randomStdError * randomStdError) ** 0.5)
        # tau、randomMean 未知情况（估计）下的新出现的study的effctsize所落的区间
        d["UL_tdPred"] = randomMean + stats.t.ppf(0.975, df) * ((T2 + randomStdError * randomStdError) ** 0.5)

        fixedMean = sum_WiYi / sum_Wi  # 固定模型元分析后得到的效应平均值
        fixedStdError = (1 / sum_Wi) ** 0.5  # 固定模型元分析的效应平均值对应的标准错
        d['fixedMean'] = fixedMean
        d['fixedStdError'] = fixedStdError
        return d

    metric_dir = t_dir
    meta_dir = m_dir
    os.chdir(metric_dir)
    print(os.getcwd())

    # 显示所有行
    pd.set_option('display.max_rows', None)

    # 显示所有列
    pd.set_option('display.max_columns', None)

    # the item of row of dataframe
    pd.set_option('display.width', 5000)

    # read one of file to get the metric names for meta-analysis
    # read_csv(path, keep_default_na=False, na_values=[""])  只有一个空字段将被识别为NaN
    df_metric_names = pd.read_csv(metric_dir + "Alves_metaThresholds.csv", keep_default_na=False, na_values=[""])

    metric_names = sorted(set(df_metric_names.metric.values.tolist()))
    print("the metric_names are ", df_metric_names.columns.values.tolist())
    print("the metric_names are ", metric_names)
    print("the len metric_names are ", len(metric_names))

    with open(metric_dir + 'List.txt') as l:
        lines = l.readlines()
    print("the files are ", lines)
    print("the number of list files is ", len(lines))

    for line in lines:
        file = line.replace("\n", "")

        # if file != "camel-1.4.0.csv":
        #     continue
        print("the current file is ", file)
        print("the current repr file is ", repr(file))
        # project_file: a csv file of a version source code from a project；Bender_threshold：stores threshold values；
        with open(metric_dir + file, 'r', encoding="ISO-8859-1") as threshold_file, \
             open(meta_dir + "Pooled_thresholds.csv", 'a+', encoding="utf-8", newline='') as Pooled_threshold, \
             open(meta_dir + "deletedList.csv", 'a+', encoding="utf-8", newline='') as deletedList:
            reader = csv.reader(threshold_file)
            writer = csv.writer(Pooled_threshold)
            writer_deletedList = csv.writer(deletedList)
            # receives the first line of a file and convert to dict generator
            fieldnames = next(reader)



if __name__ == '__main__':

    s_time = time.time()
    pooled_meta()
    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ". This is end of PooledMeta.py!\n",
          "The execution time of PooledMeta.py script is ", execution_time)
