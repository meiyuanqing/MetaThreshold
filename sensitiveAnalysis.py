#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/4/11
Time: 9:25
File: sensitiveAnalysis.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

Sensitive analysis by trim and fill method.

Reference:
[1] M.Borenstein, L.V. Hedges, J.P.T. Higgins, H.R. Rothstein. Introduction to meta-analysis, John Wiley & Sons, 2009;

"""
import time
import numpy as np
import pandas as pd
from scipy.stats import norm  # norm.cdf() the cumulative normal distribution function in Python
from scipy import stats  # 根据卡方分布计算p值: p_value=1.0-stats.chi2.cdf(chisquare,freedom_degree)


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

    print("upcaseQ = ", Q, "\n")
    print("upcaseT2 = ", T2, "\n")

    if T2 < 0:
        T2 = 0  # 20210411，若T2小于0，取0,   M.Borenstein[2009] P114
        # T2 = (- 1) * T2  # 20190719，若T2小于0，取相反数

    for i in range(study_number):
        random_weight[i] = 1 / (variance[i] + T2)  # random_weight 随机模型对应的权值
        print("the ", i, " variance is ", variance[i], " and the random_weight is ", random_weight[i])

    for i in range(study_number):
        sum_Wistar = sum_Wistar + random_weight[i]
        sum_WistarYi = sum_WistarYi + random_weight[i] * effect_size[i]
        print("the ", i, " effect_size is ", effect_size[i], " and the sum_Wistar is ", sum_Wistar,
              " and the sum_WistarYi is ", sum_WistarYi)

    print("sum_Wistar = ", sum_Wistar, "\n")
    print("sum_WistarYi = ", sum_WistarYi, "\n")

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


def getEstimatedK0(effectSizeArray, mean):
    centeredEffectSizeArray = []
    absoluteCenteredEffectSizeArray = []
    size = len(effectSizeArray)
    for i in range(size):
        centeredEffectSizeArray.append(effectSizeArray[i] - mean)
        absoluteCenteredEffectSizeArray.append(np.abs(effectSizeArray[i] - mean))
    sortedArray = sorted(absoluteCenteredEffectSizeArray)
    rank = {sortedArray[0]: 1}  # return a dict
    initialRankValue = 1
    predValue = sortedArray[0]
    for i in range(size):
        if sortedArray[i] > predValue:
            predValue = sortedArray[i]
            initialRankValue += 1
        rank[sortedArray[i]] = initialRankValue
        # print("the sortedArray[", i, "] is ", sortedArray[i])
        # print("the rank[sortedArray[", i, "]] is ", rank[sortedArray[i]])
    finalRank = []
    for i in range(size):
        if centeredEffectSizeArray[i] < 0:
            finalRank.append((-1) * rank[absoluteCenteredEffectSizeArray[i]])
        else:
            finalRank.append(rank[absoluteCenteredEffectSizeArray[i]])
    gamma = finalRank[size - 1] + finalRank[0]
    SumPositiveRank = 0
    for i in range(size):
        # print("the type of finalRank is ", type(finalRank[i]))
        # print("the repr of finalRank is ", repr(finalRank[i]))
        # print("the finalRank is ", finalRank[i])
        if finalRank[i] < 0:
            continue
        SumPositiveRank = SumPositiveRank + finalRank[i]
    R0 = int(gamma + 0.5) - 1
    temp = (4 * SumPositiveRank - size * (size + 1)) / (2 * size - 1)
    L0 = int(temp + 0.5)
    if R0 < 0:
        R0 = 0
    if L0 < 0:
        L0 = 0
    return R0, L0


# Duval and Tweedie's trim and fill method
def trimAndFill(effect_size, variance, isAUC):
    effectSizeArray = effect_size
    varianceArray = variance
    size = len(effect_size)
    # 检查是否需要切换方向，因为trim and fill方法假设miss most negative的研究
    flipFunnel = 0
    metaAnalysisForFlip = fixed_effect_meta_analysis(effectSizeArray, varianceArray)
    meanForFlip = metaAnalysisForFlip["fixedMean"]

    tempSorted = sorted(effectSizeArray)
    min = tempSorted[0] - meanForFlip
    max = tempSorted[-1] - meanForFlip

    if np.abs(min) > np.abs(max):
        flipFunnel = 1
        for i in range(size):
            effectSizeArray[i] = (-1) * effectSizeArray[i]

    # 按effect size排序
    merge = []
    for i in range(size):
        merge.append([effect_size[i], variance[i]])
    sortedMerge = sorted(merge)
    OrignalEffectSizeArray = []
    OrignalVarianceArray = []
    for i in range(len(sortedMerge)):
        OrignalEffectSizeArray.append(sortedMerge[i][0])
        OrignalVarianceArray.append(sortedMerge[i][1])
    # 迭代算法，估算k0
    metaAnalysisResult = fixed_effect_meta_analysis(OrignalEffectSizeArray, OrignalVarianceArray)
    mean = metaAnalysisResult["fixedMean"]
    RL = getEstimatedK0(OrignalEffectSizeArray, mean)
    R0 = RL[0]
    L0 = RL[1]
    k0 = L0    # 默认的情况利用L0来估算k0
    if (k0 == 0) or (k0 > size):
        result = random_effect_meta_analysis(effect_size, variance)
        result["k0"] = k0
        return result
    trimmedMean = mean
    change = 1
    count = 0
    while change and (size - k0) > 2 and (count < 1000):
        count += 1
        upperBound = size - k0 - 1
        trimmedEffectSizeArray = []
        trimmedVarianceArray = []
        for i in range(upperBound):
            trimmedEffectSizeArray.append(OrignalEffectSizeArray[i])
            trimmedVarianceArray.append(OrignalVarianceArray[i])
        trimmedMetaAnalysisResult = fixed_effect_meta_analysis(trimmedEffectSizeArray, trimmedVarianceArray)
        trimmedMean = trimmedMetaAnalysisResult["fixedMean"]
        trimmedR0_L0 = getEstimatedK0(OrignalEffectSizeArray, trimmedMean)
        trimmedR0 = trimmedR0_L0[0]
        trimmedL0 = trimmedR0_L0[1]
        k1 = trimmedL0
        if k1 == k0:
            change = 0
        k0 = k1
    filledEffectSizeArray = []
    filledVarianceArray = []

    for j in range(k0):
        imputedEffectSize = 2 * trimmedMean - OrignalEffectSizeArray[size - j - 1]
        imputedVariance = OrignalVarianceArray[size - j - 1]
        filledEffectSizeArray.append(imputedEffectSize)
        filledVarianceArray.append(imputedVariance)
    fullEffectSizeArray = filledEffectSizeArray
    fullVarianceArray = filledVarianceArray
    fullEffectSizeArray.extend(OrignalEffectSizeArray)
    fullVarianceArray.extend(OrignalVarianceArray)
    if flipFunnel:
        newSize = len(fullEffectSizeArray)
        for i in range(newSize):
            fullEffectSizeArray[i] = -1 * fullEffectSizeArray[i]

    if isAUC:
        # AUC应该在0到1之间，否则有错
        filteredFullEffectSizeArray = []
        filteredFullVarianceArray = []
        for i in range(len(fullEffectSizeArray)):
            if fullEffectSizeArray[i] < 0:
                continue
            if fullEffectSizeArray[i] > 1:
                continue
            filteredFullEffectSizeArray.append(fullEffectSizeArray[i])
            filteredFullVarianceArray.append(fullVarianceArray[i])
        result = random_effect_meta_analysis(filteredFullEffectSizeArray, filteredFullVarianceArray)
        finalk0 = len(filteredFullEffectSizeArray) - len(OrignalEffectSizeArray)
    else:
        result = random_effect_meta_analysis(fullEffectSizeArray, fullVarianceArray)
        finalk0 = len(fullEffectSizeArray) - len(OrignalEffectSizeArray)
    result["k0"] = finalk0
    result["flipFunnel"] = flipFunnel
    return result

if __name__ == '__main__':

    s_time = time.time()
    a = [1.26, 0.97, 0.79, 1.18, 1.2, 1.65, 1.45, 1.06, 1.19, 1.16, 1.66, 1.23, 1.11, 1.08, 1.03, 1.55, 1.52, 0.75,
         1.1, 2.13, 1.19, 1.62, 2.01, 1.6, 2.16, 1.66, 0.74, 0.8, 1.03, 2.07, 1.2, 2.34, 2.27, 0.79, 2.55, 1.52, 2.02]
    b = [1 / 13.63, 1 / 10.9, 1 / 8.48, 1 / 7.28, 1 / 6.11, 1 / 4.22, 1 / 4.14, 1 / 4.06, 1 / 3.77, 1 / 3.76, 1 / 3.43,
         1 / 3, 1 / 2.06, 1 / 1.92, 1 / 1.91, 1 / 1.78, 1 / 1.72, 1 / 1.72, 1 / 1.59, 1 / 1.54, 1 / 1.53, 1 / 1.53,
         1 / 1.39, 1 / 1.2, 1 / 1.1, 1 / 0.78, 1 / 0.76, 1 / 0.71, 1 / 0.63, 1 / 0.6, 1 / 0.59, 1 / 0.47, 1 / 0.43,
         1 / 0.4, 1 / 0.34, 1 / 0.28, 1 / 0.25]
    print("the len of a is ", len(a))
    print("the len of b is ", len(b))
    meta = fixed_effect_meta_analysis(a, b)
    K0 = getEstimatedK0(a, meta['fixedMean'])
    print(meta['fixedMean'], "\n", meta['fixedStdError'], "\n", K0[0], "\n", K0[1], "\n")
    meta_1 = trimAndFill(a, b, 0)
    print(meta_1['mean'], "\n", meta_1['stdError'], "\n", meta_1['pValue_Z'], "\n", meta_1['k0'],
          "\n", meta_1['flipFunnel'], "\n")
    meta_2 = random_effect_meta_analysis(a, b)
    print(meta_2['mean'], "\n", meta_2['stdError'], "\n", meta_2['pValue_Z'], "\n")

    print("############################################")

    c = [0.095, 0.277, 0.367, 0.664, 0.462, 0.185]
    d = [0.033, 0.031, 0.05, 0.011, 0.043, 0.023]
    meta_3 = fixed_effect_meta_analysis(c, d)
    print("the fixedEffectMetaAnalysis is ", meta_3['fixedMean'], "\n", meta_3['fixedStdError'], "\n")
    meta_4 = random_effect_meta_analysis(c, d)
    print("the randomEffectMetaAnalysis is ", meta_4['mean'], "\n", meta_4['stdError'], "\n", meta_4['pValue_Z'], "\n")

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ". This is end of sensitiveAnalysis.py!\n",
          "The execution time of sensitiveAnalysis.py script is ", execution_time)