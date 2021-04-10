#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/4/10
Time: 13:57
File: PearsonMeta.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

Do Random Meta-Analysis For Pearson, including sensitive analysis by trim and fill method.
"""
import time


def pearson_meta(t_dir="F:\\NJU\\MTmeta\\experiments\\supervised\\PearsonEffect\\",
                 m_dir="F:\\NJU\\MTmeta\\experiments\\supervised\\PearsonEffect\\"):
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

    # inverse Fisher Transformation
    def inverse_Fisher_Z(fisher_Z):
        rp = (np.exp(2 * fisher_Z) - 1) / (np.exp(2 * fisher_Z) + 1)
        return rp

    def getEstimatedK0(effectSizeArray, mean):
        centeredEffectSizeArray = []
        absoluteCenteredEffectSizeArray = []
        size = len(effectSizeArray)
        for i in range(size):
            centeredEffectSizeArray[i] = effectSizeArray[i] - mean
            absoluteCenteredEffectSizeArray[i] = np.abs(effectSizeArray[i] - mean)
        sortedArray = sorted(absoluteCenteredEffectSizeArray)
        rank = pd.DataFrame()
        rank[sortedArray[0]] = 1
        initialRankValue = 1
        predValue = sortedArray[0]
        for i in range(size):
            if sortedArray[i] > predValue:
                predValue = sortedArray[i]
                initialRankValue += 1
            rank[sortedArray[i]] = initialRankValue
        finalRank = []
        for i in range(size):
            if centeredEffectSizeArray[i] < 0:
                finalRank[i] = (-1) * rank[absoluteCenteredEffectSizeArray[i]]
            else:
                finalRank[i] = rank[absoluteCenteredEffectSizeArray[i]]
        gamma = finalRank[size - 1] + finalRank[0]
        SumPositiveRank = 0
        for i in range(size):
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
        size = len(effect_size)
        # 检查是否需要切换方向，因为trim and fill方法假设miss most negative的研究
        flipFunnel = 0
        metaAnalysisForFlip = fixed_effect_meta_analysis(effect_size, variance)
        meanForFlip = metaAnalysisForFlip["fixedMean"]

        tempSorted = sorted(effect_size)
        min = tempSorted[0] - meanForFlip
        max = tempSorted[-1] - meanForFlip

        if np.abs(min) > np.abs(max):
            flipFunnel = 1
            for i in range(len(effect_size)):
                effect_size[i] = (-1) * effect_size[i]

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

    metric_dir = t_dir
    meta_dir = m_dir
    os.chdir(metric_dir)
    print(os.getcwd())

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 5000)

    # read_csv(path, keep_default_na=False, na_values=[""])  只有一个空字段将被识别为NaN
    df = pd.read_csv(meta_dir + "Pearson_effects.csv", keep_default_na=False, na_values=[""])

    metric_names = sorted(set(df.metric.values.tolist()))
    print("the metric_names are ", df.columns.values.tolist())
    print("the metric_names are ", metric_names)
    print("the len metric_names are ", len(metric_names))
    k = 0
    for metric in metric_names:

        print("the current metric is ", metric)

        FisherZ_effect_size = df[df["metric"] == metric].loc[:, "Fisher_Z"].astype(float)
        print("the FisherZ_effect_size items are ", FisherZ_effect_size)
        print("the type FisherZ_effect_size items are ", type(FisherZ_effect_size))
        print("the len of FisherZ_effect_size items is ", len(FisherZ_effect_size))

        FisherZ_variance = df[df["metric"] == metric].loc[:, "Fisher_Z_variance"]
        print("the threshold_variance items are ", FisherZ_variance)
        print("the type threshold_variance items are ", type(FisherZ_variance))
        print("the len of threshold_variance items is ", len(FisherZ_variance))

        metaThreshold = pd.DataFrame()
        metaThreshold['EffectSize'] = FisherZ_effect_size
        metaThreshold['Variance'] = FisherZ_variance
        try:
            resultMetaAnalysis = random_effect_meta_analysis(
                np.array(metaThreshold[metaThreshold["EffectSize"] > 0].loc[:, "EffectSize"]),
                np.array(metaThreshold[metaThreshold["EffectSize"] > 0].loc[:, "Variance"]))

            # d["LL_CI"] = randomMean - 1.96 * randomStdError  # The 95% lower limits for the summary effect
            # d["UL_CI"] = randomMean + 1.96 * randomStdError  # The 95% upper limits for the summary effect
            meta_stdError = (inverse_Fisher_Z(resultMetaAnalysis["UL_CI"])
                             - inverse_Fisher_Z(resultMetaAnalysis["LL_CI"])) / (1.96 * 2)
            with open(meta_dir + "Pearson_effects_meta.csv", 'a+', encoding="utf-8", newline='') as f:
                writer_f = csv.writer(f)
                if os.path.getsize(meta_dir + "Pearson_effects_meta.csv") == 0:
                    writer_f.writerow(
                        ["metric", "Pearson_effects_meta", "Pearson_effects_meta_stdError", "LL_CI", "UL_CI",
                         "ZValue", "pValue_Z", "Q", "df", "pValue_Q", "I2", "tau", "number_of_effect_size"])
                writer_f.writerow([metric, inverse_Fisher_Z(resultMetaAnalysis["mean"]), meta_stdError,
                                   inverse_Fisher_Z(resultMetaAnalysis["LL_CI"]),
                                   inverse_Fisher_Z(resultMetaAnalysis["UL_CI"]),
                                   resultMetaAnalysis["ZValue"], resultMetaAnalysis["pValue_Z"],
                                   resultMetaAnalysis["Q"], resultMetaAnalysis["df"], resultMetaAnalysis["pValue_Q"],
                                   resultMetaAnalysis["I2"], resultMetaAnalysis["tau"], len(FisherZ_effect_size)])

        except Exception as err1:
            print(err1)

        k += 1
        # break


if __name__ == '__main__':

    s_time = time.time()
    pearson_meta()
    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ". This is end of Pearson.py!\n",
          "The execution time of Pearson.py script is ", execution_time)
