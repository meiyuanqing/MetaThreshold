#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/4/8
Time: 15:42
File: RocMethod.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

This script find out the cutoff of a metric value by maximizing the AUC value and ROC、BPP、MFM、GM methods.

References：
[1]  Bender, R. Quantitative risk assessment in epidemiological studies investigating threshold effects.
     Biometrical Journal, 41 (1999), 305-319.（计算VARL的SE（标准误）的参考文献P310）
[2]  Zhou, Y., et al. "An in-depth study of the potentially confounding effect of class size in fault prediction."
     ACM Trans. Softw. Eng. Methodol. (2014) 23(1): 1-51. (计算BPP、MFM(F1)值为阈值)
[3]  Shatnawi, R. (2018). Identifying Threshold Values of Change-Prone Modules.
     (计算sum(Sensitivity+Specificity)=sum(TPR+TNR)值为阈值)
"""
import time


def roc_threshold(working_dir="F:\\NJU\\MTmeta\\experiments\\supervised\\trainingData\\",
                  result_dir="F:\\NJU\\MTmeta\\experiments\\supervised\\",
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

    with open(working_dir + training_list) as l:
        lines = l.readlines()

    for line in lines:
        file = line.replace("\n", "")
        print('the file is ', file)

        # 分别处理每一个项目: f1取出要被处理的项目;
        #                  f2:用于存储每一个项目的阈值信息,f2用csv.writer写数据时没有newline参数，会多出一空行;
        #                  deletedList: 用于存储项目中某个度量logistic回归时，系数不显著或系数为零的度量及该项目名
        with open(working_directory + file, 'r', encoding="ISO-8859-1") as f1, \
                open(result_directory + "RocThreshold\\ROC_Thresholds.csv", 'a+', encoding="utf-8", newline='') as f2, \
                open(result_directory + "RocThreshold\\deletedList.csv", 'a+', encoding="utf-8") as deletedList:

            reader = csv.reader(f1)
            writer = csv.writer(f2)
            writer_deletedList = csv.writer(deletedList)
            # receives the first line of a file and convert to dict generator
            fieldnames = next(reader)
            # exclude the non metric fields (12 items) and metric values including undef and undefined (17 items)
            non_metric = ["relName", "className", "bug"]

            # metric_data stores the metric fields (102 items)
            def fun_1(m):
                return m if m not in non_metric else None

            metric_data = filter(fun_1, fieldnames)

            df = pd.read_csv(file)
            # drop all rows that have any NaN values,删除表中含有任何NaN的行,并重新设置行号
            df = df.dropna(axis=0, how='any', inplace=False).reset_index(drop=True)

            if os.path.getsize(result_directory + "RocThreshold\\ROC_Thresholds.csv") == 0:
                writer.writerow(["fileName", "metric", "Corr_metric_bug", "B_0", "B_0_pValue", "B_1", "B_1_pValue",
                                 "cov11", "cov12", "cov22", "BaseProbability_1",
                                 "auc_threshold", "auc_threshold_variance", "auc_max_value", "i_auc_max",
                                 "gm_threshold", "gm_threshold_variance", "gm_max_value", "i_gm_max",
                                 "bpp_threshold", "bpp_threshold_variance", "bpp_max_value", "i_bpp_max",
                                 "mfm_threshold", "mfm_threshold_variance", "f1_max_value", "i_f1_max",
                                 "roc_threshold", "roc_threshold_variance", "roc_max_value", "i_roc_max",
                                 "varl_threshold", "varl_threshold_variance"])

            if os.path.getsize(result_directory + "RocThreshold\\deletedList.csv") == 0:
                writer_deletedList.writerow(["fileName", "metric", "B_0_pValue", "B_0",
                                             "auc_max_value", "i_auc_max", "gm_max_value", "i_gm_max", "bpp_max_value",
                                             "i_bpp_max", "f1_max_value", "i_f1_max", "roc_max_value", "i_roc_max"])

            for metric in metric_data:

                print("the current file is ", file)
                print("the current metric is ", metric)

                # 由于bug中存储的是缺陷个数,转化为二进制存储,若x>2,则可预测bug为3个以上的阈值,其他类推
                df['bugBinary'] = df.bug.apply(lambda x: 1 if x > 0 else 0)

                # 依次用该度量的每一个值作为阈值计算出auc和GM,然后选择auc最大值的那个度量值作为阈值，即断点回归的cutoff
                # 同时计算BPP(Balanced-pf-pd)、MFM(F1)和ROC(Sensitivity+Specificity)=(TPR+TNR)值，
                # 分别定义存入五个值list,最大值和取最大值的下标值
                AUCs = []
                GMs = []
                BPPs = []
                MFMs = []
                ROCs = []

                auc_max_value = 0
                gm_max_value = 0
                bpp_max_value = 0
                f1_max_value = 0
                roc_max_value = 0

                i_auc_max = 0
                i_gm_max = 0
                i_bpp_max = 0
                i_f1_max = 0
                i_roc_max = 0

                # 判断每个度量与bug之间的关系,因为该关系会影响到断点回归时,相关系数大于零,则LATE估计值大于零,反之,则LATE估计值小于零
                Corr_metric_bug = df.loc[:, [metric, 'bug']].corr('spearman')

                # the i value in this loop, is the subscript value in the list of AUCs, GMs etc.
                for i in range(len(df)):

                    t = df.loc[i, metric]
                    if Corr_metric_bug[metric][1] < 0:
                        df['predictBinary'] = df[metric].apply(lambda x: 1 if x <= t else 0)
                    else:
                        df['predictBinary'] = df[metric].apply(lambda x: 1 if x >= t else 0)

                    # confusion_matrix()函数中需要给出label, 0和1，否则该函数算不出TP,因为不知道哪个标签是poistive.
                    c_matrix = confusion_matrix(df["bugBinary"], df['predictBinary'], labels=[0, 1])
                    tn, fp, fn, tp = c_matrix.ravel()

                    if (tn + fp) == 0:
                        tnr_value = 0
                    else:
                        tnr_value = tn / (tn + fp)

                    if (fp + tn) == 0:
                        fpr = 0
                    else:
                        fpr = fp / (fp + tn)
                    # fpr, tpr, thresholds = roc_curve(df['bugBinary'], df['predictBinary'])
                    # AUC = auc(fpr, tpr)
                    auc_value = roc_auc_score(df['bugBinary'], df['predictBinary'])
                    recall_value = recall_score(df['bugBinary'], df['predictBinary'], labels=[0, 1])
                    precision_value = precision_score(df['bugBinary'], df['predictBinary'], labels=[0, 1])
                    f1_value = f1_score(df['bugBinary'], df['predictBinary'], labels=[0, 1])

                    gm_value = (recall_value * tnr_value) ** 0.5
                    pfr = recall_value
                    pdr = fpr  # fp / (fp + tn)
                    bpp_value = 1 - (((0 - pfr) ** 2 + (1 - pdr) ** 2) * 0.5) ** 0.5
                    roc_value = recall_value + tnr_value

                    AUCs.append(auc_value)
                    GMs.append(gm_value)
                    BPPs.append(bpp_value)
                    MFMs.append(f1_value)
                    ROCs.append(roc_value)

                    # 求出上述五个list中最大值，及对应的i值，可能会有几个值相同，且为最大值，则取第一次找到那个值(i)为阈值
                    if auc_value > auc_max_value:
                        auc_max_value = auc_value
                        i_auc_max = i

                    if gm_value > gm_max_value:
                        gm_max_value = gm_value
                        i_gm_max = i

                    if bpp_value > bpp_max_value:
                        bpp_max_value = bpp_value
                        i_bpp_max = i

                    if f1_value > f1_max_value:
                        f1_max_value = f1_value
                        i_f1_max = i

                    if roc_value > roc_max_value:
                        roc_max_value = roc_value
                        i_roc_max = i

                print("auc_max_value is ", auc_max_value)
                print("gm_max_value is ", gm_max_value)
                print("bpp_max_value is ", bpp_max_value)
                print("f1_max_value is ", f1_max_value)
                print("roc_max_value is ", roc_max_value)

                print("i_auc_max is ", i_auc_max)
                print("i_gm_max is ", i_gm_max)
                print("i_bpp_max is ", i_bpp_max)
                print("i_f1_max is ", i_f1_max)
                print("i_roc_max is ", i_roc_max)

                df['intercept'] = 1.0

                # 通过 statsmodels.api 逻辑回归分类; 指定作为训练变量的列，不含目标列`bug`
                logit = sm.Logit(df['bugBinary'], df.loc[:, [metric, 'intercept']])
                # 拟合模型,disp=1 用于显示结果
                result = logit.fit(method='bfgs', disp=0)
                print(result.summary())

                pValueLogit = result.pvalues
                if pValueLogit[0] > 0.05:  # 自变量前的系数
                    writer_deletedList.writerow(
                        [file, metric, pValueLogit[0], B[0], auc_max_value, i_auc_max, gm_max_value,
                         i_gm_max, bpp_max_value, i_bpp_max, f1_max_value, i_f1_max, roc_max_value,
                         i_roc_max])
                    continue  # 若训练数据LOGIT回归系数的P值大于0.05，放弃该数据。

                B = result.params  # logit回归系数
                if B[0] == 0:  # 自变量前的系数
                    writer_deletedList.writerow(
                        [file, metric, pValueLogit[0], B[0], auc_max_value, i_auc_max, gm_max_value,
                         i_gm_max, bpp_max_value, i_bpp_max, f1_max_value, i_f1_max, roc_max_value,
                         i_roc_max])
                    continue  # 若训练数据LOGIT回归系数等于0，放弃该数据。

                # 计算auc阈值及标准差,包括其他四个类型阈值
                auc_threshold = df.loc[i_auc_max, metric]
                gm_threshold = df.loc[i_gm_max, metric]
                bpp_threshold = df.loc[i_bpp_max, metric]
                mfm_threshold = df.loc[i_f1_max, metric]
                roc_threshold = df.loc[i_roc_max, metric]
                # 计算LOGIT回归系数矩阵的协方差矩阵,因为计算aucThreshold的标准差要用到,见参考文献[1],
                # 此处借鉴VARL方法，本质上VARL也是度量值中的一个
                cov = result.cov_params()
                cov11 = cov.iloc[0, 0]
                cov12 = cov.iloc[0, 1]
                cov22 = cov.iloc[1, 1]
                auc_threshold_se = ((cov.iloc[0, 0] + 2 * auc_threshold * cov.iloc[0, 1]
                                     + auc_threshold * auc_threshold * cov.iloc[1, 1]) ** 0.5) / B[0]
                auc_threshold_variance = auc_threshold_se ** 2

                gm_threshold_se = ((cov.iloc[0, 0] + 2 * gm_threshold * cov.iloc[0, 1]
                                    + gm_threshold * gm_threshold * cov.iloc[1, 1]) ** 0.5) / B[0]
                gm_threshold_variance = gm_threshold_se ** 2

                bpp_threshold_se = ((cov.iloc[0, 0] + 2 * bpp_threshold * cov.iloc[0, 1]
                                     + bpp_threshold * bpp_threshold * cov.iloc[1, 1]) ** 0.5) / B[0]
                bpp_threshold_variance = bpp_threshold_se ** 2

                mfm_threshold_se = ((cov.iloc[0, 0] + 2 * mfm_threshold * cov.iloc[0, 1]
                                     + mfm_threshold * mfm_threshold * cov.iloc[1, 1]) ** 0.5) / B[0]
                mfm_threshold_variance = mfm_threshold_se ** 2

                roc_threshold_se = ((cov.iloc[0, 0] + 2 * roc_threshold * cov.iloc[0, 1]
                                     + roc_threshold * roc_threshold * cov.iloc[1, 1]) ** 0.5) / B[0]
                roc_threshold_variance = roc_threshold_se ** 2

                # 求VARL作为阈值，此处未用10折交叉验证的方法 VARL.threshold = (log(Porbability[1]/Porbability[2])-B[1])/B[2]
                valueOfbugBinary = df["bugBinary"].value_counts()  # 0 和 1 的各自的个数
                print("the value of valueOfbugBinary[0] is ", valueOfbugBinary[0])
                print("the value of valueOfbugBinary[1] is ", valueOfbugBinary[1])

                # 用缺陷为大于0的模块数占所有模块之比
                BaseProbability_1 = valueOfbugBinary[1] / (valueOfbugBinary[0] + valueOfbugBinary[1])

                # 计算VARL阈值及标准差
                varl_threshold = (np.log(BaseProbability_1 / (1 - BaseProbability_1)) - B[1]) / B[0]
                varl_threshold_se = ((cov.iloc[0, 0] + 2 * varl_threshold * cov.iloc[0, 1]
                                      + varl_threshold * varl_threshold * cov.iloc[1, 1]) ** 0.5) / B[0]
                varl_threshold_variance = varl_threshold_se ** 2

                # 输出每一度量的结果
                writer.writerow([file, metric, Corr_metric_bug[metric][1], B[0], pValueLogit[0], B[1], pValueLogit[1],
                                 cov11, cov12, cov22, BaseProbability_1,
                                 auc_threshold, auc_threshold_variance, auc_max_value, i_auc_max,
                                 gm_threshold, gm_threshold_variance, gm_max_value, i_gm_max,
                                 bpp_threshold, bpp_threshold_variance, bpp_max_value, i_bpp_max,
                                 mfm_threshold, mfm_threshold_variance, f1_max_value, i_f1_max,
                                 roc_threshold, roc_threshold_variance, roc_max_value, i_roc_max,
                                 varl_threshold, varl_threshold_variance])

        # break

if __name__ == '__main__':

    s_time = time.time()
    roc_threshold()
    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ". This is end of RocMethod.py!\n",
          "The execution time of Bender.py script is ", execution_time)
